"""Manual GPU gate. Interleave an identified reference checkout with the current code on the same GPU."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from types import SimpleNamespace

from benchmark.memory_study import CONFIGS, ROOT, run_case
from benchmark.resnet_before_after import _import

STEP_LIMIT = 1.10
GEMM_LIMIT = 1.20


def gemm_ratio(row):
    """register/cuBLAS of one GEMM row: SM cycles under Nsight Compute when present, else CUDA-event time."""
    keys = ("register_cycles", "cublas_cycles") if "register_cycles" in row else ("register_ms", "cublas_ms")
    if not all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) and row[k] > 0. for k in keys):
        raise ValueError("GEMM timings must be finite and positive")
    return row[keys[0]] / row[keys[1]]


def violations(step_ratios, gemm_rows):
    from benchmark.gemm_engines import GEMM_MODELS

    if set(step_ratios) != {name for name, *_ in CONFIGS} or not all(
            math.isfinite(value) and value > 0. for value in step_ratios.values()):
        raise ValueError("expected finite positive step ratios for every fixed ResNet configuration")
    expected = {(name, part, layout) for name, *_ in GEMM_MODELS
                for part, layout in (("forward", "NT"), ("dW", "TN"), ("dX", "NN"))}
    if {(r["model"], r["part"], r["layout"]) for r in gemm_rows} != expected:
        raise ValueError("both networks need forward, dW and dX GEMM measurements")
    keys = {(r["model"], r["part"], r["layout"], r["m"], r["n"], r["k"]) for r in gemm_rows}
    if len(keys) != len(gemm_rows):
        raise ValueError("duplicate GEMM shape within a measurement group")
    failed = [f"{name} step {ratio:.3f}×" for name, ratio in step_ratios.items() if ratio > STEP_LIMIT]
    for row in gemm_rows:
        ratio = gemm_ratio(row)
        if ratio > GEMM_LIMIT:
            failed.append(f"{row['model']} {row['part']} {row['layout']} {row['m']}×{row['n']}×{row['k']} {ratio:.3f}×")
    return failed


def operator_worker(args):
    _import(args.tree)
    from benchmark.cuda_ops import Operator, cases
    result = []
    for case in cases("performance", 1):
        op = Operator(case, "cuda_native_register")
        for _ in range(args.warmup):
            op.combined()
        for phase in ("forward", "backward", "combined"):
            events = [op.timed(phase)[0] for _ in range(args.steps)]
            result.append({"case": case["id"], "kind": case["kind"], "phase": phase,
                           "median_ms": statistics.median(events), "events_ms": events})
    return result


def gemm_worker(args):
    _import(args.tree)
    from benchmark.gemm_engines import gemm_part
    run = SimpleNamespace(results={}, timings=[], notes=[])
    gemm_part(args, run)
    return run.results["gemm"]


def measure_part(tree, args, part):
    cmd = [sys.executable, "-X", "utf8", "-m", "benchmark.perf_gate", "--worker", part, "--tree", str(tree),
           "--rounds", str(args.rounds), "--warmup", str(args.warmup), "--steps", str(args.steps)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", timeout=1800)
    if proc.returncode:
        raise RuntimeError(proc.stdout[-1000:] + proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_gate(args):
    from apps.autopilot.files import write_json
    from apps.autopilot.provenance import code_state
    from apps.common.environment import environment
    from apps.common.gpu_lock import GpuLock
    from MyFlows.monitoring.gpu import GpuSampler

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    trees = {"baseline": args.reference_tree.resolve(strict=True), "current": ROOT}
    states = {side: code_state(tree) for side, tree in trees.items()}
    with GpuLock(label="manual GPU performance gate", run_dir=out), GpuSampler() as sampler:
        devices = sampler.sample().get("gpus") or []
        if not devices or not devices[0].get("uuid"):
            raise RuntimeError("a GPU identity is required to select a performance baseline")
        gpu_key = hashlib.sha256(devices[0]["uuid"].encode()).hexdigest()[:16]
        protocol = {"rounds": args.rounds, "steps": args.steps, "warmup": args.warmup, "configs": CONFIGS,
                    "step_limit": STEP_LIMIT, "gemm_limit": GEMM_LIMIT, "gemm_timer": args.gemm_timer,
                    "gemm_reps": args.gemm_reps}
        archive = args.baselines/gpu_key/states["baseline"]["code_hash"]/out.name
        archive.mkdir(parents=True, exist_ok=False)
        write_json(out/"manifest.json", {"schema_version": 1, "environment": environment(), "trees": states,
                                         "gpu_key": gpu_key, "protocol": protocol})
        steps, operators = [], []
        for rep in range(args.rounds):
            sides = list(trees) if rep % 2 == 0 else list(reversed(trees))
            for name, stem, width, batch in CONFIGS:
                for side in sides:
                    row = run_case(trees[side], args, stem, width, batch)
                    steps.append({"name": name, "round": rep, "side": side, **row})
                    write_json(out/"steps.json", steps)
            for side in sides:
                operators.extend({**r, "side": side, "round": rep} for r in measure_part(trees[side], args, "operators"))
                write_json(out/"operators.json", operators)
        if args.gemm_timer == "ncu":
            from benchmark.gemm_ncu import compare, list_shapes
            gemms, gemm_meta = compare(list_shapes(), out/"gemm-ncu", args.gemm_reps)
        else:
            gemms, gemm_meta = measure_part(ROOT, args, "gemms"), {}
        write_json(out/"gemms.json", gemms)
        write_json(archive/"baseline.json", {"gpu_key": gpu_key, "source": states["baseline"], "protocol": protocol,
                                             "steps": [r for r in steps if r["side"] == "baseline"],
                                             "operators": [r for r in operators if r["side"] == "baseline"]})
    med = statistics.median
    ratios = {}
    lines = ["# GPU 性能门槛", "", f"设备 {devices[0]['name']}。同一会话内交替运行两套代码，每套 {args.rounds} 轮。"
             "每轮先预热，再取同步训练步时中位数。", "",
             "| ResNet18 | 基线 ms | 当前 ms | 比值 | 门槛 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, *_ in CONFIGS:
        groups = [[r["median_ms"] for r in steps if r["side"] == side and r["name"] == name] for side in trees]
        a,b = [med(g) for g in groups]
        ratios[name] = b/a
        lines.append(f"| {name} | {a:.3f} ± {statistics.stdev(groups[0]):.3f} | "
                     f"{b:.3f} ± {statistics.stdev(groups[1]):.3f} | {b/a:.3f} | {STEP_LIMIT:.2f} |")
    lines += ["", "| 单算子 | 阶段 | 基线 ms | 当前 ms | 比值 |", "| --- | --- | ---: | ---: | ---: |"]
    for case, phase in sorted({(r["case"], r["phase"]) for r in operators}):
        a, b = [med(r["median_ms"] for r in operators if r["side"] == side and r["case"] == case and r["phase"] == phase)
                for side in trees]
        lines.append(f"| {case} | {phase} | {a:.4f} | {b:.4f} | {b/a:.3f} |")
    failed = violations(ratios, gemms)
    if args.gemm_timer == "ncu":
        method = (f"在 Nsight Compute 下比较 register 与 cuBLAS：GPC 时钟锁定在基频（实测 SM {gemm_meta['sm_ghz_median']:.3f} GHz），"
                  f"每个 kernel 前清空缓存，每次调用的 GEMM 与 split-K 归约 kernel 按 NVTX 区间求和，两引擎交替 "
                  f"{args.gemm_reps} 次，按 sm__cycles_elapsed.max 的中位数求比值")
    else:
        method = "按完整 GEMM 的 CUDA Event 耗时比较 register 与 cuBLAS，包含 split-K 归约"
    weighted = (sum(r["count"] * r.get("register_cycles", r["register_ms"]) for r in gemms)
                / sum(r["count"] * r.get("cublas_cycles", r["cublas_ms"]) for r in gemms))
    lines += ["", f"逐形状检查覆盖 {len(gemms)} 项卷积 GEMM，{method}。每项上限为 {GEMM_LIMIT:.2f} 倍。"
              f"按一步内出现次数加权的合计比值为 {weighted:.3f}。", "",
              "| 模型 | 部分 | 布局 | M×N×K | tile / 切分 | 比值 |", "| --- | --- | --- | --- | --- | ---: |"]
    for r in sorted(gemms, key=gemm_ratio, reverse=True)[:8]:
        lines.append(f"| {r['model']} | {r['part']} | {r['layout']} | {r['m']}×{r['n']}×{r['k']} | "
                     f"{r['tile']} / {r['slices']} | {gemm_ratio(r):.3f} |")
    lines += ["", "## 超限项目", ""]
    lines += [f"- {v}" for v in failed] if failed else ["无。"]
    lines += ["", "原始比较在本目录的 steps.json、operators.json、gemms.json，本机基线按 GPU 标识和代码哈希保存。"]
    write_json(out/"result.json", {"passed": not failed, "step_ratios": ratios, "gemm_count": len(gemms),
                                   "gemm_weighted_ratio": weighted, "gemm_timer": args.gemm_timer, "gemm": gemm_meta,
                                   "violations": failed, "baseline": str(archive)})
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return int(bool(failed))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-tree", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--baselines", type=Path, default=ROOT/"runs/perf-baselines")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--gemm-timer", choices=("ncu", "events"), default="ncu",
                    help="per-shape GEMM timer: Nsight Compute cycles at locked base clocks, or CUDA events")
    ap.add_argument("--gemm-reps", type=int, default=7, help="alternating calls per engine under Nsight Compute")
    ap.add_argument("--worker", choices=("operators", "gemms"), help=argparse.SUPPRESS)
    ap.add_argument("--tree", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    args.graph_opt = False
    if args.worker:
        print(json.dumps(operator_worker(args) if args.worker == "operators" else gemm_worker(args)))
        return 0
    if not args.reference_tree or not args.out or args.rounds < 2 or min(args.steps, args.warmup) < 1 \
            or args.gemm_reps < 3:
        ap.error("supply --reference-tree, --out, at least two rounds, positive step counts and three GEMM repeats")
    return run_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
