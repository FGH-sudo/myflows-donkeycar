"""Kernel-level GEMM comparison of the register engine and cuBLAS under Nsight Compute.

CUDA events on the laptop GPU move by up to 2x between rounds of the same run as the power limit and the
clocks change, so a per-shape register/cuBLAS ratio taken from events can flip between runs. Nsight
Compute locks the GPC clock to base and flushes the caches before every kernel. Each GEMM call is wrapped
in an NVTX range; the kernels inside one range (the GEMM and any split-K reduction) are summed, and the
median over alternating repeats is compared by ``sm__cycles_elapsed.max`` (stage-3 plan, section 2.3).

  python -m benchmark.gemm_ncu --out runs/stage3/gemm-ncu-<batch>
  python -m benchmark.gemm_ncu --out <dir> --shapes TN:128:64:2560 --candidates    # every register plan
"""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

from benchmark.resnet_before_after import ROOT

METRICS = ("gpu__time_duration.sum", "sm__cycles_elapsed.max")


def ncu_options(cache="all"):
    """Locked base clocks; ``cache="all"`` flushes L1/L2 before every kernel, ``"none"`` keeps them warm."""
    if cache not in ("all", "none"):
        raise ValueError("cache control is 'all' or 'none'")
    return ("--clock-control", "base", "--cache-control", cache)


def _number(text):
    return float(str(text).replace(",", ""))


def gate_shapes():
    """Every distinct convolution GEMM of one training step of the gate's two ResNet18 networks."""
    from benchmark.gemm_engines import GEMM_MODELS, conv_gemms

    rows = []
    for model, stem, width, batch in GEMM_MODELS:
        for (part, layout, m, n, k), count in sorted(conv_gemms(stem, width, batch).items(),
                                                       key=lambda item: (item[0][0], -item[0][2] * item[0][3])):
            rows.append({"model": model, "part": part, "layout": layout, "m": m, "n": n, "k": k, "count": count})
    return rows


def parse_csv(text):
    """``{(job, label): [{"rep", "ns", "cycles", "kernels"}, ...]}`` from an NVTX-renamed raw CSV page.

    Every profiled kernel must sit inside a ``<job>:<label>:<rep>`` range; the kernels of one range are one
    call and are summed.
    """
    rows = csv.DictReader(io.StringIO("\n".join(line for line in text.splitlines() if line.startswith('"'))))
    calls = {}
    for row in rows:
        if not str(row.get("ID", "")).strip().isdigit():
            continue  # the units row
        name = row["Kernel Name"]
        if "/" not in name:
            raise ValueError(f"kernel outside a labelled NVTX range: {name}")
        label, kernel = name.split("/", 1)
        parts = label.rsplit(":", 2)
        if len(parts) != 3 or not parts[2].isdigit():
            raise ValueError(f"unexpected NVTX range name {label!r}")
        key = (parts[0], parts[1], int(parts[2]))
        call = calls.setdefault(key, {"rep": key[2], "ns": 0., "cycles": 0., "kernels": [], "first_id": int(row["ID"])})
        call["ns"] += _number(row["gpu__time_duration.sum"])
        call["cycles"] += _number(row["sm__cycles_elapsed.max"])
        call["kernels"].append(kernel)
    grouped = {}
    for (job, label, _), call in sorted(calls.items(), key=lambda item: item[1]["first_id"]):
        grouped.setdefault((job, label), []).append({k: call[k] for k in ("rep", "ns", "cycles", "kernels")})
    return grouped


def summarize(samples, reps):
    if sorted(s["rep"] for s in samples) != list(range(reps)):
        raise ValueError(f"expected repeats 0..{reps - 1}, captured {sorted(s['rep'] for s in samples)}")
    return {"cycles": statistics.median(s["cycles"] for s in samples),
            "ms": statistics.median(s["ns"] for s in samples) / 1e6,
            "rounds_cycles": [s["cycles"] for s in samples], "rounds_ms": [s["ns"] / 1e6 for s in samples],
            "kernels": sorted({k for s in samples for k in s["kernels"]}),
            "kernels_per_call": max(len(s["kernels"]) for s in samples)}


def child(args):
    """Run the labelled calls of ``--jobs`` between cudaProfilerStart/Stop (inside Nsight Compute)."""
    from MyFlows.core.device import set_device, xp
    from MyFlows.ops.cuda_native import native

    set_device("cuda")
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    nvtx, used = xp.cuda.nvtx, {}
    for job in jobs:
        layout, m, n, k = job["layout"], job["m"], job["n"], job["k"]
        ta, tb = layout == "TN", layout == "NT"
        rng = np.random.default_rng(job.get("seed", 0))
        a = xp.asarray(rng.standard_normal((k, m) if ta else (m, k)).astype(np.float32))
        b = xp.asarray(rng.standard_normal((n, k) if tb else (k, n)).astype(np.float32))
        out = xp.empty((m, n), xp.float32)
        calls = []
        for call in job["calls"]:
            plan = tuple(call["plan"]) if call.get("plan") else None
            calls.append((call["label"], lambda engine=call["engine"], plan=plan: native.gemm(
                a, b, trans_a=ta, trans_b=tb, engine=engine, out=out, plan=plan)))
        for _, fn in calls:
            for _ in range(3):
                fn()
        xp.cuda.Device().synchronize()
        used[job["id"]] = list(native.gemm_plan(m, n, k, layout=layout))
        xp.cuda.profiler.start()
        try:
            for rep in range(args.reps):
                for label, fn in (calls if rep % 2 == 0 else calls[::-1]):
                    nvtx.RangePush(f"{job['id']}:{label}:{rep}")
                    fn()
                    nvtx.RangePop()
            xp.cuda.Device().synchronize()
        finally:
            xp.cuda.profiler.stop()
        del a, b, out, calls
        xp.get_default_memory_pool().free_all_blocks()
    Path(args.result).write_text(json.dumps({"production_plans": used,
                                             "autotune": os.environ.get("MYFLOWS_GEMM_AUTOTUNE")}), encoding="utf-8")
    return 0


def capture(jobs, out, reps, tool=None, timeout=3600, cache="all"):
    """Profile ``jobs`` in a child process under Nsight Compute; return the parsed calls and child result."""
    from benchmark.nsight import find_tool

    out.mkdir(parents=True, exist_ok=True)
    tool = str(tool or find_tool("ncu"))
    jobs_path, result_path, prefix = out / "ncu-jobs.json", out / "ncu-child.json", out / "gemm"
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
    cmd = [tool, "--config-file", "off", "--target-processes", "all", "--profile-from-start", "off", "--nvtx",
           *ncu_options(cache), "--metrics", ",".join(METRICS), "--export", str(prefix), "--force-overwrite",
           sys.executable, "-X", "utf8", "-m", "benchmark.gemm_ncu", "--child", "--jobs", str(jobs_path),
           "--result", str(result_path), "--reps", str(reps)]
    with (out / "ncu.log").open("wb") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    raw = subprocess.check_output([tool, "--import", str(prefix.with_suffix(".ncu-rep")), "--page", "raw", "--csv",
                                   "--print-units", "base", "--print-nvtx-rename", "kernel",
                                   "--metrics", ",".join(METRICS)],
                                  encoding="utf-8", errors="replace", timeout=600)
    (out / "gemm.csv").write_text(raw, encoding="utf-8")
    return parse_csv(raw), json.loads(result_path.read_text(encoding="utf-8")), cmd


def compare(shapes, out, reps=7, tool=None, cache="all"):
    """Production register plan against cuBLAS for each shape; rows carry both engines' medians."""
    jobs = [{"id": f"g{i}", "layout": s["layout"], "m": s["m"], "n": s["n"], "k": s["k"], "seed": i,
             "calls": [{"label": "register", "engine": "register"}, {"label": "cublas", "engine": "cublas"}]}
            for i, s in enumerate(shapes)]
    calls, result, cmd = capture(jobs, out, reps, tool, cache=cache)
    rows = []
    for i, shape in enumerate(shapes):
        row = {**shape, "timer": "ncu", "reps": reps}
        row["tile"], row["slices"] = result["production_plans"][f"g{i}"]
        for engine in ("register", "cublas"):
            got = summarize(calls.get((f"g{i}", engine), []), reps)
            row.update({f"{engine}_{key}": value for key, value in got.items()})
        row["ratio"] = row["register_cycles"] / row["cublas_cycles"]
        rows.append(row)
    clocks = [r[f"{e}_cycles"] / (r[f"{e}_ms"] * 1e6) for r in rows for e in ("register", "cublas")]
    return rows, {"command": cmd[:cmd.index(sys.executable)], "reps": reps, "autotune": result["autotune"], "cache": cache,
                  "sm_ghz_median": statistics.median(clocks), "sm_ghz_range": [min(clocks), max(clocks)]}


def candidate_plans(layout, m, n, k, max_requested=128):
    """Every compiled register tile with each distinct effective split count it can launch for this shape."""
    from MyFlows.ops.cuda_native import native

    plans = []
    for tile, (_, *_geometry, bk) in native.tile_catalog().items():
        limit = 32 if tile in native.GEMM_TILES else max_requested
        seen = set()
        for requested in range(1, limit + 1):
            slices = native._slices(k, requested, bk)
            if slices not in seen:
                seen.add(slices)
                plans.append((tile, requested, slices))
    return plans


def prefilter(shape, plans, keep=12, rounds=5):
    """Fastest ``keep`` plans by the minimum CUDA-event time over shuffled rounds (clock drops only add time).

    Nsight Compute costs about a second per kernel on WDDM, so only the shortlist is profiled.
    """
    import random

    from benchmark.gemm_engines import _event_ms
    from MyFlows.core.device import xp
    from MyFlows.ops.cuda_native import native

    layout, m, n, k = (shape[key] for key in ("layout", "m", "n", "k"))
    ta, tb = layout == "TN", layout == "NT"
    rng = np.random.default_rng(0)
    a = xp.asarray(rng.standard_normal((k, m) if ta else (m, k)).astype(np.float32))
    b = xp.asarray(rng.standard_normal((n, k) if tb else (k, n)).astype(np.float32))
    out = xp.empty((m, n), xp.float32)
    calls = {plan: (lambda plan=plan: native.gemm(a, b, trans_a=ta, trans_b=tb, out=out, plan=plan[:2]))
             for plan in plans}
    best = {}
    order = list(plans)
    shuffle = random.Random(11)
    for _ in range(rounds):
        shuffle.shuffle(order)
        for plan in order:
            calls[plan]()
            reps = max(2, min(64, int(2.0 / max(_event_ms(xp, calls[plan], 1), 1e-3))))
            best[plan] = min(best.get(plan, float("inf")), _event_ms(xp, calls[plan], reps))
    del a, b, out, calls
    xp.get_default_memory_pool().free_all_blocks()
    ranked = sorted(plans, key=best.get)
    chosen = ranked[:keep]
    unsplit = next((p for p in ranked if p[2] == 1), None)
    if unsplit and unsplit not in chosen:
        chosen.append(unsplit)
    return chosen, {f"{t}|{r}": best[(t, r, s)] for t, r, s in plans}


def candidates(shape, out, reps=5, tool=None, max_requested=128, keep=12, cache="all"):
    from MyFlows.core.device import set_device

    set_device("cuda")
    plans = candidate_plans(shape["layout"], shape["m"], shape["n"], shape["k"], max_requested)
    plans, prefilter_ms = prefilter(shape, plans, keep)
    calls = [{"label": "cublas", "engine": "cublas"}, {"label": "production", "engine": "register"}]
    calls += [{"label": f"p{i}", "engine": "register", "plan": [tile, requested]}
              for i, (tile, requested, _) in enumerate(plans)]
    job = {"id": "c", "seed": 0, "calls": calls, **{key: shape[key] for key in ("layout", "m", "n", "k")}}
    got, result, _ = capture([job], out, reps, tool, cache=cache)
    cublas = summarize(got[("c", "cublas")], reps)
    rows = []
    for call in calls[1:]:
        summary = summarize(got[("c", call["label"])], reps)
        plan = call.get("plan") or result["production_plans"]["c"]
        slices = next((s for t, r, s in plans if [t, r] == list(plan)), None)
        rows.append({"label": call["label"], "tile": plan[0], "requested": plan[1], "slices": slices,
                     "ratio": summary["cycles"] / cublas["cycles"], **summary})
    rows.sort(key=lambda r: r["cycles"])
    return {"shape": shape, "cache": cache, "cublas": cublas, "candidates": rows, "prefilter_min_event_ms": prefilter_ms}


def _report(rows, meta):
    lines = ["# 卷积 GEMM 内核级对比（Nsight Compute）", "",
             f"GPC 时钟锁定在基频（实测 SM {meta['sm_ghz_median']:.3f} GHz），"
             f"{'每个 kernel 前清空 L1/L2' if meta.get('cache', 'all') == 'all' else '不清空缓存（与连续调用相同）'}。"
             f"每次调用包在 NVTX 区间内，区间内的 GEMM 与 split-K 归约 kernel 求和，两引擎交替 {meta['reps']} 次取中位数，"
             "比值按 sm__cycles_elapsed.max。", "",
             "| 模型 | 部分 | 布局 | M×N×K | 次数 | tile / 切分 | register µs | cuBLAS µs | 比值 |",
             "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |"]
    for r in sorted(rows, key=lambda r: -r["ratio"]):
        lines.append(f"| {r['model']} | {r['part']} | {r['layout']} | {r['m']}×{r['n']}×{r['k']} | {r['count']} | "
                     f"{r['tile']} / {r['slices']} | {r['register_ms'] * 1e3:.2f} | {r['cublas_ms'] * 1e3:.2f} | "
                     f"{r['ratio']:.3f} |")
    reg = sum(r["count"] * r["register_cycles"] for r in rows)
    blas = sum(r["count"] * r["cublas_cycles"] for r in rows)
    lines += ["", f"按出现次数加权的一步合计 register/cuBLAS 为 {reg / blas:.3f}。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--shapes", default="", help="comma list of LAYOUT:M:N:K instead of the gate's ResNet18 GEMMs")
    ap.add_argument("--candidates", action="store_true", help="measure every register plan of --shapes")
    ap.add_argument("--max-requested", type=int, default=128)
    ap.add_argument("--keep", type=int, default=12, help="plans kept by the CUDA-event prefilter for profiling")
    ap.add_argument("--cache-control", choices=("all", "none"), default="all",
                    help="flush L1/L2 before every kernel (default) or keep the caches warm between kernels")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--list-shapes", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--jobs", help=argparse.SUPPRESS)
    ap.add_argument("--result", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:
        return child(args)
    if args.list_shapes:
        from MyFlows.core.device import set_device

        set_device("cuda")
        print(json.dumps(gate_shapes()))
        return 0
    if not args.out or args.reps < 3:
        ap.error("supply --out and at least three repeats")
    from apps.autopilot.files import write_json
    from apps.autopilot.provenance import code_state
    from apps.common.environment import environment
    from apps.common.gpu_lock import GpuLock

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.shapes:
        shapes = []
        for item in args.shapes.split(","):
            layout, m, n, k = item.split(":")
            shapes.append({"model": "-", "part": "-", "layout": layout, "m": int(m), "n": int(n), "k": int(k),
                           "count": 1})
    elif args.candidates:
        ap.error("--candidates needs --shapes")
    else:
        shapes = list_shapes()
    write_json(out / "manifest.json", {"schema_version": 1, "environment": environment(), "source": code_state(ROOT),
                                       "reps": args.reps, "shapes": shapes, "ncu_options": list(ncu_options(args.cache_control))})
    with GpuLock(label="Nsight Compute GEMM comparison", run_dir=out):
        if args.candidates:
            for shape in shapes:
                name = f"{shape['layout']}-{shape['m']}x{shape['n']}x{shape['k']}"
                result = candidates(shape, out / name, args.reps, max_requested=args.max_requested, keep=args.keep,
                                    cache=args.cache_control)
                write_json(out / f"{name}.json", result)
                best = result["candidates"][0]
                print(name, "production", next(r for r in result["candidates"] if r["label"] == "production")["ratio"],
                      "best", best["tile"], best["requested"], best["slices"], best["ratio"], flush=True)
            return 0
        rows, meta = compare(shapes, out, args.reps, cache=args.cache_control)
    write_json(out / "gemms.json", rows)
    write_json(out / "summary.json", meta)
    (out / "report.md").write_text(_report(rows, meta), encoding="utf-8")
    for r in sorted(rows, key=lambda r: -r["ratio"])[:12]:
        print(r["model"], r["part"], r["layout"], r["m"], r["n"], r["k"], r["tile"], r["slices"], f"{r['ratio']:.3f}")
    return 0


def list_shapes():
    """The gate's GEMM shapes, enumerated in a short-lived CUDA process so the caller keeps no device memory."""
    proc = subprocess.run([sys.executable, "-X", "utf8", "-m", "benchmark.gemm_ncu", "--list-shapes"], cwd=ROOT,
                          capture_output=True, encoding="utf-8", errors="replace", timeout=900)
    if proc.returncode:
        raise RuntimeError(proc.stdout[-1000:] + proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    raise SystemExit(main())
