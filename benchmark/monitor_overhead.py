"""Alternate off / 0.5 s / 0.1 s monitoring on the production DonkeyCar training path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from apps.autopilot.provenance import code_state
from apps.common.environment import environment
from MyFlows.monitoring import GpuSampler, ResourceRecorder
from MyFlows.monitoring.schema import EXTERNAL_MONITOR_ENV
from tools.check_run_format import check_run

ROOT = Path(__file__).resolve().parents[1]
MODES = (("off", None), ("500ms", 0.5), ("100ms", 0.1))


def read_rows(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(run_dir: Path, warmup_steps: int) -> dict:
    steps = [row for row in read_rows(run_dir / "metrics.jsonl") if row.get("kind") == "step"][warmup_steps:]
    if not steps:
        raise ValueError("no steady-state steps remain after warmup")
    total_ms = [r.get("step_total_ms", r["step_time_ms"] + r["data_load_ms"]) for r in steps]
    begin = steps[0]["monotonic_s"] - total_ms[0] / 1000
    end = steps[-1]["monotonic_s"]
    resources = [row for row in read_rows(run_dir / "power-reference.jsonl") if begin <= row["monotonic_s"] <= end]
    pairs = [(r.get("gpu_power_w"), r.get("gpu_power_limit_w")) for r in resources]
    pairs = [(p, limit) for p, limit in pairs if p is not None and limit is not None and limit > 0]
    return {"steps": len(steps), "step_ms_median": statistics.median(total_ms),
            "step_ms_pstdev": statistics.pstdev(total_ms),
            "samples_per_s": sum(r["batch_size"] for r in steps) / (sum(total_ms) / 1000),
            "power_samples": len(pairs), "power_capped_share": sum(p >= .97 * lim for p, lim in pairs) / len(pairs) if pairs else None,
            "power_limit_min_w": min(lim for _, lim in pairs) if pairs else None,
            "power_limit_max_w": max(lim for _, lim in pairs) if pairs else None,
            "steady_start_monotonic_s": begin, "steady_end_monotonic_s": end}


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for label, _ in MODES:
        group = [row for row in rows if row["mode"] == label]
        values = [row["step_ms_median"] for row in group]
        rates = [row["samples_per_s"] for row in group]
        if values:
            out[label] = {"repeats": len(values), "median_step_ms": statistics.median(values),
                          "pstdev_step_ms": statistics.pstdev(values), "min_step_ms": min(values), "max_step_ms": max(values),
                          "median_samples_per_s": statistics.median(rates)}
    baseline = {row["repeat"]: row["step_ms_median"] for row in rows if row["mode"] == "off"}
    for label in ("500ms", "100ms"):
        paired = [100 * (row["step_ms_median"] / baseline[row["repeat"]] - 1)
                  for row in rows if row["mode"] == label and row["repeat"] in baseline]
        if paired:
            out[label].update(overhead_pct_median=statistics.median(paired), overhead_pct_pstdev=statistics.pstdev(paired),
                              overhead_pct_min=min(paired), overhead_pct_max=max(paired))
    return out


def write_report(out: Path, args, rows, groups):
    lines = ["# 训练资源监控开销", "", f"协议：cifar stem、w64、batch 2、FP32、{args.max_samples} 条样本、"
             f"1 epoch；每档 {args.repeats} 次独立训练进程，按 off → 0.5 s → 0.1 s 交替。每次丢弃前 {args.warmup_steps} 步。",
             "", "计时为 data_load_ms + step_time_ms 的逐步中位数，吞吐为相同稳态步总样本 / 总时间。"
             "每档保留相同的阶段计时、JSONL、关闭 TensorBoard；只改变资源采样。"
             "为读取关闭档的功耗墙比例，所有档次均在独立的测量进程中保持相同 0.5 s 功耗参考采样"
             "（power-reference.jsonl）；此项固定测量成本不计入训练自身监控的增量。",
             "", "| 档次 | 重复 | 中位 step ms | 重复间标准差 ms | 中位 samples/s | 配对开销中位 % | 开销标准差 % |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, value in groups.items():
        lines.append(f"| {name} | {value['repeats']} | {value['median_step_ms']:.3f} | {value['pstdev_step_ms']:.3f} | "
                     f"{value['median_samples_per_s']:.3f} | {value.get('overhead_pct_median', 0):.3f} | "
                     f"{value.get('overhead_pct_pstdev', 0):.3f} |")
    lines += ["", "| 次序 | 档次 | 稳态步数 | step ms | samples/s | 顶到功耗上限比例 | 功耗样本数 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for i, row in enumerate(rows):
        cap = "缺测" if row["power_capped_share"] is None else f"{row['power_capped_share']:.1%}"
        lines.append(f"| {i + 1} | {row['mode']} | {row['steps']} | {row['step_ms_median']:.3f} | {row['samples_per_s']:.3f} | {cap} | {row['power_samples']} |")
    passed = groups.get("500ms", {}).get("overhead_pct_median", float("inf")) < 5 and args.repeats >= 3
    lines += ["", f"默认 0.5 s 档的 <5% 门槛：{'通过' if passed else '尚未通过（不减少验收要求）'}。"
              "小于重复波动的负开销不解释为加速。功耗墙、桌面负载和热状态会造成离散度。",
              "", f"原始数据：`{out.relative_to(ROOT).as_posix()}/`（本地，不纳入 Git）；"
              "manifest.json 保存命令、代码与环境，summary.json 保存逐次统计；各子目录保留完整运行和格式检查。",
              "", "复现命令：", "", "```powershell", f".venv/Scripts/python.exe -X utf8 -m benchmark.monitor_overhead --out {out.relative_to(ROOT).as_posix()} "
              f"--max-samples {args.max_samples} --repeats {args.repeats} --warmup-steps {args.warmup_steps}", "```", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/semester_2026_fall/monitoring/overhead-001"))
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--data", default="mycar/data")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.max_samples < 2 * (args.warmup_steps + 1) or args.warmup_steps < 0:
        parser.error("need positive repeats and more than warmup_steps batches")
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT):
        parser.error("experiment output must be inside the repository")
    if (out / "manifest.json").exists():
        parser.error("output already has a manifest; use a new directory to preserve previous evidence")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "protocol": vars(args) | {"out": out.relative_to(ROOT).as_posix()},
                "source": code_state(ROOT), "environment": environment(), "commands": [], "started_unix_s": time.time()}
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop(EXTERNAL_MONITOR_ENV, None)
    rows = []
    for repeat in range(args.repeats):
        for label, interval in MODES:
            directory = out / f"r{repeat + 1}-{label}"
            directory.mkdir()
            rel = directory.relative_to(ROOT).as_posix()
            command = [sys.executable, "-X", "utf8", "-m", "apps.train", "--device", "cuda", "--data", args.data,
                       "--stem", "cifar", "--base-width", "64", "--batch", "2", "--max-samples", str(args.max_samples),
                       "--epochs", "1", "--seed", "1", "--no-tensorboard", "--checkpoint-every", "0",
                       "--phase-timing-every", "20", "--run-dir", rel, "--log-dir", rel,
                       "--out", f"{rel}/checkpoint", "--best-out", f"{rel}/best", "--stop-file", f"{rel}/STOP",
                       "--monitor", "off" if interval is None else "on"]
            if interval is not None:
                command += ["--monitor-interval-s", str(interval)]
            manifest["commands"].append([".venv/Scripts/python.exe", *command[1:]])
            (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(f"[{repeat + 1}/{args.repeats}] {label}: {rel}", flush=True)
            with GpuSampler() as sampler, (directory / "process.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                try:
                    with ResourceRecorder(sampler, interval_s=0.5, root_pid=process.pid,
                                          jsonl_path=directory / "power-reference.jsonl"):
                        code = process.wait()
                except BaseException:
                    from apps.console.jobs import kill_tree

                    kill_tree(process.pid)
                    raise
            if code:
                raise RuntimeError(f"training exited {code}; see {rel}/process.log")
            checked = check_run(directory)
            (directory / "format-check.json").write_text(json.dumps(checked, indent=2), encoding="utf-8")
            if not checked["valid"]:
                raise RuntimeError(checked["errors"])
            rows.append({"mode": label, "repeat": repeat, "run": rel, **summarize(directory, args.warmup_steps)})
            groups = aggregate(rows)
            (out / "summary.json").write_text(json.dumps({"rows": rows, "groups": groups}, indent=2), encoding="utf-8")
            write_report(out, args, rows, groups)
            print(json.dumps(rows[-1]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
