"""Run a complete small DonkeyCar training trajectory and compare 0/2 data-loader workers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

from benchmark.monitor_overhead import ROOT, read_rows
from benchmark.stage1_common import source_state
from tools.check_run_format import check_run


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def train(out, workers, args):
    directory = out / f"workers-{workers}"
    directory.mkdir()
    rel = directory.relative_to(ROOT).as_posix()
    command = [sys.executable, "-X", "utf8", "-m", "apps.train", "--device", "cuda", "--seed", "1",
               "--batch", "2", "--base-width", "64", "--stem", "cifar", "--lr", ".001",
               "--max-samples", str(args.samples), "--sample-seed", "1", "--val-size", str(args.val_size),
               "--epochs", str(args.epochs), "--num-workers", str(workers), "--phase-timing-every", "10",
               "--tb-log-interval", "10", "--tb-grad-interval", "0", "--tb-param-interval", "0",
               "--tb-activation-interval", "0", "--tb-image-interval", "0", "--checkpoint-every", "0",
               "--run-dir", rel, "--log-dir", rel, "--logdir", f"{rel}/tensorboard", "--out", f"{rel}/checkpoint",
               "--best-out", f"{rel}/best", "--stop-file", f"{rel}/STOP", "--split-out", f"{rel}/split.json"]
    write(directory / "command.json", [".venv/Scripts/python.exe", *command[1:]])
    print(f"data loader workers={workers}: {rel}", flush=True)
    with (directory / "process.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, check=True, timeout=3600)
    checked = check_run(directory)
    write(directory / "format-check.json", checked)
    if not checked["valid"]:
        raise RuntimeError(checked["errors"])
    return directory


def trends(directory, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_rows(directory / "metrics.jsonl")
    steps = [r for r in rows if r["kind"] == "step"]
    epochs = [r for r in rows if r["kind"] == "epoch"]
    hardware = read_rows(directory / "resources.jsonl")
    origin = min(steps[0]["monotonic_s"], hardware[0]["monotonic_s"])
    panels = [
        ("GPU utilization", "%", [(hardware, "gpu_utilization_pct", "device")]),
        ("GPU memory", "MiB", [(hardware, "gpu_device_memory_mib", "device"), (steps, "cupy_pool_used_mib", "CuPy used"), (steps, "cupy_pool_total_mib", "CuPy reserved")]),
        ("Peak GPU memory", "MiB", [(hardware, "gpu_device_memory_peak_mib", "device sampled"), (steps, "used_peak_mib", "CuPy sampled")]),
        ("GPU temperature", "deg C", [(hardware, "gpu_temperature_c", "GPU")]),
        ("GPU power", "W", [(hardware, "gpu_power_w", "draw"), (hardware, "gpu_power_limit_w", "limit")]),
        ("CPU utilization", "%", [(hardware, "cpu_utilization_pct", "host")]),
        ("Step incl. load", "ms", [(steps, "step_total_ms", "wall")]),
        ("Forward", "ms", [(steps, "forward_ms", "device")]),
        ("Backward", "ms", [(steps, "backward_ms", "device")]),
        ("Optimizer", "ms", [(steps, "optimizer_ms", "device")]),
        ("Data loading", "ms", [(steps, "data_load_ms", "wall")]),
        ("Throughput incl. load", "samples/s", [(steps, "samples_per_sec_total", "batch / step")]),
    ]
    peak = 0.
    for row in steps:
        peak = max(peak, row["cupy_pool_used_mib"])
        row["used_peak_mib"] = peak
    fig, axes = plt.subplots(4, 3, figsize=(13, 11), constrained_layout=True, sharex=True)
    stats = []
    for ax, (title, unit, lines) in zip(axes.flat, panels):
        for data, key, label in lines:
            samples = [(r["monotonic_s"] - origin, r[key]) for r in data if r.get(key) is not None]
            if not samples:
                raise ValueError(f"no values for required trend {key}")
            times, values = np.asarray(samples).T
            ax.plot(times, values, linewidth=.8, label=label)
            stats.append({"metric": title, "field": key, "unit": unit, "n": len(values), "median": float(np.median(values)),
                          "p95": float(np.percentile(values, 95)), "max": float(np.max(values))})
        for epoch in epochs:
            ax.axvline(epoch["monotonic_s"] - origin, color="black", alpha=.2, linewidth=.6, linestyle="--")
        ax.set(title=title, ylabel=unit)
        ax.grid(alpha=.17)
        if len(lines) > 1:
            ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Elapsed time (s)")
    fig.savefig(out / "trends.png", dpi=150)
    plt.close(fig)
    return stats


def summarize(directory, warmup):
    rows = read_rows(directory / "metrics.jsonl")
    steps = [r for r in rows if r["kind"] == "step"][warmup:]
    epochs = [r for r in rows if r["kind"] == "epoch"]
    if not steps or not epochs:
        raise ValueError("run does not cover a complete epoch after warmup")
    total = sum(r["step_total_ms"] for r in steps)
    return {"steps": len(steps), "data_share": sum(r["data_load_ms"] for r in steps) / total,
            "median_data_ms": statistics.median(r["data_load_ms"] for r in steps),
            "samples_per_s": sum(r["batch_size"] for r in steps) * 1000 / total,
            "epoch_wall_s": [r["epoch_time_s"] for r in epochs],
            "epoch_metrics": [[r["loss"], r["val_loss"], r["val_angle_mae"]] for r in epochs]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/semester_2026_fall/monitoring/trends-001"))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args(argv)
    if not 0 < args.val_size < args.samples or args.epochs < 1 or args.warmup < 0:
        parser.error("invalid sample/validation/epoch budget")
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT):
        parser.error("--out must be within the workspace")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "manifest.json", {"schema_version": 1, "source": source_state(), "samples": args.samples,
                                  "epochs": args.epochs, "val_size": args.val_size, "warmup": args.warmup})
    summaries = [summarize(train(out, n, args), args.warmup) for n in (0, 2)]
    if summaries[0]["epoch_metrics"] != summaries[1]["epoch_metrics"]:
        raise RuntimeError("data-loader worker count changed the training trajectory")
    stats = trends(out / "workers-0", out)
    write(out / "summary.json", {"workers_0": summaries[0], "workers_2": summaries[1], "trends": stats})
    lines = ["# 完整训练过程的资源与阶段指标", "",
             f"{args.samples} 条真实样本，其中 {args.val_size} 条用于验证，训练 {args.epochs} 轮。cifar stem、w64、"
             "batch 2、FP32、种子 1。两种数据加载方式的各轮训练 loss、验证 loss 与角度 MAE 逐值相同。",
             "", "![完整训练过程的 12 项指标](trends.png)", "",
             "虚线标明每轮结束。表中列出 workers=0 从启动到结束的全部有效样本，阶段耗时按选定步采样。", "",
             "| 字段 | 单位 | 样本数 | 中位数 | P95 | 最大值 |", "| --- | --- | --- | --- | --- | --- |"]
    for row in stats:
        lines.append(f"| {row['field']} | {row['unit']} | {row['n']} | {row['median']:.3f} | {row['p95']:.3f} | {row['max']:.3f} |")
    lines += ["", f"流水线比较剔除前 {args.warmup} 个 step，数据读取占比为数据读取总耗时除以相同步的总耗时。", "",
              "| loader workers | 稳态步数 | 数据占比 | 中位读取 ms | samples/s |", "| --- | --- | --- | --- | --- |"]
    for n, row in zip((0, 2), summaries):
        lines.append(f"| {n} | {row['steps']} | {row['data_share']:.2%} | {row['median_data_ms']:.3f} | {row['samples_per_s']:.3f} |")
    sha = hashlib.sha256((out / "summary.json").read_bytes()).hexdigest()
    lines += ["", f"原始目录为 `{out.relative_to(ROOT).as_posix()}/`，每组含 config、metrics、resources、result、TensorBoard 与格式检查。"
              f"summary.json 的 SHA-256 为 `{sha}`。", "",
              f"复现命令为 `python -m benchmark.monitor_training --out runs/stage3/trends-repro --samples {args.samples} "
              f"--val-size {args.val_size} --epochs {args.epochs} --warmup {args.warmup}`。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
