"""Interleave training measurements and find the WDDM residency boundary in fresh processes."""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from benchmark.resnet_before_after import _build, _import

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (("cifar-w64-b2", "cifar", 64, 2), ("cifar-w16-b16", "cifar", 16, 16),
           ("imagenet-w16-b128", "imagenet", 16, 128))


def wddm_memory():
    if os.name != "nt":
        return None
    prefix = rf"\GPU Process Memory(pid_{os.getpid()}_*)"
    names = ("Local Usage", "Non Local Usage")
    proc = subprocess.run(["typeperf", *(prefix + "\\" + name for name in names), "-sc", "1"],
                          capture_output=True, encoding="mbcs", errors="replace", timeout=15)
    rows = [row for row in csv.reader(proc.stdout.splitlines()) if len(row) > 1]
    if proc.returncode or len(rows) < 2:
        return None
    headers, values = rows[:2]
    return {name: sum(float(v) for k, v in zip(headers[1:], values[1:]) if k.endswith("\\" + name)) / 2**20
            for name in names}


def worker(args):
    ms = _import(args.tree)
    from MyFlows.core.device import xp
    from MyFlows.monitoring.gpu import GpuSampler

    model, x, y, loss, graph = _build(ms, args, "cuda", "float32", "cuda_native_register")
    if args.graph_opt:
        graph = ms.Graph(loss, optimize=True)
    model.train()
    opt = ms.Adam(graph, learning_rate=.001)
    pool = xp.get_default_memory_pool()
    rng = np.random.default_rng(51)
    xb = rng.normal(size=x.value.shape).astype(np.float32)
    yb = rng.normal(size=y.value.shape).astype(np.float32)
    idle = wddm_memory() if args.capacity else None
    times, powers, peaks, used = [], [], [], []
    with GpuSampler() as sampler:
        for step in range(args.warmup + args.steps):
            xp.cuda.Device().synchronize()
            start = time.perf_counter()
            x.value, y.value = xb, yb
            try:
                opt.one_step()
                opt.update()
                value = float(loss.value)
            except xp.cuda.memory.OutOfMemoryError:
                if args.capacity:
                    return {"resident": False, "reason": "allocation failed", "pool_mib": pool.total_bytes()/2**20}
                raise
            elapsed = (time.perf_counter() - start) * 1000.
            if not np.isfinite(value):
                raise ValueError("nonfinite training loss")
            peaks.append(pool.total_bytes() / 2**20)
            used.append(pool.used_bytes() / 2**20)
            if step >= args.warmup:
                times.append(elapsed)
                gpu = sampler.sample()["gpus"][0]
                powers.append({k: gpu.get(k) for k in ("power_w", "power_limit_w", "memory_used_mib", "temperature_c")})
            if args.capacity and step == 0 and idle:
                first = wddm_memory()
                if first and first["Non Local Usage"] > idle["Non Local Usage"] + 1.:
                    return {"resident": False, "reason": "nonlocal growth", "idle": idle, "wddm": first,
                            "pool_mib": max(peaks), "first_ms": elapsed}
    after = wddm_memory() if args.capacity else None
    if args.capacity and not (idle and after):
        raise RuntimeError("the WDDM nonlocal usage counter is required to certify batch capacity")
    return {"median_ms": statistics.median(times), "steps_ms": times, "pool_mib": max(peaks),
            "pool_used_at_step_end_mib": max(used), "power": powers, "loss": value,
            "idle": idle, "wddm": after,
            "resident": after["Non Local Usage"] <= idle["Non Local Usage"] + 1. if after else None}


def run_case(tree, args, stem, width, batch, capacity=False):
    cmd = [sys.executable, "-X", "utf8", "-m", "benchmark.memory_study", "--worker", "--tree", str(tree),
           "--batch", str(batch), "--stem", stem, "--width", str(width),
           "--warmup", str(1 if capacity else args.warmup), "--steps", str(2 if capacity else args.steps)]
    if capacity:
        cmd.append("--capacity")
    if args.graph_opt:
        cmd.append("--graph-opt")
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", timeout=240)
    if run.returncode:
        raise RuntimeError(run.stdout[-1000:] + run.stderr[-2000:])
    return json.loads(run.stdout.strip().splitlines()[-1])


def study(args):
    from apps.common.gpu_lock import GpuLock
    from apps.autopilot.provenance import code_state
    from benchmark.transfer_study import write

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    trees = {"before": args.old.resolve(), "after": ROOT}
    manifest = {"trees": {key: code_state(path) for key, path in trees.items()},
                "protocol": {k: v for k, v in vars(args).items() if k not in ("tree", "out", "old")},
                "environment": {k: v for k, v in os.environ.items() if k.startswith("MYFLOWS_")}}
    write(out/"manifest.json", manifest)
    rows, capacity = [], []
    with GpuLock(label="ResNet memory comparison", run_dir=out):
        for name, stem, width, batch in CONFIGS:
            for repeat in range(args.rounds):
                for side, tree in trees.items():
                    result = run_case(tree, args, stem, width, batch)
                    rows.append({"name": name, "repeat": repeat, "side": side, **result})
                    print(name, repeat, side, result["median_ms"], result["pool_mib"], flush=True)
                    write(out/"timings.json", rows)
        if not args.skip_capacity:
            for side, tree in trees.items():
                low, high = 0, None
                batch = 16
                while high is None or high - low > 1:
                    result = run_case(tree, args, "cifar", 16, batch, capacity=True)
                    capacity.append({"side": side, "batch": batch, **result})
                    print(side, "batch", batch, "resident", result["resident"], flush=True)
                    write(out/"capacity.json", capacity)
                    if result["resident"]:
                        low = batch
                    else:
                        high = batch
                    batch = low + 8 if high is None else (low + high)//2
    lines = ["# ResNet18 梯度与显存测量", "", f"FP32、register、120×160，训练模式 BN 和 Adam。"
             f"每次预热 {args.warmup} 步，再测 {args.steps} 步。每个配置交替启动旧、新代码各 {args.rounds} 次。"
             "每个进程的步时取中位数，下表列进程间中位数与样本标准差。CuPy 池列为单次进程申请峰值的中位数。", "",
             "| 配置 | 旧步时 ms | 新步时 ms | 旧池 MiB | 新池 MiB | 旧 / 新触及功耗墙 |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name, *_ in CONFIGS:
        times, pools, caps = [], [], []
        for side in trees:
            group = [r for r in rows if r["name"] == name and r["side"] == side]
            medians = [r["median_ms"] for r in group]
            times.append(f"{statistics.median(medians):.2f} ± {statistics.stdev(medians):.2f}")
            pools.append(f"{statistics.median(r['pool_mib'] for r in group):.1f}")
            powers = [p for r in group for p in r["power"] if p["power_w"] is not None and p["power_limit_w"]]
            caps.append(f"{statistics.mean(p['power_w'] >= .97*p['power_limit_w'] for p in powers):.0%}" if powers else "缺测")
        lines.append("| " + " | ".join((name, *times, *pools, " / ".join(caps))) + " |")
    if capacity:
        lines += ["", "另用 cifar stem、w16 逐步增大 batch，在首次出现非本地显存后按整数二分。"
                  "每次独立进程，记录 WDDM 的进程 Local Usage 和 Non Local Usage，容许相对空模型增加 1 MiB 的计数波动。"]
        for side in trees:
            group = [r for r in capacity if r["side"] == side]
            resident = max((r["batch"] for r in group if r["resident"]), default=0)
            lines.append(f"- {side} 的最大驻留 batch 为 {resident}，相邻的 {resident+1} 已超出。")
    lines += ["", "原始测量保存在 timings.json、capacity.json 和 manifest.json。"]
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--old", type=Path)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--graph-opt", action="store_true")
    ap.add_argument("--skip-capacity", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--capacity", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--tree", type=Path, default=ROOT)
    ap.add_argument("--stem", default="cifar")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args(argv)
    args.height, args.width_px = 120, 160
    if args.worker:
        print(json.dumps(worker(args)), flush=True)
    elif args.old and args.out and min(args.steps, args.warmup) > 0 and args.rounds >= 2:
        study(args)
    else:
        ap.error("--old, --out, positive step counts and at least two repeats are required")


if __name__ == "__main__":
    main()
