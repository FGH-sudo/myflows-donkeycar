"""Interleaved optimizer-phase timing inside real FP32 ResNet18 forward/backward/update steps."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from benchmark.resnet_before_after import ROOT, _build, _import


def measure(args):
    ms = _import(args.tree)
    from MyFlows.core.device import asnumpy, xp

    model, x, y, loss, graph = _build(ms, args, "cuda", "float32", "cuda_native_register")
    model.train()
    opt = ms.Adam(graph, learning_rate=.001)
    rng = np.random.default_rng(71)
    x.value = rng.normal(size=x.value.shape).astype(np.float32)
    y.value = rng.normal(size=y.value.shape).astype(np.float32)
    start, end = xp.cuda.Event(), xp.cuda.Event()
    rows = []
    for i in range(args.warmup + args.steps):
        opt.one_step()
        xp.cuda.get_current_stream().synchronize()
        wall = time.perf_counter()
        start.record()
        opt.update()
        end.record()
        end.synchronize()
        elapsed = 1000*(time.perf_counter()-wall)
        if i >= args.warmup:
            rows.append({"cuda_ms": xp.cuda.get_elapsed_time(start, end), "wall_ms": elapsed})
    sha = hashlib.sha256()
    for node in model.params:
        for value in (node.value, opt.v[node], opt.s[node]):
            sha.update(asnumpy(value).tobytes())
        sha.update(str(opt.parameter_steps[node]).encode())
    return {"phases": rows, "loss": float(loss.value), "tensors": len(opt.v), "state_sha256": sha.hexdigest()}


def main(argv=None):
    from apps.autopilot.files import write_json
    from apps.autopilot.provenance import code_state
    from apps.common.gpu_lock import GpuLock

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--tree", type=Path, default=ROOT)
    ap.add_argument("--stem", default="cifar")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    args=ap.parse_args(argv)
    args.height, args.width_px = 120, 160
    if args.worker:
        print(json.dumps(measure(args)))
        return
    if not args.out or not args.old or args.rounds < 2:
        ap.error("--out, --old and two or more rounds are required")
    out=args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    trees = {"before": args.old.resolve(), "after": ROOT}
    write_json(out/"manifest.json", {"source": {k:code_state(p) for k,p in trees.items()},
                                     "steps": args.steps, "warmup": args.warmup, "rounds": args.rounds})
    import os
    rows=[]
    with GpuLock(label="Adam phase comparison",run_dir=out):
        for index in range(args.rounds):
            for side in trees if index%2 == 0 else reversed(trees):
                cmd=[sys.executable,"-X","utf8","-m","benchmark.adam_study","--worker","--tree",str(trees[side]),
                     "--steps",str(args.steps),"--warmup",str(args.warmup)]
                data=subprocess.check_output(cmd,cwd=ROOT,env={**os.environ,"MYFLOWS_MULTI_ADAM":str(int(side=="after"))},
                                             encoding="utf-8",timeout=240)
                row={"side":side,"round":index,**json.loads(data.strip().splitlines()[-1])}
                rows.append(row)
                write_json(out/"timings.json",rows)
                print(side,index,statistics.median(v["wall_ms"] for v in row["phases"]),flush=True)
    if len({r["loss"] for r in rows}) != 1:
        raise RuntimeError("training trajectory changed")
    if len({r["state_sha256"] for r in rows}) != 1:
        raise RuntimeError("model parameters, Adam moments or step counters differ")
    med = statistics.median
    lines=["# Adam 多张量更新", "", f"cifar stem、w64、batch 2、120×160、FP32 register。每步完整训练，"
           f"在更新前同步，再分别记录 CUDA Event 与主机至同步返回的耗时。两种实现在同会话交替 {args.rounds} 次，"
           f"预热 {args.warmup} 步，测量 {args.steps} 步。", "",
           "| 更新 | 参数张量 | CUDA ms | 主机同步 ms |", "| --- | ---: | ---: | ---: |"]
    for side in trees:
        group=[r for r in rows if r["side"]==side]
        values=[]
        for key in ("cuda_ms","wall_ms"):
            vals=[med(v[key] for v in r["phases"]) for r in group]
            values.append(f"{med(vals):.3f} ± {statistics.stdev(vals):.3f}")
        lines.append(f"| {side} | {group[0]['tensors']} | {values[0]} | {values[1]} |")
    lines+=["","表中为进程间中位数及样本标准差。所有进程终点 loss 相同，"
            "全部参数、Adam 的一阶矩与二阶矩、逐参数更新计数的联合 SHA-256 均为 "
            f"`{rows[0]['state_sha256']}`。逐步耗时见 timings.json。"]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
