"""Compare MyFlows and PyTorch on the same ResNet18 NPZ, road images and batch lists."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def orders(train, batch, epochs, seed, timing_steps):
    rng = np.random.default_rng(seed)
    used = len(train) // batch * batch
    if used == 0:
        raise ValueError("the training split does not contain a full batch")
    training = np.stack([rng.permutation(train)[:used].reshape(-1, batch) for _ in range(epochs)]).astype(np.int32)
    timing = training[0][np.arange(timing_steps) % len(training[0])]
    return training, timing


def initial_state(path, seed):
    from MyFlows.core.device import asnumpy, set_device
    from MyFlows.core.dtype import set_default_dtype
    from MyFlows.utils.initializers import make_initializer
    from MyFlows.utils.pretrained import build_model

    set_device("cpu")
    set_default_dtype(np.float32)
    model = build_model("resnet18", 2, dropout=0., initializer=make_initializer("kaiming_normal", seed=seed))
    arrays = {}
    for key, target, kind in model.torchvision_tensors():
        if kind == "buffer":
            layer, attr = target
            value = asnumpy(getattr(layer, attr))
        else:
            value = asnumpy(target.value)
        arrays[key] = np.ascontiguousarray(value.T if kind == "param_t" else value)
        if key.endswith("running_mean"):
            arrays[key.removesuffix("running_mean") + "num_batches_tracked"] = np.asarray(0, dtype=np.int64)
    np.savez(path, **arrays)


def prepare(directory, args):
    from apps.common.donkey_data import load_donkey_index
    from apps.common.image_preprocess import imread_nchw
    from apps.common.splits import build_time_block_split, frame_numbers

    directory.mkdir()
    data_dir = (ROOT / args.data).resolve()
    index = load_donkey_index(data_dir, fixed_throttle=.5, angle_scale=1.)
    frames = frame_numbers([str(row[0]) for row in index])
    indices = sorted(range(len(index)), key=lambda i: (frames[i], str(index[i][0])))[:args.samples]
    if len(indices) != args.samples:
        raise ValueError("not enough source images for the requested protocol")
    rows = [index[i] for i in indices]
    block = min(50, max(1, args.val_size))
    split = build_time_block_split([frames[i] for i in indices], val_size=args.val_size, test_size=0,
                                   block_size=block, gap=min(5, block // 5), seed=42)
    start = time.perf_counter()
    images = np.lib.format.open_memmap(directory / "images.npy", mode="w+", dtype=np.float32,
                                       shape=(len(rows), 3, 120, 160))
    targets = np.empty((len(rows), 2), np.float32)
    for i, (rel, angle, throttle, *_) in enumerate(rows):
        images[i] = imread_nchw(data_dir / rel, (160, 120), np.float32)[0]
        targets[i] = (angle, throttle)
    images.flush()
    del images
    np.save(directory / "targets.npy", targets)
    prepare_s = time.perf_counter() - start
    write(directory / "split.json", split)
    write(directory / "samples.json", [[str(row[0]).replace("\\", "/"), float(row[1]), float(row[2])] for row in rows])
    protocol = {"schema_version": 1, "samples": len(rows), "train": len(split["train"]), "val": len(split["val"]),
                "graph_opt": getattr(args, "graph_opt", False),
                "block_size": block, "gap": min(5, block // 5), "split_seed": 42, "batch": args.batch,
                "epochs": args.epochs, "warmup": args.warmup, "timed_steps": args.timed_steps, "seeds": args.seeds,
                "input": "RGB FP32 NCHW [0,1], OpenCV INTER_LINEAR, 120x160", "architecture": "torchvision ResNet18, 2 outputs",
                "optimizer": {"name": "Adam", "lr": .001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.},
                "tail": "common drop_last for training; mask padded validation rows", "data_prepare_s": prepare_s,
                "validation_uses_all_rows": True, "sha256": {}}
    for seed in args.seeds:
        seed_dir = directory / f"seed-{seed}"
        seed_dir.mkdir()
        training, timing = orders(split["train"], args.batch, args.epochs, seed, max(args.warmup, args.timed_steps))
        np.savez(seed_dir / "batches.npz", training=training, timing=timing, val=np.asarray(split["val"], dtype=np.int32))
        initial_state(seed_dir / "initial.npz", seed)
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            protocol["sha256"][path.relative_to(directory).as_posix()] = sha256(path)
    write(directory / "protocol.json", protocol)
    return protocol


def alignment(myflows, pytorch):
    per_tensor, totals = {}, {}
    with ExitStack() as stack:
        left, right = [stack.enter_context(np.load(p, allow_pickle=False)) for p in (myflows, pytorch)]
        if set(left.files) != set(right.files):
            raise ValueError(f"alignment tensor names differ: {sorted(set(left.files) ^ set(right.files))}")
        if {k[5:] for k in left.files if k.startswith("grad/")} != {k[8:] for k in left.files if k.startswith("updated/")}:
            raise ValueError("a trainable tensor is missing its gradient or update")
        for name in sorted(left.files):
            a, b = left[name], right[name]
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"{name}: invalid shape or nonfinite values")
            diff = np.subtract(a, b, dtype=np.float64).ravel()
            ref = b.astype(np.float64, copy=False).ravel()
            squared, norm = float(diff @ diff), float(ref @ ref)
            maximum = float(np.abs(diff).max())
            floor = 1e-6
            relative = float((np.abs(diff) / np.maximum(np.abs(ref), floor)).max())
            group = name.split("/")[0]
            per_tensor[name] = {"shape": list(a.shape), "max_abs": maximum, "max_rel_floor_1e-6": relative,
                                "relative_l2": (squared / norm)**.5 if norm else None}
            record = totals.setdefault(group, {"squared": 0., "reference_squared": 0., "max_abs": 0., "max_rel": 0., "tensors": 0})
            record["squared"] += squared
            record["reference_squared"] += norm
            record["max_abs"] = max(record["max_abs"], maximum)
            record["max_rel"] = max(record["max_rel"], relative)
            record["tensors"] += 1
    for record in totals.values():
        record["relative_l2"] = (record["squared"] / record["reference_squared"])**.5 if record["reference_squared"] else None
    required = {"grad", "updated", "logits", "loss"}
    if set(totals) != required or totals["grad"]["tensors"] != totals["updated"]["tensors"]:
        raise ValueError("expected logits, loss and a gradient/update for every trainable tensor")
    # FP32 training-mode BN and cuDNN convolutions may place near-zero ReLU values on opposite sides.
    # Preserve all elementwise discrepancies, including near-zero gradients amplified by the first Adam step.
    passed = (totals["logits"]["max_abs"] <= .001 and totals["loss"]["max_abs"] <= .001
              and totals["grad"]["relative_l2"] is not None and totals["grad"]["relative_l2"] <= .02
              and totals["updated"]["max_abs"] <= .00201)
    return {"passed": passed, "groups": totals, "tensors": per_tensor,
            "thresholds": {"logits_abs": .001, "loss_abs": .001, "gradient_relative_l2": .02, "updated_max_abs": .00201}}


def summarize_resources(rows, start, end):
    selected = [row for row in rows if start <= row["monotonic_s"] <= end]
    def stat(key, fn):
        values = [r[key] for r in selected if r.get(key) is not None]
        return fn(values) if values else None
    return {"samples": len(selected), "gpu_util_pct_median": stat("gpu_utilization_pct", statistics.median),
            "gpu_device_mib_sampled_peak": stat("gpu_device_memory_mib", max),
            "cpu_util_pct_mean": stat("cpu_utilization_pct", statistics.fmean),
            "rss_bytes_sampled_peak": stat("process_rss_total_bytes", max)}


def run_worker(out, fixture, env_info, framework, mode, seed, batch):
    from MyFlows.monitoring import GpuSampler, ResourceRecorder
    from tools.check_run_format import check_run

    directory = out / f"{framework}-{mode}-s{seed}"
    directory.mkdir()
    command = [sys.executable, "-X", "utf8", "-m", "benchmark.framework_worker", "--fixture", str(fixture),
               "--run", str(directory), "--framework", framework, "--mode", mode, "--seed", str(seed)]
    write(directory / "config.json", {"schema_version": 1, "kind": "single", "task": "framework_resnet18",
                                     "args": {"framework": framework, "mode": mode, "seed": seed, "batch": batch, "device": "cuda"},
                                     "monitoring": {"enabled": True, "external": True, "interval_s": .5}})
    write(directory / "command.json", command)
    print(directory.name, flush=True)
    start = time.perf_counter()
    with GpuSampler() as sampler, (directory / "process.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            with ResourceRecorder(sampler, interval_s=.5, jsonl_path=directory / "resources.jsonl", root_pid=proc.pid) as recorder:
                code = proc.wait(timeout=1800)
            if recorder.error:
                raise RuntimeError(recorder.error)
        except BaseException:
            from apps.console.jobs import kill_tree
            kill_tree(proc.pid)
            raise
    if code:
        write(directory / "result.json", {"schema_version": 1, "status": "failed", "exit_code": code,
                                         "environment": env_info, "failure": {"kind": "worker_exit", "log": "process.log"}})
        raise RuntimeError(f"{directory.name} exited {code}, see process.log")
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    resources = [json.loads(line) for line in (directory / "resources.jsonl").read_text(encoding="utf-8").splitlines()]
    result.update(environment=env_info, process_wall_s=time.perf_counter()-start,
                  resources=summarize_resources(resources, result["measurement_start_s"], result["measurement_end_s"]))
    write(directory / "result.json", result)
    checked = check_run(directory)
    if not checked["valid"]:
        raise RuntimeError(checked["errors"])
    return result


def report(out, protocol, results, alignments):
    lines = ["# ResNet18 同条件跨框架比较", "",
             f"同一台 GPU 上，以 {protocol['samples']} 帧道路图像、{protocol['batch']} 的 batch 和相同的 FP32 初值比较 MyFlows 与 PyTorch。"
             f"固定时间段划分得到 {protocol['train']} 个训练样本和 {protocol['val']} 个验证样本。每轮共享批次清单并共同丢弃训练尾批。", "",
             "两端均为 ImageNet stem、w64、无卷积 bias、带 padding 的 maxpool、训练模式 BN 和两输出回归头。"
             "Adam 的 lr=0.001、β=(0.9, 0.999)、ε=1e−8。预处理、样本顺序与 NPZ 初值的 SHA-256 在 fixture/protocol.json。", "",
             "| seed | 输出最大绝对差 | loss 绝对差 | 全部梯度相对 L2 | 梯度最大绝对差 | 一步后参数最大绝对差 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for key, comparison in alignments.items():
        g = comparison["groups"]
        lines.append(f"| {key} | {g['logits']['max_abs']:.3g} | {g['loss']['max_abs']:.3g} | {g['grad']['relative_l2']:.3g} | "
                     f"{g['grad']['max_abs']:.3g} | {g['updated']['max_abs']:.3g} |")
    lines += ["", "alignment-s*.json 保存每个张量的形状、最大绝对差、相对 L2 和带 1e−6 分母下限的逐元素最大相对差。", "",
              "| 模式 | 框架 | step ms | samples/s | GPU % | 整卡峰值 MiB | 分配器已用峰值 MiB | CPU % | RSS 峰值 MiB |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mode in ("fixed", "e2e"):
        for framework in ("myflows", "torch", "torch-autotuned"):
            group = [r for r in results if r["mode"] == mode and r["framework"] == framework]
            if not group:
                continue
            def fmt(vals):
                vals = [v for v in vals if v is not None]
                return f"{statistics.median(vals):.2f}" if vals else "缺测"
            vals = [fmt([r["median_step_ms"] for r in group]), fmt([r["samples_per_s"] for r in group])]
            for k in ("gpu_util_pct_median", "gpu_device_mib_sampled_peak"):
                vals.append(fmt([r["resources"][k] for r in group]))
            vals += [fmt([r["allocator"]["peak_used_mib"] for r in group]), fmt([r["resources"]["cpu_util_pct_mean"] for r in group]),
                     fmt([r["resources"]["rss_bytes_sampled_peak"]/2**20 if r["resources"]["rss_bytes_sampled_peak"] is not None else None for r in group])]
            lines.append(f"| {mode} | {framework} | " + " | ".join(vals) + " |")
    lines += ["", "fixed 重用已传入设备的 batch。e2e 包含从共同 NPY 取 batch 与 H2D。每步在计时边界同步。"
              "各计时进程先预热，再从共同初值重建模型、BN 和优化器，并重置批次顺序。默认 torch 行启用 cuDNN deterministic，"
              "autotuned 行单独启用 cuDNN benchmark。所有 torch 行关闭 TF32。时间表对共同种子取中位数。",
              "", "| 框架 | 实际训练样本/seed | 最终验证角度 MAE 各 seed | mean ± sample std | 末轮训练 / 验证 MSE 均值 | 初始化 / 训练墙钟均值 s |",
              "| --- | --- | --- | --- | --- | --- |"]
    for framework in ("myflows", "torch"):
        group = [r for r in results if r["mode"] == "quality" and r["framework"] == framework]
        maes = [r["epochs"][-1]["val_angle_mae"] for r in group]
        if maes:
            dispersion = f"{statistics.stdev(maes):.5f}" if len(maes) > 1 else "单 seed"
            last = [r["epochs"][-1] for r in group]
            lines.append(f"| {framework} | {group[0]['actual_samples']} | {', '.join(f'{v:.5f}' for v in maes)} | "
                         f"{statistics.mean(maes):.5f} ± {dispersion} | {statistics.mean(r['loss'] for r in last):.6g} / "
                         f"{statistics.mean(r['val_loss'] for r in last):.6g} | {statistics.mean(r['init_s'] for r in group):.2f} / "
                         f"{statistics.mean(sum(e['train_wall_s'] for e in r['epochs']) for r in group):.2f} |")
    lines += ["", "MyFlows 的卷积累加顺序、FP64 通道规约和显式舍入 Adam 与 cuDNN、PyTorch 的内核实现有差异。"
              "接近零的激活可能在 ReLU 产生不同的门控，微小的梯度符号差异也会被 Adam 首步放大。"
              "PyTorch 显存取分配器精确高水位，CuPy 已用峰值取阶段边界采样，两者与统一 NVML 设备采样分列。"
              "CPU 为主机利用率，RSS 为进程树采样峰值。", "",
              f"原始归档位于 `{out.relative_to(ROOT).as_posix()}/`。独立进程的 result.json 包含初始化、预热、训练总时间和各轮质量。"
              f"共同图像预处理耗时 {protocol['data_prepare_s']:.2f} s。summary.json SHA-256 为 `{sha256(out/'summary.json')}`。", "",
              f"复现命令为 `python -m benchmark.compare_frameworks_resnet --out runs/stage3/frameworks-repro --samples {protocol['samples']} "
              f"--batch {protocol['batch']} --val-size {protocol['val']} --epochs {protocol['epochs']} --seeds {','.join(map(str,protocol['seeds']))} "
              f"--timed-steps {protocol['timed_steps']} --warmup {protocol['warmup']}"
              + (" --graph-opt" if protocol.get("graph_opt") else "") + "`。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    from apps.common.environment import environment
    from apps.common.gpu_lock import GpuLock
    from benchmark.stage1_common import source_state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/semester_2026_fall/frameworks/resnet18-001"))
    parser.add_argument("--data", default="mycar/data")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--graph-opt", action="store_true", help="measure MyFlows graph fusion on the shared protocol")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--timed-steps", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args(argv)
    args.seeds = [int(s) for s in args.seeds.split(",")]
    if not 0 < args.val_size < args.samples or min(args.batch, args.epochs, args.warmup, args.timed_steps) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("invalid dataset, seeds or measurement budget")
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT):
        parser.error("output must be in the workspace")
    out.mkdir(parents=True, exist_ok=False)
    fixture = out / "fixture"
    protocol = prepare(fixture, args)
    env_info = environment()
    write(out / "manifest.json", {"schema_version": 1, "source": source_state(), "environment": env_info,
                                 "protocol_sha256": sha256(fixture / "protocol.json")})
    results, alignments = [], {}
    with GpuLock(label="cross-framework ResNet18 comparison", run_dir=out):
        for seed in args.seeds:
            for framework in ("myflows", "torch"):
                results.append(run_worker(out, fixture, env_info, framework, "align", seed, args.batch))
            a, b = [out / f"{name}-align-s{seed}" / "alignment.npz" for name in ("myflows", "torch")]
            comparison = alignment(a, b)
            write(out / f"alignment-s{seed}.json", comparison)
            alignments[seed] = comparison
            if not comparison["passed"] or comparison["groups"]["grad"]["tensors"] != 62:
                raise RuntimeError(f"seed {seed}: single-batch alignment failed, inspect alignment-s{seed}.json")
        for seed in args.seeds:
            for mode in ("fixed", "e2e", "quality"):
                frameworks = ("myflows", "torch") if mode == "quality" else ("myflows", "torch", "torch-autotuned")
                for framework in frameworks:
                    results.append(run_worker(out, fixture, env_info, framework, mode, seed, args.batch))
                    write(out / "summary.json", {"schema_version": 1, "task": "framework_resnet18", "protocol": protocol, "runs": results})
    report(out, protocol, results, alignments)


if __name__ == "__main__":
    main()
