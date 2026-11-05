"""Verify frozen-stage gradients byte for byte and alternate actual training runs across two checkouts.

Run --part reference on the pre-change tree, --part verify after the change, then --part speed.
The two trees have separate MyFlows imports and native builds. --smoke uses a shorter separate archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("numpy", "cupy", "cuda_native_register", "cuda_native_cublas")
FREEZE_LEVELS = {"resnet18": range(6), "mobilenet_v2": (0, 3), "alexnet": (0, 5)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(array):
    array = np.ascontiguousarray(array)
    sha = hashlib.sha256(array.dtype.str.encode() + str(array.shape).encode())
    sha.update(memoryview(array).cast("B"))
    return sha.hexdigest()


def numeric_worker(args):
    tree = args.tree.resolve()
    sys.path.insert(0, str(tree))
    import MyFlows as ms
    from MyFlows.core.device import asnumpy, set_device
    from MyFlows.utils.checkpoint import named_state
    from MyFlows.utils.initializers import make_initializer
    from MyFlows.utils.pretrained import build_model, freeze_stages, frozen_fingerprint, load_torchvision_weights

    if not Path(ms.__file__).resolve().is_relative_to(tree):
        raise RuntimeError("MyFlows was imported from a different checkout")
    device = "cpu" if args.backend == "numpy" else "cuda"
    set_device(device)
    ms.set_default_dtype(np.float32)
    # Use the production torchvision topology and converted source weights, including full AlexNet FC layers.
    model = build_model(args.arch, 2, dropout=0., initializer=make_initializer("kaiming_normal", seed=7),
                        conv_backend=args.backend, input_norm="imagenet")
    source = ROOT / "runs" / "pretrained" / "converted" / f"{args.arch}.npz"
    load = load_torchvision_weights(model, source, replace_head=True)
    freeze_stages(model, args.level)
    frozen_before = frozen_fingerprint(model)
    rng = np.random.default_rng(17)
    size = 64 if args.arch == "alexnet" else 32
    inputs = rng.uniform(0, 1, (2, 3, size, size)).astype(np.float32)
    targets = rng.normal(0, .2, (2, 2)).astype(np.float32)
    x, y = ms.Variable(inputs, requires_grad=False), ms.Variable(targets, requires_grad=False)
    model.train(True)
    logits = model(x)
    loss = ms.MSELoss(logits, y)
    graph = ms.Graph(loss)
    opt = ms.Adam(graph, learning_rate=.001)
    params, buffers = named_state(model)
    initial = {name: digest(asnumpy(p.value)) for name, p in params.items()}
    opt.one_step()
    trainable = {name: p for name, p in params.items() if p.trainable}
    gradients = {name: digest(asnumpy(opt.get_gradient(p))) for name, p in trainable.items()}
    frozen_with_grad = [name for name, p in params.items() if not p.trainable and p.grad is not None]
    opt.update()
    frozen_after = frozen_fingerprint(model)
    write(args.output, {"schema_version": 1, "arch": args.arch, "freeze": args.level, "backend": args.backend,
                        "input_sha256": digest(inputs), "target_sha256": digest(targets), "source": load.get("source"),
                        "initial": initial, "gradients": gradients, "loss": float(asnumpy(loss.value)),
                        "logits": digest(asnumpy(logits.value)), "updated": {k: digest(asnumpy(p.value)) for k, p in trainable.items()},
                        "buffers": {k: digest(asnumpy(getattr(layer, attr))) for k, (layer, attr) in buffers.items()},
                        "frozen_before": frozen_before, "frozen_after": frozen_after, "frozen_with_grad": frozen_with_grad})


def compare_numeric(before, after):
    errors = []
    for key in ("arch", "freeze", "backend", "input_sha256", "target_sha256", "logits", "loss", "frozen_before", "frozen_after"):
        if before[key] != after[key]:
            errors.append(key)
    for group in ("initial", "gradients", "updated", "buffers"):
        keys = before[group].keys() | after[group].keys()
        errors.extend(f"{group}.{key}" for key in sorted(keys) if before[group].get(key) != after[group].get(key))
    if after["frozen_with_grad"]:
        errors.append("frozen parameters still have gradients")
    if after["frozen_before"] != after["frozen_after"]:
        errors.append("frozen fingerprint changed")
    return errors


def numerics(args, out):
    cases = [(arch, level, backend) for arch, levels in FREEZE_LEVELS.items() for level in levels for backend in BACKENDS]
    if args.smoke:
        cases = [("resnet18", 3, "numpy"), ("resnet18", 3, "cuda_native_register")]
    reference = args.part == "reference"
    tree = args.old if reference else ROOT
    side = "before" if reference else "after"
    results = []
    for arch, level, backend in cases:
        name = f"{arch}-k{level}-{backend}.json"
        target = out / "numeric" / side / name
        if target.exists():
            raise FileExistsError(f"{target} exists, select a new output directory")
        command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", "--tree", str(tree),
                   "--arch", arch, "--level", str(level), "--backend", backend, "--output", str(target)]
        print(side, arch, level, backend, flush=True)
        subprocess.run(command, cwd=tree, check=True, timeout=600)
        if not reference:
            after = read(target)
            errors = compare_numeric(read(out / "numeric" / "before" / name), after)
            results.append({"case": name, "trainable_tensors": len(after["gradients"]), "errors": errors})
            write(out / "numerical-checks.json", results)
    if any(row["errors"] for row in results):
        raise RuntimeError("bitwise verification failed, see numerical-checks.json")


def timed_run(tree, run_dir, level, size, args):
    run_dir.mkdir(parents=True)
    command = [sys.executable, "-X", "utf8", "-m", "apps.train", "--device", "cuda", "--dtype", "float32",
               "--conv-backend", "cuda_native_register", "--data", str(ROOT / "mycar" / "data"), "--batch", "2",
               "--epochs", "1", "--seed", "1", "--max-samples", str(args.samples), "--sample-seed", "1",
               "--pretrained", str(ROOT / "runs/pretrained/converted/resnet18.npz"), "--freeze-stages", str(level),
               "--image-size", size, "--phase-timing-every", "1", "--monitor", "on", "--no-tensorboard",
               "--checkpoint-every", "0", "--run-dir", str(run_dir), "--log-dir", str(run_dir),
               "--out", str(run_dir / "checkpoint"), "--best-out", str(run_dir / "best"), "--stop-file", str(run_dir / "STOP")]
    write(run_dir / "command.json", command)
    with (run_dir / "process.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=tree, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    from tools.check_run_format import check_run

    checked = check_run(run_dir)
    if not checked["valid"]:
        raise ValueError(checked["errors"])
    steps = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    steps = [row for row in steps if row["kind"] == "step"][args.warmup:]
    result = read(run_dir / "result.json")
    if result["status"] != "succeeded" or not result["transfer"]["frozen_unchanged"]:
        raise ValueError("run or frozen fingerprint failed")
    resources = [json.loads(line) for line in (run_dir / "resources.jsonl").read_text(encoding="utf-8").splitlines()]
    window = [row for row in resources if steps[0]["monotonic_s"] - steps[0]["step_total_ms"]/1000 <= row["monotonic_s"] <= steps[-1]["monotonic_s"]]
    watts = [(r["gpu_power_w"], r["gpu_power_limit_w"]) for r in window if r["gpu_power_w"] is not None and r["gpu_power_limit_w"]]
    return {**{key: statistics.median(row[key] for row in steps) for key in
               ("forward_ms", "backward_ms", "optimizer_ms", "data_load_ms", "step_total_ms", "samples_per_sec_total")},
            "steps": len(steps), "samples": result["samples"], "pool_peak_mib": result["memory"]["peak_pool_mb"],
            "used_peak_mib": result["memory"]["peak_used_mb"], "loss": result["metrics"]["last_train_loss"],
            "power_capped_share": sum(p >= .97 * lim for p, lim in watts)/len(watts) if watts else None}


def speed(args, out):
    rows = []
    sizes = ("120x160",) if args.smoke else ("120x160", "224x224")
    levels = (3,) if args.smoke else (0, 2, 3, 4, 5)
    for size in sizes:
        for level in levels:
            for repeat in range(1, args.repeats + 1):
                for side, tree in (("before", args.old), ("after", ROOT)):
                    label = f"{size}-k{level}-r{repeat}-{side}"
                    print(label, flush=True)
                    row = timed_run(tree, out / "speed" / label, level, size, args)
                    rows.append({"size": size, "level": level, "repeat": repeat, "side": side, **row})
                    write(out / "speed.json", rows)


def report(out):
    checks, rows = read(out / "numerical-checks.json"), read(out / "speed.json")
    speed_manifest = read(out / "speed-manifest.json")
    if len(checks) != 40 or any(row["errors"] for row in checks):
        raise ValueError("expected the full passing numerical matrix")
    repeats = speed_manifest["repeats"]
    if repeats < 2 or len(rows) != 2 * 5 * 2 * repeats:
        raise ValueError("expected repeated A/B measurements at both sizes and five freeze levels")
    lines = ["# 冻结阶段反向截断的数值与性能对照", "",
             f"40 组同后端逐字节比较通过，覆盖 {sum(row['trainable_tensors'] for row in checks):,} 组可训练参数。"
             "每组核对输入、初值、logits、loss、全部可训练梯度、一步 Adam 后的可训练参数、BN buffer 和冻结指纹。"
             "截断后的冻结参数梯度均为 null。", "",
             "| 模型 | 冻结级别 | 对比后端 | 输入 |",
             "| --- | --- | --- | --- |",
             "| torchvision ResNet18 | 0–5 | NumPy、CuPy、register、cuBLAS | 2×3×32×32 |",
             "| MobileNetV2 | 0、3 | 同上 | 2×3×32×32 |",
             "| AlexNet | 0、5 | 同上 | 2×3×64×64 |", "",
             "比较使用相同的转换权重、确定性初始化与批次。每次在独立进程导入指定代码树，原生路径分别使用该树编译的库。"
             "numeric/before 与 numeric/after 含每个张量的类型、形状和原始字节组成的 SHA-256。", "",
             f"实际计时使用 apps.train、w64 torchvision ResNet18、batch 2、register、FP32、同一批 {speed_manifest['samples']} 条真实图像。"
             f"每轮去除前 {speed_manifest['warmup']} 步，CUDA Event 每步计时，资源每 0.5 s 采样。每个配置按旧版/新版交替 {repeats} 轮。"
             "表中为各轮稳态中位数的中位数，括号为重复间总体标准差。显存为 CuPy 池的申请峰值。", "",
             "| 输入 | 冻结段 | 反向 ms 旧→新 | 单步总 ms 旧→新 | samples/s 旧→新 | 池峰值 MiB 旧→新 | 配对反向耗时降低 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    def values(group, key):
        return [row[key] for row in group]
    for size in ("120x160", "224x224"):
        for level in (0, 2, 3, 4, 5):
            before, after = [sorted((row for row in rows if row["size"] == size and row["level"] == level and row["side"] == side),
                                   key=lambda row: row["repeat"]) for side in ("before", "after")]
            if len(before) != repeats or len(after) != repeats or any(a["loss"] != b["loss"] for a, b in zip(before, after)):
                raise ValueError(f"unpaired or divergent training run: {size} k{level}")
            def pair(key, *, spread=False):
                entries = []
                for group in (before, after):
                    vals = values(group, key)
                    entries.append(f"{statistics.median(vals):.2f}" + (f" ({statistics.pstdev(vals):.2f})" if spread else ""))
                return " → ".join(entries)
            reduction = statistics.median(1 - b["backward_ms"]/a["backward_ms"] for a, b in zip(before, after))
            lines.append(f"| {size} | {level} | {pair('backward_ms', spread=True)} | {pair('step_total_ms', spread=True)} | "
                         f"{pair('samples_per_sec_total')} | {pair('pool_peak_mib')} | {reduction:.1%} |")
    old = speed_manifest["old"]["repos"]
    new = speed_manifest["current"]["repos"]
    lines += ["", "初始时间线中，冻结前三段后，首层卷积在 5 个捕获步中仍调用 5 次 backward。"
              "Graph.backward 的全节点清零与反向循环，以及 Conv2D_Op.backward 的无条件 dW，保留了冻结前缀的工作。"
              "新实现由可训练叶子与显式保留的中间节点计算梯度需求，再分配梯度和选择 dX/dW/db。"
              "前向同时释放对应的 im2col 缓冲。", "",
              "规划中冻结 3 / 4 段的约 55% / 77% 是按阶段计算量得到的估算。实测还包含梯度清零与累加、"
              "不同规模 GEMM 的执行效率以及主机发射开销。CPU 读取和 H2D 也计入单步总时长，因此整步收益和反向收益分别报告。", "",
              f"旧代码根仓库 / MyFlows 为 `{old['root']['head'][:8]}` / `{old['MyFlows']['head'][:8]}`，"
              f"正式计时代码为 `{new['root']['head'][:8]}` / `{new['MyFlows']['head'][:8]}`。"
              "原始文件在本报告同目录，speed.json 保存全部运行及功耗墙比例，speed/ 含每步指标、资源与命令。", "",
              "使用独立的旧代码树并初始化其 MyFlows 子模块，在新目录按序运行：", "", "```powershell",
              "python -m benchmark.freeze_before_after --part reference --old <旧代码树> --out runs/stage3/freeze-repro",
              "python -m benchmark.freeze_before_after --part verify --old <旧代码树> --out runs/stage3/freeze-repro",
              "python -m benchmark.freeze_before_after --part speed --old <旧代码树> --out runs/stage3/freeze-repro",
              "python -m benchmark.freeze_before_after --part report --out runs/stage3/freeze-repro", "```", ""]
    for name in ("speed.json", "numerical-checks.json", "speed-manifest.json"):
        lines.append(f"- `{name}` SHA-256 `{hashlib.sha256((out / name).read_bytes()).hexdigest()}`")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("reference", "verify", "speed", "report"))
    parser.add_argument("--old", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/semester_2026_fall/monitoring/freeze-001"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tree", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--arch", help=argparse.SUPPRESS)
    parser.add_argument("--level", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--backend", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        numeric_worker(args)
        return
    if args.part == "report":
        report((ROOT / args.out).resolve())
        return
    if not args.part or not args.old or args.repeats < 1 or args.warmup < 0 or args.samples <= 2 * args.warmup:
        parser.error("requires --part, --old and a valid measurement budget")
    args.old = args.old.resolve()
    if args.old == ROOT or not (args.old / "MyFlows/core/graph.py").is_file():
        parser.error("--old must be a separate checkout with MyFlows populated")
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT):
        parser.error("expected an output directory inside the workspace")
    from apps.autopilot.provenance import code_state

    manifest = out / f"{args.part}-manifest.json"
    if manifest.exists():
        parser.error("this part already has an archive, keep it and select a new --out")
    write(manifest, {"schema_version": 1, "old": code_state(args.old), "current": code_state(ROOT),
                     "part": args.part, "smoke": args.smoke, "samples": args.samples,
                     "warmup": args.warmup, "repeats": args.repeats})
    if args.part == "speed":
        speed(args, out)
    else:
        numerics(args, out)


if __name__ == "__main__":
    main()
