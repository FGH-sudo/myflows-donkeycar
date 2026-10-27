"""ResNet18 before/after: the current tree against another one (e.g. a worktree of the previous commit).

Every measurement runs in a fresh worker process that imports MyFlows from ``--tree``, so the
previous implementation, its own native DLL included, runs on the same machine, data and
configuration:

* gradients: one training-mode step from identical float32 parameters on several batches
  (``--seeds``), compared with an FP64 NumPy reference computed by the current tree, together with
  the ReLU inputs whose sign differs from FP64;
* speed: per-step time (forward, backward, Adam update and the loss read-back) for fixed
  configurations, plus the CuPy memory-pool size afterwards.

  git worktree add --detach <old> HEAD
  git -C MyFlows worktree add --detach <old>/MyFlows HEAD
  (cd <old> && python tools/build_native_cuda.py)
  python -m benchmark.resnet_before_after --old <old> --out-dir docs/experiments/semester_2026_fall/cuda_lecture19/resnet-001

Only APIs present in both trees are used: conv backends are assigned to the layers after
construction and float32 is forced by converting parameters, as the distributed task used to do.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEED_CONFIGS = (
    # name, stem, width, batch, bn, dtype, backend
    ("w16-frozen-bn", "imagenet", 16, 128, "frozen", "float32", "cuda_native_cublas"),
    ("w16-train-bn", "imagenet", 16, 128, "train", "float32", "cuda_native_cublas"),
    ("w64-train-bn", "imagenet", 64, 32, "train", "float32", "cuda_native_cublas"),
    ("w64-train-bn-cupy", "imagenet", 64, 32, "train", "float32", "cupy"),
    ("apps-train-default", "cifar", 64, 2, "train", "default", "default"),
)
GRAD_CONFIGS = (
    # name, device, dtype, backend
    ("numpy-fp32", "cpu", "float32", "numpy"),
    ("cupy-fp32", "cuda", "float32", "cupy"),
    ("native-fp32", "cuda", "float32", "cuda_native_cublas"),
)


# ------------------------------------------------------------------------------------------------
# Worker (imports MyFlows from --tree)
# ------------------------------------------------------------------------------------------------
def _import(tree):
    sys.path.insert(0, str(Path(tree).resolve()))
    import MyFlows as ms

    if not Path(ms.__file__).resolve().is_relative_to(Path(tree).resolve()):
        raise RuntimeError(f"imported {ms.__file__}, expected a module under {tree}")
    return ms


def _build(ms, args, device, dtype, backend):
    from MyFlows.core.device import set_device, xp
    from MyFlows.core.node import Variable
    from MyFlows.layers.layer import Conv2D
    from MyFlows.layers.resnet import BatchNorm2d
    from MyFlows.ops.convolution import MaxPool2d_Op
    from MyFlows.utils.initializers import make_initializer

    set_device(device)
    model = ms.ResNet18(in_channels=3, output_dim=2, stem=args.stem, base_width=args.width, name="resnet18",
                        initializer=make_initializer("kaiming_normal", seed=0))
    forced = {"float32": np.float32, "float64": np.float64}.get(dtype)
    for layer in model.sub_layers:
        if isinstance(layer, Conv2D) and backend != "default":
            layer.backend = backend
        if isinstance(layer, BatchNorm2d) and forced is not None:
            layer.running_mean = xp.asarray(layer.running_mean, dtype=forced)
            layer.running_var = xp.asarray(layer.running_var, dtype=forced)
    if forced is not None:
        for node in model.params:
            node.value = xp.asarray(node.value, dtype=forced)
    shape = (args.batch, 3, args.height, args.width_px)
    input_dtype = np.float64 if forced is np.float64 else np.float32  # apps/train feeds float32 images
    try:
        x = Variable(xp.zeros(shape, input_dtype), name="X", requires_grad=False)
        y = Variable(xp.zeros((args.batch, 2), input_dtype), name="y", requires_grad=False)
    except TypeError:  # the previous Variable has no requires_grad
        x = Variable(xp.zeros(shape, input_dtype), name="X")
        y = Variable(xp.zeros((args.batch, 2), input_dtype), name="y")
    loss = ms.MSELoss(model(x), y)
    graph = ms.Graph(loss)
    for node in graph.nodes:
        if isinstance(node, MaxPool2d_Op) and backend != "default":
            node.backend = backend
    return model, x, y, loss, graph


def worker_reference(args):
    """FP64 NumPy reference on the current tree; writes the shared float32 fixture and FP64 grads."""
    ms = _import(args.tree)
    model, x, y, loss, graph = _build(ms, args, "cpu", "float64", "numpy")
    rng = np.random.default_rng(args.seed)
    xb = rng.standard_normal(x.value.shape).astype(np.float32)
    yb = (0.5 * rng.standard_normal(y.value.shape)).astype(np.float32)
    params = [np.asarray(p.value, dtype=np.float32) for p in model.params]
    for node, value in zip(model.params, params):
        node.value = value.astype(np.float64)
    x.value, y.value = xb.astype(np.float64), yb.astype(np.float64)
    model.train(True)
    graph.forward()
    graph.backward()
    np.savez(args.fixture, x=xb, y=yb, **{f"param_{i}": p for i, p in enumerate(params)})
    np.savez(args.output, loss=float(loss.value), **{f"grad_{i}": p.grad for i, p in enumerate(model.params)})


def worker_grads(args):
    ms = _import(args.tree)
    from MyFlows.core.device import asnumpy, xp

    model, x, y, loss, graph = _build(ms, args, args.device, args.dtype, args.backend)
    fixture = np.load(args.fixture)
    for i, node in enumerate(model.params):
        node.value = xp.asarray(fixture[f"param_{i}"].astype(node.value.dtype))
    x.value, y.value = xp.asarray(fixture["x"]), xp.asarray(fixture["y"])
    model.train(True)
    graph.forward()
    graph.backward()
    np.savez(args.output, loss=float(loss.value),
             **{f"grad_{i}": asnumpy(p.grad).astype(np.float64) for i, p in enumerate(model.params)})


def worker_relu(args):
    """Forward error of every ReLU input (this configuration against FP64 NumPy of the same tree) and
    how many inputs land on the other side of zero, where the local gradient jumps between 0 and dy."""
    ms = _import(args.tree)
    from MyFlows.core.device import asnumpy, xp
    from MyFlows.ops.activation import ReLU

    fixture = np.load(args.fixture)

    def relu_inputs(device, dtype, backend):
        model, x, y, loss, graph = _build(ms, args, device, dtype, backend)
        for i, node in enumerate(model.params):
            node.value = xp.asarray(fixture[f"param_{i}"].astype(node.value.dtype))
        x.value, y.value = (xp.asarray(fixture[k].astype(x.value.dtype)) for k in ("x", "y"))
        model.train(True)
        graph.forward()
        return [asnumpy(n.parents[0].value).astype(np.float64) for n in graph.sorted_nodes if isinstance(n, ReLU)]

    reference = relu_inputs("cpu", "float64", "numpy")
    values = relu_inputs(args.device, args.dtype, args.backend)
    errors = [float(np.abs(a - b).max() / np.abs(b).max()) for a, b in zip(values, reference)]
    print(json.dumps({"relu_inputs": len(values), "relu_error_median": float(np.median(errors)),
                      "relu_error_max": float(np.max(errors)),
                      "sign_flips": sum(int(((a > 0) != (b > 0)).sum()) for a, b in zip(values, reference))}))


def worker_speed(args):
    ms = _import(args.tree)
    from MyFlows.core.device import xp

    model, x, y, loss, graph = _build(ms, args, "cuda", args.dtype, args.backend)
    if args.bn == "frozen":
        model.eval()
    else:
        model.train(True)
    opt = ms.Adam(graph, learning_rate=1e-3)
    rng = np.random.default_rng(0)
    xb = rng.standard_normal(x.value.shape).astype(np.float32)
    yb = (0.5 * rng.standard_normal(y.value.shape)).astype(np.float32)
    times = []
    for step in range(args.warmup + args.steps):
        xp.cuda.Device().synchronize()
        start = time.perf_counter()
        x.value, y.value = xb, yb  # host batch, copied to the device each step as in training
        opt.one_step()
        opt.update()
        float(loss.value)
        times.append((time.perf_counter() - start) * 1000.0)
    measured = times[args.warmup:]
    pool = xp.get_default_memory_pool()
    print(json.dumps({"median_ms": float(np.median(measured)), "mean_ms": float(np.mean(measured)),
                      "p90_ms": float(np.percentile(measured, 90)), "steps": len(measured),
                      "pool_total_mib": pool.total_bytes() / 2 ** 20,
                      "param_dtype": str(model.params[0].value.dtype)}))


# ------------------------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------------------
def _worker(tree, mode, extra, timeout=3600):
    command = [sys.executable, "-X", "utf8", "-m", "benchmark.resnet_before_after", "--worker", mode,
               "--tree", str(tree), *map(str, extra)]
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace", timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{mode} worker failed for {tree}: {result.stdout[-3000:]}")
    return result.stdout


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def _gradient_rows(seed, run, trees, configs=GRAD_CONFIGS):
    """One training-mode step on the batch of ``seed`` for every tree and configuration."""
    common = ["--stem", "imagenet", "--width", 16, "--batch", 8, "--height", 120, "--width-px", 160,
              "--seed", seed]
    fixture, reference = run.path / f"grad-s{seed}-fixture.npz", run.path / f"grad-s{seed}-fp64.npz"
    _worker(ROOT, "reference", common + ["--fixture", fixture, "--output", reference])
    expected = np.load(reference)
    count = sum(1 for key in expected.files if key.startswith("grad_"))
    norms = np.array([np.linalg.norm(expected[f"grad_{i}"]) for i in range(count)])
    # Conv biases feeding a training-mode BatchNorm have an exactly zero gradient (BN removes the
    # channel mean); their FP64 values are rounding noise, so they are reported as a noise level.
    zero = norms <= 1e-9 * norms.max()
    typical = float(np.median(norms[~zero]))
    rows = []
    for tree_name, tree in trees:
        for name, device, dtype, backend in configs:
            output = run.path / f"grad-s{seed}-{tree_name}-{name}.npz"
            _worker(tree, "grads", common + ["--fixture", fixture, "--output", output, "--device", device,
                                             "--dtype", dtype, "--backend", backend])
            got = np.load(output)
            rel = np.array([np.linalg.norm(got[f"grad_{i}"] - expected[f"grad_{i}"]) / max(norms[i], 1e-300)
                            for i in range(count)])
            noise = max((float(np.linalg.norm(got[f"grad_{i}"])) for i in np.flatnonzero(zero)), default=0.0)
            rows.append({"seed": seed, "tree": tree_name, "config": name,
                         "loss_rel": abs(float(got["loss"]) - float(expected["loss"])) / abs(float(expected["loss"])),
                         "grad_rel_median": float(np.median(rel[~zero])), "grad_rel_p90": float(np.percentile(rel[~zero], 90)),
                         "grad_rel_max": float(np.max(rel[~zero])), "zero_grad_noise": noise / typical,
                         "grad_rel": rel.tolist()})
            trace = _worker(tree, "relu", common + ["--fixture", fixture, "--device", device, "--dtype", dtype,
                                                    "--backend", backend])
            rows[-1].update(json.loads(trace.strip().splitlines()[-1]))
            print("grads", f"seed={seed}", tree_name, name, f"median={rows[-1]['grad_rel_median']:.2e} "
                  f"max={rows[-1]['grad_rel_max']:.2e} noise={rows[-1]['zero_grad_noise']:.2e} "
                  f"relu_err={rows[-1]['relu_error_median']:.2e} flips={rows[-1]['sign_flips']}", flush=True)
    return rows, zero


def _span(rows, key):
    return f"{min(r[key] for r in rows):.1e}–{max(r[key] for r in rows):.1e}" if rows else "—"


def gradients(args, run, trees, configs=GRAD_CONFIGS):
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for seed in seeds:
        seed_rows, zero = _gradient_rows(seed, run, trees, configs)
        rows += seed_rows
    run.results["gradients"] = {"seeds": seeds, "rows": rows, "zero_gradient_params": np.flatnonzero(zero).tolist()}
    first = [r for r in rows if r["seed"] == seeds[0]]
    run.notes.append(
        f"## 单步梯度相对 FP64 的误差（ResNet18 w16，imagenet stem，8×3×120×160，训练模式 BN，seed {seeds[0]}）\n\n"
        f"相同 float32 参数与 batch。{int((~zero).sum())} 个参数统计相对误差 ‖g − g64‖₂/‖g64‖₂ 的中位数、90 分位与最大值；"
        f"另 {int(zero.sum())} 个是紧接训练模式 BN 的卷积 bias，真实梯度恰为 0（BN 减去了通道均值），"
        "“零梯度噪声”列给出其中最大的 ‖g‖₂，以其余参数 ‖g64‖₂ 的中位数为单位（纯舍入噪声）。"
        f"最后两列是前向的对照：{first[0]['relu_inputs']} 个 ReLU 输入各自的最大相对误差（max|x − x64| / max|x64|）的中位数，"
        "以及与同一代码 FP64 结果符号相反的 ReLU 输入个数。符号相反的元素上局部梯度在 0 与 dy 之间跳变，"
        "一个这样的元素就能让该 ReLU 之前（靠近输入一侧）各层的参数梯度出现 1e-3–1e-2 量级的偏差，这与累加误差无关。\n\n"
        + _table(["代码", "配置", "loss 相对误差", "中位数", "90 分位", "最大值", "零梯度噪声",
                  "ReLU 输入误差", "异号 ReLU 输入"],
                 [[r["tree"], r["config"], f"{r['loss_rel']:.2e}", f"{r['grad_rel_median']:.2e}",
                   f"{r['grad_rel_p90']:.2e}", f"{r['grad_rel_max']:.2e}", f"{r['zero_grad_noise']:.2e}",
                   f"{r['relu_error_median']:.2e}", r["sign_flips"]] for r in first]))
    if len(seeds) == 1:
        return
    summary = []
    for tree_name, _ in trees:
        for name, *_ in configs:
            group = [r for r in rows if r["tree"] == tree_name and r["config"] == name]
            clean = [r for r in group if not r["sign_flips"]]
            flipped = [r for r in group if r["sign_flips"]]
            summary.append([tree_name, name, f"{len(flipped)}/{len(group)}", sum(r["sign_flips"] for r in group),
                            _span(clean, "grad_rel_median"), _span(clean, "grad_rel_max"),
                            _span(flipped, "grad_rel_median"), _span(flipped, "grad_rel_max"),
                            _span(group, "relu_error_median")])
    run.notes.append(
        f"## 多个 batch 的复验（seed {', '.join(map(str, seeds))}）\n\n"
        "参数相同，batch 由各个种子生成。离 0 足够近的 ReLU 输入在 float32 中可能落到 0 的另一侧；是否出现取决于 batch，"
        "所以按有无异号分开给出各 batch 的范围：无异号 batch 的误差反映累加精度，有异号 batch 的误差主要来自异号本身。\n\n"
        + _table(["代码", "配置", "有异号的 batch", "异号输入合计", "无异号：中位数", "无异号：最大值",
                  "有异号：中位数", "有异号：最大值", "ReLU 输入误差"], summary)
        + "\n\n逐 batch：\n\n"
        + _table(["seed", "代码", "配置", "中位数", "最大值", "ReLU 输入误差", "异号 ReLU 输入"],
                 [[r["seed"], r["tree"], r["config"], f"{r['grad_rel_median']:.2e}", f"{r['grad_rel_max']:.2e}",
                   f"{r['relu_error_median']:.2e}", r["sign_flips"]] for r in rows]))


def speed(args, run, trees):
    rows = []
    for name, stem, width, batch, bn, dtype, backend in SPEED_CONFIGS:
        for tree_name, tree in trees:
            extra = ["--stem", stem, "--width", width, "--batch", batch, "--height", 120, "--width-px", 160,
                     "--bn", bn, "--dtype", dtype, "--backend", backend if backend != "default" else
                     ("default" if tree_name == "before" else "cuda_native_cublas"),
                     "--warmup", args.warmup, "--steps", args.steps]
            line = _worker(tree, "speed", extra).strip().splitlines()[-1]
            rows.append({"config": name, "tree": tree_name, **json.loads(line)})
            print("speed", name, tree_name, line, flush=True)
    run.results["speed"] = rows
    table = []
    for name, *_ in SPEED_CONFIGS:
        before = next(r for r in rows if r["config"] == name and r["tree"] == "before")
        after = next(r for r in rows if r["config"] == name and r["tree"] == "after")
        table.append([name, f"{before['median_ms']:.1f}", f"{after['median_ms']:.1f}",
                      f"{before['median_ms'] / after['median_ms']:.2f}×", f"{before['pool_total_mib']:.0f}",
                      f"{after['pool_total_mib']:.0f}", f"{before['param_dtype']} → {after['param_dtype']}"])
    run.notes.append(
        f"## 每步耗时（中位数，{args.steps} 步，预热 {args.warmup} 步）\n\n"
        "一步 = 主机 batch 拷到 GPU、前向、反向、Adam 更新、读回 loss。配置：w16 为 base_width=16、batch 128；"
        "w64 为 base_width=64、batch 32；输入 3×120×160。apps-train-default 为 apps/train 的默认配置"
        "（cifar stem、w64、batch 2）：修改前该脚本未指定 dtype 与后端，模型参数为 float64、卷积走 CuPy；"
        "修改后为 float32 + cuda_native_cublas。\n\n"
        + _table(["配置", "修改前 ms", "修改后 ms", "加速", "修改前显存池 MiB", "修改后显存池 MiB", "参数 dtype"], table))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", choices=("reference", "grads", "relu", "speed"), help=argparse.SUPPRESS)
    parser.add_argument("--tree", default=str(ROOT), help=argparse.SUPPRESS)
    parser.add_argument("--old", help="root of the code tree to compare against (with its native DLL built)")
    parser.add_argument("--out-dir")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    for name, default in (("--stem", "imagenet"), ("--bn", "train"), ("--dtype", "float32"),
                          ("--backend", "cuda_native_cublas"), ("--device", "cuda")):
        parser.add_argument(name, default=default, help=argparse.SUPPRESS)
    for name, default in (("--width", 16), ("--batch", 8), ("--height", 120), ("--width-px", 160)):
        parser.add_argument(name, type=int, default=default, help=argparse.SUPPRESS)
    parser.add_argument("--fixture", help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--seeds", default="1,2,3,4,5,6",
                        help="comma list of seeds; each makes one batch for the gradient comparison")
    parser.add_argument("--skip", default="", help="comma list of parts to skip: gradients,speed")
    args = parser.parse_args(argv)
    if args.worker:
        return {"reference": worker_reference, "grads": worker_grads, "relu": worker_relu,
                "speed": worker_speed}[args.worker](args)
    if not args.old or not args.out_dir:
        parser.error("--old and --out-dir are required")
    from benchmark.stage1_common import RunArtifacts, command

    old = Path(args.old).resolve(strict=True)
    trees = (("before", old), ("after", ROOT))
    with RunArtifacts(args.out_dir, vars(args)) as run:
        run.manifest["trees"] = {name: {"path": str(path), "myflows_commit": command(["git", "rev-parse", "HEAD"],
                                                                                     cwd=path / "MyFlows")}
                                 for name, path in trees}
        skip = set(filter(None, args.skip.split(",")))
        if "gradients" not in skip:
            gradients(args, run, trees)
        if "speed" not in skip:
            speed(args, run, trees)
        run.notes.append("“修改前”为 --old 指向的代码树（含其自身编译的原生 DLL），“修改后”为当前工作区；"
                         "两者在同一台机器、同一 Python 环境中逐个进程运行。")


if __name__ == "__main__":
    main()
