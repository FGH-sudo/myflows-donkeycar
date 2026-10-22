"""Register-tiled GEMM engine against cuBLAS in the native convolution (current tree).

* gemm: the convolution GEMMs of one ResNet18 training step (forward rows = cols W^T, dW = dY^T cols,
  dX = dY W) per distinct shape, both engines timed with CUDA events in interleaved rounds, with
  their error against FP64 and the tile shape / split-K the register engine picks;
* gradients: one training-mode step on several batches against FP64 NumPy, with the ReLU inputs that
  change sign (the workers of benchmark.resnet_before_after);
* speed: per-step time of ResNet18 configurations, engines interleaved over several rounds.

  python -m benchmark.gemm_engines --out-dir docs/experiments/semester_2026_fall/cuda_lecture19/engine-001
"""

import argparse
import json
import math
from collections import Counter

import numpy as np

from benchmark.resnet_before_after import ROOT, _table, _worker, gradients

ENGINES = (("cublas", "cuda_native_cublas"), ("register", "cuda_native_register"))
LABELS = {"cublas": "cuBLAS", "register": "寄存器分块"}
GEMM_MODELS = (
    # name, stem, base width, batch (input 3 x 120 x 160)
    ("w16", "imagenet", 16, 128),
    ("w64", "imagenet", 64, 32),
)
SPEED_CONFIGS = (
    # name, stem, base width, batch, bn
    ("w16-frozen-bn", "imagenet", 16, 128, "frozen"),
    ("w16-train-bn", "imagenet", 16, 128, "train"),
    ("w64-train-bn", "imagenet", 64, 32, "train"),
    ("apps-train-default", "cifar", 64, 2, "train"),
)
GRAD_CONFIGS = tuple((f"native-{engine}", "cuda", "float32", backend) for engine, backend in ENGINES)


def conv_gemms(stem, width, batch):
    """Count of every (part, layout, M, N, K) GEMM in the native convolutions of one training step."""
    import MyFlows as ms
    from MyFlows.core.device import set_device, xp
    from MyFlows.core.node import Variable
    from MyFlows.ops.cuda_native import native
    from MyFlows.utils.initializers import make_initializer

    set_device("cuda")
    calls = []
    original = native.conv2d_backward

    def record(x, weight, grad_y, cols, **kwargs):
        calls.append((weight.shape, grad_y.shape, kwargs.get("need_input_grad", True)))
        return original(x, weight, grad_y, cols, **kwargs)

    native.conv2d_backward = record  # Conv2D_Op imports it at call time
    try:
        model = ms.ResNet18(in_channels=3, output_dim=2, stem=stem, base_width=width, name="resnet18",
                            conv_backend="cuda_native_register",
                            initializer=make_initializer("kaiming_normal", seed=0))
        x = Variable(xp.zeros((batch, 3, 120, 160), np.float32), name="X", requires_grad=False)
        y = Variable(xp.zeros((batch, 2), np.float32), name="y", requires_grad=False)
        graph = ms.Graph(ms.MSELoss(model(x), y))
        model.train(True)
        graph.forward()
        graph.backward()
    finally:
        native.conv2d_backward = original
    gemms = Counter()
    for (co, ci, kh, kw), (n, _, oh, ow), need_dx in calls:
        m, k = n * oh * ow, ci * kh * kw
        gemms[("forward", "NT", m, co, k)] += 1
        gemms[("dW", "TN", co, k, m)] += 1
        if need_dx:
            gemms[("dX", "NN", m, k, co)] += 1
    return gemms


def _event_ms(xp, fn, reps):
    start, end = xp.cuda.Event(), xp.cuda.Event()
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return xp.cuda.get_elapsed_time(start, end) / reps


def gemm_part(args, run):
    from MyFlows.core.device import xp
    from MyFlows.ops.cuda_native import native

    rng = np.random.default_rng(0)
    rows = []
    for model_name, stem, width, batch in GEMM_MODELS:
        for (part, layout, m, n, k), count in sorted(conv_gemms(stem, width, batch).items(),
                                                       key=lambda item: (item[0][0], -item[0][2] * item[0][3])):
            trans_a, trans_b = layout == "TN", layout == "NT"
            a = rng.standard_normal((k, m) if trans_a else (m, k)).astype(np.float32)
            b = rng.standard_normal((n, k) if trans_b else (k, n)).astype(np.float32)
            exact = (a.T if trans_a else a).astype(np.float64) @ (b.T if trans_b else b).astype(np.float64)
            da, db, out = xp.asarray(a), xp.asarray(b), xp.empty((m, n), xp.float32)
            calls = {engine: (lambda engine=engine: native.gemm(da, db, trans_a=trans_a, trans_b=trans_b,
                                                                engine=engine, out=out))
                     for engine, _ in ENGINES}
            row = {"model": model_name, "part": part, "layout": layout, "m": m, "n": n, "k": k, "count": count}
            row["tile"], row["slices"] = native.gemm_plan(m, n, k)
            for engine, _ in ENGINES:
                got = xp.asnumpy(calls[engine]()).astype(np.float64)
                row[f"{engine}_error"] = float(np.abs(got - exact).max() / np.abs(exact).max())
            # Enough back-to-back calls for ~20 ms per measurement; engines alternate first per round.
            reps = max(3, min(200, math.ceil(20.0 / max(_event_ms(xp, calls["cublas"], 1), 1e-3))))
            samples = {engine: [] for engine, _ in ENGINES}
            for round_index in range(args.rounds):
                for engine, _ in (ENGINES if round_index % 2 == 0 else ENGINES[::-1]):
                    samples[engine].append(_event_ms(xp, calls[engine], reps))
            for engine, _ in ENGINES:
                row[f"{engine}_ms"] = float(np.median(samples[engine]))
                run.timings.append({"part": "gemm", "model": model_name, "gemm": part, "shape": f"{m}x{n}x{k}",
                                    "engine": engine, "reps": reps, "ms": row[f"{engine}_ms"]})
            rows.append(row)
            print("gemm", model_name, part, layout, (m, n, k), row["tile"], row["slices"],
                  f"cublas={row['cublas_ms']:.3f} register={row['register_ms']:.3f}",
                  f"err {row['cublas_error']:.1e}/{row['register_error']:.1e}", flush=True)
            del da, db, out
            xp.get_default_memory_pool().free_all_blocks()
    run.results["gemm"] = rows

    totals = []
    for model_name, *_ in GEMM_MODELS:
        for part in ("forward", "dW", "dX", "合计"):
            group = [r for r in rows if r["model"] == model_name and part in (r["part"], "合计")]
            cublas = sum(r["count"] * r["cublas_ms"] for r in group)
            register = sum(r["count"] * r["register_ms"] for r in group)
            totals.append([model_name, part, sum(r["count"] for r in group), f"{cublas:.2f}", f"{register:.2f}",
                           f"{register / cublas:.2f}"])
    run.notes.append(
        "## 卷积 GEMM 逐形状对比（一次训练步）\n\n"
        "取 ResNet18（imagenet stem，输入 3×120×160）一次训练步中原生卷积的全部 GEMM：前向 rows = cols·Wᵀ（NT）、"
        "dW = dYᵀ·cols（TN）、dX = dY·W（NN，输入层不需要 dX）。同形状的 GEMM 只测一次，按出现次数加权求和。"
        f"计时为 CUDA event 包住连续多次调用的平均（每次约 20 ms），两个引擎交替先后共 {args.rounds} 轮取中位数；"
        "误差为 max|C − C64| / max|C64|，操作数为标准正态随机数。\n\n"
        + _table(["模型", "GEMM", "次数", "cuBLAS ms", "寄存器分块 ms", "寄存器/cuBLAS"], totals)
        + "\n\n逐形状（tile 与 split 为寄存器分块引擎的选择，split = 1 表示不切分 K）：\n\n"
        + _table(["模型", "GEMM", "次数", "M × N × K", "tile", "split", "cuBLAS ms", "寄存器分块 ms", "比值",
                  "cuBLAS 误差", "寄存器分块误差"],
                 [[r["model"], f"{r['part']} ({r['layout']})", r["count"], f"{r['m']}×{r['n']}×{r['k']}", r["tile"],
                   r["slices"], f"{r['cublas_ms']:.3f}", f"{r['register_ms']:.3f}",
                   f"{r['register_ms'] / r['cublas_ms']:.2f}", f"{r['cublas_error']:.1e}", f"{r['register_error']:.1e}"]
                  for r in rows]))


def speed_part(args, run):
    rows = []
    for round_index in range(args.rounds):
        for name, stem, width, batch, bn in SPEED_CONFIGS:
            for engine, backend in (ENGINES if round_index % 2 == 0 else ENGINES[::-1]):
                extra = ["--stem", stem, "--width", width, "--batch", batch, "--height", 120, "--width-px", 160,
                         "--bn", bn, "--dtype", "float32", "--backend", backend,
                         "--warmup", args.warmup, "--steps", args.steps]
                line = _worker(ROOT, "speed", extra).strip().splitlines()[-1]
                rows.append({"config": name, "engine": engine, "round": round_index, **json.loads(line)})
                run.timings.append({"part": "speed", "config": name, "engine": engine, "round": round_index,
                                    "ms": rows[-1]["median_ms"]})
                print("speed", name, engine, round_index, line, flush=True)
    run.results["speed"] = rows
    table = []
    for name, *_ in SPEED_CONFIGS:
        medians = {engine: [r["median_ms"] for r in rows if r["config"] == name and r["engine"] == engine]
                   for engine, _ in ENGINES}
        mid = {engine: float(np.median(values)) for engine, values in medians.items()}
        pool = {engine: max(r["pool_total_mib"] for r in rows if r["config"] == name and r["engine"] == engine)
                for engine, _ in ENGINES}
        table.append([name, f"{mid['cublas']:.1f}", f"{mid['register']:.1f}", f"{mid['register'] / mid['cublas']:.2f}",
                      " / ".join(f"{v:.1f}" for v in medians["cublas"]),
                      " / ".join(f"{v:.1f}" for v in medians["register"]),
                      f"{pool['cublas']:.0f} / {pool['register']:.0f}"])
    run.notes.append(
        f"## ResNet18 每步耗时（{args.rounds} 轮交替，每轮 {args.steps} 步取中位数，预热 {args.warmup} 步）\n\n"
        "一步 = 主机 batch 拷到 GPU、前向、反向、Adam 更新、读回 loss，配置同 resnet-001：w16 为 base_width=16、"
        "batch 128；w64 为 base_width=64、batch 32；apps-train-default 为 cifar stem、w64、batch 2。"
        "两种后端只有卷积 GEMM 的引擎不同，im2col、尾处理、池化与其余算子相同。\n\n"
        + _table(["配置", "cuBLAS ms", "寄存器分块 ms", "寄存器/cuBLAS", "cuBLAS 各轮", "寄存器分块各轮",
                  "显存池 MiB（cuBLAS / 寄存器）"], table))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="1,2,3,4,5,6",
                        help="comma list of seeds; each makes one batch for the gradient comparison")
    parser.add_argument("--rounds", type=int, default=3, help="interleaved timing rounds of both engines")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--skip", default="", help="comma list of parts to skip: gemm,gradients,speed")
    args = parser.parse_args(argv)
    from benchmark.stage1_common import RunArtifacts

    with RunArtifacts(args.out_dir, vars(args)) as run:
        skip = set(filter(None, args.skip.split(",")))
        if "gemm" not in skip:
            gemm_part(args, run)
        if "gradients" not in skip:
            gradients(args, run, (("当前", ROOT),), GRAD_CONFIGS)
        if "speed" not in skip:
            speed_part(args, run)
        run.notes.append("全部在当前工作区运行：cuda_native_cublas 与 cuda_native_register 只有 GEMM 引擎不同"
                         "（cublasSgemm，IEEE FP32 / 第 19 讲寄存器分块 kernel，FP32 顺序 FMA，"
                         "小输出按 K 切分后 FP64 归约）。")


if __name__ == "__main__":
    main()
