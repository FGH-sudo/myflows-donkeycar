"""Six-seed numerical and small DonkeyCar training studies of experimental convolution/GEMM backends."""

import argparse
import json
from pathlib import Path
import statistics

from benchmark.resnet_before_after import ROOT, _gradient_rows
from benchmark.transfer_study import WEIGHTS, attempt, init_study, split_for, write


class NumericalRun:
    def __init__(self, path):
        self.path = path
        self.config = {}


def numerics(out, old, modes):
    run = NumericalRun(out / "numeric")
    run.path.mkdir(exist_ok=True)
    rows = []
    for seed in range(1, 7):
        before, _ = _gradient_rows(
            seed, run, [("baseline", old)], [("cublas", "cuda", "float32", "cuda_native_cublas")]
        )
        current, _ = _gradient_rows(
            seed, run, [("experiment", ROOT)], [(b, "cuda", "float32", b) for b in modes]
        )
        rows.extend(before + current)
        write(out / "gradients.json", rows)
    return rows


def quality(out, source, modes):
    split = split_for("pilot", out)
    common = {
        "device": "cuda",
        "dtype": "float32",
        "batch": 8,
        "epochs": 2,
        "lr": 0.0003,
        "max_samples": 2000,
        "split_file": str(split),
        "data": str(ROOT / "mycar/data"),
        "pretrained": str(WEIGHTS),
        "image_size": "120x160",
        "input_norm": "imagenet",
        "phase_timing_every": 10,
        "no_tensorboard": True,
        "checkpoint_every": 0,
        "initial_validation": True,
        "num_workers": 0,
    }
    rows = []
    for seed in (1, 2, 3):
        for backend in ("cuda_native_cublas", *modes):
            run, result = attempt(
                ROOT, source, out / backend / f"s{seed}", {**common, "conv_backend": backend, "seed": seed}
            )
            steps = [json.loads(s) for s in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            steps = [r for r in steps if r["kind"] == "step"][8:]
            rows.append(
                {
                    "backend": backend,
                    "seed": seed,
                    "val_mae": result["metrics"]["val_angle_mae_final"],
                    "pool_mib": result["memory"]["peak_pool_mb"],
                    "step_ms": statistics.median(r["step_total_ms"] for r in steps),
                    "samples_s": 1000
                    * sum(r["batch_size"] for r in steps)
                    / sum(r["step_total_ms"] for r in steps),
                    "loss_scaling": result.get("loss_scaling"),
                    "epochs": result["epochs"],
                    "run": run.relative_to(ROOT).as_posix(),
                }
            )
            write(out / "quality.json", rows)
    return rows


def winograd_counts(out):
    import numpy as np
    from MyFlows.core.device import get_device, set_device, xp
    from MyFlows.core.node import Variable
    from MyFlows.ops.convolution import Conv2D_Op
    from benchmark.gemm_engines import _event_ms

    previous = get_device()
    set_device("cuda")
    rows = []
    for n, ci, co, h, w in ((2, 16, 16, 31, 37), (8, 32, 64, 32, 32), (2, 64, 64, 120, 160)):
        rng = np.random.default_rng(13)
        x = Variable(xp.asarray(rng.normal(size=(n, ci, h, w)).astype(np.float32)), requires_grad=False)
        weight = Variable(
            xp.asarray(rng.normal(size=(co, ci, 3, 3)).astype(np.float32)), trainable=False, requires_grad=False
        )
        times = {b: [] for b in ("cuda_native_cublas", "winograd")}
        ops = {b: Conv2D_Op(x, weight, padding=1, backend=b) for b in times}
        for i in range(5):
            for b in times if i % 2 == 0 else reversed(times):
                call = lambda: ops[b].forward(x.value, weight.value)
                call()
                times[b].append(float(_event_ms(xp, call, 8)))
        a, b = [xp.asnumpy(ops[key].value) for key in times]
        rows.append(
            {
                "shape": [n, ci, co, h, w],
                "direct_products": n * ci * co * h * w * 9,
                "winograd_products": n * ci * co * ((h + 1) // 2) * ((w + 1) // 2) * 16,
                "relative_max_diff": float(np.max(np.abs(a - b)) / np.max(np.abs(a))),
                "times_ms": times,
            }
        )
    set_device(previous)
    write(out / "convolution.json", rows)
    return rows


def report(out, modes, gradients, runs, multiplications):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lines = [
        "# " + ("Winograd F(2×2,3×3) 卷积" if multiplications else "TF32、FP16 与 BF16 训练"),
        "",
        "FP32 主参数、Adam 和相同 ImageNet 初值。数值对拍使用 6 个固定 batch，在 w16 的 ResNet18 "
        "逐参数比较 FP64。单列 ReLU 符号改变的样本，全部逐参数误差在 gradients.json。",
        "",
        "| 后端 | 无异号 batch 数 | 无异号梯度相对误差中位数范围 | 异号 batch 数 | 异号梯度相对误差中位数范围 |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for b in ("cublas", *modes):
        g = [r for r in gradients if r["config"] == b]
        groups = [[r["grad_rel_median"] for r in g if bool(r["sign_flips"]) == f] for f in (False, True)]

        def span(v):
            return f"{min(v):.2e}–{max(v):.2e}" if v else "—"

        lines.append(f"| {b} | {len(groups[0])} | {span(groups[0])} | {len(groups[1])} | {span(groups[1])} |")
    lines += [
        "",
        "前 2,000 个真实道路样本沿用共同时间段划分，batch 8、2 轮、学习率 0.0003。"
        "下表对三个相同种子汇总。步时剔除最初 8 步，显存为 CuPy 池申请峰值。",
        "",
        "| 后端 | step ms | samples/s | 池峰值 MiB | 最终验证 MAE 均值 ± 样本标准差 | 跳过的溢出更新 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for b in ("cuda_native_cublas", *modes):
        group = [r for r in runs if r["backend"] == b]
        med = lambda k: statistics.median(r[k] for r in group)
        mae = [r["val_mae"] for r in group]
        skipped = sum((r["loss_scaling"] or {}).get("skipped_steps", 0) for r in group)
        lines.append(
            f"| {b} | {med('step_ms'):.2f} | {med('samples_s'):.2f} | {med('pool_mib'):.1f} | "
            f"{statistics.mean(mae):.5f} ± {statistics.stdev(mae):.5f} | {skipped} |"
        )
        for ax, key in zip(axes, ("loss", "val_angle_mae")):
            vals = np.array([[e[key] for e in r["epochs"]] for r in group])
            ax.plot([1, 2], vals.mean(axis=0), "o-", label=b)
            ax.set(xticks=[1, 2], xlabel="Epoch", ylabel=key)
    axes[0].legend(fontsize=7)
    fig.savefig(out / "curves.png", dpi=140)
    plt.close(fig)
    lines += ["", "![同种子训练与验证曲线](curves.png)"]
    if multiplications:
        lines += [
            "",
            "乘法数统计跨输入通道的核心乘积，边界 tile 也计入。变换和归约开销均包含在实测时间内。"
            "滤波器变换还使用常数 1/2。[Lavin 与 Gray，CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Lavin_Fast_Algorithms_for_CVPR_2016_paper.html)",
            "",
            "| N/Cin/Cout/H/W | 常规乘积 | Winograd 乘积 | cuBLAS ms | Winograd ms | 输出最大相对差 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in multiplications:
            lines.append(
                f"| {'/'.join(map(str, r['shape']))} | {r['direct_products']:,} | {r['winograd_products']:,} | "
                f"{statistics.median(r['times_ms']['cuda_native_cublas']):.4f} | "
                f"{statistics.median(r['times_ms']['winograd']):.4f} | {r['relative_max_diff']:.2e} |"
            )
    else:
        lines += [
            "",
            "混合模式的卷积展开矩阵与 Linear 的 GEMM 操作数使用 16 位存储，cuBLAS 以 FP32 累加并输出。"
            "TF32 使用 FAST_TF32 计算模式。FP16/BF16 在反向前缩放 loss，更新前反缩放并检查非有限值。"
            "[NVIDIA cuBLAS GEMMEx](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex)",
        ]
    lines += ["", "逐步训练文件、配置、检查点及其来源在各后端的 seed 子目录。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    from apps.common.gpu_lock import GpuLock

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=("precision", "winograd"))
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = args.out.resolve()
    modes = ("winograd",) if args.kind == "winograd" else ("cuda_tf32", "cuda_fp16", "cuda_bf16")
    source = init_study(out, ROOT, args.kind)
    with GpuLock(label=f"{args.kind} numerical comparison", run_dir=out):
        gradients = numerics(out, args.old.resolve(), modes)
        multiplications = winograd_counts(out) if args.kind == "winograd" else []
    report(out, modes, gradients, quality(out, source, modes), multiplications)


if __name__ == "__main__":
    main()
