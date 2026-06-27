#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线评估当前 MyFlows ResNet-18 在 DonkeyCar tub 上的预测能力：
- angle/throttle 回归误差（MSE/MAE 等，见 MyFlows.utils.metrics）
- angle 符号正确率（用于判断“方向反了”的问题）
- 预测幅度分布（用于判断“拐弯幅度太小”）

使用方式（在项目根目录 D:\\DL\\testmyflow）：
  # 走 MyFlows 图推理（GPU=CUDA+CuPy）；checkpoint 可为 .onnx 或 .json/.npz 基名
  python -m apps.eval.eval_myflows_donkey --checkpoint mycar/models/myflow_resnet18_best \\
      --max-samples 2000 --batch 4 --device auto

  更快的大规模评估请优先用 apps.eval.eval_myflows_donkey_onnx（ONNX Runtime + onnxruntime-gpu）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import MyFlows as ms
from apps.common.donkey_data import load_donkey_index
from apps.common.image_preprocess import imread_nchw, read_rgb
from apps.train.common.checkpoints import checkpoint_stem as normalize_checkpoint_stem
from tools.device_runtime import myflows_asnumpy, print_myflows_device, resolve_myflows_device
from MyFlows.utils.metrics import DonkeyRegressionEvaluator


def _checkpoint_stem(path: Path) -> Path:
    return normalize_checkpoint_stem(path.with_suffix("") if path.suffix == ".onnx" else path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate MyFlows model on tub data.")
    ap.add_argument("--data", type=str, default="mycar/data", help="tub 根目录")
    ap.add_argument(
        "--catalog",
        type=str,
        default="catalog_generated.catalog",
        help="评估用 catalog（默认 catalog_generated.catalog）",
    )
    ap.add_argument("--checkpoint", type=str, default="mycar/models/myflow_resnet18_best.onnx")
    ap.add_argument("--max-samples", type=int, default=2000, help="最多评估多少条(0=catalog 全部)")
    ap.add_argument("--batch", type=int, default=4, help="评估 batch（固定 batch 会用于一次构图）")
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="MyFlows 计算设备：auto 有 GPU 则用 CUDA(CuPy)",
    )

    # 附加：部署侧变换（只用于解释，不用于主指标计算）
    ap.add_argument("--steering-scale", type=float, default=2.5, help="部署侧 steering_scale（附加解释用）")
    ap.add_argument("--angle-sign-eps", type=float, default=1e-3, help="真值角度绝对值小于该阈值时认为“接近直行”")
    ap.add_argument("--zero-pred-eps", type=float, default=0.05, help="接近直行时预测绝对值小于该阈值认为正确")

    args = ap.parse_args()

    data_dir = (ROOT / args.data).resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"数据目录不存在: {data_dir}")

    checkpoint_path = (ROOT / args.checkpoint).resolve()
    checkpoint_stem = _checkpoint_stem(checkpoint_path)

    index = load_donkey_index(data_dir, fixed_throttle=0.5, angle_scale=1.0, catalog_name=args.catalog)
    if not index:
        raise SystemExit("未找到带 cam/image_array 的 tub 行，请确认已采数且 catalog 有图名。")

    if args.max_samples and args.max_samples > 0:
        index = index[: args.max_samples]

    # 用第一条样本确定输入尺寸
    first_p = data_dir / index[0][0]
    first_rgb = read_rgb(first_p)
    h0, w0 = first_rgb.shape[:2]
    h, w = h0, w0
    dtype = np.float32

    device = resolve_myflows_device(args.device)
    print_myflows_device(device, args.device)

    B = int(args.batch)
    n = len(index)
    steps = (n + B - 1) // B

    # 构图：输出 [angle, throttle]
    x_var = ms.Variable(np.zeros((B, 3, h, w), dtype=dtype), name="X")
    model = ms.ResNet18(
        in_channels=3,
        num_classes=2,
        stem="cifar",
        base_width=64,
        name="resnet18_donkey",
    )
    out = model(x_var)
    graph = ms.Graph(out)
    model.eval()

    # 加载权重（checkpoint_stem.json + checkpoint_stem.npz）
    t0 = time.time()
    ms.load_checkpoint([model], None, str(checkpoint_stem))
    graph.forward()
    print(
        f"loaded checkpoint={checkpoint_stem} in {time.time() - t0:.2f}s, samples={n}, input={w}x{h}, batch={B}"
    )

    evaluator = DonkeyRegressionEvaluator(
        near_zero_eps=float(args.angle_sign_eps),
        zero_pred_eps=float(args.zero_pred_eps),
    )

    for s in range(steps):
        sl = s * B
        bi = index[sl : sl + B]
        if len(bi) < B:
            if len(bi) == 0:
                break
            pad = int(B - len(bi))
            bi = bi + [bi[-1]] * pad

        x_batch = np.zeros((B, 3, h, w), dtype=dtype)
        y = np.zeros((B, 2), dtype=dtype)

        for r, (rel, a_true, t_true) in enumerate(bi):
            p = data_dir / rel
            x_batch[r] = imread_nchw(p, (w, h), dtype=dtype)[0]
            y[r, 0] = a_true
            y[r, 1] = t_true

        x_var.value = x_batch
        graph.forward()

        pred = np.asarray(myflows_asnumpy(out.value))
        evaluator.update(y, pred)

    metrics = evaluator.compute()
    print()
    print(evaluator.format_summary(metrics))
    print("Interpretation:")
    print("- mean_abs_angle(pred) 若明显低于 true -> 拐弯幅度偏小（部署 scale 可缓解）")
    print("- angle_sign_accuracy 若明显低 -> 方向预测本身有问题（部署 scale 无法根治）")
    print(
        f"- (部署侧辅助) steering_scale={float(args.steering_scale)} "
        "不会改变符号，只影响饱和与幅度"
    )


if __name__ == "__main__":
    main()
