#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 ONNX Runtime 的离线评估（快速版）：
- angle/throttle MSE
- angle 符号正确率（判断“方向反了”）
- mean_abs_angle(pred/true)（判断“拐弯幅度太小”）

示例（项目根目录）:
  # 快速抽查（推荐）；GPU 需 pip install onnxruntime-gpu
  python -m apps.eval.eval_myflows_donkey_onnx --checkpoint mycar/models/myflow_resnet18_best.onnx \\
      --max-samples 2000 --device auto

  # 全量评估（10177 条 catalog，较慢）
  python -m apps.eval.eval_myflows_donkey_onnx --checkpoint mycar/models/myflow_resnet18_best.onnx \\
      --max-samples 0 --device auto

  注: 导出的 ONNX 输入 batch 常为 1，--batch 8 可能被忽略；--max-samples 0 表示评全部样本。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from apps.common.donkey_data import load_donkey_index
from apps.common.image_preprocess import imread_nchw, read_rgb
from tools.device_runtime import create_ort_inference_session, print_ort_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="mycar/data")
    ap.add_argument(
        "--catalog",
        type=str,
        default="catalog_generated.catalog",
        help="评估用的 catalog 文件名（默认仅生成路网 catalog_generated.catalog）",
    )
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--max-samples", type=int, default=1000, help="评估条数上限(0=catalog 全部)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="ONNX Runtime 设备：auto/cuda 优先 GPU（CUDA 或 DirectML）",
    )
    ap.add_argument("--angle-sign-eps", type=float, default=1e-3)
    ap.add_argument("--zero-pred-eps", type=float, default=0.05)
    ap.add_argument(
        "--int8-model",
        type=str,
        default=None,
        help="可选 INT8 ONNX；与 FP32 对比 MSE 与推理耗时",
    )
    args = ap.parse_args()

    data_dir = (ROOT / args.data).resolve()
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    int8_path = (ROOT / args.int8_model).resolve() if args.int8_model else None

    if not data_dir.is_dir():
        raise SystemExit(f"数据目录不存在: {data_dir}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"ONNX 不存在: {checkpoint_path}")

    index = load_donkey_index(data_dir, fixed_throttle=0.5, angle_scale=1.0, catalog_name=args.catalog)
    if not index:
        raise SystemExit("未找到带 cam/image_array 的 tub 行")
    if args.max_samples and args.max_samples > 0:
        index = index[: args.max_samples]

    first_p = data_dir / index[0][0]
    first_rgb = read_rgb(first_p)
    h0, w0 = first_rgb.shape[:2]
    w, h = w0, h0

    def _run_eval(model_path: Path, label: str) -> dict:
        sess, providers = create_ort_inference_session(model_path, args.device)
        if label == "FP32":
            print_ort_device(providers, args.device)
        input_name = sess.get_inputs()[0].name
        input_type = sess.get_inputs()[0].type
        input_shape = sess.get_inputs()[0].shape
        dtype = np.float32 if "float" in input_type and "double" not in input_type else np.float64

        B = int(args.batch)
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 1:
            b0 = input_shape[0]
            if isinstance(b0, int) and b0 == 1:
                B = 1
        n = len(index)
        steps = (n + B - 1) // B

        angle_se = 0.0
        throttle_se = 0.0
        sign_correct = 0
        sign_total = 0
        near_zero_true_total = 0
        near_zero_true_correct = 0
        pred_abs_sum = 0.0
        true_abs_sum = 0.0

        t0 = time.time()
        for s in range(steps):
            sl = s * B
            bi = index[sl : sl + B]
            if not bi:
                break
            if len(bi) < B:
                bi = bi + [bi[-1]] * (B - len(bi))

            x_batch = np.zeros((B, 3, h, w), dtype=dtype)
            y = np.zeros((B, 2), dtype=np.float64)

            for r, (rel, a_true, t_true) in enumerate(bi):
                p = data_dir / rel
                x_batch[r] = imread_nchw(p, (w, h), dtype=dtype)[0]
                y[r, 0] = a_true
                y[r, 1] = t_true

            out = sess.run(None, {input_name: x_batch})[0]
            out = np.asarray(out).reshape(-1, 2)
            a_pred = out[:, 0]
            t_pred = out[:, 1]
            a_true = y[:, 0]
            t_true = y[:, 1]

            angle_se += float(np.mean((a_pred - a_true) ** 2))
            throttle_se += float(np.mean((t_pred - t_true) ** 2))

            for ai_pred, ai_true in zip(a_pred, a_true):
                sign_total += 1
                if abs(float(ai_true)) < float(args.angle_sign_eps):
                    near_zero_true_total += 1
                    if abs(float(ai_pred)) < float(args.zero_pred_eps):
                        near_zero_true_correct += 1
                        sign_correct += 1
                    continue
                sign_true = 1.0 if ai_true > 0 else -1.0
                sign_pred = 1.0 if ai_pred > 0 else -1.0
                if sign_true == sign_pred:
                    sign_correct += 1

            pred_abs_sum += float(np.mean(np.abs(a_pred)))
            true_abs_sum += float(np.mean(np.abs(a_true)))

        runtime = time.time() - t0
        return {
            "label": label,
            "path": str(model_path),
            "angle_mse": angle_se / max(1, steps),
            "throttle_mse": throttle_se / max(1, steps),
            "angle_sign_accuracy": sign_correct / max(1, sign_total),
            "near_zero_accuracy": near_zero_true_correct / max(1, near_zero_true_total),
            "mean_abs_angle_pred": pred_abs_sum / max(1, steps),
            "mean_abs_angle_true": true_abs_sum / max(1, steps),
            "runtime_s": runtime,
            "samples": n,
            "batch": B,
        }

    results = [_run_eval(checkpoint_path, "FP32")]
    if int8_path and int8_path.is_file():
        results.append(_run_eval(int8_path, "INT8"))
    elif int8_path:
        print(f"警告: INT8 模型不存在，跳过: {int8_path}")

    print("\n=== ONNX Eval Summary ===")
    print(f"input={w}x{h}")
    for r in results:
        print(f"\n--- {r['label']} ({r['path']}) ---")
        print(f"angle_mse={r['angle_mse']:.6f} throttle_mse={r['throttle_mse']:.6f}")
        print(
            f"angle_sign_accuracy={r['angle_sign_accuracy']:.4f} "
            f"(near-zero={r['near_zero_accuracy']:.4f})"
        )
        print(
            f"mean_abs_angle: true={r['mean_abs_angle_true']:.4f} "
            f"pred={r['mean_abs_angle_pred']:.4f}"
        )
        print(f"runtime={r['runtime_s']:.2f}s samples={r['samples']} batch={r['batch']}")


if __name__ == "__main__":
    main()

