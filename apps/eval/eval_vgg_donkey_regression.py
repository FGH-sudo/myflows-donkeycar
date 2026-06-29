#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线评估 VGG-11 DonkeyCar 控制回归模型。

示例:
  python -m apps.eval.eval_vgg_donkey_regression --checkpoint mycar/models/vgg11_regression_best \\
      --max-samples 2000 --device auto
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import MyFlows as ms
from apps.common.donkey_data import load_donkey_index
from apps.common.image_preprocess import imread_nchw
from apps.common.splits import load_split, select_split
from apps.train.common.checkpoints import checkpoint_stem
from tools.device_runtime import myflows_asnumpy, print_myflows_device, resolve_myflows_device
from MyFlows.layers.vgg import VGG11
from MyFlows.utils.metrics import DonkeyRegressionEvaluator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="mycar/data")
    ap.add_argument("--catalog", type=str, default="catalog_generated.catalog")
    ap.add_argument("--checkpoint", type=str, default="mycar/models/vgg11_regression_best")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--split-file", type=str, default=None)
    ap.add_argument("--split", type=str, default="all", choices=("train", "val", "test", "all"))
    ap.add_argument("--fixed-throttle", type=float, default=0.5)
    ap.add_argument("--force-fixed-throttle", action="store_true", help="忽略 catalog 中的 user/throttle，强制使用 --fixed-throttle")
    ap.add_argument("--angle-scale", type=float, default=1.0)
    ap.add_argument("--angle-sign-eps", type=float, default=1e-3)
    ap.add_argument("--zero-pred-eps", type=float, default=0.05)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="MyFlows 计算设备：auto 有 GPU 则用 CUDA(CuPy)",
    )
    args = ap.parse_args()

    data_dir = (ROOT / args.data).resolve()
    stem = checkpoint_stem((ROOT / args.checkpoint).resolve())

    index = load_donkey_index(
        data_dir,
        args.fixed_throttle,
        args.angle_scale,
        args.catalog,
        force_fixed_throttle=bool(args.force_fixed_throttle),
    )
    if args.split_file:
        payload = load_split((ROOT / args.split_file).resolve())
        index = select_split(index, payload["splits"], args.split)
        print(f"[split] file={(ROOT / args.split_file).resolve()} split={args.split} samples={len(index)}")
        if not index:
            raise SystemExit(f"split {args.split!r} 为空")
    if args.max_samples > 0:
        index = index[: args.max_samples]

    first = data_dir / index[0][0]
    img0 = cv2.imread(str(first), cv2.IMREAD_COLOR)
    h0, w0 = img0.shape[:2]
    B = args.batch

    device = resolve_myflows_device(args.device)
    print_myflows_device(device, args.device)

    x_var = ms.Variable(np.zeros((B, 3, h0, w0), dtype=np.float32), name="X")
    model = VGG11(3, output_dim=2, image_h=h0, image_w=w0, name="vgg11_donkey")
    pred_node = model(x_var)
    graph = ms.Graph(pred_node)
    model.eval()

    t0 = time.time()
    ms.load_checkpoint([model], None, str(stem))
    print(f"loaded in {time.time()-t0:.2f}s, n={len(index)}")

    evaluator = DonkeyRegressionEvaluator(
        near_zero_eps=float(args.angle_sign_eps),
        zero_pred_eps=float(args.zero_pred_eps),
    )
    steps = (len(index) + B - 1) // B

    for s in range(steps):
        batch = index[s * B : (s + 1) * B]
        real_count = len(batch)
        if len(batch) < B:
            batch = batch + [batch[-1]] * (B - len(batch))
        xb = np.zeros((B, 3, h0, w0), dtype=np.float64)
        yb = np.zeros((B, 2), dtype=np.float32)
        for r, (rel, angle, throttle) in enumerate(batch):
            xb[r] = imread_nchw(data_dir / rel, (w0, h0))[0]
            yb[r, 0] = angle
            yb[r, 1] = throttle
        x_var.value = xb
        graph.forward()
        preds = np.asarray(myflows_asnumpy(pred_node.value), dtype=np.float32)
        evaluator.update(yb[:real_count], preds[:real_count])

    metrics = evaluator.compute()
    print()
    print(evaluator.format_summary(metrics))


if __name__ == "__main__":
    main()
