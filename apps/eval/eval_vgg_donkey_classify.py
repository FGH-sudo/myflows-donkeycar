#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线评估 VGG-11 角度分类模型。

示例:
  python -m apps.eval.eval_vgg_donkey_classify --checkpoint mycar/models/vgg11_classify_best \\
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
from apps.train.common.checkpoints import checkpoint_stem
from tools.device_runtime import myflows_asnumpy, print_myflows_device, resolve_myflows_device
from MyFlows.layers.vgg import VGG11, angle_to_class


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="mycar/data")
    ap.add_argument("--catalog", type=str, default="catalog_generated.catalog")
    ap.add_argument("--checkpoint", type=str, default="mycar/models/vgg11_classify_best")
    ap.add_argument("--num-classes", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--angle-scale", type=float, default=1.0)
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

    raw = load_donkey_index(data_dir, 0.5, args.angle_scale, args.catalog)
    index = [(rel, angle_to_class(a, args.num_classes)) for rel, a, _ in raw]
    if args.max_samples > 0:
        index = index[: args.max_samples]

    first = data_dir / index[0][0]
    img0 = cv2.imread(str(first), cv2.IMREAD_COLOR)
    h0, w0 = img0.shape[:2]
    B = args.batch
    K = args.num_classes

    device = resolve_myflows_device(args.device)
    print_myflows_device(device, args.device)

    x_var = ms.Variable(np.zeros((B, 3, h0, w0), dtype=np.float32), name="X")
    model = VGG11(3, K, image_h=h0, image_w=w0, name="vgg11_donkey")
    logits = model(x_var)
    graph = ms.Graph(logits)
    model.eval()

    t0 = time.time()
    ms.load_checkpoint([model], None, str(stem))
    print(f"loaded in {time.time()-t0:.2f}s, n={len(index)}")

    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    steps = (len(index) + B - 1) // B

    for s in range(steps):
        batch = index[s * B : (s + 1) * B]
        real_count = len(batch)
        if len(batch) < B:
            batch = batch + [batch[-1]] * (B - len(batch))
        xb = np.zeros((B, 3, h0, w0), dtype=np.float64)
        yb = np.zeros((B, 1), dtype=np.float64)
        for r, (rel, cls) in enumerate(batch):
            xb[r] = imread_nchw(data_dir / rel, (w0, h0))[0]
            yb[r, 0] = cls
        x_var.value = xb
        graph.forward()
        preds = np.argmax(myflows_asnumpy(logits.value), axis=1)
        y_true_all.extend(yb[:real_count, 0].astype(int).tolist())
        y_pred_all.extend(preds[:real_count].tolist())

    yt = np.array(y_true_all)
    yp = np.array(y_pred_all)
    acc = ms.accuracy(yt, yp, num_classes=K)
    cm = ms.confusion_matrix(yt, yp, num_classes=K)
    prf = ms.precision_recall_f1(yt, yp, num_classes=K, average="macro")
    print(f"accuracy={acc:.4f}")
    print("confusion_matrix:\n", cm)
    print("macro P/R/F1:", prf)
    print(ms.classification_report(yt, yp, num_classes=K))


if __name__ == "__main__":
    main()
