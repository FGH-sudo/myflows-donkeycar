#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 DonkeyCar 图像回归子集上对比 MyFlows / PyTorch / PaddlePaddle 的 VGG11 训练耗时与内存。

用法:
  python benchmark/compare_frameworks.py --data mycar/data --samples 64 --epochs 2 --batch 8 --device cuda

依赖（按需安装）:
  pip install torch psutil matplotlib
  pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.common.donkey_data import load_donkey_index
from apps.common.image_preprocess import imread_nchw, pad_fixed_batch
from MyFlows.layers.vgg import VGG11, vgg_fc_input_dim
from tools.device_runtime import myflows_scalar_float, resolve_myflows_device


def load_donkey_regression_subset(
    data_dir: str | Path,
    *,
    catalog: str = "catalog_generated.catalog",
    samples: int = 64,
    image_h: int = 120,
    image_w: int = 160,
    fixed_throttle: float = 0.5,
    angle_scale: float = 1.0,
    sample_seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """读取轻量 DonkeyCar 回归子集，返回 NCHW 图像和 [angle, throttle] 标签。"""
    data_path = Path(data_dir)
    rows = load_donkey_index(data_path, fixed_throttle, angle_scale, catalog)
    if not rows:
        raise SystemExit(f"未找到 DonkeyCar 样本: {data_path}")

    if samples and samples > 0 and samples < len(rows):
        if sample_seed is None:
            rows = rows[:samples]
        else:
            rng = np.random.default_rng(int(sample_seed))
            indices = rng.choice(len(rows), size=int(samples), replace=False)
            rows = [rows[int(i)] for i in indices]

    n = len(rows)
    x = np.zeros((n, 3, int(image_h), int(image_w)), dtype=np.float64)
    y = np.zeros((n, 2), dtype=np.float64)
    for i, (rel, angle, throttle) in enumerate(rows):
        x[i] = imread_nchw(data_path / rel, (int(image_w), int(image_h)), np.float64)[0]
        y[i, 0] = float(angle)
        y[i, 1] = float(throttle)
    return x, y


def _peak_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        if tracemalloc.is_tracing():
            return tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        return 0.0


def bench_myflows(x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch: int, lr: float, device: str) -> dict:
    try:
        import MyFlows as ms
    except ImportError:
        return {"framework": "MyFlows", "error": "not installed"}

    try:
        resolved_device = resolve_myflows_device(device)
    except Exception as exc:
        return {"framework": "MyFlows", "error": str(exc)}
    if device == "cuda" and resolved_device != "cuda":
        return {"framework": "MyFlows", "error": "cuda unavailable", "device": resolved_device}

    bsz = int(batch)
    _, _, h, w = x_train.shape
    x_var = ms.Variable(np.zeros((bsz, 3, h, w), dtype=np.float32), name="X")
    y_var = ms.Variable(np.zeros((bsz, 2), dtype=np.float32), name="y")
    model = VGG11(in_channels=3, output_dim=2, image_h=h, image_w=w, name="vgg11_bench")
    pred = model(x_var)
    loss_node = ms.MSELoss(pred, y_var, name="loss")
    graph = ms.Graph(loss_node)
    opt = ms.Adam(graph, learning_rate=lr)

    gc.collect()
    t0 = time.perf_counter()
    model.train(True)
    for _ in range(epochs):
        for i in range(0, x_train.shape[0], bsz):
            x_batch = x_train[i : i + bsz].astype(np.float32, copy=False)
            y_batch = y_train[i : i + bsz].astype(np.float32, copy=False)
            x_batch, y_batch = pad_fixed_batch(x_batch, y_batch, bsz)
            x_var.value = x_batch
            y_var.value = y_batch
            opt.one_step()
            opt.update()
    elapsed = time.perf_counter() - t0
    return {
        "framework": "MyFlows",
        "time_s": elapsed,
        "peak_mb": _peak_mb(),
        "final_loss": myflows_scalar_float(loss_node.value),
        "device": ms.get_device(),
    }


def bench_pytorch(x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch: int, lr: float, device: str) -> dict:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {"framework": "PyTorch", "error": "not installed"}

    if device == "cuda" and not torch.cuda.is_available():
        return {"framework": "PyTorch", "error": "cuda unavailable"}
    torch_device = torch.device("cuda" if device in ("cuda", "auto") and torch.cuda.is_available() else "cpu")

    _, _, h, w = x_train.shape

    class TorchVGG11Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            layers: list[nn.Module] = []
            in_ch = 3
            for num_convs, out_ch in ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512)):
                for _ in range(num_convs):
                    layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
                    layers.append(nn.ReLU())
                    in_ch = out_ch
                layers.append(nn.MaxPool2d(2, 2))
            self.features = nn.Sequential(*layers)
            fc_in = vgg_fc_input_dim(h, w, 512)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(fc_in, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 2),
            )

        def forward(self, x):
            return self.head(self.features(x))

    model = TorchVGG11Regressor().to(torch_device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    xt = torch.from_numpy(x_train.astype(np.float32))
    yt = torch.from_numpy(y_train.astype(np.float32))

    gc.collect()
    t0 = time.perf_counter()
    model.train()
    loss = None
    for _ in range(epochs):
        for i in range(0, x_train.shape[0], batch):
            xb = xt[i : i + batch].to(torch_device)
            yb = yt[i : i + batch].to(torch_device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    elapsed = time.perf_counter() - t0
    final = float(loss.item()) if loss is not None else 0.0
    return {"framework": "PyTorch", "time_s": elapsed, "peak_mb": _peak_mb(), "final_loss": final, "device": str(torch_device)}


def bench_paddle(x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch: int, lr: float, device: str) -> dict:
    try:
        import paddle
        import paddle.nn as nn
    except ImportError:
        return {"framework": "PaddlePaddle", "error": "not installed"}

    if device == "cuda" and not paddle.device.is_compiled_with_cuda():
        return {"framework": "PaddlePaddle", "error": "cuda unavailable"}
    if device == "cpu":
        paddle.set_device("cpu")
    elif device in ("cuda", "auto"):
        paddle.set_device("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")
    else:
        paddle.set_device("cpu")
    active_device = paddle.get_device()

    _, _, h, w = x_train.shape

    class PaddleVGG11Regressor(nn.Layer):
        def __init__(self):
            super().__init__()
            layers = []
            in_ch = 3
            for num_convs, out_ch in ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512)):
                for _ in range(num_convs):
                    layers.append(nn.Conv2D(in_ch, out_ch, 3, padding=1))
                    layers.append(nn.ReLU())
                    in_ch = out_ch
                layers.append(nn.MaxPool2D(2, 2))
            self.features = nn.Sequential(*layers)
            self.fc1 = nn.Linear(vgg_fc_input_dim(h, w, 512), 4096)
            self.fc2 = nn.Linear(4096, 4096)
            self.fc3 = nn.Linear(4096, 2)

        def forward(self, x):
            x = self.features(x)
            x = paddle.flatten(x, 1)
            x = paddle.nn.functional.relu(self.fc1(x))
            x = paddle.nn.functional.relu(self.fc2(x))
            return self.fc3(x)

    model = PaddleVGG11Regressor()
    opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
    crit = nn.MSELoss()
    xt = paddle.to_tensor(x_train.astype(np.float32))
    yt = paddle.to_tensor(y_train.astype(np.float32))

    gc.collect()
    t0 = time.perf_counter()
    model.train()
    loss = None
    for _ in range(epochs):
        for i in range(0, x_train.shape[0], batch):
            pred = model(xt[i : i + batch])
            loss = crit(pred, yt[i : i + batch])
            loss.backward()
            opt.step()
            opt.clear_grad()
    elapsed = time.perf_counter() - t0
    final = float(loss.numpy()) if loss is not None else 0.0
    return {"framework": "PaddlePaddle", "time_s": elapsed, "peak_mb": _peak_mb(), "final_loss": final, "device": active_device}


BENCHMARK_RUNNERS = (bench_myflows, bench_pytorch, bench_paddle)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="mycar/data")
    ap.add_argument("--catalog", type=str, default="catalog_generated.catalog")
    ap.add_argument("--samples", type=int, default=64, help="使用的 DonkeyCar 子集大小；0=全部")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--image-h", type=int, default=120)
    ap.add_argument("--image-w", type=int, default=160)
    ap.add_argument("--fixed-throttle", type=float, default=0.5)
    ap.add_argument("--angle-scale", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--out", type=str, default="benchmark/results.csv")
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("auto", "cpu", "cuda"),
        help="cuda 为严格 GPU 模式；auto 会在可用时使用 GPU，否则回退 CPU",
    )
    args = ap.parse_args()

    x, y = load_donkey_regression_subset(
        (ROOT / args.data).resolve(),
        catalog=args.catalog,
        samples=args.samples,
        image_h=args.image_h,
        image_w=args.image_w,
        fixed_throttle=args.fixed_throttle,
        angle_scale=args.angle_scale,
        sample_seed=args.sample_seed,
    )
    runners = list(BENCHMARK_RUNNERS)
    rows: list[dict] = []

    print(
        f"Benchmark: task=donkey_regression model=VGG11 samples={len(x)} "
        f"batch={args.batch} epochs={args.epochs} device={args.device}"
    )
    for fn in runners:
        try:
            row = fn(x, y, args.epochs, args.batch, args.lr, args.device)
        except Exception as exc:
            row = {"framework": fn.__name__, "error": str(exc)}
        row.update({"task": "donkey_regression", "model": "VGG11", "samples": len(x)})
        rows.append(row)
        print(row)

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
