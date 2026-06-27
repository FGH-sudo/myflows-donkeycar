#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在相同合成数据上对比 MyFlows / PyTorch / TensorFlow / Paddle 小 CNN 训练耗时与内存。

用法:
  python benchmark/compare_frameworks.py --epochs 3 --batch 8 --samples 128 --device auto

依赖（按需安装）:
  pip install torch tensorflow paddlepaddle psutil
  pip install cupy-cuda12x   # MyFlows GPU
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

import MyFlows as ms
from tools.device_runtime import resolve_myflows_device
from MyFlows.core.graph import Graph
from MyFlows.core.node import Variable
from MyFlows.layers.layer import Conv2D, Dense, Flatten, MaxPool2d
from MyFlows.ops.activation import ReLU
from MyFlows.ops.loss import CrossEntropy
from MyFlows.train.opt import Adam


def _synthetic_batch(n: int, h: int = 32, w: int = 32, num_classes: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3, h, w), dtype=np.float64) * 0.1
    y = rng.integers(0, num_classes, size=(n, 1), dtype=np.int64)
    return x, y


def _peak_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        if tracemalloc.is_tracing():
            return tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        return 0.0


def bench_myflows(x_train, y_train, epochs: int, batch: int, lr: float, device: str) -> dict:
    resolve_myflows_device(device)
    B = batch
    n = x_train.shape[0]
    steps = (n + B - 1) // B
    h, w = x_train.shape[2], x_train.shape[3]
    K = int(y_train.max()) + 1

    x_var = Variable(x_train[:B].copy(), name="X")
    y_var = Variable(y_train[:B].copy(), name="y")
    conv = Conv2D(3, 16, 3, padding=1, activation=ReLU)
    pool = MaxPool2d(2, 2)
    flat = Flatten()
    dense1 = Dense(16 * (h // 2) * (w // 2), 32, activation=ReLU)
    dense2 = Dense(32, K)
    logits = dense2(dense1(flat(pool(conv(x_var)))))
    loss = CrossEntropy(logits, y_var)
    graph = Graph(loss)
    opt = Adam(graph, learning_rate=lr)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(epochs):
        for s in range(steps):
            sl = slice(s * B, (s + 1) * B)
            x_var.value = x_train[sl]
            y_var.value = y_train[sl]
            opt.one_step()
            opt.update()
    elapsed = time.perf_counter() - t0
    peak = _peak_mb()
    tracemalloc.stop()
    from tools.device_runtime import myflows_scalar_float

    return {
        "framework": "MyFlows",
        "time_s": elapsed,
        "peak_mb": peak,
        "final_loss": myflows_scalar_float(loss.value),
        "device": ms.get_device(),
    }


def bench_pytorch(x_train, y_train, epochs: int, batch: int, lr: float, device: str) -> dict:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {"framework": "PyTorch", "error": "not installed"}

    if device == "cpu":
        torch_device = torch.device("cpu")
    elif device in ("cuda", "gpu", "auto"):
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device("cpu")
    n, _, h, w = x_train.shape
    K = int(y_train.max()) + 1

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(16 * (h // 2) * (w // 2), 32),
                nn.ReLU(),
                nn.Linear(32, K),
            )

        def forward(self, x):
            return self.net(x)

    model = SmallCNN().to(torch_device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    xt = torch.from_numpy(x_train.astype(np.float32))
    yt = torch.from_numpy(y_train.reshape(-1).astype(np.int64))

    gc.collect()
    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for i in range(0, n, batch):
            xb = xt[i : i + batch].to(torch_device)
            yb = yt[i : i + batch].to(torch_device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    elapsed = time.perf_counter() - t0
    with torch.no_grad():
        final = float(crit(model(xt[:batch].to(torch_device)), yt[:batch]).item())
    return {
        "framework": "PyTorch",
        "time_s": elapsed,
        "peak_mb": _peak_mb(),
        "final_loss": final,
        "device": str(torch_device),
    }


def bench_tensorflow(x_train, y_train, epochs: int, batch: int, lr: float, device: str) -> dict:
    try:
        import tensorflow as tf
    except ImportError:
        return {"framework": "TensorFlow", "error": "not installed"}

    tf.keras.backend.clear_session()
    if device in ("cuda", "gpu", "auto"):
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception:
                        pass
        except Exception:
            pass
    n, _, h, w = x_train.shape
    K = int(y_train.max()) + 1
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", input_shape=(h, w, 3)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(K),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    x_hwc = np.transpose(x_train, (0, 2, 3, 1)).astype(np.float32)
    y_flat = y_train.reshape(-1).astype(np.int32)

    gc.collect()
    t0 = time.perf_counter()
    hist = model.fit(x_hwc, y_flat, epochs=epochs, batch_size=batch, verbose=0)
    elapsed = time.perf_counter() - t0
    return {
        "framework": "TensorFlow",
        "time_s": elapsed,
        "peak_mb": _peak_mb(),
        "final_loss": float(hist.history["loss"][-1]),
    }


def bench_paddle(x_train, y_train, epochs: int, batch: int, lr: float, device: str) -> dict:
    try:
        import paddle
        import paddle.nn as nn
    except ImportError:
        return {"framework": "Paddle", "error": "not installed"}

    if device == "cpu":
        paddle.set_device("cpu")
    elif device in ("cuda", "gpu", "auto"):
        paddle.set_device("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")
    else:
        paddle.set_device("cpu")
    n, _, h, w = x_train.shape
    K = int(y_train.max()) + 1

    class SmallCNN(nn.Layer):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2D(3, 16, 3, padding=1)
            self.pool = nn.MaxPool2D(2, 2)
            self.fc1 = nn.Linear(16 * (h // 2) * (w // 2), 32)
            self.fc2 = nn.Linear(32, K)

        def forward(self, x):
            x = paddle.nn.functional.relu(self.conv(x))
            x = self.pool(x)
            x = paddle.flatten(x, 1)
            x = paddle.nn.functional.relu(self.fc1(x))
            return self.fc2(x)

    model = SmallCNN()
    opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
    crit = nn.CrossEntropyLoss()
    xt = paddle.to_tensor(x_train.astype(np.float32))
    yt = paddle.to_tensor(y_train.reshape(-1).astype(np.int64))

    gc.collect()
    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for i in range(0, n, batch):
            logits = model(xt[i : i + batch])
            loss = crit(logits, yt[i : i + batch])
            loss.backward()
            opt.step()
            opt.clear_grad()
    elapsed = time.perf_counter() - t0
    return {"framework": "Paddle", "time_s": elapsed, "peak_mb": _peak_mb(), "final_loss": float(loss.numpy())}


def estimate_flops_myflows(n, h, w, steps, epochs) -> float:
    """粗算：一次 step 主要卷积+全连接乘加次数（近似）。"""
    conv_ops = n * 16 * 3 * 3 * 3 * h * w
    fc_ops = n * (16 * (h // 2) * (w // 2) * 32 + 32 * 5)
    return float(conv_ops + fc_ops) * steps * epochs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", type=str, default="benchmark/results.csv")
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="各框架尽量使用 GPU（不可用则回退 CPU）",
    )
    args = ap.parse_args()

    x, y = _synthetic_batch(args.samples, seed=42)
    runners = [bench_myflows, bench_pytorch, bench_tensorflow, bench_paddle]
    rows: list[dict] = []

    print(f"Benchmark: samples={args.samples} batch={args.batch} epochs={args.epochs} device={args.device}")
    for fn in runners:
        try:
            row = fn(x, y, args.epochs, args.batch, args.lr, args.device)
        except Exception as exc:
            row = {"framework": fn.__name__, "error": str(exc)}
        rows.append(row)
        print(row)

    rows.append({
        "framework": "MyFlows_FLOPs_est",
        "flops_est": estimate_flops_myflows(
            args.samples, 32, 32, (args.samples + args.batch - 1) // args.batch, args.epochs
        ),
    })

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
