# -*- coding: utf-8 -*-
"""图像读取、预处理与 batch 工具。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_nchw(path: str | Path, size_wh: tuple[int, int], dtype=np.float64) -> np.ndarray:
    """读取图片并返回 (1,3,H,W) float，H=size_wh[1]，W=size_wh[0]。"""
    w, h = size_wh
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读取图片失败: {path}")
    img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img.astype(dtype) / 255.0
    return np.transpose(arr, (2, 0, 1))[np.newaxis, ...]


def read_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def pad_fixed_batch(x_batch: np.ndarray, y_batch: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    if x_batch.shape[0] >= batch_size:
        return x_batch, y_batch
    pad = int(batch_size - x_batch.shape[0])
    x_pad = np.repeat(x_batch[-1:], pad, axis=0)
    y_pad = np.repeat(y_batch[-1:], pad, axis=0)
    return np.concatenate([x_batch, x_pad], axis=0), np.concatenate([y_batch, y_pad], axis=0)
