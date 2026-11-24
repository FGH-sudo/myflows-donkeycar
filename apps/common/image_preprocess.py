# -*- coding: utf-8 -*-
"""图像读取、预处理与 batch 工具。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def parse_roi_crop(spec: str | tuple[int, ...] | list[int] | None) -> tuple[int, int, int, int] | None:
    """解析 ROI 边距裁剪配置：None 或 TOP,BOTTOM[,LEFT,RIGHT] -> (top, bottom, left, right) | None。"""
    if spec is None:
        return None
    if isinstance(spec, str):
        text = spec.strip()
        if not text or text.lower() == "none":
            return None
        parts = [part.strip() for part in text.split(",")]
        if len(parts) not in (2, 4):
            raise ValueError(f"roi_crop 必须是 TOP,BOTTOM 或 TOP,BOTTOM,LEFT,RIGHT，收到 {spec!r}")
        try:
            vals = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"roi_crop 必须全部为整数，收到 {spec!r}") from exc
    elif isinstance(spec, (tuple, list)):
        if len(spec) not in (2, 4):
            raise ValueError(f"roi_crop 必须包含 2 个或 4 个整数，收到 {spec!r}")
        parsed = []
        for item in spec:
            if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
                raise ValueError(f"roi_crop 元素必须为整数，收到 {spec!r}")
            parsed.append(int(item))
        vals = tuple(parsed)
    else:
        raise ValueError(f"不支持的 roi_crop 类型: {type(spec).__name__}")

    if len(vals) == 2:
        t, b = vals
        l, r = 0, 0
    else:
        t, b, l, r = vals
    if min(t, b, l, r) < 0:
        raise ValueError(f"roi_crop 边距不能为负数，收到 {spec!r}")
    if (t, b, l, r) == (0, 0, 0, 0):
        return None
    return (t, b, l, r)


def format_roi_crop(crop: tuple[int, int, int, int] | None) -> str | None:
    """将标准 4 元组 ROI 边距格式化为紧凑字符串；空裁剪返回 None。"""
    if crop is None:
        return None
    t, b, l, r = (int(v) for v in crop)
    if (t, b, l, r) == (0, 0, 0, 0):
        return None
    return f"{t},{b}" if (l == 0 and r == 0) else f"{t},{b},{l},{r}"


def crop_rgb(img: np.ndarray, roi_crop: tuple[int, int, int, int] | None) -> np.ndarray:
    """按 (top, bottom, left, right) 像素边距裁剪 HxWxC 图像。"""
    if roi_crop is None:
        return img
    t, b, l, r = (int(v) for v in roi_crop)
    h, w = img.shape[:2]
    y1 = h - b
    x1 = w - r
    if min(t, b, l, r) < 0 or t >= y1 or l >= x1:
        raise ValueError(f"ROI 裁剪 {(t, b, l, r)} 超出图像尺寸 {(h, w)}")
    return img[t:y1, l:x1]


def imread_nchw(
    path: str | Path,
    size_wh: tuple[int, int],
    dtype=np.float64,
    roi_crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """读取图片并返回 (1,3,H,W) float，H=size_wh[1]，W=size_wh[0]。"""
    w, h = size_wh
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读取图片失败: {path}")
    if roi_crop is not None:
        img = crop_rgb(img, roi_crop)
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
