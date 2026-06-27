# -*- coding: utf-8 -*-
"""Grad-CAM 解释目标选择。"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / max(float(np.sum(exp)), 1e-12)


def target_output_index(value: str) -> int:
    lowered = str(value).strip().lower()
    if lowered in ("angle", "steering", "0"):
        return 0
    if lowered in ("throttle", "1"):
        return 1
    return int(lowered)


def select_target(model_type: str, raw_outputs: np.ndarray, target_output: str, target_class: str) -> tuple[int, str]:
    if model_type == "vgg":
        probs = softmax(raw_outputs)
        target_index = int(np.argmax(probs)) if str(target_class).lower() == "pred" else int(target_class)
        return target_index, f"class_{target_index}"
    target_index = target_output_index(target_output)
    target_label = "angle" if target_index == 0 else f"output_{target_index}"
    return target_index, target_label
