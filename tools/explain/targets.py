# -*- coding: utf-8 -*-
"""Grad-CAM 解释目标选择。"""

from __future__ import annotations

import numpy as np


def target_output_index(value: str) -> int:
    lowered = str(value).strip().lower()
    if lowered in ("angle", "steering", "0"):
        return 0
    if lowered in ("throttle", "1"):
        return 1
    return int(lowered)


def select_target(model_type: str, raw_outputs: np.ndarray, target_output: str) -> tuple[int, str]:
    target_index = target_output_index(target_output)
    target_label = "angle" if target_index == 0 else "throttle" if target_index == 1 else f"output_{target_index}"
    return target_index, target_label
