# -*- coding: utf-8 -*-
"""训练 checkpoint 路径工具。"""

from __future__ import annotations

from pathlib import Path


DEFAULT_RESNET_CHECKPOINT_OUT = "mycar/logs/myflow_resnet18_checkpoint"
DEFAULT_RESNET_BEST_OUT = "mycar/models/myflow_resnet18_best"


def checkpoint_stem(path: Path) -> Path:
    """MyFlows JSON+NPZ checkpoint 使用同名 .json/.npz 双文件。"""
    if path.suffix in (".json", ".npz"):
        return path.with_suffix("")
    return path


def with_run_id(path_str: str, run_id: str) -> str:
    """在路径主干名后追加 _<run_id>，避免多次全新训练互相覆盖。"""
    path = Path(path_str)
    suffix = f"_{run_id}"
    if path.name.endswith(suffix):
        return str(path)
    return str(path.parent / f"{path.name}{suffix}")


def checkpoint_score_to_loss(score: float | None) -> float | None:
    """Return positive regression loss from new loss checkpoints or old -loss ones."""
    if score is None:
        return None
    value = float(score)
    if value == 0.0:
        return None
    return -value if value < 0.0 else value
