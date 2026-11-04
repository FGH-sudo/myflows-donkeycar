# -*- coding: utf-8 -*-
"""训练 checkpoint 路径与分数工具。"""

from __future__ import annotations

from pathlib import Path

from MyFlows.utils.checkpoint import read_checkpoint_metadata


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
    return -value if value < 0.0 else value


def saved_best_loss(best_base: Path) -> float | None:
    """续训时的最佳 loss：取最佳模型自己记录的分数，没有最佳模型时为 None。

    训练断点里的分数是当时的训练 loss，不能拿来和验证 loss 比较；``load_checkpoint`` 找不到断点时
    返回的 0.0 也不是分数。"""
    meta = read_checkpoint_metadata(best_base)
    return checkpoint_score_to_loss(meta.get("score_value", meta.get("best_loss"))) if meta else None
