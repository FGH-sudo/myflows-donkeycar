# -*- coding: utf-8 -*-
"""Deterministic train/validation/test split helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SPLIT_NAMES = ("train", "val", "test")


def _count_from_ratio(total: int, ratio: float) -> int:
    ratio = max(0.0, float(ratio or 0.0))
    return int(round(total * ratio))


def _resolve_count(total: int, size: int | None, ratio: float) -> int:
    if size is not None and int(size) > 0:
        return int(size)
    return _count_from_ratio(total, ratio)


def build_split(
    rows: Sequence[Any] | int,
    *,
    val_ratio: float = 0.0,
    test_ratio: float = 0.0,
    val_size: int | None = None,
    test_size: int | None = None,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Build a deterministic split over row indices.

    Fixed sizes take precedence over ratios. Returned values are original row
    indices, not row objects, so the same split can be reused across train/eval.
    """

    total = int(rows if isinstance(rows, int) else len(rows))
    if total < 0:
        raise ValueError("row count must be non-negative")

    n_val = _resolve_count(total, val_size, val_ratio)
    n_test = _resolve_count(total, test_size, test_ratio)
    if n_val < 0 or n_test < 0:
        raise ValueError("split sizes must be non-negative")
    if n_val + n_test > total:
        raise ValueError("validation + test split is larger than dataset")

    indices = np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)

    test = sorted(indices[:n_test].astype(int).tolist())
    val = sorted(indices[n_test : n_test + n_val].astype(int).tolist())
    held = set(test) | set(val)
    train = [i for i in range(total) if i not in held]
    return {"train": train, "val": val, "test": test}


def split_enabled(
    *,
    split_file: str | Path | None = None,
    val_ratio: float = 0.0,
    test_ratio: float = 0.0,
    val_size: int | None = None,
    test_size: int | None = None,
) -> bool:
    return bool(
        split_file
        or float(val_ratio or 0.0) > 0
        or float(test_ratio or 0.0) > 0
        or int(val_size or 0) > 0
        or int(test_size or 0) > 0
    )


def save_split(
    path: str | Path,
    splits: dict[str, list[int]],
    *,
    source_count: int,
    seed: int,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "source_count": int(source_count),
        "seed": int(seed),
        "splits": {name: list(map(int, splits.get(name, []))) for name in SPLIT_NAMES},
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_split(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != 1:
        raise ValueError("unsupported split file version")
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("split file missing splits")
    payload["splits"] = {name: list(map(int, splits.get(name, []))) for name in SPLIT_NAMES}
    return payload


def select_split(rows: Sequence[Any], splits: dict[str, list[int]], name: str) -> list[Any]:
    split_name = str(name or "all").lower()
    if split_name == "all":
        return list(rows)
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"unknown split: {name!r}")
    selected = []
    total = len(rows)
    for idx in splits.get(split_name, []):
        if idx < 0 or idx >= total:
            raise IndexError(f"split index {idx} outside dataset of size {total}")
        selected.append(rows[idx])
    return selected


def resolve_splits(
    rows: Sequence[Any],
    *,
    split_file: str | Path | None = None,
    split_out: str | Path | None = None,
    val_ratio: float = 0.0,
    test_ratio: float = 0.0,
    val_size: int | None = None,
    test_size: int | None = None,
    seed: int = 42,
) -> tuple[dict[str, list[int]] | None, Path | None]:
    """Load or build splits if requested; otherwise return ``(None, None)``."""

    if split_file:
        payload = load_split(split_file)
        return payload["splits"], Path(split_file)
    if not split_enabled(
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        val_size=val_size,
        test_size=test_size,
    ):
        return None, None

    splits = build_split(
        rows,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )
    saved = save_split(split_out, splits, source_count=len(rows), seed=seed) if split_out else None
    return splits, saved
