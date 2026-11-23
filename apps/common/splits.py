# -*- coding: utf-8 -*-
"""Deterministic train/validation/test split helpers."""

from __future__ import annotations

import hashlib
import json
import math
import operator
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SPLIT_NAMES = ("train", "val", "test")
# What a saved time-block split promises (its ``contract`` field). Legacy: training rows keep the gap from
# validation and test rows. Separated: validation and test rows keep it from each other as well.
TIME_BLOCK_LEGACY = 1
TIME_BLOCK_SEPARATED = 2


def _count_from_ratio(total: int, ratio: float) -> int:
    ratio = float(ratio)
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("split ratio must be finite and within [0, 1]")
    return int(round(total * ratio))


def _count(value, name, minimum=0):
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, bool) or count < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return count


def _resolve_count(total: int, size: int | None, ratio: float) -> int:
    if size is not None:
        size = _count(size, "split size")
        if size:
            return size
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


def build_time_block_split(
    frames: Sequence[int],
    *,
    val_size: int,
    test_size: int = 0,
    block_size: int = 50,
    gap: int = 5,
    seed: int = 42,
    separate_held_out: bool = True,
) -> dict[str, list[int]]:
    """Split by contiguous time blocks, so neighbouring frames never land on both sides.

    ``frames[i]`` is the time position of row ``i`` (DonkeyCar: the frame index in the image name). Rows are
    ordered by it and cut into blocks of ``block_size``; whole blocks are shuffled with ``seed`` and handed to
    the test set until it holds ``test_size`` rows, then to validation until ``val_size``, the rest to training.
    Training rows within ``gap`` of either held-out set are purged, including gaps spanning several small
    blocks. Returns row indices like build_split, plus ``"purged"``.

    ``separate_held_out`` (the default) also keeps validation blocks more than ``gap`` rows from test blocks and
    rejects requests with too few separated validation blocks. ``False`` reproduces the splits written before
    that rule existed (``TIME_BLOCK_LEGACY``): validation takes the next blocks of the shuffle, so a validation
    block may touch a test block.
    """
    total = len(frames)
    val_size = _count(val_size, "val_size")
    test_size = _count(test_size, "test_size")
    block_size = _count(block_size, "block_size", 1)
    gap = _count(gap, "gap")
    if val_size + test_size > total:
        raise ValueError("validation + test split is larger than dataset")
    order = sorted(range(total), key=lambda i: (frames[i], i))
    blocks = [order[k:k + block_size] for k in range(0, total, block_size)]
    rng = np.random.default_rng(int(seed))
    shuffled = list(rng.permutation(len(blocks)))
    owner = ["train"] * len(blocks)
    counts = {"test": 0, "val": 0}
    test_intervals = []
    for name, want in (("test", test_size), ("val", val_size)):
        for b in shuffled:
            if counts[name] >= want:
                break
            if owner[b] != "train":
                continue
            first, last = b * block_size, b * block_size + len(blocks[b]) - 1
            if separate_held_out and name == "val" and any(first <= end + gap and last >= start - gap
                                                           for start, end in test_intervals):
                continue
            owner[b] = name
            counts[name] += len(blocks[b])
            if name == "test":
                test_intervals.append((first, last))
        if counts[name] < want:
            raise ValueError("not enough separated blocks; reduce split sizes or gap, or change seed")
    owners = [owner[b] for b, rows in enumerate(blocks) for _ in rows]
    purged = set()
    for positions in (range(total), range(total - 1, -1, -1)):
        nearest = None
        for pos in positions:
            if owners[pos] != "train":
                nearest = pos
            elif nearest is not None and abs(pos - nearest) <= gap:
                purged.add(pos)
    split = {"train": [], "val": [], "test": [], "purged": []}
    for pos, row in enumerate(order):
        split["purged" if pos in purged else owners[pos]].append(int(row))
    return {name: sorted(rows) for name, rows in split.items()}


def frame_numbers(rels: Sequence[str]) -> list[int]:
    """DonkeyCar frame index of each image name (``<index>_<angle>.jpg``); the row position when absent."""
    out = []
    for i, rel in enumerate(rels):
        head = Path(str(rel)).name.split("_", 1)[0]
        out.append(int(head) if head.isdigit() else i)
    return out


def time_block_contract(payload: dict) -> int:
    """Which separation a saved time-block split promises: ``TIME_BLOCK_SEPARATED`` or ``TIME_BLOCK_LEGACY``.

    Files written before the contract was recorded carry no ``contract`` field and are legacy splits.
    """
    contract = payload.get("contract", TIME_BLOCK_LEGACY)
    if type(contract) is not int or contract not in (TIME_BLOCK_LEGACY, TIME_BLOCK_SEPARATED):
        raise ValueError("unsupported time-block split contract")
    return contract


def validate_time_split_gap(frames, splits, gap, *, separate_held_out: bool = True):
    """Reject an existing split whose rows from different sets are too close in time.

    Training rows must always stay more than ``gap`` rows from validation and test rows. Validation and test
    rows must keep that distance from each other too, unless ``separate_held_out`` is False (a legacy split).
    """
    gap = _count(gap, "gap")
    owners = {row: name for name in SPLIT_NAMES for row in splits.get(name, [])}
    previous: dict[str, int] = {}
    for pos, row in enumerate(sorted(range(len(frames)), key=lambda i: (frames[i], i))):
        name = owners.get(row)
        if name is None:
            continue
        for other, last in previous.items():
            if other == name or pos - last > gap:
                continue
            if separate_held_out or "train" in (name, other):
                raise ValueError(f"time-block split keeps {name} and {other} rows within {gap} rows of each "
                                 "other; regenerate it")
        previous[name] = pos


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
    kind: str = "random",
    meta: dict | None = None,
    source_paths: Sequence[str | Path] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "source_count": int(source_count),
        "seed": int(seed),
        "kind": kind,
        **(meta or {}),
        "splits": {name: list(map(int, splits.get(name, []))) for name in SPLIT_NAMES},
    }
    if source_paths is not None:
        if len(source_paths) != source_count:
            raise ValueError("source_paths must match the split source_count")
        payload["source_order_hash"] = _source_order_hash(source_paths)
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out.parent, prefix=".split-",
                                         suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    return out


def _source_order_hash(paths: Sequence[str | Path]) -> str:
    names = json.dumps([Path(path).as_posix() for path in paths], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(names.encode("utf-8")).hexdigest()


def load_split(path: str | Path, *, source_count: int | None = None,
               source_paths: Sequence[str | Path] | None = None, allow_prefix: bool = False) -> dict[str, Any]:
    """Read a positional split and validate its dataset when the ordered source paths are available."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("unsupported split file version")
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("split file missing splits")
    total = payload.get("source_count")
    if type(total) is not int or total < 0:
        raise ValueError("split source_count must be a non-negative integer")
    if source_count is not None and total != source_count:
        raise ValueError(f"split source_count {total} does not match dataset size {source_count}")
    if source_paths is not None:
        if allow_prefix:
            source_paths = source_paths[:total]
        if len(source_paths) != total:
            raise ValueError(f"split source_count {total} does not match available dataset size {len(source_paths)}")
        if "source_order_hash" in payload and payload["source_order_hash"] != _source_order_hash(source_paths):
            raise ValueError("split source_order_hash does not match the dataset image order")
    seen = set()
    for name in SPLIT_NAMES:
        indices = splits.get(name, [])
        if not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < total for i in indices):
            raise ValueError(f"split {name} must contain integer indices within [0, {total})")
        if len(set(indices)) != len(indices) or seen.intersection(indices):
            raise ValueError(f"split {name} contains duplicate or overlapping indices")
        seen.update(indices)
    payload["splits"] = {name: splits.get(name, []) for name in SPLIT_NAMES}
    if payload.get("kind") == "time_block":
        contract = time_block_contract(payload)
        gap = _count(payload.get("gap", 0), "gap")
        if source_paths is not None:
            validate_time_split_gap(frame_numbers(source_paths), payload["splits"], gap,
                                    separate_held_out=contract >= TIME_BLOCK_SEPARATED)
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


def select_saved_split(rows: Sequence[tuple[Path, float, float]], path: str | Path, name: str) -> list:
    """Select from a saved split over a full or prefix catalog, as used by offline tools."""
    payload = load_split(path, source_paths=[rel for rel, *_ in rows], allow_prefix=True)
    return select_split(rows, payload["splits"], name)


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
    source_paths: Sequence[str | Path] | None = None,
) -> tuple[dict[str, list[int]] | None, Path | None]:
    """Load or build splits if requested; otherwise return ``(None, None)``."""

    if split_file:
        payload = load_split(split_file, source_count=len(rows), source_paths=source_paths)
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
    saved = save_split(split_out, splits, source_count=len(rows), seed=seed,
                       source_paths=source_paths) if split_out else None
    return splits, saved


def entries_hash(entries: Sequence[Any]) -> str:
    """Fingerprint of a sample list ``(relative path, angle, throttle)``: checks that two runs trained or
    validated on the same samples with the same labels."""

    digest = hashlib.sha256()
    for rel, angle, throttle in entries:
        digest.update(f"{Path(rel).as_posix()}\t{float(angle):.9g}\t{float(throttle):.9g}\n".encode("utf-8"))
    return digest.hexdigest()
