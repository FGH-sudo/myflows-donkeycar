"""The campaign's fixed data split: every trial validates on the same samples (spec §6.5 post-check)."""

from __future__ import annotations

from pathlib import Path

from apps.common.donkey_data import load_donkey_index
from apps.common.splits import (build_split, build_time_block_split, entries_hash, frame_numbers, load_split,
                                save_split, select_split)

from .provenance import data_hash, file_hash

CATALOG = "catalog_generated.catalog"


def load_index(repo_root: Path, max_samples: int) -> list:
    """The same sample list apps.train builds from its defaults (catalog first, sorted, first N)."""
    data_dir = Path(repo_root) / "mycar" / "data"
    index = load_donkey_index(data_dir, fixed_throttle=0.5, angle_scale=1.0, catalog_name=CATALOG)
    if max_samples and max_samples > 0:
        index = index[:min(int(max_samples), len(index))]
    return index


def ensure_split(repo_root: Path, splits_dir: Path, *, max_samples: int, val_size: int, split_seed: int,
                 kind: str = "random", test_size: int = 0, block_size: int = 50, gap: int = 5) -> dict:
    """Create (or reuse) the split of the first ``max_samples`` samples and describe it.

    ``random`` draws validation (and test) rows uniformly; ``time_block`` hands out contiguous blocks of
    frames and purges the frames next to a boundary (apps.common.splits.build_time_block_split), so the
    held-out sets are not near-duplicates of training frames. The test set never takes part in selection.
    """
    repo_root = Path(repo_root)
    index = load_index(repo_root, max_samples)
    if not index:
        raise ValueError("mycar/data 中没有可用样本")
    if val_size + test_size >= len(index):
        raise ValueError(f"验证集 {val_size} 条加测试集 {test_size} 条不小于样本总数 {len(index)}")
    tag = "" if kind == "random" else f"-tb{block_size}g{gap}"
    name = f"donkey-{max_samples or 'all'}-val{val_size}" + (f"-test{test_size}" if test_size else "") + f"{tag}-s{split_seed}.json"
    path = Path(splits_dir) / name
    meta = {}
    if kind == "time_block":
        splits = build_time_block_split(frame_numbers([rel for rel, *_ in index]), val_size=val_size,
                                        test_size=test_size, block_size=block_size, gap=gap, seed=split_seed)
        meta = {"block_size": block_size, "gap": gap, "purged": len(splits.pop("purged"))}
    elif kind == "random":
        splits = build_split(index, val_size=val_size, test_size=test_size, seed=split_seed)
    else:
        raise ValueError(f"未知的划分方式：{kind}")
    if path.exists():
        saved = load_split(path, source_count=len(index))
        if saved["splits"] != splits:
            raise ValueError(f"已有划分文件 {path} 与当前数据不一致；请换 split_seed 或删除旧文件")
    else:
        save_split(path, splits, source_count=len(index), seed=split_seed, kind=kind, meta=meta)
    try:
        rel = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        rel = str(path)
    return describe_split(repo_root, rel, index=index)


def describe_split(repo_root: Path, split_file: str, *, index: list | None = None,
                   max_samples: int = 0) -> dict:
    path = Path(split_file)
    path = path if path.is_absolute() else Path(repo_root) / path
    if index is None:
        index = load_index(repo_root, max_samples)
    payload = load_split(path, source_count=len(index))
    val = select_split(index, payload["splits"], "val")
    train = select_split(index, payload["splits"], "train")
    test = select_split(index, payload["splits"], "test")
    return {"split_file": split_file, "split_hash": file_hash(path), "val_hash": entries_hash(val) if val else None,
            "test_hash": entries_hash(test) if test else None, "train_count": len(train), "val_count": len(val),
            "test_count": len(test), "source_count": len(index), "kind": payload.get("kind", "random"),
            "purged": payload.get("purged", 0), "block_size": payload.get("block_size"),
            "data_hash": data_hash(Path(repo_root) / "mycar" / "data", CATALOG)}
