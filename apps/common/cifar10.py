"""Read the official CIFAR-10 binary batches and build reproducible stratified splits."""

from pathlib import Path

import numpy as np

RECORD_BYTES = 1 + 3 * 32 * 32
ROWS_PER_FILE = 10_000
CLASSES = 10


def _batch(path, *, rows=ROWS_PER_FILE):
    expected = rows * RECORD_BYTES
    if path.stat().st_size != expected:
        raise ValueError(f"{path.name}: expected {expected} bytes")
    data = np.fromfile(path, dtype=np.uint8)
    if data.size != expected:
        raise ValueError(f"{path.name}: truncated binary batch")
    records = data.reshape(rows, RECORD_BYTES)
    labels = records[:, 0].astype(np.int64)
    if np.any(labels >= CLASSES):
        raise ValueError(f"{path.name}: CIFAR-10 labels must be in [0,9]")
    return np.ascontiguousarray(records[:, 1:].reshape(rows, 3, 32, 32)), labels


def load(root, split):
    directory = Path(root)
    if (directory / "cifar-10-batches-bin").is_dir():
        directory = directory / "cifar-10-batches-bin"
    if split == "train":
        files = [f"data_batch_{i}.bin" for i in range(1, 6)]
    elif split == "test":
        files = ["test_batch.bin"]
    else:
        raise ValueError("choose the official train or test archive")
    batches = [_batch(directory / name) for name in files]
    images = np.concatenate([x for x, _ in batches])
    labels = np.concatenate([y for _, y in batches])
    expected = labels.size // CLASSES
    if not np.all(np.bincount(labels, minlength=CLASSES) == expected):
        raise ValueError(f"unexpected class counts in official {split} set")
    return images, labels


def stratified_split(labels, *, val_size=5000, seed=42):
    labels = np.asarray(labels)
    if val_size % CLASSES or not 0 < val_size < len(labels):
        raise ValueError("validation size must be a positive multiple of ten and leave training examples")
    if labels.ndim != 1 or np.any((labels < 0) | (labels >= CLASSES)):
        raise ValueError("invalid labels")
    rng = np.random.default_rng(seed)
    per_class = val_size // CLASSES
    train, val = [], []
    for cls in range(CLASSES):
        indices = rng.permutation(np.flatnonzero(labels == cls))
        if len(indices) <= per_class:
            raise ValueError(f"not enough training examples for class {cls}")
        train.extend(indices[per_class:])
        val.extend(indices[:per_class])
    return np.sort(train).astype(np.int64), np.sort(val).astype(np.int64)


def balanced_subset(indices, labels, size, *, seed=42):
    indices = np.asarray(indices, dtype=np.int64)
    if not 0 < size <= len(indices):
        raise ValueError("subset size must be positive and fit within the selected split")
    labels = np.asarray(labels)[indices]
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in range(CLASSES):
        count = size // CLASSES + (cls < size % CLASSES)
        available = indices[labels == cls]
        if len(available) < count:
            raise ValueError(f"not enough examples for balanced class {cls}")
        chosen.extend(rng.choice(available, count, replace=False))
    return np.sort(chosen).astype(np.int64)
