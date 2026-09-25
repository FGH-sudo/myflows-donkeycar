"""Incremental readers for run artifacts (JSONL tails, JSON files, TensorBoard scalars)."""

from __future__ import annotations

import csv
import json
import threading
from collections.abc import Iterable
from pathlib import Path


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_csv_rows(path: Path) -> list[dict]:
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []
    return [{k: _number(v) for k, v in row.items()} for row in rows]


def _number(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return value


def tail_text(path: Path, max_bytes: int = 64 * 1024) -> str:
    try:
        with Path(path).open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return ""
    for encoding in ("utf-8", "gbk"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", errors="replace")
    if size > max_bytes:
        text = text.split("\n", 1)[-1]
    return text


class JsonlTail:
    """Reads a growing JSONL file from a byte offset.

    A trailing line without ``\\n`` is kept in a buffer until it is completed, so rows
    written by line-buffered writers are never parsed half-way. Rows are projected to
    ``fields`` (plus scalar values of nested dicts listed in ``nested``) to bound memory.
    """

    def __init__(self, path: Path, fields: Iterable[str] | None = None, nested: Iterable[str] = ()):
        self.path = Path(path)
        self.fields = set(fields) if fields is not None else None
        self.nested = tuple(nested)
        self.rows: list[dict] = []
        self._offset = 0
        self._pending = b""
        self._lock = threading.Lock()

    def _project(self, row: dict) -> dict:
        if self.fields is None:
            return row
        out = {k: v for k, v in row.items() if k in self.fields}
        for key in self.nested:
            value = row.get(key)
            if isinstance(value, dict):
                for sub, sub_value in value.items():
                    if isinstance(sub_value, (int, float, str)) or sub_value is None:
                        out[f"{key}_{sub}"] = sub_value
            elif value is not None and key in self.fields:
                out[key] = value
        return out

    def poll(self) -> list[dict]:
        with self._lock:
            try:
                size = self.path.stat().st_size
            except OSError:
                return []
            if size < self._offset:
                self.rows, self._offset, self._pending = [], 0, b""
            if size == self._offset:
                return []
            with self.path.open("rb") as f:
                f.seek(self._offset)
                chunk = f.read(size - self._offset)
            self._offset += len(chunk)
            data = self._pending + chunk
            lines = data.split(b"\n")
            self._pending = lines.pop()
            new_rows = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    new_rows.append(self._project(json.loads(line)))
                except ValueError:
                    continue
            self.rows.extend(new_rows)
            return new_rows


def downsample_indices(n: int, max_points: int | None) -> list[int]:
    if not max_points or n <= max_points:
        return list(range(n))
    stride = n / float(max_points)
    picked = sorted({int(i * stride) for i in range(max_points)} | {n - 1})
    return picked


def columns(rows: list[dict], fields: Iterable[str], indices: list[int] | None = None) -> dict[str, list]:
    selected = rows if indices is None else [rows[i] for i in indices]
    return {field: [row.get(field) for row in selected] for field in fields}


TB_STEP_TAGS = {
    "train/loss_step": "loss",
    "train/loss_running_mean": "running_loss",
    "train/accuracy_step": "accuracy",
    "train/data_load_ms": "data_load_ms",
    "train/train_step_ms": "train_step_ms",
    "train/step_time_ms": "step_time_ms",
    "train/samples_per_sec": "samples_per_sec",
    "gradients/global_norm": "grad_norm",
}
TB_EPOCH_TAGS = {
    "train/loss_epoch": "loss",
    "train/accuracy_epoch": "accuracy",
    "val/loss_epoch": "val_loss",
    "val/accuracy_epoch": "val_accuracy",
    "train/lr": "learning_rate",
    "train/epoch_time_s": "epoch_time_s",
}


def read_tensorboard(log_dir: Path) -> dict:
    """Load the scalar tags used by TrainingDashboard; histograms and images are skipped."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0, "tensors": 16, "histograms": 1,
                                                        "images": 1, "audio": 1, "compressedHistograms": 1})
    acc.Reload()
    tags = acc.Tags()
    scalar_tags = set(tags.get("scalars", []))

    def merge(mapping):
        by_step: dict[int, dict] = {}
        for tag, name in mapping.items():
            if tag not in scalar_tags:
                continue
            for event in acc.Scalars(tag):
                row = by_step.setdefault(int(event.step), {"step": int(event.step), "unix_s": event.wall_time})
                row[name] = float(event.value)
        return [by_step[k] for k in sorted(by_step)]

    steps = merge(TB_STEP_TAGS)
    epochs = [dict(row, epoch=row.pop("step")) for row in merge(TB_EPOCH_TAGS)]
    config = None
    tensor_tags = set(tags.get("tensors", []))
    config_tag = next((t for t in ("config/training_args/text_summary", "config/training_args") if t in tensor_tags), None)
    if config_tag:
        try:
            events = acc.Tensors(config_tag)
            raw = events[-1].tensor_proto.string_val[0].decode("utf-8")
            text = raw.strip().removeprefix("```json").removesuffix("```").strip()
            config = json.loads(text)
        except Exception:
            config = None
    return {"steps": steps, "epochs": epochs, "config": config, "scalar_tags": sorted(scalar_tags)}
