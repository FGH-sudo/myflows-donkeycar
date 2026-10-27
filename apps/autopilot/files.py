"""Atomic JSON writes and tolerant JSON/JSONL reads used by every AutoPilot store."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

_append_lock = threading.Lock()


def clean(value: Any) -> Any:
    """Replace NaN/Inf by None so the files stay valid JSON for browsers."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path: str | Path, value: Any) -> Path:
    """Write via a temporary file and ``os.replace`` so readers never see a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    replace(tmp, path)
    return path


def replace(src: Path, dst: Path, attempts: int = 100) -> None:
    """``os.replace`` that waits out readers: Windows refuses to replace a file another handle has open."""
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.01 + 0.002 * attempt)


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(clean(row), ensure_ascii=False, default=str) + "\n"
    with _append_lock, path.open("a", encoding="utf-8") as f:
        f.write(line)


def read_jsonl(path: str | Path) -> list[dict]:
    """All complete rows; a torn last line (writer still busy) is skipped."""
    rows = []
    try:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if not line.endswith("\n"):
                    break
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows
