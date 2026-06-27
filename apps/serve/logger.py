# -*- coding: utf-8 -*-
"""Serving 结构化 JSONL 日志。"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping


class JsonlLogger:
  def __init__(self, log_file: str | Path | None, enabled: bool = True):
    self.path = Path(log_file) if log_file else None
    self.enabled = bool(enabled and self.path is not None)
    self.lock = threading.Lock()
    if self.enabled:
      self.path.parent.mkdir(parents=True, exist_ok=True)

  def write(self, record: Mapping[str, Any]) -> None:
    if not self.enabled or self.path is None:
      return
    with self.lock:
      with self.path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), ensure_ascii=False, default=str))
        f.write("\n")
