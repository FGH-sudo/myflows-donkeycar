# -*- coding: utf-8 -*-
"""Serving 运行指标统计。"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Mapping


class ServeMetrics:
  def __init__(self, max_latencies: int = 1000):
    self.started_at = time.time()
    self.total_requests = 0
    self.total_errors = 0
    self.latencies = deque(maxlen=int(max_latencies))
    self.lock = threading.Lock()

  def record(self, latency_ms: float, status_code: int = 200, ok: bool | None = None) -> None:
    success = bool(ok) if ok is not None else int(status_code) < 400
    with self.lock:
      self.total_requests += 1
      if not success:
        self.total_errors += 1
      self.latencies.append(float(latency_ms))

  def snapshot(self, model: Mapping[str, Any] | None = None) -> dict[str, Any]:
    with self.lock:
      latencies = sorted(self.latencies)
      total_requests = int(self.total_requests)
      total_errors = int(self.total_errors)

    def pct(q: float) -> float:
      if not latencies:
        return 0.0
      idx = min(len(latencies) - 1, max(0, int(q * (len(latencies) - 1))))
      return float(latencies[idx])

    data: dict[str, Any] = {
        "total_requests": total_requests,
        "total_errors": total_errors,
        "error_rate": total_errors / max(total_requests, 1),
        "mean_latency_ms": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "p50_latency_ms": pct(0.50),
        "p95_latency_ms": pct(0.95),
        "p99_latency_ms": pct(0.99),
        "uptime_s": time.time() - self.started_at,
    }
    if model is not None:
      data["model"] = dict(model)
    return data
