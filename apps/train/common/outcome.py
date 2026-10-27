# -*- coding: utf-8 -*-
"""How a training run ended: exit codes, failure kinds and ``result.json``.

Callers (the console, AutoPilot's Executor) read the exit code first and ``result.json`` for details, so a
run that dies of a non-finite loss or out of GPU memory is told apart from any other error without parsing
the log.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

RESULT_SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_ERROR = 1          # any other exception, including bad arguments reported with SystemExit(message)
EXIT_NONFINITE = 3      # loss became NaN or Inf
EXIT_OOM = 4            # GPU (or host) memory exhausted
EXIT_LOCK_TIMEOUT = 5   # gave up waiting for the GPU lock

FAILURE_EXIT_CODES = {"nan": EXIT_NONFINITE, "oom": EXIT_OOM, "lock_timeout": EXIT_LOCK_TIMEOUT,
                      "error": EXIT_ERROR}
EXIT_CODE_FAILURES = {code: kind for kind, code in FAILURE_EXIT_CODES.items()}


class NonFiniteLossError(FloatingPointError):
    """Training produced a NaN/Inf loss; the run stops without overwriting its last healthy checkpoint."""

    def __init__(self, step: int, epoch: int, loss: float):
        super().__init__(f"非有限 loss: {loss} (epoch {epoch}, global step {step})")
        self.step, self.epoch, self.loss = step, epoch, loss


def is_oom(exc: BaseException | str) -> bool:
    """CuPy's OutOfMemoryError, a CUDA allocation failure, host MemoryError, or their text in a worker error."""
    text = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    if isinstance(exc, MemoryError):
        return True
    return any(token in text for token in ("OutOfMemoryError", "cudaErrorMemoryAllocation", "out of memory",
                                           "显存不足", "MemoryError"))


def classify(exc: BaseException | str) -> str:
    """Failure kind of an exception or of an error message relayed from a distributed worker."""
    if isinstance(exc, NonFiniteLossError):
        return "nan"
    if isinstance(exc, TimeoutError) and "GPU 锁" in str(exc):
        return "lock_timeout"
    text = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    if "NonFiniteLossError" in text or "非有限 loss" in text:
        return "nan"
    if is_oom(exc):
        return "oom"
    return "error"


def write_result(run_dir: str | Path | None, result: dict) -> None:
    """Write ``<run_dir>/result.json`` atomically (temporary file + replace)."""
    if run_dir is None:
        return
    path = Path(run_dir) / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": RESULT_SCHEMA_VERSION, "written_unix_s": time.time(), **result}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_result(run_dir: str | Path) -> dict | None:
    try:
        return json.loads((Path(run_dir) / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
