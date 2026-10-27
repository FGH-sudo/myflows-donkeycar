"""Machine-wide GPU lock for training entries.

Only one training may use the GPU at a time, whether the console, AutoPilot or a terminal started it. Every
training entry takes this lock before it touches the GPU and keeps it until it exits; a distributed run is
locked by the parent process that starts the workers.

The lock is an OS file lock, so it is released when the holder exits, even if it crashes. Who holds it is
recorded next to it (pid + process creation time); a record whose process is gone or was reused is stale.

``MYFLOWS_GPU_LOCK`` overrides the lock path; ``0``/``off`` disables locking (tests, CPU-only machines).
Waiting and acquiring are written to ``<run_dir>/gpu_lock.jsonl``: the AutoPilot Verifier starts its timeout
and hang clocks at the ``acquired`` event, so time spent queueing is never counted against a trial.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ENV_PATH = "MYFLOWS_GPU_LOCK"
EVENTS_FILE = "gpu_lock.jsonl"
WINDOWS = os.name == "nt"


def default_path() -> Path | None:
    """Lock path from the environment, or the per-machine default; ``None`` when locking is disabled."""
    raw = os.environ.get(ENV_PATH, "").strip()
    if raw.lower() in ("0", "off", "none", "false"):
        return None
    return Path(raw) if raw else Path(tempfile.gettempdir()) / "myflows" / "gpu.lock"


def _holder_path(path: Path) -> Path:
    return path.with_name(path.name + ".holder.json")


def _process_created(pid: int) -> float | None:
    try:
        import psutil

        return psutil.Process(pid).create_time()
    except Exception:  # NoSuchProcess, AccessDenied, psutil missing
        return None


def read_holder(path: Path | None = None) -> dict | None:
    """The live holder's record, or ``None`` when the lock is free or the record is stale."""
    path = default_path() if path is None else Path(path)
    if path is None:
        return None
    try:
        holder = json.loads(_holder_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    created = _process_created(int(holder.get("pid") or 0))
    if created is None or abs(created - float(holder.get("pid_create_time") or 0.0)) > 1.0:
        return None
    return holder


class GpuLock:
    """Hold the machine-wide GPU lock; use as a context manager or call acquire()/release()."""

    def __init__(self, path: Path | None = None, *, label: str = "", run_dir: str | Path | None = None,
                 poll_s: float = 0.5, say=None):
        self.path = default_path() if path is None else Path(path)
        self.label = label
        self.run_dir = Path(run_dir) if run_dir else None
        self.poll_s = float(poll_s)
        self.say = say or (lambda message: print(message, file=sys.stderr, flush=True))
        self._handle = None
        self.wait_s = 0.0
        self.acquired_unix_s: float | None = None

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def _event(self, event: str, **fields) -> None:
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        row = {"event": event, "unix_s": time.time(), "monotonic_s": time.perf_counter(), "pid": os.getpid(),
               **fields}
        with (self.run_dir / EVENTS_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _try_lock(self) -> bool:
        handle = self._handle
        handle.seek(0)
        try:
            if WINDOWS:
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def acquire(self, timeout_s: float | None = None) -> GpuLock:
        """Block until the lock is ours; ``TimeoutError`` after ``timeout_s`` (``None`` waits forever)."""
        if self.path is None:
            self._event("disabled")
            self.acquired_unix_s = time.time()
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        started = time.perf_counter()
        holder = None if self._try_lock() else (read_holder(self.path) or {})
        self._event("wait", lock=str(self.path), holder=holder)
        if holder is not None:
            who = f"pid {holder.get('pid')} · {holder.get('label') or holder.get('run_dir') or '未知任务'}" \
                if holder else "另一个进程"
            self.say(f"[gpu-lock] GPU 正被占用（{who}），排队等待: {self.path}")
            while not self._try_lock():
                if timeout_s is not None and time.perf_counter() - started > timeout_s:
                    self._handle.close()
                    self._handle = None
                    self._event("timeout", waited_s=time.perf_counter() - started)
                    raise TimeoutError(f"等待 GPU 锁超过 {timeout_s:.0f} s: {self.path}")
                time.sleep(self.poll_s)
        self.wait_s = time.perf_counter() - started
        self.acquired_unix_s = time.time()
        record = {"pid": os.getpid(), "pid_create_time": _process_created(os.getpid()), "label": self.label,
                  "run_dir": str(self.run_dir) if self.run_dir else None, "acquired_unix_s": self.acquired_unix_s,
                  "argv": sys.argv[:12]}
        tmp = _holder_path(self.path).with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, _holder_path(self.path))
        self._event("acquired", wait_s=self.wait_s)
        if holder is not None:
            self.say(f"[gpu-lock] 已获得 GPU 锁（等待 {self.wait_s:.1f} s）")
        return self

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            holder = _holder_path(self.path)
            current = json.loads(holder.read_text(encoding="utf-8"))
            if current.get("pid") == os.getpid():
                holder.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
        try:
            self._handle.seek(0)
            if WINDOWS:
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self._handle.close()
        self._handle = None
        self._event("released")

    def __enter__(self) -> GpuLock:
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def read_events(run_dir: str | Path) -> list[dict]:
    """The lock events of one run, oldest first."""
    rows = []
    try:
        with (Path(run_dir) / EVENTS_FILE).open(encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def acquired_unix_s(run_dir: str | Path) -> float | None:
    """When the run got the GPU (its clocks start here), or ``None`` while it is still waiting."""
    for row in read_events(run_dir):
        if row.get("event") in ("acquired", "disabled"):
            return float(row["unix_s"])
    return None
