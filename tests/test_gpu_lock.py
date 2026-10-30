"""apps.common.gpu_lock: one training on the GPU at a time, released when the holder dies."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import psutil

from apps.common import gpu_lock
from apps.common.gpu_lock import GpuLock, acquired_unix_s, read_events, read_holder

ROOT = Path(__file__).resolve().parents[1]

HOLDER = """
import os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from apps.common.gpu_lock import GpuLock
lock = GpuLock(Path(sys.argv[1]), label="holder", run_dir=sys.argv[2]).acquire()
print("locked", os.getpid(), flush=True)
time.sleep(120)
"""


def _wait(predicate, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class GpuLockTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.path = self.root / "gpu.lock"
        self.procs = []

    def tearDown(self):
        for proc in self.procs:
            for child in psutil.Process(proc.pid).children(recursive=True) if proc.poll() is None else []:
                child.kill()
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)
        self._tmp.cleanup()

    def holder(self) -> subprocess.Popen:
        proc = subprocess.Popen([sys.executable, "-c", HOLDER, str(self.path), str(self.root / "holder-run"),
                                 str(ROOT)], stdout=subprocess.PIPE, text=True)
        self.procs.append(proc)
        word, pid = proc.stdout.readline().split()
        self.assertEqual(word, "locked")
        # On Windows a venv python.exe is a launcher; the interpreter holding the lock is its child.
        return psutil.Process(int(pid))

    def test_second_process_waits_and_times_out_while_held(self):
        proc = self.holder()
        holder = read_holder(self.path)
        self.assertEqual((holder["pid"], holder["label"]), (proc.pid, "holder"))
        lock = GpuLock(self.path, run_dir=self.root / "waiter", poll_s=0.05, say=lambda _m: None)
        with self.assertRaises(TimeoutError):
            lock.acquire(timeout_s=0.5)
        events = read_events(self.root / "waiter")
        self.assertEqual([e["event"] for e in events], ["wait", "timeout"])
        self.assertEqual(events[0]["holder"]["pid"], proc.pid)
        self.assertIsNone(acquired_unix_s(self.root / "waiter"))

    def test_holder_death_releases_the_lock_and_wait_is_recorded(self):
        proc = self.holder()
        lock = GpuLock(self.path, run_dir=self.root / "waiter", poll_s=0.05, say=lambda _m: None)
        killer = threading_timer(0.6, proc.kill)
        started = time.time()
        lock.acquire(timeout_s=30)
        killer.join()
        try:
            self.assertGreater(lock.wait_s, 0.4)
            events = read_events(self.root / "waiter")
            self.assertEqual([e["event"] for e in events], ["wait", "acquired"])
            self.assertEqual(events[0]["holder"]["pid"], proc.pid)
            self.assertGreaterEqual(acquired_unix_s(self.root / "waiter"), started + 0.4)
            # The dead holder's record is stale; ours replaced it.
            self.assertEqual(read_holder(self.path)["pid"], os.getpid())
        finally:
            lock.release()
        self.assertIsNone(read_holder(self.path))

    def test_stale_record_is_ignored_and_free_lock_is_taken_at_once(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        (self.root / "gpu.lock.holder.json").write_text(json.dumps({"pid": 999999, "pid_create_time": 1.0}),
                                                        encoding="utf-8")
        self.assertIsNone(read_holder(self.path))
        with GpuLock(self.path, run_dir=self.root / "run") as lock:
            self.assertLess(lock.wait_s, 0.5)
            self.assertEqual(read_holder(self.path)["pid"], os.getpid())
        events = read_events(self.root / "run")
        self.assertEqual([e["event"] for e in events], ["wait", "acquired", "released"])
        self.assertIsNone(events[0]["holder"])

    def test_environment_can_disable_locking(self):
        with mock.patch.dict(os.environ, {gpu_lock.ENV_PATH: "off"}):
            lock = GpuLock(run_dir=self.root / "run").acquire()
            self.assertFalse(lock.enabled)
            lock.release()
        self.assertIsNotNone(acquired_unix_s(self.root / "run"))

    def test_cpu_start_is_recorded_without_taking_the_gpu_lock(self):
        with GpuLock(self.path, label="GPU holder"), \
                GpuLock(self.path, enabled=False, run_dir=self.root / "cpu") as lock:
            self.assertFalse(lock.enabled)
            self.assertEqual(read_holder(self.path)["label"], "GPU holder")
        self.assertEqual([e["event"] for e in read_events(self.root / "cpu")], ["disabled"])
        self.assertIsNotNone(acquired_unix_s(self.root / "cpu"))


def threading_timer(delay, fn):
    import threading

    thread = threading.Thread(target=lambda: (time.sleep(delay), fn()), daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    unittest.main()
