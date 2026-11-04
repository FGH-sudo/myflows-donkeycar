"""Process lifetime and console ownership checks for the production sampler."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import psutil

from apps.console.jobs import JobManager
from MyFlows.monitoring.schema import EXTERNAL_MONITOR_ENV

ROOT = Path(__file__).resolve().parents[1]


class MonitorProcessTest(unittest.TestCase):
    def test_sampler_exits_after_parent_abrupt_exit(self):
        script = """
import os, sys
from MyFlows.monitoring.record import MonitorProcess
monitor = MonitorProcess(sys.argv[1], 0.1).start()
print(monitor.process.pid, flush=True)
os._exit(0)
"""
        with tempfile.TemporaryDirectory() as tmp:
            parent = subprocess.run([sys.executable, "-c", script, tmp], cwd=ROOT, capture_output=True,
                                    text=True, timeout=30)
            self.assertEqual(parent.returncode, 0, parent.stderr)
            pid = int(parent.stdout.strip().splitlines()[-1])
            deadline = time.monotonic() + 6
            while psutil.pid_exists(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(psutil.pid_exists(pid), "sampler outlived the training process")
            self.assertTrue((Path(tmp) / "resources.jsonl").is_file())

    def test_console_marks_external_sampling_and_closes_recorder(self):
        class Recorder:
            stopped = False

            def stop(self):
                self.stopped = True

        recorder = Recorder()
        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(Path(tmp) / "jobs", ROOT, poll_interval_s=.02,
                                 resource_recorder_factory=lambda job, pid: recorder).start()
            try:
                code = f"import os; print(os.environ.get('{EXTERNAL_MONITOR_ENV}'), flush=True)"
                job = manager.submit(job_type="single", label="monitor", task="fixture",
                                     command=[sys.executable, "-c", code], extra={"record_resources": True})
                deadline = time.monotonic() + 5
                while manager.get(job["id"])["status"] in ("pending", "running") and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(manager.get(job["id"])["status"], "succeeded")
                self.assertEqual(Path(job["log_path"]).read_text(encoding="utf-8").strip(), "1")
                self.assertTrue(recorder.stopped)
            finally:
                manager.shutdown()


if __name__ == "__main__":
    unittest.main()
