import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmark.cuda_ops import run_matrix
from benchmark.profile_cuda import execute
from benchmark.stage1_common import RunArtifacts, digest


class Stage1ArtifactsTest(unittest.TestCase):
    def test_cpu_cli_records_outputs_and_refuses_existing_run(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            cmd = [sys.executable, "-X", "utf8", "-m", "benchmark.cuda_ops", "--suite", "correctness",
                   "--backends", "numpy", "--case", "T0", "--out-dir", str(path)]
            first = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(first.returncode, 0, first.stderr)
            expected = ("config.json", "manifest.json", "results.json", "timings.csv", "stdout.log", "report.md")
            self.assertTrue(all((path / name).is_file() for name in expected))
            self.assertFalse((path / "source-at-run.zip").exists())
            result = json.loads((path / "results.json").read_text())
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["correctness"][0]["actual_backend"], "numpy")
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            for name, expected_hash in manifest["artifacts_sha256"].items():
                self.assertEqual(digest(path / name), expected_hash)
            old_manifest = digest(path / "manifest.json")
            second = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(digest(path / "manifest.json"), old_manifest)

    def test_required_gpu_failure_is_recorded_not_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            args = argparse.Namespace(suite="correctness", seed=0, case="T0", backends=["cuda_c"])
            with patch("benchmark.stage1_common.environment", return_value={}), patch("benchmark.stage1_common.source_state", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                    with RunArtifacts(path, {}) as run:
                        with patch("benchmark.cuda_ops.set_device", side_effect=RuntimeError("CUDA unavailable")):
                            run_matrix(args, run)
            result = json.loads((path / "results.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertIn("CUDA unavailable", result["error"])

    def test_profiler_failure_overrides_workload_success(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "profiler.log"
            completed = subprocess.CompletedProcess([], 1, stdout="workload status=passed\nERR_NVGPUCTRPERM")
            with patch("benchmark.profile_cuda.subprocess.run", return_value=completed):
                with self.assertRaisesRegex(RuntimeError, "profiler exited 1"):
                    execute(["ncu"], log)
            self.assertIn("ERR_NVGPUCTRPERM", log.read_text())
