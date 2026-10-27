"""apps.train.train_myflows_donkey: seeds, result.json, exit codes, phase timing and the GPU lock (spec P1-P8)."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from apps.common.gpu_lock import read_events
from apps.train.common import outcome
from tests.test_train_distributed import _tub

ROOT = Path(__file__).resolve().parents[1]


def _cuda() -> bool:
    try:
        from MyFlows.core.device import cuda_available

        return cuda_available()
    except Exception:
        return False


def _train(work: Path, name: str, *extra: str, device: str = "cpu", env: dict | None = None):
    out = work / name
    command = [sys.executable, "-X", "utf8", "-m", "apps.train.train_myflows_donkey", "--data", str(work / "tub"),
               "--catalog", "missing.catalog", "--device", device, "--base-width", "4", "--batch", "6",
               "--epochs", "2", "--val-size", "5", "--no-tensorboard", "--checkpoint-every", "0",
               "--log-dir", str(out), "--out", str(out / "ckpt"), "--best-out", str(out / "best"),
               "--stop-file", str(out / "STOP"), "--run-dir", str(out / "run"), "--sync-timeout", "60", *extra]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=600, check=False, env={**os.environ, **(env or {})})
    return result, out


def _params(out: Path) -> dict:
    with np.load(out / "ckpt.npz") as saved:
        return {key: saved[key] for key in saved.files if key.startswith(("param_", "buffer_"))}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TrainOutcomeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls.tmp.name)
        _tub(cls.work)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ok(self, name, *extra, **kwargs):
        result, out = _train(self.work, name, *extra, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-4000:])
        return out, json.loads((out / "run" / "result.json").read_text(encoding="utf-8"))

    def test_same_seed_is_bitwise_reproducible_with_augmentation_dropout_and_mixup(self):
        noisy = ("--augment", "--dropout", "0.3", "--mixup")
        first, result = self.ok("seed-a", "--seed", "11", *noisy)
        second, _ = self.ok("seed-b", "--seed", "11", *noisy)
        other, _ = self.ok("seed-c", "--seed", "12", *noisy)
        a, b, c = _params(first), _params(second), _params(other)
        for key, value in a.items():
            np.testing.assert_array_equal(value, b[key], err_msg=key)
        self.assertTrue(any(not np.array_equal(value, c[key]) for key, value in a.items()))
        config = json.loads((first / "run" / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["seeds"]["init"], 11)
        self.assertEqual(config["seeds"], result["seeds"])
        self.assertNotEqual(config["seeds"]["augment"][0], config["seeds"]["dropout"][0])

    def test_result_json_metrics_phases_and_samples(self):
        out, result = self.ok("result", "--seed", "3", "--phase-timing-every", "2")
        self.assertEqual((result["status"], result["exit_code"], result["failure"]), ("succeeded", 0, None))
        self.assertEqual(result["schema_version"], outcome.RESULT_SCHEMA_VERSION)
        self.assertEqual(result["samples"], {"train": 15, "val": 5, "trained": 36, "expected_trained": 36})
        self.assertEqual(result["epochs_completed"], 2)
        self.assertEqual(result["backend"]["actual"], "numpy")
        self.assertIsNotNone(result["data"]["val_hash"])
        maes = [row["val_angle_mae"] for row in result["epochs"]]
        self.assertEqual(result["metrics"]["best_val_angle_mae"], min(maes))
        best = result["metrics"]["best_epoch"]
        self.assertEqual(result["metrics"]["val_angle_mae"], maes[best - 1])
        rows = _rows(out / "run" / "metrics.jsonl")
        steps = [r for r in rows if r["kind"] == "step"]
        timed = [r["step"] for r in steps if "forward_ms" in r]
        self.assertEqual(timed, [1, 3, 5])  # global steps 0, 2, 4 counted from zero
        self.assertTrue(all(r["backward_ms"] > 0 and r["optimizer_ms"] >= 0 for r in steps if "forward_ms" in r))
        epochs = [r for r in rows if r["kind"] == "epoch"]
        self.assertEqual([r["val_angle_mae"] for r in epochs], maes)
        legacy = json.loads((out / "run" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(legacy["status"], "passed")

    def test_non_finite_loss_exits_with_its_own_code(self):
        result, out = _train(self.work, "nan", "--lr", "1e30", "--seed", "1")
        self.assertEqual(result.returncode, outcome.EXIT_NONFINITE, result.stdout[-3000:] + result.stderr[-3000:])
        payload = json.loads((out / "run" / "result.json").read_text(encoding="utf-8"))
        self.assertEqual((payload["status"], payload["failure"]["kind"]), ("failed", "nan"))
        self.assertEqual(json.loads((out / "run" / "results.json").read_text(encoding="utf-8"))["failure_kind"], "nan")

    def test_distributed_run_writes_result_phases_and_classifies_nan(self):
        out, payload = self.ok("ring", "--distributed", "ring", "--workers", "2", "--seed", "5",
                               "--phase-timing-every", "1")
        self.assertEqual(payload["parallel"]["workers"], 2)
        self.assertIsNotNone(payload["metrics"]["val_angle_mae"])
        rank_rows = _rows(out / "run" / "rank-1" / "steps.jsonl")
        self.assertTrue(all("forward_ms" in r and "optimizer_ms" in r for r in rank_rows))
        result, out = _train(self.work, "ring-nan", "--distributed", "ring", "--workers", "2", "--lr", "1e30")
        self.assertEqual(result.returncode, outcome.EXIT_NONFINITE, result.stdout[-3000:] + result.stderr[-3000:])
        payload = json.loads((out / "run" / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["failure"]["kind"], "nan")

    def test_bad_arguments_still_leave_a_result(self):
        result, out = _train(self.work, "bad", "--split-file", str(self.work / "missing-split.json"))
        self.assertEqual(result.returncode, outcome.EXIT_ERROR)
        payload = json.loads((out / "run" / "result.json").read_text(encoding="utf-8"))
        self.assertEqual((payload["status"], payload["failure"]["kind"]), ("failed", "error"))

    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_gpu_run_takes_the_lock_and_reports_peak_memory(self):
        lock_path = self.work / "gpu.lock"
        out, payload = self.ok("gpu", "--seed", "2", "--phase-timing-every", "1", device="cuda",
                               env={"MYFLOWS_GPU_LOCK": str(lock_path)})
        memory = payload["memory"]
        self.assertEqual(memory["source"], "cupy_memory_pool")
        self.assertGreater(memory["peak_used_mb"], 0)
        self.assertGreaterEqual(memory["peak_pool_mb"], memory["peak_used_mb"])
        self.assertEqual([e["event"] for e in read_events(out / "run")], ["wait", "acquired", "released"])
        self.assertIsNotNone(payload["timing"]["lock_acquired_unix_s"])
        steps = [r for r in _rows(out / "run" / "metrics.jsonl") if r["kind"] == "step"]
        self.assertTrue(all("gpu_mem_used_mb" in r and "forward_ms" in r for r in steps))


class ClassifyTest(unittest.TestCase):
    def test_failure_kinds(self):
        self.assertEqual(outcome.classify(outcome.NonFiniteLossError(3, 1, float("nan"))), "nan")
        self.assertEqual(outcome.classify("rank 1: NonFiniteLossError: 非有限 loss: inf"), "nan")
        self.assertEqual(outcome.classify("rank 0: RuntimeError: 显存不足（Out of memory allocating 1 bytes）"), "oom")
        self.assertEqual(outcome.classify(MemoryError()), "oom")
        self.assertEqual(outcome.classify(TimeoutError("等待 GPU 锁超过 5 s")), "lock_timeout")
        self.assertEqual(outcome.classify(ValueError("bad")), "error")
        self.assertEqual(outcome.FAILURE_EXIT_CODES["oom"], outcome.EXIT_OOM)


if __name__ == "__main__":
    unittest.main()
