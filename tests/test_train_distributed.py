"""apps.train.train_myflows_donkey: one training loop for single process, PS and Ring AllReduce."""

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from apps.train.train_myflows_donkey import _epoch_samples
from MyFlows.data.pipeline import MultiprocessDataLoader

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SAMPLES = 15  # 20 images, 5 held out for validation
BATCH = 6
STEPS_PER_EPOCH = 3


def _tub(directory: Path, count: int = 20) -> Path:
    """Tiny tub without catalog: the angle is encoded in the file name; bright stripes follow it."""
    images = directory / "tub" / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(count):
        angle = float(rng.uniform(-1.0, 1.0))
        image = np.full((24, 32, 3), 40, np.uint8)
        image[:, max(1, int(16 + 12 * angle)) - 1:][:, :6] = 220
        cv2.imwrite(str(images / f"{index}_{angle:.4f}.jpg"), image)
    return directory / "tub"


def _train(work: Path, name: str, *extra: str, epochs: int = 2) -> tuple[subprocess.CompletedProcess, Path]:
    out = work / name
    command = [sys.executable, "-X", "utf8", "-m", "apps.train.train_myflows_donkey", "--data", str(work / "tub"),
               "--catalog", "missing.catalog", "--device", "cpu", "--base-width", "4", "--batch", str(BATCH),
               "--epochs", str(epochs), "--val-size", "5", "--seed", "7", "--no-tensorboard",
               "--checkpoint-every", "0", "--log-dir", str(out), "--out", str(out / "ckpt"),
               "--best-out", str(out / "best"), "--stop-file", str(work / f"{name}.STOP"),
               "--run-dir", str(out / "run"), "--sync-timeout", "60", *extra]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=600, check=False)
    return result, out


def _parameters(out: Path) -> dict:
    with np.load(out / "ckpt.npz") as saved:
        return {key: saved[key] for key in saved.files if key.startswith(("param_", "buffer_"))}


class EpochSamplesTest(unittest.TestCase):
    def test_single_rank_order_matches_the_shuffled_loader_with_padding(self):
        dataset = [(f"sample-{i}", float(i), 0.0, 4, 4, "float32") for i in range(10)]
        for epoch in (0, 3):
            loader_order = list(range(10))
            random.Random(epoch).shuffle(loader_order)
            expected = [dataset[i] for i in loader_order] + [dataset[loader_order[-1]]] * 2
            self.assertEqual(_epoch_samples(dataset, epoch, [4], 0), expected)
            # MultiprocessDataLoader(shuffle=True, seed=epoch) visits the samples in the same order.
            seen = [item for x_items, _ in MultiprocessDataLoader(dataset, 4, shuffle=True, seed=epoch,
                                                                 load_fn=lambda s: (s, None)) for item in x_items]
            self.assertEqual(seen, expected[:10])

    def test_ranks_take_disjoint_consecutive_slices_of_every_global_batch(self):
        dataset = list(range(7))
        shards = [_epoch_samples(dataset, 1, [2, 1], rank) for rank in (0, 1)]
        merged = [item for step in range(3) for item in shards[0][2 * step:2 * step + 2] + shards[1][step:step + 1]]
        self.assertEqual(merged, _epoch_samples(dataset, 1, [3], 0))


class DistributedTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls.tmp.name)
        _tub(cls.work)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _passed(self, name, *extra, epochs=2):
        result, out = _train(self.work, name, *extra, epochs=epochs)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-4000:])
        return out, json.loads((out / "run" / "results.json").read_text(encoding="utf-8"))

    def test_one_worker_ps_and_ring_reproduce_single_process_training(self):
        reference, _ = self._passed("single")
        expected = _parameters(reference)
        for name, extra in (("ps-1", ("--distributed", "ps", "--workers", "1")),
                            ("ring-1", ("--distributed", "ring", "--workers", "1"))):
            with self.subTest(name):
                out, results = self._passed(name, *extra)
                self.assertEqual(results["steps"], 2 * STEPS_PER_EPOCH)
                # Bit for bit: the conv biases ahead of BN get noise-level gradients that Adam turns into
                # full-size steps, so any rounding difference would show up there first.
                for key, value in _parameters(out).items():
                    np.testing.assert_array_equal(value, expected[key], err_msg=key)

    def test_ps_and_ring_workers_agree_and_save_standard_checkpoints(self):
        runs = {}
        for name, extra in (("ps-2", ("--distributed", "ps", "--workers", "2")),
                            ("ps-socket-3", ("--distributed", "ps", "--workers", "3", "--transport", "socket_json")),
                            ("ring-2", ("--distributed", "ring", "--workers", "2"))):
            with self.subTest(name):
                out, results = self._passed(name, *extra)
                self.assertEqual(results["steps"], 2 * STEPS_PER_EPOCH)
                self.assertIsNotNone(results["best_loss"])
                meta = json.loads((out / "ckpt.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["training"]["global_step"], 2 * STEPS_PER_EPOCH)
                self.assertTrue(meta["training"]["epoch_complete"])
                self.assertTrue((out / "best.npz").exists())
                runs[name] = _parameters(out)
        # Ring scales each rank's tensors by n_i / N like the PS, and two-term sums commute: same bits.
        for key, value in runs["ps-2"].items():
            np.testing.assert_array_equal(runs["ring-2"][key], value, err_msg=key)

    def test_stop_file_stops_every_worker_and_resume_continues_mid_epoch(self):
        (self.work / "resume.STOP").write_text("stop", encoding="utf-8")
        out, results = self._passed("resume", "--distributed", "ring", "--workers", "2")
        self.assertTrue(results["stopped_by_file"])
        self.assertEqual(results["steps"], 1)
        meta = json.loads((out / "ckpt.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["epoch"], meta["training"]["next_step"], meta["training"]["epoch_complete"]),
                         (0, 1, False))
        self.assertFalse((self.work / "resume.STOP").exists())
        out, results = self._passed("resume", "--distributed", "ps", "--workers", "3", "--resume")
        self.assertEqual(results["steps"], 2 * STEPS_PER_EPOCH - 1)
        meta = json.loads((out / "ckpt.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["training"]["global_step"], 2 * STEPS_PER_EPOCH)


if __name__ == "__main__":
    unittest.main()
