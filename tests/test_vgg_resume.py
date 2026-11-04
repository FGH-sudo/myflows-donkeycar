"""apps.train.train_vgg_donkey_regression --resume: the best model is compared with its own score."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _tub(directory: Path, count: int = 6) -> Path:
    """Tiny tub without catalog; VGG pools five times, so the images are 32×32."""
    images = directory / "tub" / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(count):
        angle = float(rng.uniform(-1.0, 1.0))
        cv2.imwrite(str(images / f"{index}_{angle:.4f}.jpg"), rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    return directory / "tub"


def _train(work: Path) -> subprocess.CompletedProcess:
    """One more epoch (``--epochs`` counts the epochs this run adds) on 4 training and 2 validation images."""
    command = [sys.executable, "-X", "utf8", "-m", "apps.train.train_vgg_donkey_regression", "--data",
               str(work / "tub"), "--catalog", "missing.catalog", "--device", "cpu", "--batch", "2",
               "--val-size", "2", "--epochs", "1", "--no-tensorboard", "--log-dir", str(work / "logs"),
               "--out", str(work / "ckpt"), "--best-out", str(work / "best"), "--resume"]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=600, check=False)


def _set_score(path: Path, value: float) -> None:
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(score_value=value, best_loss=value)
    path.write_text(json.dumps(meta), encoding="utf-8")


class VggResumeTest(unittest.TestCase):
    def test_resume_compares_with_the_best_models_own_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _tub(work)
            # Nothing to resume yet: training starts from scratch and still saves its best model
            result = _train(work)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
            self.assertTrue((work / "best.json").exists(), result.stdout[-3000:])
            # The training checkpoint records the training loss, the best model its validation loss;
            # only the latter decides whether epoch 2 replaces the best model.
            _set_score(work / "ckpt.json", 1e-12)
            _set_score(work / "best.json", 1e9)
            result = _train(work)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
            best = json.loads((work / "best.json").read_text(encoding="utf-8"))
            self.assertEqual(best["epoch"], 1)
            self.assertLess(best["score_value"], 1e9)


if __name__ == "__main__":
    unittest.main()
