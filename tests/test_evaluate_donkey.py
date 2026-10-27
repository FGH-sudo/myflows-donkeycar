"""apps.train.evaluate_donkey: the held-out evaluation reproduces training's own validation numbers."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from apps.common.donkey_data import load_donkey_index
from apps.common.splits import build_split, save_split
from tests.test_train_distributed import _tub
from tests.test_train_outcome import _train

ROOT = Path(__file__).resolve().parents[1]


class EvaluateDonkeyTest(unittest.TestCase):
    def test_best_checkpoint_scores_like_training_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _tub(work)
            index = load_donkey_index(work / "tub", fixed_throttle=0.5, angle_scale=1.0, catalog_name="missing.catalog")
            split_file = work / "split.json"
            save_split(split_file, build_split(index, val_size=5, test_size=5, seed=3), source_count=len(index), seed=3)
            result, out = _train(work, "run", "--seed", "2", "--split-file", str(split_file))
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
            trained = json.loads((out / "run" / "result.json").read_text(encoding="utf-8"))
            args = {"data": str(work / "tub"), "catalog": "missing.catalog", "device": "cpu", "base_width": 4,
                    "batch": 6, "seed": 2, "split_file": str(split_file)}
            items = work / "items.json"
            items.write_text(json.dumps({"split": "val", "items": [
                {"key": "best", "checkpoint": str(out / "best"), "args": args}]}), encoding="utf-8")
            done = subprocess.run([sys.executable, "-X", "utf8", "-m", "apps.train.evaluate_donkey", "--items", str(items),
                                   "--out", str(work / "eval.json"), "--run-dir", str(work / "eval")],
                                  cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
            self.assertEqual(done.returncode, 0, done.stdout[-3000:] + done.stderr[-3000:])
            scored = json.loads((work / "eval.json").read_text(encoding="utf-8"))["results"][0]
            self.assertEqual(scored["samples"], 5)
            self.assertEqual(scored["angle_mae"], trained["metrics"]["val_angle_mae"])
            self.assertTrue((work / "eval" / "gpu_lock.jsonl").exists())
            items.write_text(json.dumps({"split": "test", "items": [
                {"key": "best", "checkpoint": str(out / "best"), "args": args}]}), encoding="utf-8")
            done = subprocess.run([sys.executable, "-X", "utf8", "-m", "apps.train.evaluate_donkey", "--items", str(items),
                                   "--out", str(work / "test.json")], cwd=ROOT, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=600)
            self.assertEqual(done.returncode, 0, done.stderr[-3000:])
            self.assertEqual(json.loads((work / "test.json").read_text(encoding="utf-8"))["results"][0]["samples"], 5)


if __name__ == "__main__":
    unittest.main()
