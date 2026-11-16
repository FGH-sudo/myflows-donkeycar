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


class OnnxEvaluateDonkeyTest(unittest.TestCase):
    def test_exported_input_dimensions_override_camera_size_and_requested_batch(self):
        import cv2
        import numpy as np
        import onnx
        from onnx import TensorProto, helper

        for batch in (1, 2):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                (work / "images").mkdir()
                for i in range(3):
                    cv2.imwrite(str(work / "images" / f"{i}_0.2000.jpg"), np.zeros((16, 32, 3), np.uint8))
                graph = helper.make_graph([
                    helper.make_node("ReduceMean", ["image"], ["mean"], axes=[1, 2, 3], keepdims=0),
                    helper.make_node("Unsqueeze", ["mean", "axis"], ["column"]),
                    helper.make_node("Concat", ["column", "column"], ["prediction"], axis=1),
                ], "small-input", [helper.make_tensor_value_info("image", TensorProto.FLOAT, [batch, 3, 4, 6])],
                    [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [batch, 2])],
                    [helper.make_tensor("axis", TensorProto.INT64, [1], [1])])
                model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
                onnx.save(model, work / "model.onnx")
                done = subprocess.run([
                    sys.executable, "-X", "utf8", "-m", "apps.eval.eval_myflows_donkey_onnx", "--data", str(work),
                    "--checkpoint", str(work / "model.onnx"), "--device", "cpu", "--batch", "8",
                    "--report-json", str(work / "score.json"),
                ], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
                self.assertEqual(done.returncode, 0, done.stdout[-2000:] + done.stderr[-2000:])
                score = json.loads((work / "score.json").read_text(encoding="utf-8"))["results"][0]
                self.assertEqual((score["samples"], score["batch"], score["input_w"], score["input_h"]), (3, batch, 6, 4))
                self.assertAlmostEqual(score["angle_mae"], .2)


if __name__ == "__main__":
    unittest.main()
