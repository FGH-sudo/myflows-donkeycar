import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import scripts.run_quantize_eval as quant_eval


def _write_image(path: Path) -> None:
    cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))


class QuantizeEvalRegressionTest(unittest.TestCase):
    def test_load_rows_reuses_split_and_forces_fixed_throttle(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            images_dir = data_dir / "images"
            images_dir.mkdir()
            records = []
            for idx, angle in enumerate([0.1, -0.2, 0.0, 0.3]):
                name = f"{idx}_{angle:.4f}.jpg"
                _write_image(images_dir / name)
                records.append(
                    {
                        "cam/image_array": name,
                        "user/angle": angle,
                        "user/throttle": 0.5,
                    }
                )
            (data_dir / "catalog_generated.catalog").write_text(
                "\n".join(json.dumps(row) for row in records) + "\n",
                encoding="utf-8",
            )
            split_file = data_dir / "split.json"
            split_file.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "source_count": 4,
                        "seed": 42,
                        "splits": {"train": [0], "val": [2], "test": [1, 3]},
                    }
                ),
                encoding="utf-8",
            )

            rows = quant_eval._load_rows(
                data_dir,
                max_samples=0,
                catalog="catalog_generated.catalog",
                split_file=split_file,
                split="test",
                fixed_throttle=0.2,
                angle_scale=1.0,
                force_fixed_throttle=True,
            )

        self.assertEqual([p.name for p, _, _ in rows], ["1_-0.2000.jpg", "3_0.3000.jpg"])
        self.assertEqual([throttle for _, _, throttle in rows], [0.2, 0.2])

    def test_eval_regression_reports_angle_throttle_and_overall_mse(self):
        class FakeSession:
            def __init__(self):
                self.outputs = [
                    np.asarray([[0.2, 0.25]], dtype=np.float32),
                    np.asarray([[-0.1, 0.1]], dtype=np.float32),
                ]

            def run(self, _outputs, _feed):
                return [self.outputs.pop(0)]

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            p0 = data_dir / "a.jpg"
            p1 = data_dir / "b.jpg"
            _write_image(p0)
            _write_image(p1)
            rows = [(p0, 0.1, 0.2), (p1, -0.3, 0.2)]

            metrics = quant_eval._eval_regression(FakeSession(), rows, 8, 8, "input")

        self.assertAlmostEqual(metrics["angle_mse"], 0.025)
        self.assertAlmostEqual(metrics["throttle_mse"], 0.00625)
        self.assertAlmostEqual(metrics["overall_mse"], 0.015625)
        self.assertEqual(metrics["n"], 2)
        self.assertNotIn("mse", metrics)


if __name__ == "__main__":
    unittest.main()
