import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from benchmark.compare_frameworks import load_donkey_regression_subset
import benchmark.compare_frameworks as compare_frameworks
import benchmark.plot_compare as plot_compare


class CompareFrameworksRegressionDataTest(unittest.TestCase):
    def test_loads_donkey_images_and_regression_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            images_dir = data_dir / "images"
            images_dir.mkdir()
            for idx, angle in enumerate((-0.25, 0.5)):
                img = np.full((8, 8, 3), 32 + idx * 80, dtype=np.uint8)
                cv2.imwrite(str(images_dir / f"{idx}_{angle:.4f}.jpg"), img)

            records = [
                {"cam/image_array": "images/0_-0.2500.jpg", "user/angle": -0.25, "user/throttle": 0.4},
                {"cam/image_array": "images/1_0.5000.jpg", "user/angle": 0.5, "user/throttle": 0.6},
            ]
            with (data_dir / "catalog_generated.catalog").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            x, y = load_donkey_regression_subset(
                data_dir,
                catalog="catalog_generated.catalog",
                samples=2,
                image_h=8,
                image_w=8,
                fixed_throttle=0.5,
                angle_scale=1.0,
                sample_seed=None,
            )

        self.assertEqual(x.shape, (2, 3, 8, 8))
        self.assertEqual(y.shape, (2, 2))
        np.testing.assert_allclose(y, np.array([[-0.25, 0.4], [0.5, 0.6]], dtype=np.float64))
        self.assertTrue(np.all(np.isfinite(x)))

    def test_benchmark_runners_include_myflows_pytorch_and_paddle_only(self):
        names = [runner.__name__ for runner in compare_frameworks.BENCHMARK_RUNNERS]
        self.assertEqual(names, ["bench_myflows", "bench_pytorch", "bench_paddle"])
        source = Path(compare_frameworks.__file__).read_text(encoding="utf-8")
        for forbidden in ("Tensor" + "Flow", "tensor" + "flow"):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("flo" + "ps_est", source)

    def test_plot_metrics_are_time_and_rss_only(self):
        metric_columns = [panel["column"] for panel in plot_compare.METRIC_PANELS]
        self.assertEqual(metric_columns, ["time_s", "peak_mb"])


if __name__ == "__main__":
    unittest.main()
