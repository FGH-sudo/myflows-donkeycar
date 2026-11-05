import tempfile
from pathlib import Path
import unittest

import numpy as np

from benchmark.compare_frameworks_resnet import alignment, orders, summarize_resources


class FrameworkProtocolTest(unittest.TestCase):
    def test_common_batch_lists_drop_the_same_tail_and_never_use_validation(self):
        training, timing = orders([0, 2, 3, 5, 6, 8, 9], 3, 2, 7, 6)
        self.assertEqual(training.shape, (2, 2, 3))
        self.assertEqual(timing.shape, (6, 3))
        for epoch in training:
            self.assertEqual(len(set(epoch.ravel())), 6)
            self.assertTrue(set(epoch.ravel()).issubset({0, 2, 3, 5, 6, 8, 9}))
        np.testing.assert_array_equal(orders([0, 2, 3, 5, 6, 8, 9], 3, 2, 7, 6)[0], training)
        np.testing.assert_array_equal(timing[:2], training[0])

    def test_alignment_can_fail_on_gradients_despite_identical_prediction_and_step(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = (Path(temp) / name for name in ("a.npz", "b.npz"))
            tensors = {"logits": np.array([[1., 2.]]), "loss": np.array(1.),
                       "grad/w": np.array([1., 2.]), "updated/w": np.ones(2)}
            np.savez(a, **tensors)
            np.savez(b, **tensors)
            self.assertTrue(alignment(a, b)["passed"])
            np.savez(b, **(tensors | {"grad/w": np.array([2., 1.])}))
            self.assertFalse(alignment(a, b)["passed"])
            tensors.pop("grad/w")
            np.savez(a, **tensors)
            with self.assertRaises(ValueError):
                alignment(a, b)

    def test_peak_rss_is_over_the_measurement_window(self):
        rows = [dict(monotonic_s=t, process_rss_total_bytes=rss, gpu_utilization_pct=util,
                     gpu_device_memory_mib=300, cpu_utilization_pct=5)
                for t, rss, util in [(0, 900, 0), (1, 100, 70), (2, 200, 90), (3, 150, 80)]]
        report = summarize_resources(rows, 1, 3)
        self.assertEqual((report["samples"], report["rss_bytes_sampled_peak"], report["gpu_util_pct_median"]), (3, 200, 80))
