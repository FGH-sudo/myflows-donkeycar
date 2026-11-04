"""Benchmark accounting must use the same samples/time and a steady-state power window."""

import json
from pathlib import Path
import tempfile
import unittest

from benchmark.monitor_overhead import aggregate, summarize


class MonitorOverheadTest(unittest.TestCase):
    def test_warmup_and_power_window_do_not_bias_steady_throughput(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(kind="step", step_time_ms=1000, data_load_ms=1000, batch_size=2, monotonic_s=1),
                    dict(kind="step", step_time_ms=40, data_load_ms=10, batch_size=2, monotonic_s=2),
                    dict(kind="step", step_time_ms=30, data_load_ms=20, batch_size=2, monotonic_s=3)]
            (root / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            power = [dict(monotonic_s=t, gpu_power_w=p, gpu_power_limit_w=50) for t, p in ((1, 0), (2, 49), (3, 25), (4, 0))]
            (root / "power-reference.jsonl").write_text("".join(json.dumps(r) + "\n" for r in power), encoding="utf-8")
            result = summarize(root, 1)
            self.assertEqual((result["step_ms_median"], result["samples_per_s"]), (50, 40))
            self.assertEqual((result["power_samples"], result["power_capped_share"]), (2, .5))

    def test_overhead_is_paired_by_repeat_not_ratio_of_unpaired_samples(self):
        rows = [dict(mode=mode, repeat=i, step_ms_median=ms, samples_per_s=2e3 / ms)
                for i, pair in enumerate(((10, 11), (100, 110), (20, 22)))
                for mode, ms in zip(("off", "500ms"), pair)]
        result = aggregate(rows)["500ms"]
        self.assertAlmostEqual(result["overhead_pct_median"], 10)
        self.assertEqual(result["repeats"], 3)


if __name__ == "__main__":
    unittest.main()
