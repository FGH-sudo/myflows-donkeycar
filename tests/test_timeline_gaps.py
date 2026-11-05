"""Synthetic Nsight sqlite: stream overlap, idle fragments, phase denominators and source attribution."""

import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from tools.timeline_gaps import analyze, write_outputs


class TimelineGapsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "capture.sqlite"
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executescript("""
                CREATE TABLE StringIds (id INTEGER, value TEXT);
                CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, shortName INTEGER, deviceId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, deviceId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER, end INTEGER, deviceId INTEGER);
                INSERT INTO StringIds VALUES (1, 'conv_kernel'), (2, 'node:forward:1:conv');
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (250000, 350000, 1, 0), (0, 1000000, 1, 1);
                INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (300000, 450000, 0);
                INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (800000, 850000, 0);
                INSERT INTO NVTX_EVENTS VALUES (0, 1000000, 'step 1', NULL, 99),
                    (0, 200000, 'data', NULL, 99), (200000, 700000, 'forward', NULL, 99),
                    (700000, 900000, 'backward', NULL, 99), (900000, 1000000, 'optimizer', NULL, 99),
                    (200000, 600000, NULL, 2, 99);
            """)
        self.mapping = {"ranges": {name: {"file": "module.py", "line": index + 1} for index, name in enumerate(
            ("step 1", "data", "forward", "backward", "optimizer", "node:forward:1:conv"))}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_device_busy_is_union_and_all_phases_share_step_denominator(self):
        report = analyze(self.path, self.mapping, threshold_ms=.1)
        self.assertEqual(report["step_wall_ms"], 1)
        self.assertAlmostEqual(sum(r["wall_share_of_steps"] for r in report["phases"]), 1)
        self.assertAlmostEqual(sum(r["gpu_idle_ms"] for r in report["phases"]), .75)
        self.assertAlmostEqual(sum(r["duration_ms"] for r in report["gaps"]), .75)
        self.assertEqual(report["steps"][0]["kernel_count"], 1)
        self.assertEqual(report["steps"][0]["kernel_mean_us"], 100)
        self.assertEqual(report["unmapped_gaps"], 0)
        self.assertTrue(any(r["node"] == "node:forward:1:conv" for r in report["gaps"]))
        self.assertTrue(any(r["duration_ms"] < .1 for r in report["gaps"]))
        write_outputs(report, self.root / "analysis", plot=False)
        saved = json.loads((self.root / "analysis" / "timeline.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["step_wall_ms"], 1)
        self.assertNotIn("activities", saved)

    def test_threshold_filters_table_not_phase_accounting(self):
        report = analyze(self.path, self.mapping, threshold_ms=2)
        self.assertEqual(report["gaps"], [])
        self.assertAlmostEqual(report["steps"][0]["gpu_idle_ms"], .75)

    def test_missing_source_is_reported_and_multiple_workers_require_selection(self):
        self.assertGreater(analyze(self.path)["unmapped_gaps"], 0)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("INSERT INTO NVTX_EVENTS VALUES (0, 1000000, 'step 1', NULL, 100)")
            connection.commit()
        with self.assertRaisesRegex(ValueError, "multiple training threads"):
            analyze(self.path)
        self.assertEqual(analyze(self.path, thread=99)["thread"], 99)


if __name__ == "__main__":
    unittest.main()
