"""New-format validation and real historical run compatibility across consumers."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from apps.autopilot.evidence import summarize_attempt
from apps.autopilot.verifier import OnlineMonitor
from apps.console.readers import JsonlTail
from MyFlows.monitoring.schema import record, schema_version
from tests.test_console_backend import make_registry
from tools.check_run_format import check_run, validate_record

FIXTURE = Path(__file__).parent / "fixtures" / "monitoring_v0"


class MonitoringFormatTest(unittest.TestCase):
    def test_real_legacy_run_still_feeds_console_and_autopilot(self):
        result = check_run(FIXTURE)
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(result["warnings"])
        rows = JsonlTail(FIXTURE / "metrics.jsonl").poll()
        self.assertEqual(schema_version(rows[0]), 0)
        summary = summarize_attempt(FIXTURE)
        self.assertEqual(summary["steps"], 2)
        self.assertGreater(summary["throughput_sps"], 0)
        self.assertGreater(summary["gpu_util_median"], 0)
        monitor = OnlineMonitor(FIXTURE, {"hang_timeout_s": 1e9, "window_steps": 10}, 1e9,
                                clock=lambda: rows[1]["unix_s"])
        self.assertIsNone(monitor.check())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "console" / "legacy"
            shutil.copytree(FIXTURE, dest)
            job = {"id": "legacy", "type": "single", "label": "fixture", "status": "succeeded", "run_dir": str(dest)}
            registry = make_registry(root, jobs=[job])
            ref = registry.resolve("console/legacy")
            self.assertEqual(registry.steps(ref)["ranks"]["0"]["loss"], [row["loss"] for row in rows[:2]])
            self.assertEqual(registry.detail(ref)["epochs"][0]["loss"], rows[2]["loss"])

    def test_unknown_versions_and_invalid_numbers_fail(self):
        self.assertTrue(validate_record({"schema_version": 2}, "config"))
        self.assertTrue(validate_record({"schema_version": True}, "config"))
        row = record({"kind": "step", "step": 1, "loss": float("nan")}, clocks=True)
        self.assertTrue(any("loss" in e for e in validate_record(row, "metrics")))
        row = record({"sampling_status": "unavailable", "gpus": [], "interval_s": 0.5}, clocks=True)
        errors = validate_record(row, "resources")
        self.assertTrue(any("null requires a reason" in error for error in errors))

    def test_partial_jsonl_is_detected_and_v1_result_requires_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(record({"kind": "single"})), encoding="utf-8")
            (root / "metrics.jsonl").write_text('{"schema_version": 1', encoding="utf-8")
            (root / "result.json").write_text(json.dumps(record({"status": "failed", "exit_code": 1})), encoding="utf-8")
            errors = check_run(root)["errors"]
            self.assertTrue(any("incomplete JSONL" in error for error in errors))
            self.assertTrue(any("environment" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
