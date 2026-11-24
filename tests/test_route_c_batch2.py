# -*- coding: utf-8 -*-
"""Route C Batch 2 parent tests: benchmark._common, --sync-bn PS training, AutoPilot closed-loop wiring."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from apps.autopilot.dsh.agent_tools import AgentTools
from apps.autopilot.service import AutoPilotService
from apps.autopilot.spec import SpecError, validate_spec
from apps.console.jobs import JobManager
from apps.train import train_myflows_donkey as train_mod
from benchmark import _common
from tests.system.harness import Harness, ROOT


class BenchmarkCommonTest(unittest.TestCase):
    def test_common_helpers_and_refactored_benchmark_aliases(self):
        stats = _common.summarize_ms([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["median_ms"], 3.0)
        self.assertEqual(stats["p95_ms"], 5.0)
        with self.assertRaises(ValueError):
            _common.summarize_ms([])
        with self.assertRaises(ValueError):
            _common.summarize_ms([1.0, float("nan")])

        p = argparse.ArgumentParser()
        _common.add_common_worker_args(p, default_warmup=3, default_steps=9)
        ns = p.parse_args([])
        self.assertEqual((ns.tree, ns.warmup, ns.steps), (_common.ROOT, 3, 9))

        pool = _common.pool_stats_mib()
        self.assertEqual(set(pool.keys()), {"used_mib", "total_mib"})

        for mod_name in (
            "benchmark.resnet_before_after",
            "benchmark.ops_before_after",
            "benchmark.memory_study",
            "benchmark.adam_study",
            "benchmark.perf_gate",
        ):
            mod = importlib.import_module(mod_name)
            self.assertEqual(mod.ROOT, _common.ROOT)

        import benchmark.resnet_before_after as rba
        import benchmark.memory_study as ms_study
        import benchmark.adam_study as adam_study
        import benchmark.perf_gate as pg

        self.assertIs(rba._import, _common.import_tree)
        self.assertIs(rba._build, _common.build_resnet_workload)
        self.assertIs(ms_study._build, _common.build_resnet_workload)
        self.assertIs(adam_study._build, _common.build_resnet_workload)
        self.assertIs(pg._import, _common.import_tree)

        with mock.patch("subprocess.run", return_value=subprocess.CompletedProcess(["x"], 0, stdout='noise\n{"ok": 42}\n', stderr="")):
            self.assertEqual(_common.run_worker_json("benchmark.fake", ["--worker"]), {"ok": 42})


class TrainDonkeySyncBnTest(unittest.TestCase):
    def test_sync_bn_default_preserves_fingerprint_and_runs_ps_training(self):
        base_args = train_mod.build_parser().parse_args([])
        sync_args = train_mod.build_parser().parse_args(["--sync-bn"])
        self.assertEqual(train_mod.trajectory_fingerprint(base_args),
                         train_mod.trajectory_fingerprint(train_mod.build_parser().parse_args([])))
        self.assertNotEqual(train_mod.trajectory_fingerprint(base_args),
                            train_mod.trajectory_fingerprint(sync_args))

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            images = work / "tub" / "images"
            images.mkdir(parents=True)
            rng = np.random.default_rng(11)
            for idx in range(16):
                angle = -0.6 + 1.2 * (idx / 15.0)
                img = np.full((24, 32, 3), 40, dtype=np.uint8)
                cx = int(np.clip(16 + angle * 8, 2, 28))
                img[:, cx - 2 : cx + 2] = 210
                cv2.imwrite(str(images / f"{idx}_{angle:.4f}.jpg"), img)

            out = work / "ps_syncbn"
            proc = subprocess.run(
                [
                    sys.executable, "-X", "utf8", "-m", "apps.train.train_myflows_donkey",
                    "--data", str(work / "tub"), "--catalog", "missing.catalog",
                    "--device", "cpu", "--base-width", "4", "--batch", "4", "--epochs", "1",
                    "--val-size", "4", "--seed", "5", "--no-tensorboard", "--checkpoint-every", "0",
                    "--distributed", "ps", "--workers", "2", "--transport", "socket_json",
                    "--sync-bn", "--log-dir", str(out), "--out", str(out / "ckpt"),
                    "--best-out", str(out / "best"), "--stop-file", str(work / "STOP"),
                    "--run-dir", str(out / "run"), "--sync-timeout", "90",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
            logs = "\n".join([proc.stdout, *(p.read_text(encoding="utf-8", errors="replace") for p in out.rglob("*.log"))])
            self.assertIn("SyncBatchNorm2d 已启用", logs)
            res = json.loads((out / "run" / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(res["status"], "passed")
            self.assertTrue((out / "best.json").is_file())
            self.assertTrue((out / "best.npz").is_file())


class AutoPilotClosedLoopTest(unittest.TestCase):
    def test_validate_spec_closed_loop_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "mycar" / "data" / "images").mkdir(parents=True)
            (repo / "mycar" / "data" / "images" / "0_0.0000.jpg").write_bytes(b"0")
            spec = validate_spec({"closed_loop": {"enabled": True, "sim": "fake", "speed_mph": 10, "seconds": 15}}, repo)
            self.assertTrue(spec["closed_loop"]["enabled"])
            self.assertEqual(spec["closed_loop"]["seconds"], 15.0)
            for bad in (
                {"closed_loop": {"enabled": "yes"}},
                {"closed_loop": {"track": "moon"}},
                {"closed_loop": {"speed_mph": -1}},
                {"closed_loop": {"device": "tpu"}},
                {"closed_loop": {"max_models": 99}},
            ):
                with self.assertRaises(SpecError):
                    validate_spec(bad, repo)

    def test_campaign_runs_closed_loop_wires_llm_advisor_reporter_and_recovery(self):
        captured: dict = {}

        class _CapturingAdvisor:
            def consult(self, service: AutoPilotService, store, reason: str):
                tools = AgentTools(service)
                info = tools.call("get_campaign", {"campaign_id": store.campaign_id}, source="test_advisor")
                captured.setdefault("reasons", []).append(reason)
                if reason == "report":
                    captured["report_campaign_view"] = info
                    cl = info.get("closed_loop_evaluation") or {}
                    tools.call(
                        "write_note",
                        {
                            "campaign_id": store.campaign_id,
                            "title": "LLM 解读",
                            "text": f"闭环验证组数={len((cl.get('groups') or {}))}",
                        },
                        source="test_advisor",
                    )

        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(
                Path(tmp),
                scenario={"write_fake_onnx": True, "step_s": 0.005},
                start=False,
                drive_module="tests.system.fixtures.fake_drive",
            )
            try:
                h.jobs = JobManager(h.tmp / "console", h.repo, python=sys.executable, poll_interval_s=0.05).start()
                h.service = AutoPilotService(
                    h.tmp / "autopilot",
                    jobs=h.jobs,
                    repo_root=h.repo,
                    python=sys.executable,
                    module="tests.system.fixtures.fake_train",
                    poll_s=0.05,
                    advisor=_CapturingAdvisor(),
                    record_environment=False,
                    eval_module="tests.system.fixtures.fake_evaluate",
                    drive_module="tests.system.fixtures.fake_drive",
                ).start()

                done = h.run(
                    timeout=120.0,
                    campaign_id="cl-campaign-01",
                    fixed={"max_samples": 40, "epochs": 1, "val_size": 8, "test_size": 8,
                           "split_kind": "time_block", "block_size": 4},
                    space={"lr": [0.001, 0.0003], "batch": [1]},
                    strategy={"max_candidates": 2, "seed": 0},
                    budget={"max_trials": 6},
                    llm={"enabled": True},
                    closed_loop={"enabled": True, "sim": "fake", "track": "lake", "speed_mph": 12.0, "seconds": 25.0},
                )
                self.assertEqual(done["status"], "completed")
                cl_eval = done["state"].get("closed_loop_evaluation")
                self.assertIsNotNone(cl_eval)
                self.assertEqual(set(cl_eval["groups"].keys()), {"best", "baseline"})
                for grp in ("best", "baseline"):
                    item = cl_eval["groups"][grp]
                    self.assertTrue(item["passed"])
                    self.assertEqual(item["excursions"], 0)
                    self.assertTrue(Path(item["summary_path"]).is_file())

                self.assertIn("report", captured.get("reasons", []))
                self.assertEqual(
                    captured["report_campaign_view"]["closed_loop_evaluation"]["groups"]["best"]["mean_abs_cte"],
                    0.14,
                )

                report = h.service.report("cl-campaign-01")["markdown"]
                self.assertIn("### 闭环仿真验证（Closed-Loop）", report)
                self.assertIn("闭环仿真验证（最佳候选", report)
                self.assertIn("闭环验证组数=2", report)

                # Restart with existing closed_loop_evaluation must not re-run fake_drive
                started_before = [e for e in h.service.events("cl-campaign-01", kinds=("closed_loop_eval",)) if e["status"] == "started"]
                self.assertEqual(len(started_before), 2)
                h.service.store("cl-campaign-01").update_state(
                    lambda s: s.update(status="finishing", finishing_as="completed", report=None)
                )
                h.service.shutdown()
                h.jobs.shutdown()
                h.jobs = JobManager(h.tmp / "console", h.repo, python=sys.executable, poll_interval_s=0.05).start()
                h.service = AutoPilotService(
                    h.tmp / "autopilot",
                    jobs=h.jobs,
                    repo_root=h.repo,
                    python=sys.executable,
                    module="tests.system.fixtures.fake_train",
                    poll_s=0.05,
                    record_environment=False,
                    eval_module="tests.system.fixtures.fake_evaluate",
                    drive_module="tests.system.fixtures.fake_drive",
                ).start()
                recovered = h.wait("cl-campaign-01")
                self.assertEqual(recovered["status"], "completed")
                started_after = [e for e in h.service.events("cl-campaign-01", kinds=("closed_loop_eval",)) if e["status"] == "started"]
                self.assertEqual(len(started_after), 2)
            finally:
                h.close()

    def test_real_udacity_drive_fake_sim_smoke_through_service(self):
        import MyFlows as ms
        from MyFlows.core.device import set_device
        from MyFlows.core.dtype import set_default_dtype
        from apps.train.common.regression_model import build_regression_resnet
        from tools.export_resnet_onnx import export_resnet_onnx

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            ckpt_dir = work / "trial_run" / "checkpoints"
            ckpt_dir.mkdir(parents=True)
            ckpt_stem = ckpt_dir / "best"
            try:
                set_device("cpu")
                set_default_dtype(np.float32)
                x = ms.Variable(np.zeros((1, 3, 48, 80), dtype=np.float32), name="X", requires_grad=False)
                model = build_regression_resnet(
                    "resnet18", "global", 48, 80, in_channels=3, output_dim=2, stem="cifar", base_width=4
                )
                model.eval()
                ms.Graph(model(x)).forward()
                ms.save_checkpoint([model], None, 0, filepath=str(ckpt_stem), loss=0.1)
            finally:
                set_device("cpu")
                set_default_dtype(np.float64)

            onnx_path = export_resnet_onnx(
                ckpt_stem, batch_size=1, image_w=80, image_h=48, base_width=4,
                device="cpu", roi_crop="16,8", dynamic_batch=True,
            )
            self.assertTrue(onnx_path.is_file())

            h = Harness(work, start=True)
            try:
                done = h.service._evaluate_closed_loop  # sanity verify service wires real apps.sim.udacity_drive cleanly
                self.assertTrue(callable(done))
                spec = validate_spec({
                    "campaign_id": "real-fake-sim-01",
                    "fixed": {"max_samples": 40, "epochs": 1, "val_size": 8},
                    "llm": {"enabled": False},
                    "closed_loop": {"enabled": True, "sim": "fake", "track": "lake", "speed_mph": 12.0, "seconds": 1.5, "device": "cpu"},
                }, h.repo)
                created = h.service.create_campaign(spec)
                store = h.service.store(created["campaign_id"])
                trial = {
                    "trial_id": "t001",
                    "phase": "baseline",
                    "status": "done",
                    "params": {"lr": 0.001, "batch": 2, "num_workers": 0, "seed": 1},
                    "origin": {"strategy": "baseline", "delta": {}},
                    "verdict": {"status": "valid", "reasons": []},
                    "result": {"metrics": {"val_angle_mae": 0.12}, "summary": {"throughput_sps": 10.0}},
                    "attempts": [{"attempt": 1, "run_dir": str(work / "trial_run"), "status": "succeeded"}],
                }
                store.save_trial(trial)
                doc = store.load()
                ev = h.service._evidence(store, doc, [trial])
                ok = h.service._evaluate_closed_loop(store, ev)
                self.assertTrue(ok)
                cl = store.load()["state"]["closed_loop_evaluation"]
                self.assertIsNotNone(cl)
                self.assertIn("baseline", cl["groups"])
                item = cl["groups"]["baseline"]
                self.assertEqual(item["environment"], "udacity-fake-stadium")
                self.assertEqual(item["termination"], "time_limit")
                self.assertTrue(Path(item["summary_path"]).is_file())
            finally:
                h.close()


if __name__ == "__main__":
    unittest.main()
