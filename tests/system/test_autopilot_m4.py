"""M4 specialists: Operator Optimization (lecture page-9 table, kernel hot spots, conv-backend recheck) and
PS Training (data-parallel matrix, sync share, stragglers, keep-single-process conclusion)."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from apps.autopilot.agents.operator_opt import OperatorOptimization, kernel_hotspots, lecture_table
from apps.autopilot.agents.ps_training import PsTraining
from apps.autopilot.evidence import Evidence, summarize_ranks
from apps.autopilot.spec import default_spec
from tests.system.harness import ROOT, Harness


def write_matmul(root: Path) -> Path:
    speed = {"numpy": 1.0, "cupy": 8.0, "basic-fp32": 0.6, "tiled16-fp32": 0.9, "shared16-fp32": 1.2,
             "register-fp32": 10.0}
    rows = []
    for n in (1024, 2048, 4096):
        base = n ** 3 / 1e9  # ms of NumPy at this size, any monotone value will do
        for name, factor in speed.items():
            rows.append({"size": n, "implementation": name, "stable_mean_ms": base / factor, "median_ms": base / factor,
                         "max_abs_vs_numpy": 0.0 if name == "numpy" else 2e-4, "rel_vs_numpy": 0.0 if name == "numpy" else 3e-7,
                         "rel_rms": 1e-6, "gflops": 100 * factor})
    ncu = [{"implementation": name, "kernel": f"k_{name}", "gpu__time_duration.sum": 1e6 / speed[name],
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 10 * speed[name],
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": 0}
           for name in ("basic-fp32", "shared16-fp32", "register-fp32")]
    path = root / "runs" / "benchmarks" / "matmul-001" / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows, "ncu": ncu, "status": "passed"}), encoding="utf-8")
    return path


def write_capture(root: Path) -> Path:
    path = root / "runs" / "benchmarks" / "nsys-001" / "capture.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, shortName INT, demangledName INT)")
    con.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "gemm_register"), (2, "bn_stats"), (3, "relu")])
    rows = [(0, 700, 1, 1)] * 3 + [(0, 200, 2, 2)] * 2 + [(0, 50, 3, 3)]
    con.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)", rows)
    con.commit()
    con.close()
    return path


class OperatorOptimizationUnitTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def evidence(self):
        from apps.autopilot.spec import default_spec

        return Evidence(spec=default_spec("dev"), state={}, trials=[])

    def test_tables_hot_spots_and_advice(self):
        write_matmul(self.root)
        write_capture(self.root)
        agent = OperatorOptimization(repo_root=self.root)
        findings = {f.kind: f for f in agent.diagnose(self.evidence())}
        table = findings["operator_table"].data["rows"]
        self.assertEqual(len(table), 18)
        at = {r["implementation"]: r for r in table if r["size"] == 4096}
        self.assertAlmostEqual(at["cupy"]["speedup"], 8.0)
        self.assertIn("4096", findings["operator_table"].message)
        self.assertIn("寄存器分块", findings["memory_bound_kernels"].message)
        hot = findings["kernel_hotspots"].data
        self.assertEqual([k["name"] for k in hot["kernels"]], ["gemm_register", "bn_stats", "relu"])
        self.assertAlmostEqual(hot["kernels"][0]["share"], 2100 / 2550)
        section = agent.report_section(self.evidence(), [f.to_dict() for f in findings.values()])
        self.assertIn("课件第 9 页", section)
        self.assertIn("| 4096 | CuPy |", section)
        self.assertIn("gpu__time_duration.sum", section)
        self.assertIn("gemm_register", section)

    def test_missing_files_and_real_results(self):
        agent = OperatorOptimization(repo_root=self.root)
        self.assertEqual(agent.available(None), (True, None))
        self.assertEqual(agent.diagnose(self.evidence()), [])
        self.assertIn("matmul_compare", agent.report_section(self.evidence(), []))
        real = ROOT / "docs" / "experiments" / "semester_2026_fall" / "cuda_lecture19" / "matmul-001" / "results.json"
        if real.exists():
            rows = lecture_table(json.loads(real.read_text(encoding="utf-8")))
            self.assertEqual({r["size"] for r in rows}, {1024, 2048, 4096})
            self.assertTrue(all(r["speedup"] for r in rows))
        self.assertIsNone(kernel_hotspots(self.root / "missing.sqlite"))


class RankSummaryTest(unittest.TestCase):
    def test_sync_share_throughput_and_straggler(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for rank, compute in ((0, 0.010), (1, 0.030)):
                (run / f"rank-{rank}").mkdir()
                with (run / f"rank-{rank}" / "steps.jsonl").open("w") as fh:
                    for step in range(10):
                        wait = 0.030 - compute + 0.005
                        fh.write(json.dumps({"rank": rank, "step": step, "step_wall_s": 0.001 + compute + wait,
                                             "data_s": 0.001, "gradient_sync_s": wait, "update_confirm_s": 0.0,
                                             "n_samples": 2}) + "\n")
            summary = summarize_ranks(run)
            self.assertEqual(summary["workers"], 2)
            self.assertAlmostEqual(summary["throughput_sps"], 2 * 2 / 0.036, places=3)
            self.assertAlmostEqual(summary["rank_step_cv"], 0.010 / 0.021, places=3)  # compute includes data
            self.assertGreater(summary["sync_share"], 0.3)
            self.assertIsNone(summarize_ranks(run / "rank-0"))


class PsMemoryFitTest(unittest.TestCase):
    """Measured on the RTX 4060 laptop (8 GB): CuPy pool peaks and device-wide use of single-process runs."""

    def test_workers_fit_by_their_share_of_the_batch(self):
        from apps.autopilot.agents.ps_training import memory_fit

        width64_batch4 = {"mem_peak_mb": 3127.0, "mem_total_mb": 8187.5, "device_mem_max_mb": 4992.0}
        self.assertTrue(memory_fit(width64_batch4, 2)[0])
        self.assertTrue(memory_fit(width64_batch4, 4)[0])
        # Two width-64 workers at batch 4 *each* (global 8) overflowed 8 GB in practice: 2 × 3127 MiB.
        self.assertFalse(memory_fit({**width64_batch4, "mem_peak_mb": 6254.0, "device_mem_max_mb": 8119.0}, 2)[0])
        self.assertTrue(memory_fit({}, 2)[0], "unknown memory: apps.train's per-worker cap fails fast instead")

    def test_nothing_fits_concludes_keep_single_process(self):
        spec = default_spec("dev")
        spec["fixed"].update(device="cuda", base_width=16)
        trial = {"trial_id": "t001", "phase": "baseline", "status": "done", "params": {"lr": 0.002, "batch": 32},
                 "verdict": {"status": "valid"}, "result": {"metrics": {"val_angle_mae": 0.1}}}
        ev = Evidence(spec=spec, state={}, trials=[trial])
        ev.summary = lambda t: {"mem_peak_mb": 6130.0, "mem_total_mb": 8187.5, "device_mem_max_mb": 7887.0,
                                "throughput_sps": 157.0}
        ps = PsTraining()
        self.assertEqual(ps.propose([], ev), [])
        findings = [f.__dict__ for f in ps.diagnose(ev)]
        conclusion = next(f for f in findings if f["kind"] == "ps_conclusion")
        self.assertTrue(conclusion["data"]["keep_single_process"] and conclusion["data"]["memory_bound"])
        self.assertIn("显存上限内放不下数据并行", conclusion["message"])
        self.assertIn("保留单进程", ps.report_section(ev, findings))


    def test_each_layout_is_measured_once(self):
        spec = default_spec("dev")
        spec["fixed"].update(device="cuda", base_width=16)
        done = {"status": "done", "verdict": {"status": "valid"}, "result": {"metrics": {"val_angle_mae": 0.1}}}
        trials = [{"trial_id": "t001", "phase": "baseline", "params": {"lr": 0.002, "batch": 8}, **done},
                  {"trial_id": "t002", "phase": "exploit", **done, "result": {"metrics": {"val_angle_mae": 0.2}},
                   "params": {"lr": 0.002, "batch": 8, "distributed": "ps", "workers": 2, "transport": "grpc_proto"}}]
        ev = Evidence(spec=spec, state={}, trials=trials)
        ev.summary = lambda t: {"mem_peak_mb": 1500.0, "mem_total_mb": 8187.5, "device_mem_max_mb": 2500.0}
        layouts = {(p.delta["distributed"], p.delta["workers"], p.delta["transport"]) for p in PsTraining().propose([], ev)}
        self.assertNotIn(("ps", 2, "grpc_proto"), layouts)
        self.assertIn(("ring", 2, "grpc_proto"), layouts)


class SpecialistCampaignTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.h = Harness(Path(self._tmp.name), {"objective": {"lr_opt": -3.0, "seed_noise": 0.0},
                                                "parallel": {"sync_ms": {"ps": 60.0, "ring": 40.0}, "straggler": True}})

    def tearDown(self):
        self.h.close()
        self._tmp.cleanup()

    def test_ps_matrix_and_keep_single_process(self):
        campaign = self.h.run(campaign_id="ps-campaign", timeout=240,
                              baseline={"batch": 4, "seeds": [1, 2]}, space={"lr": [0.001, 0.0003], "batch": [4]},
                              strategy={"max_candidates": 1, "seed": 0}, budget={"max_trials": 10},
                              specialists=["benchmark", "ps_training"])
        self.assertEqual(campaign["status"], "completed", campaign.get("stop_reason"))
        trials = self.h.service.trials("ps-campaign")
        parallel = sorted((t["params"]["distributed"], t["params"]["workers"], t["params"]["transport"])
                          for t in trials if t["params"].get("distributed", "none") != "none")
        self.assertEqual(parallel, [("ps", 2, "grpc_proto"), ("ps", 4, "grpc_proto"),
                                    ("ring", 2, "grpc_proto"), ("ring", 4, "grpc_proto")],
                         "socket_json is skipped at width 64; workers never exceed the global batch")
        findings = self.h.service.findings("ps-campaign")["findings"]
        kinds = {f["kind"] for f in findings if f["agent"] == "ps_training"}
        self.assertTrue({"sync_bound", "straggler", "ps_conclusion"} <= kinds, kinds)
        conclusion = next(f for f in findings if f["kind"] == "ps_conclusion")
        self.assertTrue(conclusion["data"]["keep_single_process"])
        self.assertIn("保留单进程", conclusion["message"])
        chains = [c for c in self.h.service.findings("ps-campaign")["chains"] if c["proposal"]["agent"] == "ps_training"]
        self.assertEqual(len(chains), 4)
        self.assertTrue(all(c["decision"] == "rejected" for c in chains))
        self.assertEqual(self.h.service.get_campaign("ps-campaign")["state"]["adopted"], {})
        report = self.h.service.report("ps-campaign")["markdown"]
        self.assertIn("### PS Training：数据并行对比", report)
        self.assertIn("RING × 4 worker（grpc_proto）", report)

    def test_conv_backend_recheck(self):
        campaign = self.h.run(campaign_id="op-campaign", timeout=180, space={"lr": [0.001], "batch": [1]},
                              strategy={"max_candidates": 1, "seed": 0}, budget={"max_trials": 8},
                              specialists=["benchmark", "operator_opt"])
        self.assertEqual(campaign["status"], "completed", campaign.get("stop_reason"))
        rechecks = [t for t in self.h.service.trials("op-campaign") if t["params"].get("conv_backend")]
        self.assertEqual([t["params"]["conv_backend"] for t in rechecks], ["cuda_native_cublas"])
        self.assertEqual(rechecks[0]["origin"]["kind"], "throughput")
        chain = next(c for c in self.h.service.findings("op-campaign")["chains"] if c["trial_id"] == rechecks[0]["trial_id"])
        self.assertEqual(chain["decision"], "rejected", "equal throughput is not adopted")
        report = self.h.service.report("op-campaign")["markdown"]
        self.assertIn("### Operator Optimization", report)


if __name__ == "__main__":
    unittest.main()
