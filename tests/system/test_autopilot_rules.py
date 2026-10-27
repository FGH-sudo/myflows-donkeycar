"""AutoPilot rule core without processes: spec validation, candidate sampling, pre-check, online rules,
the knowledge store and post-check (spec §6, §13 layer one)."""

import json
import math
import tempfile
import time
import unittest
from pathlib import Path

from apps.autopilot.agents.base import Proposal
from apps.autopilot.memory import KnowledgeBase, SEED_FACTS
from apps.autopilot.planner import confirm_seeds
from apps.autopilot.spec import SpecError, default_spec, grid, resolve_args, validate_spec
from apps.autopilot.strategies.candidate_list import stratified
from apps.autopilot.verifier import OnlineMonitor, postcheck, precheck, replay_online_findings
from apps.common.gpu_lock import EVENTS_FILE
from tests.system.harness import make_tub


def _write_rows(path: Path, rows):
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class SpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.repo = Path(cls.tmp.name)
        make_tub(cls.repo, count=4)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_defaults_follow_the_spec_decisions(self):
        spec = validate_spec({}, self.repo)
        self.assertEqual((spec["baseline"]["lr"], spec["baseline"]["batch"], spec["baseline"]["num_workers"]),
                         (0.001, 2, 0))
        self.assertEqual(spec["baseline"]["seeds"], [1, 2])
        self.assertEqual(validate_spec({"mode": "formal"}, self.repo)["baseline"]["seeds"], [1, 2, 3])
        self.assertEqual(spec["space"], {"lr": [0.0001, 0.0003, 0.001], "batch": [1, 2, 4]})
        self.assertEqual(spec["llm"]["max_calls"], 30)
        self.assertEqual(spec["llm"]["max_total_tokens"], 414000)
        self.assertIsNone(validate_spec({"llm": {"max_total_tokens": None}}, self.repo)["llm"]["max_total_tokens"])

    def test_default_candidate_count_shrinks_to_a_small_space(self):
        small = {"space": {"lr": [0.001, 0.002], "batch": [8]}}
        self.assertEqual(validate_spec(small, self.repo)["strategy"]["max_candidates"], 2)
        with self.assertRaises(SpecError) as ctx:  # an explicit count over the grid is still a mistake
            validate_spec({**small, "strategy": {"max_candidates": 3}}, self.repo)
        self.assertIn("不能大于 2", str(ctx.exception))

    def test_invalid_specs_are_rejected_with_reasons(self):
        cases = [
            ({"unknown": 1}, "不支持的字段"),
            ({"space": {"lr": [0.001], "batch": [2], "freeze_stages": [1]}}, "fixed.pretrained"),
            ({"space": {"lr": [0.001], "batch": [2], "weight_decay": [0.0]}}, "TPE"),
            ({"space": {"lr": [0.001], "batch": [2], "num_workers": [0, 2]}}, "吞吐"),
            ({"llm": {"max_calls": 31}}, "不能大于 30"),
            ({"baseline": {"seeds": [1, 1]}}, "不能重复"),
            ({"fixed": {"lr": 0.1}}, "fixed 不能包含 lr"),
            ({"campaign_id": "Bad Id"}, "campaign_id"),
            ({"fixed": {"device": "cpu", "conv_backend": "cuda_native_register"}}, "基线配置无效"),
            ({"objective": {"metric": "val_accuracy"}}, "不支持目标指标"),
            ({"mode": "formal", "fixed": {"split_kind": "random"}}, "按时间段划分"),
            ({"mode": "formal", "fixed": {"test_size": 0}}, "测试集"),
            ({"mode": "formal", "baseline": {"seeds": [1, 2]}}, "3 个基线 seed"),
        ]
        for raw, message in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(SpecError) as ctx:
                    validate_spec(raw, self.repo)
                self.assertIn(message, str(ctx.exception))

    def test_trial_arguments_use_the_fixed_split(self):
        spec = validate_spec({}, self.repo)
        with self.assertRaises(SpecError):
            resolve_args(spec, {"lr": 0.001, "batch": 2, "seed": 1})
        spec["fixed"]["split_file"] = "runs/autopilot/splits/x.json"
        args = resolve_args(spec, {"lr": 0.001, "batch": 2, "seed": 1, "num_workers": 0})
        self.assertEqual(args["split_file"], "runs/autopilot/splits/x.json")
        self.assertNotIn("val_size", args)
        self.assertEqual((args["checkpoint_every"], args["no_tensorboard"]), (0, True))

    def test_confirmation_seeds_avoid_the_explore_seed(self):
        self.assertEqual(confirm_seeds(default_spec("dev")), [2, 3])
        self.assertEqual(confirm_seeds(default_spec("formal")), [2, 3, 4])


class CandidateListTest(unittest.TestCase):
    def test_stratified_sample_covers_every_value_and_is_reproducible(self):
        space = {"lr": [1e-4, 3e-4, 1e-3], "batch": [1, 2, 4]}
        baseline = {"lr": 1e-3, "batch": 2}
        picked = stratified(space, 6, seed=0, exclude=[baseline])
        self.assertEqual(len(picked), 6)
        self.assertNotIn(baseline, picked)
        self.assertEqual({p["lr"] for p in picked}, set(space["lr"]))
        self.assertEqual({p["batch"] for p in picked}, set(space["batch"]))
        for key in ("lr", "batch"):
            counts = [sum(1 for p in picked if p[key] == v) for v in space[key]]
            self.assertLessEqual(max(counts) - min(counts), 1, key)
        self.assertEqual(picked, stratified(space, 6, seed=0, exclude=[baseline]))
        self.assertEqual(len(grid(space)), 9)


class PrecheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.repo = Path(cls.tmp.name)
        make_tub(cls.repo, count=4)
        cls.spec = validate_spec({}, cls.repo)
        cls.spec["fixed"]["split_file"] = "split.json"
        (cls.repo / "split.json").write_text("{}", encoding="utf-8")
        cls.knowledge = KnowledgeBase(cls.repo / "knowledge.jsonl")
        cls.knowledge.ensure_seeded()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, params, delta=None, trials=(), fingerprint="fp", budget=None):
        base = {"lr": 0.001, "batch": 2, "num_workers": 0, "seed": 1}
        return precheck({**base, **params}, spec=self.spec, repo_root=self.repo, knowledge=self.knowledge,
                        trials=list(trials), fingerprint=fingerprint, delta=delta, budget_state=budget)

    def test_accepts_a_valid_trial(self):
        result = self.check({"lr": 0.0003}, {"lr": 0.0003})
        self.assertTrue(result["passed"], result["reasons"])
        self.assertEqual(result["args"]["lr"], 0.0003)

    def test_rejections(self):
        cases = [
            ({"lr": 0.01}, {"lr": 0.01}, "超出搜索空间"),
            ({"optimizer": "sgd"}, {"optimizer": "sgd"}, "不在可调范围内"),
            ({"freeze_stages": 2}, {"freeze_stages": 2}, "能力不可用"),
            ({"weight_decay": 0.01}, {"weight_decay": 0.01}, "不在可调范围内"),
            ({"num_workers": -1}, {"num_workers": -1}, "参数校验失败"),
        ]
        for params, delta, message in cases:
            with self.subTest(params=params):
                result = self.check(params, delta)
                self.assertFalse(result["passed"])
                self.assertTrue(any(message in r for r in result["reasons"]), result["reasons"])

    def test_duplicates_budget_and_known_oom(self):
        done = {"trial_id": "t001", "fingerprint": "fp", "status": "done", "verdict": {"status": "valid"}}
        self.assertIn("重复的指纹", self.check({}, trials=[done])["reasons"][0])
        infra = {"trial_id": "t001", "fingerprint": "fp", "status": "done",
                 "verdict": {"status": "failed", "kind": "crash"}}
        self.assertTrue(self.check({}, trials=[infra])["passed"])
        budget = {"trials_committed": self.spec["budget"]["max_trials"], "run_s": 0.0}
        self.assertIn("预算", self.check({}, budget=budget)["reasons"][0])
        self.knowledge.record_oom("donkey_resnet18_single", {"batch": 4, "stem": "cifar", "base_width": 64},
                                  evidence="test", code_hash=None)
        result = self.check({"batch": 4}, {"batch": 4})
        self.assertFalse(result["passed"])
        self.assertIn("经验库预测会失败", result["reasons"][0])
        self.assertTrue(self.check({"batch": 1}, {"batch": 1})["passed"])

    def test_knowledge_store_imports_resource_facts_only(self):
        kinds = {fact["kind"] for fact in self.knowledge.facts()}
        self.assertEqual(kinds, {"resource"})
        self.assertGreaterEqual(len(self.knowledge.facts()), len(SEED_FACTS))
        two_workers = {"distributed": "ring", "workers": 2, "batch": 8, "stem": "cifar", "base_width": 64}
        self.assertEqual(self.knowledge.predicted_failure("donkey_resnet18_single", two_workers)["id"],
                         "seed-two-w64-workers-batch4")
        self.assertIsNone(self.knowledge.predicted_failure("donkey_resnet18_single", {**two_workers, "batch": 4}))


class OnlineMonitorTest(unittest.TestCase):
    ONLINE = {"divergence_windows": 3, "divergence_factor": 3.0, "window_steps": 2, "hang_timeout_s": 5,
              "gpu_idle_pct": 35}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run = Path(self._tmp.name)
        self.now = 1000.0

    def tearDown(self):
        self._tmp.cleanup()

    def monitor(self, timeout=100):
        return OnlineMonitor(self.run, self.ONLINE, timeout, clock=lambda: self.now)

    def acquire(self, at):
        _write_rows(self.run / EVENTS_FILE, [{"event": "wait", "unix_s": at - 50}, {"event": "acquired", "unix_s": at}])

    def steps(self, losses, start=1):
        _write_rows(self.run / "metrics.jsonl", [{"kind": "step", "step": start + i, "loss": loss}
                                                 for i, loss in enumerate(losses)])

    def test_clocks_wait_for_the_gpu_lock(self):
        monitor = self.monitor(timeout=10)
        self.now = 5000.0  # far beyond hang and timeout, but the trial is still queued for the GPU
        self.assertIsNone(monitor.check())
        self.acquire(4999.0)
        self.assertIsNone(monitor.check())
        self.now = 5006.0
        self.assertEqual(monitor.check().kind, "hang")

    def test_timeout_counts_from_the_lock(self):
        self.acquire(1000.0)
        monitor = self.monitor(timeout=10)
        for t in range(1, 12):
            self.now = 1000.0 + t
            self.steps([1.0], start=t)
            stop = monitor.check()
        self.assertEqual(stop.kind, "timeout")

    def test_nan_and_divergence(self):
        self.acquire(1000.0)
        monitor = self.monitor()
        self.steps([1.0, 0.9, float("nan")])
        self.assertEqual(monitor.check().kind, "nan")
        self.run.joinpath("metrics.jsonl").unlink()
        monitor = self.monitor()
        self.steps([1.0 * 1.5 ** i for i in range(12)])
        stop = monitor.check()
        self.assertEqual(stop.kind, "diverged")
        self.assertIn("3 个窗口", stop.message)
        self.assertEqual((stop.step, stop.since_step), (8, 3), "the rise starts after the first window's trough")

    def test_explosion_is_divergence_and_records_where_it_began(self):
        self.acquire(1000.0)
        monitor = self.monitor()
        # Adam with a far too large lr: the second update blows the loss up, then it only decays slowly.
        self.steps([0.6] + [5e4 * 0.9 ** i for i in range(9)])
        stop = monitor.check()
        self.assertEqual((stop.kind, stop.step, stop.since_step), ("diverged", 6, 1))
        self.assertIn("爆炸", stop.message)
        self.assertEqual({k: stop.to_dict()[k] for k in ("step", "since_step")}, {"step": 6, "since_step": 1})

    def test_a_spike_that_recovers_is_not_divergence(self):
        self.acquire(1000.0)
        monitor = self.monitor()
        # Measured with lr 0.5: one huge step, then back near the start within two windows.
        self.steps([0.6, 4000.0, 60.0, 1.5, 1.9, 4.7, 1.6, 6.0, 2.3, 4.5, 2.2, 1.2])
        self.assertIsNone(monitor.check())

    def test_rising_but_small_loss_is_not_divergence(self):
        self.acquire(1000.0)
        monitor = self.monitor()
        self.steps([1.0 + 0.05 * i for i in range(12)])
        self.assertIsNone(monitor.check())

    def timed_steps(self, start_unix, count, *, every=0.5, first=1):
        _write_rows(self.run / "metrics.jsonl", [{"kind": "step", "step": first + i, "loss": 1.0,
                                                  "unix_s": start_unix + every * i} for i in range(count)])

    def samples(self, start_unix, count, util, *, every=1.0):
        _write_rows(self.run / "resources.jsonl", [{"unix_s": start_unix + every * i, "gpu_utilization_pct": util}
                                                   for i in range(count)])

    def test_oom_in_log_and_gpu_idle_finding(self):
        self.acquire(1000.0)
        monitor = self.monitor()
        self.now = 1012.0
        self.timed_steps(1000.0, 24)  # steps keep coming every 0.5 s while the GPU sits at 20 %
        self.samples(1000.25, 12, 20.0)
        self.assertIsNone(monitor.check())
        self.assertEqual([f["kind"] for f in monitor.findings], ["gpu_idle"])
        (self.run / "train.log").write_text("cupy.cuda.memory.OutOfMemoryError: Out of memory\n", encoding="utf-8")
        self.assertEqual(monitor.check().kind, "oom")

    def test_an_epoch_boundary_is_not_gpu_idle_training(self):
        # formal-full-002: 95 % while training; the per-epoch validation and checkpoint left the GPU idle for 19 s,
        # and the old rule took the epoch row written after it as "steps still advancing".
        self.acquire(1000.0)
        monitor = self.monitor()
        self.timed_steps(1000.0, 20)
        self.samples(1000.25, 10, 95.0)
        self.samples(1010.0, 20, 10.0)
        _write_rows(self.run / "metrics.jsonl", [{"kind": "epoch", "epoch": 1, "unix_s": 1025.0}])
        self.timed_steps(1030.0, 4, first=21)
        self.samples(1030.25, 2, 95.0)
        self.now = 1032.0
        self.assertIsNone(monitor.check())
        self.assertEqual(monitor.findings, [])

    def test_replay_rechecks_a_finished_attempt(self):
        self.acquire(1000.0)
        self.timed_steps(1000.0, 20)
        self.samples(1000.25, 10, 95.0)
        self.samples(1010.0, 20, 10.0)  # epoch-end validation
        _write_rows(self.run / "metrics.jsonl", [{"kind": "epoch", "epoch": 1, "unix_s": 1025.0}])
        self.timed_steps(1030.0, 4, first=21)
        self.assertEqual(replay_online_findings(self.run, self.ONLINE, 100), [])
        self.timed_steps(1032.0, 30, first=25)  # then the loop really starves the GPU
        self.samples(1032.25, 14, 15.0)
        self.assertEqual([f["kind"] for f in replay_online_findings(self.run, self.ONLINE, 100)], ["gpu_idle"])

    def test_slow_data_bound_steps_still_count_as_running(self):
        self.acquire(1000.0)
        monitor = self.monitor()
        self.timed_steps(1000.0, 12, every=3.0)  # 3 s per step, GPU mostly waiting for data
        self.samples(1000.5, 30, 12.0)
        self.now = 1034.0
        self.assertIsNone(monitor.check())
        self.assertEqual([f["kind"] for f in monitor.findings], ["gpu_idle"])


class PowerLimitEvidenceTest(unittest.TestCase):
    """Throttling compares each sample with the limit enforced at that moment (it moves on a laptop)."""

    def summary(self, samples):
        from apps.autopilot.evidence import summarize_attempt

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            _write_rows(run / "metrics.jsonl", [{"kind": "step", "step": i + 1, "loss": 1.0, "unix_s": 100.0 + i,
                                                 "data_load_ms": 1.0, "step_time_ms": 10.0, "batch_size": 4}
                                                for i in range(len(samples))])
            _write_rows(run / "resources.jsonl", [
                {"unix_s": 100.0 + i, "gpu_temperature_c": 60.0, "gpu_power_w": power,
                 "gpus": [{"power_w": power, "power_limit_w": limit}]} for i, (power, limit) in enumerate(samples)])
            return summarize_attempt(run)

    def test_a_moving_limit_alone_is_not_throttling(self):
        # The enforced limit jumps between 60 and 80 W while the GPU draws well below it.
        out = self.summary([(55.0, 80.0 if i % 3 else 60.0) for i in range(30)])
        self.assertEqual((out["power_limit_w"], out["power_limit_min_w"], out["power_limited"]), (80.0, 60.0, False))
        self.assertFalse(out["throttled"])

    def test_drawing_the_enforced_limit_is_power_limited(self):
        out = self.summary([(79.5, 80.0) if i % 2 else (60.0, 60.0) for i in range(30)])
        self.assertEqual(out["power_capped_share"], 1.0)
        self.assertTrue(out["power_limited"] and out["throttled"])


class GpuMonitorMemoryTest(unittest.TestCase):
    def evidence(self, summary):
        from apps.autopilot.evidence import Evidence

        trial = {"trial_id": "t001", "phase": "baseline", "status": "done", "params": {"lr": 0.002, "batch": 32},
                 "verdict": {"status": "valid"}, "result": {"metrics": {"val_angle_mae": 0.1}}}
        ev = Evidence(spec=default_spec(), state={}, trials=[trial])
        ev.summary = lambda t: {"steps": 30, **summary}
        return ev

    def test_memory_pressure_counts_the_whole_gpu(self):
        from apps.autopilot.agents.gpu_monitor import GpuMonitor

        # Measured: width 16, batch 32 — the process pool is 75 % but the desktop brings the GPU to 96 %.
        ev = self.evidence({"mem_share": 0.749, "device_mem_share": 0.964})
        monitor = GpuMonitor()
        findings = [f for f in monitor.diagnose(ev) if f.kind == "memory_pressure"]
        self.assertEqual(len(findings), 1)
        self.assertIn("96%", findings[0].message)
        self.assertEqual(findings[0].evidence[0]["metric"], "device_mem_share")
        self.assertEqual([p.delta for p in monitor.propose(findings, ev)], [{"batch": 16}])
        self.assertFalse([f for f in monitor.diagnose(self.evidence({"mem_share": 0.38, "device_mem_share": 0.61}))
                          if f.kind == "memory_pressure"])


class PostcheckTest(unittest.TestCase):
    def test_valid_and_each_invalid_reason(self):
        spec = default_spec()
        split = {"val_hash": "v", "train_count": 30}
        trial = {"resolved_args": {"conv_backend": "cuda_native_register"}}
        good = {"status": "succeeded", "metrics": {"val_angle_mae": 0.1, "val_loss": 0.2}, "data": {"val_hash": "v"},
                "samples": {"train": 30, "trained": 30, "expected_trained": 30},
                "backend": {"actual": "cuda_native_register"}, "memory": {"peak_pool_mb": 100.0}}
        attempt = {"status": "succeeded", "failure": None}
        self.assertEqual(postcheck(trial, attempt, good, spec=spec, split=split)["status"], "valid")
        bad = {**good, "metrics": {"val_angle_mae": math.inf, "val_loss": 0.2}, "data": {"val_hash": "other"},
               "samples": {"train": 29, "trained": 10, "expected_trained": 30}, "backend": {"actual": "cupy"}}
        verdict = postcheck(trial, attempt, bad, spec=spec, split=split)
        self.assertEqual(verdict["status"], "invalid")
        self.assertEqual(len(verdict["reasons"]), 5)
        failed = postcheck(trial, {"status": "failed", "failure": {"kind": "oom", "message": "x"}}, None,
                           spec=spec, split=split)
        self.assertEqual((failed["status"], failed["kind"]), ("failed", "oom"))


class ProposalTest(unittest.TestCase):
    def test_proposal_round_trips_to_dict(self):
        p = Proposal(delta={"num_workers": 2}, rationale="r", agent="gpu_monitor", kind="throughput")
        self.assertEqual(p.to_dict()["delta"], {"num_workers": 2})
        self.assertLess(time.time() - 0, time.time() + 1)



class ConclusionTest(unittest.TestCase):
    def trial(self, tid, phase, seed, value, status="valid"):
        return {"trial_id": tid, "phase": phase, "status": "done", "verdict": {"status": status},
                "params": {"lr": 0.001, "batch": 2, "num_workers": 0, "seed": seed},
                "result": {"metrics": {"val_angle_mae": value}}}

    def conclude(self, trials, **state):
        from apps.autopilot.evidence import Evidence
        from apps.autopilot.reporter import confirmation, conclusion

        ev = Evidence(spec=default_spec("dev"), state=state, trials=trials)
        return conclusion(ev, confirmation(ev))

    def test_a_campaign_stopped_before_any_candidate_finished_draws_no_conclusion(self):
        baseline = [self.trial("t001", "baseline", 1, 0.03), self.trial("t002", "baseline", 2, 0.02)]
        text = self.conclude(baseline + [self.trial("t003", "explore", 1, None, status="cancelled")],
                             status="stopped", phase="explore")
        self.assertIn("campaign 在探索阶段结束（stopped）", text)
        self.assertIn("无法判断能否超越人工基线", text)
        self.assertNotIn("未被超越", text)

    def test_candidates_compared_before_a_stop_are_marked_partial(self):
        trials = [self.trial("t001", "baseline", 1, 0.03), self.trial("t002", "baseline", 2, 0.02),
                  self.trial("t003", "explore", 1, 0.04)]
        self.assertIn("只比较了已跑完的候选", self.conclude(trials, status="stopped", phase="explore"))
        self.assertNotIn("提前结束", self.conclude(trials, status="completed", phase="done"))


if __name__ == "__main__":
    unittest.main()
