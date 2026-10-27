"""E3 in campaigns: exact rollback on NaN/divergence (bounded), OOM → half batch, TPE proposals (inside the
space, never repeated, conditional regularisation), and the time-block split with its independent test set."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apps.autopilot.evidence import Evidence
from apps.autopilot.spec import default_spec, in_space, validate_spec
from apps.autopilot.strategies.tpe import TPEStrategy
from apps.common.splits import build_time_block_split, frame_numbers
from tests.system.harness import Harness

ROLLBACK = {"enabled": True, "state_every": 1, "max_rollbacks": 2, "lr_factor": 0.5}


class RollbackTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.h.close()
        self._tmp.cleanup()

    def harness(self, rules):
        self.h = Harness(Path(self._tmp.name), {"objective": {"lr_opt": -3.0}, "rules": rules})
        return self.h

    def run_one(self, cid, **spec):
        return self.h.run(campaign_id=cid, timeout=180, baseline={"batch": 1}, space={"lr": [0.001, 0.004], "batch": [1]},
                          strategy={"max_candidates": 1, "seed": 0}, budget={"max_trials": 6}, **spec)

    def test_nan_rolls_back_to_the_last_healthy_state_with_half_lr(self):
        h = self.harness([{"when": {"lr": 0.004}, "behavior": "nan"}])
        campaign = self.run_one("nan-rollback", rollback=ROLLBACK)
        self.assertEqual(campaign["status"], "completed")
        trial = next(t for t in h.service.trials("nan-rollback") if t["params"]["lr"] == 0.004)
        self.assertEqual([a["cause"] for a in trial["attempts"]], ["first", "rollback"])
        rb = trial["rollbacks"][0]
        self.assertEqual((rb["kind"], rb["from_step"], rb["from_attempt"]), ("nan", 4, 1))
        self.assertEqual((rb["lr_before"], rb["lr"]), (0.004, 0.002))
        self.assertEqual(trial["verdict"]["status"], "valid")
        command = (Path(trial["attempts"][1]["run_dir"]) / "command.json").read_text(encoding="utf-8")
        self.assertIn("--resume-state", command)
        self.assertIn("step-000000004", command)
        samples = trial["result"]["samples"]
        self.assertEqual(samples["trained"], samples["expected_trained"])
        self.assertFalse(any((Path(a["run_dir"]) / "states").exists() for a in trial["attempts"]),
                         "states are dropped once the trial is decided")
        events = h.service.events("nan-rollback", kinds=("rollback",))
        self.assertEqual(len(events), 1)
        report = h.service.report("nan-rollback")["markdown"]
        self.assertIn("异常与精确回滚", report)
        self.assertIn("0.004 → 0.002", report)

    def test_rollbacks_are_bounded(self):
        h = self.harness([{"when": {"lr": 0.004}, "behavior": "nan"}, {"when": {"lr": 0.002}, "behavior": "nan"}])
        self.run_one("bounded", rollback={**ROLLBACK, "max_rollbacks": 1})
        trial = next(t for t in h.service.trials("bounded") if t["params"]["lr"] == 0.004)
        self.assertEqual(len(trial["rollbacks"]), 1)
        self.assertEqual((trial["verdict"]["status"], trial["verdict"]["kind"]), ("failed", "nan"))
        skipped = h.service.events("bounded", kinds=("rollback_skipped",))
        self.assertIn("上限", skipped[-1]["reason"])

    def test_divergence_rolls_back_before_the_rise(self):
        h = self.harness([{"when": {"lr": 0.004}, "behavior": "diverge"}])
        self.run_one("diverge-rollback", rollback=ROLLBACK, fixed={"max_samples": 40, "epochs": 2, "val_size": 10})
        trial = next(t for t in h.service.trials("diverge-rollback") if t["params"]["lr"] == 0.004)
        rb = trial["rollbacks"][0]
        self.assertEqual(rb["kind"], "diverged")
        self.assertLessEqual(rb["from_step"], 4, "the state must predate the loss passing 3× its start")
        self.assertEqual(trial["verdict"]["status"], "valid")

    def test_explosion_on_the_first_updates_rolls_back_to_the_initial_weights(self):
        h = self.harness([{"when": {"lr": 0.004}, "behavior": "explode"}])
        self.run_one("explode-rollback", rollback={**ROLLBACK, "state_every": 5, "lr_factor": 0.1},
                     fixed={"max_samples": 40, "epochs": 2, "val_size": 10})
        trial = next(t for t in h.service.trials("explode-rollback") if t["params"]["lr"] == 0.004)
        failure = trial["attempts"][0]["failure"]
        self.assertEqual((failure["kind"], failure["since_step"]), ("diverged", 1))
        self.assertIn("爆炸", failure["message"])
        rb = trial["rollbacks"][0]
        self.assertEqual((rb["from_step"], rb["lr"]), (0, 0.004 * 0.1))
        self.assertIn("初始权重", rb["reason"])
        self.assertEqual(trial["verdict"]["status"], "valid")

    def test_without_rollback_the_trial_just_fails(self):
        h = self.harness([{"when": {"lr": 0.004}, "behavior": "nan"}])
        self.run_one("no-rollback")
        trial = next(t for t in h.service.trials("no-rollback") if t["params"]["lr"] == 0.004)
        self.assertEqual(len(trial["attempts"]), 1)
        self.assertEqual(trial["verdict"]["kind"], "nan")
        self.assertNotIn("state_every", trial["resolved_args"])

    def test_oom_derives_a_half_batch_trial(self):
        h = self.harness([{"when": {"batch": 4}, "behavior": "oom"}])
        h.run(campaign_id="oom-half", timeout=180, baseline={"batch": 1}, space={"lr": [0.001], "batch": [1, 4]},
              strategy={"max_candidates": 1, "seed": 0}, budget={"max_trials": 6}, rollback=ROLLBACK)
        trials = h.service.trials("oom-half")
        derived = [t for t in trials if (t.get("origin") or {}).get("strategy") == "rollback"]
        self.assertEqual([t["params"]["batch"] for t in derived], [2])
        self.assertEqual(derived[0]["verdict"]["status"], "valid")
        self.assertEqual(len(h.service.events("oom-half", kinds=("oom_derived",))), 1)


class TPEStrategyTest(unittest.TestCase):
    def spec(self, **space):
        raw = {"campaign_id": "tpe-unit", "strategy": {"name": "tpe", "max_candidates": 6, "seed": 3},
               "space": {"lr": [0.0001, 0.001], "batch": [1, 2, 4], **space},
               "fixed": {"max_samples": 40, "epochs": 1, "val_size": 10, "split_file": "x.json", "device": "cpu",
                         "conv_backend": "numpy"}}
        spec = default_spec("dev")
        spec.update({k: v for k, v in raw.items() if k not in ("fixed", "strategy", "space")})
        spec["strategy"].update(raw["strategy"])
        spec["space"] = raw["space"]
        spec["fixed"].update(raw["fixed"])
        return spec

    def trial(self, tid, params, value):
        return {"trial_id": tid, "phase": "explore", "status": "done", "params": {**params, "seed": 1},
                "verdict": {"status": "valid"}, "result": {"metrics": {"val_angle_mae": value}}}

    def test_proposals_are_inside_unique_and_conditional(self):
        spec = self.spec(weight_decay=[0.0001, 0.01], dropout=[0.1, 0.5], augment=[False, True])
        tpe = TPEStrategy(spec)
        trials = [self.trial("t001", {"lr": 0.001, "batch": 2, "num_workers": 0}, 0.2)]
        seen = set()
        for i in range(12):
            ev = Evidence(spec=spec, state={}, trials=trials)
            proposal = tpe.ask(ev)
            self.assertIsNotNone(proposal)
            params = proposal.delta
            for key, value in params.items():
                self.assertTrue(in_space(spec, key, value), (key, value))
            self.assertFalse(params.get("weight_decay", 0) > 0 and params.get("dropout", 0) > 0,
                             "one regularisation branch at a time")
            key = tuple(sorted(params.items()))
            self.assertNotIn(key, seen)
            seen.add(key)
            trials.append(self.trial(f"t{i + 2:03d}", {**params, "num_workers": 0}, 0.1 + 0.01 * i))
        self.assertTrue(any(p.startswith("TPE") for p in [proposal.rationale]))

    def test_regularisation_keys_need_the_tpe_strategy(self):
        with self.assertRaises(Exception):
            validate_spec({"space": {"lr": [0.001], "batch": [2], "dropout": [0.1]}}, Path("."))


class TPECampaignTest(unittest.TestCase):
    def test_tpe_campaign_runs_its_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(Path(tmp), {"objective": {"lr_opt": -3.5}})
            try:
                campaign = h.run(campaign_id="tpe-campaign", timeout=240,
                                 strategy={"name": "tpe", "max_candidates": 3, "seed": 1},
                                 space={"lr": [0.0001, 0.001], "batch": [1, 2], "weight_decay": [0.0001, 0.01]},
                                 budget={"max_trials": 8})
                self.assertEqual(campaign["status"], "completed", campaign.get("stop_reason"))
                tpe = [t for t in h.service.trials("tpe-campaign") if t["origin"]["strategy"] == "tpe"]
                self.assertEqual(len(tpe), 3)
                self.assertEqual(len({t["fingerprint"] for t in tpe}), 3)
                self.assertTrue(all("weight_decay" in t["resolved_args"] for t in tpe))
            finally:
                h.close()


class TimeBlockSplitTest(unittest.TestCase):
    def test_blocks_keep_neighbours_apart(self):
        frames = frame_numbers([f"{i}_0.1.jpg" for i in (5, 3, 1, 4, 2)] + ["x.jpg"])
        self.assertEqual(frames, [5, 3, 1, 4, 2, 5])
        split = build_time_block_split(list(range(2000)), val_size=200, test_size=200, block_size=40, gap=3, seed=7)
        self.assertGreaterEqual(len(split["val"]), 200)
        self.assertGreaterEqual(len(split["test"]), 200)
        held = set(split["val"]) | set(split["test"])
        self.assertGreaterEqual(min(abs(a - b) for a in split["train"] for b in held), 4)
        self.assertEqual(sorted(split["train"] + split["val"] + split["test"] + split["purged"]), list(range(2000)))

    def test_campaign_with_time_block_split_evaluates_the_test_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(Path(tmp), {"objective": {"lr_opt": -3.5}})
            try:
                campaign = h.run(campaign_id="test-set", timeout=240,
                                 fixed={"max_samples": 40, "epochs": 1, "val_size": 8, "test_size": 8,
                                        "split_kind": "time_block", "block_size": 4},
                                 space={"lr": [0.001, 0.0003], "batch": [1]}, strategy={"max_candidates": 1, "seed": 0})
                self.assertEqual(campaign["status"], "completed")
                split = campaign["state"]["provenance"]["split"]
                self.assertEqual((split["kind"], split["test_count"] >= 8), ("time_block", True))
                test = campaign["state"]["test_evaluation"]
                self.assertEqual(set(test["groups"]), {"baseline", "best"})
                self.assertGreaterEqual(len(test["groups"]["baseline"]["angle_mae"]), 2)
                report = h.service.report("test-set")["markdown"]
                self.assertIn("### 独立测试集", report)
                self.assertIn("time_block", report)
            finally:
                h.close()


if __name__ == "__main__":
    unittest.main()
