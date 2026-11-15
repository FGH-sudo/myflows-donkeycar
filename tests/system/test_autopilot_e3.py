"""E3 in campaigns: exact rollback on NaN/divergence (bounded), OOM → half batch, TPE proposals (inside the
space, never repeated, conditional regularisation), and the time-block split with its independent test set."""

from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from apps.autopilot.data import describe_split, ensure_split, load_index
from apps.autopilot.evidence import Evidence
from apps.autopilot.memory import KnowledgeBase
from apps.autopilot.service import CampaignError
from apps.autopilot.spec import default_spec, in_space, validate_spec
from apps.autopilot.strategies.tpe import TPEStrategy, record_rejection, rejected_proposals
from apps.common.splits import (TIME_BLOCK_LEGACY, TIME_BLOCK_SEPARATED, build_time_block_split, frame_numbers,
                                load_split, save_split, validate_time_split_gap)
from tests.system.harness import Harness, make_tub

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

    def test_recovered_rollback_keeps_its_state_and_lr_on_infrastructure_retry(self):
        h = self.harness([])
        cid = "recovered-rollback"
        h.service.create_campaign(h.spec(campaign_id=cid))
        queued = h.service.submit_trials(cid, [{"delta": {"lr": 0.001, "batch": 2}}], source="user")[0]
        self.assertTrue(queued["queued"], queued)
        store = h.service.store(cid)
        trial = store.load_trial(queued["trial_id"])
        overrides = {"resume_state": "healthy-state", "lr": 0.0005}
        trial.update(status="running", attempts=[{
            "attempt": 1, "status": "running", "cause": "rollback", "overrides": overrides,
            "run_dir": str(store.attempt_dir(trial["trial_id"], 1))}])
        store.save_trial(trial)

        def fail(_store, _trial, attempt, _spec, **_kwargs):
            attempt.update(status="failed", failure={"kind": "crash", "message": "injected crash"}, run_s=0.1)
            return attempt

        def submit(_store, record, no, *, cause, overrides):
            attempt = {"attempt": no, "cause": cause, "overrides": overrides, "status": "running",
                       "run_dir": str(store.attempt_dir(record["trial_id"], no))}
            record["attempts"].append(attempt)
            return attempt

        with mock.patch.object(h.service.executor, "wait", side_effect=fail), \
                mock.patch.object(h.service.executor, "submit", side_effect=submit) as dispatched:
            h.service._run_trial(store, store.load_trial(trial["trial_id"]))
        self.assertEqual(dispatched.call_count, 1)
        self.assertEqual(dispatched.call_args.kwargs, {"cause": "retry", "overrides": overrides})

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

    def test_refused_suggestions_and_known_failures_are_not_proposed_again(self):
        spec = self.spec()
        trials = [self.trial("t001", {"lr": 0.001, "batch": 2, "num_workers": 0}, 0.2)]
        first = TPEStrategy(spec).ask(Evidence(spec=spec, state={}, trials=trials))
        # Same history and nothing recorded: the same suggestion comes back (what stalled TPE campaigns).
        self.assertEqual(TPEStrategy(spec).ask(Evidence(spec=spec, state={}, trials=trials)).delta, first.delta)
        state = {}
        seen = [first.delta]
        for _ in range(6):
            record_rejection(state, {**seen[-1], "num_workers": 0, "seed": 1}, ["预检拒绝"])
            proposal = TPEStrategy(spec).ask(Evidence(spec=spec, state=state, trials=trials))
            self.assertIsNotNone(proposal)
            self.assertNotIn(proposal.delta, seen)
            seen.append(proposal.delta)
        self.assertEqual(len(rejected_proposals(state)), 6)
        record_rejection(state, {**seen[0], "num_workers": 0, "seed": 9}, ["again"])
        self.assertEqual(len(rejected_proposals(state)), 6, "one record per configuration")

        with tempfile.TemporaryDirectory() as tmp:
            knowledge = KnowledgeBase(Path(tmp) / "knowledge.jsonl")
            knowledge.add({"id": "test-oom-b4", "kind": "resource", "task": spec["task"], "model": {},
                           "when": {"batch_min": 4}, "outcome": "oom", "fact": "batch 4 时显存不足"})
            tpe = TPEStrategy(spec, knowledge=knowledge)
            history = list(trials)
            for i in range(10):
                proposal = tpe.ask(Evidence(spec=spec, state={}, trials=history))
                self.assertIsNotNone(proposal)
                self.assertLess(proposal.delta["batch"], 4, "the knowledge base predicts batch 4 fails")
                history.append(self.trial(f"t{i + 2:03d}", {**proposal.delta, "num_workers": 0}, 0.1 + 0.01 * i))


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

    def run_with_oom_fact(self, cid, **patches):
        """A TPE campaign whose every suggestion (batch 4 or 8) is refused by an out-of-memory fact."""
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(Path(tmp), {"objective": {"lr_opt": -3.5}})
            try:
                h.service.knowledge.add({"id": "test-oom-b4", "kind": "resource", "task": "donkey_resnet18_single",
                                         "model": {}, "when": {"batch_min": 4}, "outcome": "oom",
                                         "fact": "batch 4 时显存不足"})
                with mock.patch.multiple(TPEStrategy, **patches) if patches else contextlib.nullcontext():
                    campaign = h.run(campaign_id=cid, timeout=240,
                                     strategy={"name": "tpe", "max_candidates": 3, "seed": 1},
                                     space={"lr": [0.0001, 0.001], "batch": [4, 8]}, budget={"max_trials": 8})
                return campaign, h.service.trials(cid), h.service.events(cid, kinds=("precheck",))
            finally:
                h.close()

    def test_refused_suggestions_use_up_candidates_and_the_campaign_goes_on(self):
        # Without the knowledge filter every suggestion reaches the pre-check and is refused there.
        campaign, trials, prechecks = self.run_with_oom_fact("tpe-refused", _known_failure=lambda self, params: False)
        self.assertEqual(campaign["status"], "completed", campaign.get("stop_reason"))
        self.assertNotEqual(campaign["stop_reason"], "没有可运行的候选")
        phases = [entry["phase"] for entry in campaign["state"]["phase_log"]]
        self.assertEqual(phases, ["explore", "exploit", "confirm", "done"])
        rejected = campaign["state"]["tpe"]["rejected"]
        self.assertEqual(len(rejected), 3, "three refusals use up the three candidates")
        self.assertEqual(len({(r["params"]["lr"], r["params"]["batch"]) for r in rejected}), 3)
        self.assertTrue(all("经验库" in r["reasons"][0] for r in rejected))
        self.assertEqual(sum(1 for e in prechecks if not e["passed"]), 3)
        self.assertEqual({t["phase"] for t in trials}, {"baseline"})

    def test_known_failures_never_reach_the_precheck(self):
        campaign, trials, prechecks = self.run_with_oom_fact("tpe-known")
        self.assertEqual(campaign["status"], "completed", campaign.get("stop_reason"))
        self.assertNotEqual(campaign["stop_reason"], "没有可运行的候选")
        self.assertIn("TPE 没有新的候选", [entry["note"] for entry in campaign["state"]["phase_log"]])
        self.assertTrue(all(e["passed"] for e in prechecks))
        self.assertEqual({t["phase"] for t in trials}, {"baseline"})


class TimeBlockSplitTest(unittest.TestCase):
    def test_formal_campaign_checks_the_actual_external_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(Path(tmp))
            try:
                for kind, test in (("random", list(range(30, 40))), ("time_block", [])):
                    path = h.repo / f"{kind}.json"
                    save_split(path, {"train": list(range(20)), "val": list(range(20, 30)), "test": test},
                               source_count=40, seed=42, kind=kind)
                    spec = h.spec(campaign_id=f"formal-{kind}", mode="formal", baseline={"seeds": [1, 2, 3]},
                                  fixed={"split_file": str(path), "split_kind": "time_block", "test_size": 10})
                    with self.assertRaisesRegex(CampaignError, "实际划分"):
                        h.service.create_campaign(spec)
                path = h.repo / "random.json"
                h.service.create_campaign(h.spec(campaign_id="change-mode", baseline={"seeds": [1, 2, 3]},
                                                 fixed={"split_file": str(path), "split_kind": "time_block",
                                                        "test_size": 10}))
                with self.assertRaisesRegex(CampaignError, "实际划分"):
                    h.service.update_campaign("change-mode", {"mode": "formal"})
                store = h.service.store("change-mode")
                legacy = store.load()
                legacy["spec"]["mode"] = "formal"
                store.save(legacy)
                with self.assertRaisesRegex(CampaignError, "实际划分"):
                    h.service.start_campaign("change-mode")
                store.update_state(lambda state: state.update(status="running"))
                with mock.patch.object(h.service.executor, "submit") as dispatched:
                    h.service._drive_safe("change-mode")
                dispatched.assert_not_called()
                self.assertEqual(h.service.get_campaign("change-mode")["status"], "failed")
            finally:
                h.close()

    def test_blocks_keep_neighbours_apart(self):
        frames = frame_numbers([f"{i}_0.1.jpg" for i in (5, 3, 1, 4, 2)] + ["x.jpg"])
        self.assertEqual(frames, [5, 3, 1, 4, 2, 5])
        split = build_time_block_split(list(range(2000)), val_size=200, test_size=200, block_size=40, gap=3, seed=7)
        self.assertGreaterEqual(len(split["val"]), 200)
        self.assertGreaterEqual(len(split["test"]), 200)
        held = set(split["val"]) | set(split["test"])
        self.assertGreaterEqual(min(abs(a - b) for a in split["train"] for b in held), 4)
        self.assertEqual(sorted(split["train"] + split["val"] + split["test"] + split["purged"]), list(range(2000)))
        self.assertGreaterEqual(min(abs(a - b) for a in split["val"] for b in split["test"]), 4)

    def test_held_out_sets_are_separated_and_the_legacy_assignment_is_reproducible(self):
        frames = list(range(1000))
        kwargs = {"val_size": 200, "test_size": 200, "block_size": 20, "gap": 4, "seed": 3}
        separated = build_time_block_split(frames, **kwargs)
        legacy = build_time_block_split(frames, **kwargs, separate_held_out=False)
        distance = lambda split: min(abs(a - b) for a in split["val"] for b in split["test"])
        self.assertGreater(distance(separated), 4)
        self.assertEqual(distance(legacy), 1, "this seed puts a validation block next to a test block")
        self.assertEqual(separated["test"], legacy["test"], "test blocks are the first blocks of the shuffle in both")
        for split in (separated, legacy):
            held = set(split["val"]) | set(split["test"])
            self.assertGreater(min(abs(a - b) for a in split["train"] for b in held), 4)
            self.assertEqual(sorted(sum((split[k] for k in ("train", "val", "test", "purged")), [])), frames)
        validate_time_split_gap(frames, separated, 4)
        validate_time_split_gap(frames, legacy, 4, separate_held_out=False)
        with self.assertRaisesRegex(ValueError, "val and test|test and val"):
            validate_time_split_gap(frames, legacy, 4)
        leaking = {**legacy, "train": [*legacy["train"], legacy["purged"][0]]}
        with self.assertRaisesRegex(ValueError, "train"):
            validate_time_split_gap(frames, leaking, 4, separate_held_out=False)
        # Blocks smaller than the gap: training rows two blocks away from a held-out block are purged too.
        small = build_time_block_split(list(range(200)), val_size=20, test_size=20, block_size=2, gap=5, seed=0)
        held = set(small["val"]) | set(small["test"])
        self.assertGreater(min(abs(a - b) for a in small["train"] for b in held), 5)
        with self.assertRaisesRegex(ValueError, "not enough separated blocks"):
            build_time_block_split(list(range(40)), val_size=20, test_size=20, block_size=10, gap=3, seed=0)
        for bad in ({"val_size": -1}, {"val_size": 1.5}, {"val_size": True}, {"block_size": 0}, {"gap": -1}):
            with self.assertRaises(ValueError, msg=bad):
                build_time_block_split(list(range(40)), **{"val_size": 4, **bad})

    def test_split_files_record_their_contract_and_legacy_files_stay_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            make_tub(repo, 200)
            splits = Path(tmp) / "splits"
            made = ensure_split(repo, splits, max_samples=200, val_size=40, split_seed=3, kind="time_block",
                                test_size=40, block_size=10, gap=2)
            self.assertEqual(made["contract"], TIME_BLOCK_SEPARATED)
            self.assertEqual(Path(made["split_file"]).name, "donkey-200-val40-test40-tb10g2c2-s3.json")
            self.assertEqual(load_split(made["split_file"])["contract"], TIME_BLOCK_SEPARATED)
            again = ensure_split(repo, splits, max_samples=200, val_size=40, split_seed=3, kind="time_block",
                                 test_size=40, block_size=10, gap=2)
            self.assertEqual(again["split_hash"], made["split_hash"], "an existing file is reused")

            # A file written before the contract existed (no ``contract``, old name, old assignment).
            index = load_index(repo, 200)
            frames = frame_numbers([rel for rel, *_ in index])
            old = build_time_block_split(frames, val_size=40, test_size=40, block_size=10, gap=2, seed=3,
                                         separate_held_out=False)
            purged = len(old.pop("purged"))
            path = splits / "donkey-200-val40-test40-tb10g2-s3.json"
            save_split(path, old, source_count=200, seed=3, kind="time_block",
                       meta={"block_size": 10, "gap": 2, "purged": purged})
            with self.assertRaises(ValueError):
                validate_time_split_gap(frames, old, 2)  # this seed has touching validation and test blocks
            described = describe_split(repo, str(path))
            self.assertEqual((described["contract"], described["val_count"], described["test_count"]),
                             (TIME_BLOCK_LEGACY, 40, 40))

            # A hand-made file that claims the separated contract is held to it, and a training row next
            # to a held-out row is refused under either contract.
            save_split(path, old, source_count=200, seed=3, kind="time_block",
                       meta={"block_size": 10, "gap": 2, "contract": TIME_BLOCK_SEPARATED})
            with self.assertRaisesRegex(ValueError, "regenerate"):
                describe_split(repo, str(path))
            order = sorted(range(200), key=lambda i: (frames[i], i))
            first_val = min(order.index(row) for row in old["val"])
            neighbour = order[first_val - 1] if first_val else order[max(order.index(r) for r in old["val"]) + 1]
            leaking = {**old, "train": sorted({*old["train"], neighbour} - set(old["val"]) - set(old["test"]))}
            if neighbour not in old["val"] and neighbour not in old["test"]:
                save_split(path, leaking, source_count=200, seed=3, kind="time_block", meta={"gap": 2})
                with self.assertRaisesRegex(ValueError, "train"):
                    describe_split(repo, str(path))
            save_split(path, old, source_count=200, seed=3, kind="time_block", meta={"gap": 2, "contract": 7})
            with self.assertRaisesRegex(ValueError, "contract"):
                describe_split(repo, str(path))

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
