"""M6 (after E2): the Transfer Learning specialist and the freeze_stages search.

freeze_stages enters the search space only after the capability check (fixed.pretrained set and the converted
weights present); candidates and TPE search it; the Verifier rejects it elsewhere and invalidates a run whose
frozen tensors changed; the specialist reports the load, compares frozen levels and proposes the neighbouring
levels around the best trial; a campaign with the fake trainer runs the whole search.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apps.autopilot.agents.transfer import TransferLearning
from apps.autopilot.evidence import Evidence
from apps.autopilot.spec import SpecError, default_spec, in_space, resolve_args, unavailable_keys, validate_spec
from apps.autopilot.strategies.candidate_list import CandidateList
from apps.autopilot.strategies.tpe import TPEStrategy
from apps.autopilot.verifier import postcheck, precheck
from tests.system.harness import Harness, make_tub


def converted(repo: Path, arch: str = "resnet18") -> Path:
    path = repo / "runs" / "pretrained" / "converted" / f"{arch}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stand-in")
    return path


def transfer_spec(**space) -> dict:
    return {"campaign_id": "m6-unit", "fixed": {"max_samples": 40, "epochs": 1, "val_size": 10, "device": "cpu",
                                                "conv_backend": "numpy", "pretrained": "imagenet", "arch": "resnet18"},
            "space": {"lr": [0.001], "batch": [2], **space}, "strategy": {"max_candidates": 2},
            "llm": {"enabled": False}}


class CapabilityTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        make_tub(self.repo, count=4)

    def tearDown(self):
        self._tmp.cleanup()

    def test_freeze_stages_needs_pretrained_weights(self):
        raw = transfer_spec(freeze_stages=[0, 2])
        without = {**raw, "fixed": {k: v for k, v in raw["fixed"].items() if k not in ("pretrained", "arch")}}
        with self.assertRaisesRegex(SpecError, "fixed.pretrained"):
            validate_spec(without, self.repo)
        with self.assertRaisesRegex(SpecError, "预训练权重不存在"):
            validate_spec(raw, self.repo)
        converted(self.repo)
        spec = validate_spec(raw, self.repo)
        self.assertEqual(spec["space"]["freeze_stages"], [0, 2])
        self.assertEqual(unavailable_keys(spec, self.repo), {})
        plain = validate_spec({**without, "space": {"lr": [0.001], "batch": [2]}, "strategy": {"max_candidates": 1}},
                              self.repo)
        self.assertIn("freeze_stages", unavailable_keys(plain, self.repo))

    def test_range_and_fixed_conflicts(self):
        converted(self.repo)
        with self.assertRaisesRegex(SpecError, "0 到 5"):
            validate_spec(transfer_spec(freeze_stages=[0, 6]), self.repo)
        raw = transfer_spec(freeze_stages=[0, 2])
        raw["fixed"]["freeze_stages"] = 1
        with self.assertRaisesRegex(SpecError, "同时出现"):
            validate_spec(raw, self.repo)

    def test_zero_is_the_baseline_fingerprint(self):
        converted(self.repo)
        spec = validate_spec(transfer_spec(freeze_stages=[0, 2]), self.repo)
        spec["fixed"]["split_file"] = "split.json"
        base = resolve_args(spec, {"lr": 0.001, "batch": 2, "seed": 1})
        self.assertEqual(resolve_args(spec, {"lr": 0.001, "batch": 2, "seed": 1, "freeze_stages": 0}), base)
        self.assertEqual(resolve_args(spec, {"lr": 0.001, "batch": 2, "seed": 1, "freeze_stages": 2})["freeze_stages"], 2)

    def test_candidates_and_tpe_search_freeze_stages(self):
        converted(self.repo)
        spec = validate_spec(transfer_spec(freeze_stages=[0, 1, 3, 5], **{"lr": [0.0001, 0.001]}), self.repo)
        spec["strategy"]["max_candidates"] = 4
        baseline = {"lr": 0.001, "batch": 2, "num_workers": 0, "seed": 1}
        points = [p.delta for p in CandidateList().proposals(spec, exclude=[baseline])]
        self.assertEqual(len(points), 4)
        self.assertGreaterEqual(len({p["freeze_stages"] for p in points}), 3)
        self.assertNotIn({"lr": 0.001, "batch": 2, "freeze_stages": 0}, points)  # the baseline (0 = default)
        spec["strategy"]["name"] = "tpe"
        tpe = TPEStrategy(spec)
        trials = [{"trial_id": "t001", "phase": "baseline", "status": "done", "params": baseline,
                   "verdict": {"status": "valid"}, "result": {"metrics": {"val_angle_mae": 0.2}}}]
        for i in range(5):
            proposal = tpe.ask(Evidence(spec=spec, state={}, trials=trials))
            self.assertIn(proposal.delta["freeze_stages"], (0, 1, 3, 5))
            self.assertTrue(all(in_space(spec, k, v) for k, v in proposal.delta.items()))
            trials.append({"trial_id": f"t{i + 2:03d}", "phase": "explore", "status": "done",
                           "params": {**proposal.delta, "num_workers": 0, "seed": 1}, "verdict": {"status": "valid"},
                           "result": {"metrics": {"val_angle_mae": 0.1 + 0.01 * i}}})

    def test_precheck_rejects_freeze_stages_without_the_capability(self):
        plain = validate_spec({"campaign_id": "m6-plain", "fixed": {"max_samples": 40, "epochs": 1, "val_size": 10,
                                                                    "device": "cpu", "conv_backend": "numpy"},
                               "space": {"lr": [0.001], "batch": [2]}, "strategy": {"max_candidates": 1}}, self.repo)
        plain["fixed"]["split_file"] = "split.json"
        check = precheck({"lr": 0.001, "batch": 2, "seed": 1, "freeze_stages": 2}, spec=plain, repo_root=self.repo,
                         knowledge=None, trials=[], fingerprint=None, delta={"freeze_stages": 2})
        self.assertFalse(check["passed"])
        self.assertTrue(any("能力不可用" in r for r in check["reasons"]))

    def test_postcheck_invalidates_changed_frozen_tensors(self):
        spec = default_spec("dev")
        result = {"status": "succeeded", "metrics": {"val_angle_mae": 0.1, "val_loss": 0.2},
                  "transfer": {"frozen_unchanged": False, "load_report": {"missing": []}}}
        verdict = postcheck({"resolved_args": {}}, {"status": "succeeded"}, result, spec=spec, split={})
        self.assertEqual(verdict["status"], "invalid")
        self.assertIn("冻结", verdict["reasons"][0])


def trial(tid, freeze, value, *, phase="explore", lr=0.001, batch=2, epochs=None):
    params = {"lr": lr, "batch": batch, "num_workers": 0, "seed": 1}
    if freeze:
        params["freeze_stages"] = freeze
    return {"trial_id": tid, "phase": phase, "status": "done", "params": params, "verdict": {"status": "valid"},
            "result": {"metrics": {"val_angle_mae": value}, "epochs": epochs or []}}


def summary(freeze):
    return {"run_dir": "r", "sources": {"transfer": "result.json#transfer", "result": "result.json"},
            "step_ms_median": 100.0 - 5 * freeze,
            "transfer": {"arch": "resnet18", "freeze_stages": freeze, "frozen": ["stem", "layer1", "layer2"][:freeze],
                         "trainable_values": 11_000_000 - freeze, "frozen_unchanged": True, "loaded": 102,
                         "expected": 102, "success_rate": 1.0, "missing": 0, "replaced_head": ["fc.weight", "fc.bias"],
                         "source_sha256": "f37072fd47e8", "npz_sha256": "ed6abcb7", "input_norm": "imagenet"}}


class SpecialistTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        make_tub(self.repo, count=4)
        converted(self.repo)
        self.spec = validate_spec(transfer_spec(freeze_stages=[0, 1, 2, 3, 4]), self.repo)

    def tearDown(self):
        self._tmp.cleanup()

    def evidence(self, trials):
        return Evidence(spec=self.spec, state={}, trials=trials,
                        summaries={t["trial_id"]: summary(t["params"].get("freeze_stages", 0)) for t in trials})

    def test_available_follows_the_capability(self):
        agent = TransferLearning(repo_root=self.repo)
        self.assertEqual(agent.available(None), (True, None))
        self.assertTrue(agent.available(self.evidence([]))[0])
        plain = Evidence(spec=default_spec("dev"), state={}, trials=[])
        ok, why = agent.available(plain)
        self.assertFalse(ok)
        self.assertIn("fixed.pretrained", why)

    def test_compares_levels_and_proposes_the_neighbours_of_the_best(self):
        agent = TransferLearning(repo_root=self.repo)
        ev = self.evidence([trial("t001", 0, 0.20, phase="baseline"), trial("t002", 2, 0.15)])
        findings = {f.kind: f for f in agent.diagnose(ev)}
        self.assertEqual(findings["pretrained_load"].data["loaded"], 102)
        comparison = findings["freeze_comparison"]
        self.assertEqual(comparison.severity, "action")
        self.assertEqual(comparison.data["best_level"], 2)
        self.assertEqual(comparison.data["neighbours"], [1, 3])  # not over-fitted: fewer frozen first
        proposals = agent.propose(list(findings.values()), ev)
        self.assertEqual([p.delta for p in proposals], [{"freeze_stages": 1}, {"freeze_stages": 3}])
        self.assertTrue(all(p.base_trial == "t002" for p in proposals))
        text = agent.report_section(ev, [f.to_dict() for f in findings.values()])
        self.assertIn("| 2 | stem, layer1 |", text)
        self.assertIn("102/102", text)

    def test_overfitting_tries_freezing_more_first(self):
        agent = TransferLearning(repo_root=self.repo)
        epochs = [{"epoch": 1, "val_loss": 0.10}, {"epoch": 2, "val_loss": 0.13}]
        ev = self.evidence([trial("t001", 0, 0.20, phase="baseline"), trial("t002", 2, 0.15, epochs=epochs)])
        comparison = next(f for f in agent.diagnose(ev) if f.kind == "freeze_comparison")
        self.assertEqual(comparison.data["neighbours"], [3, 1])
        self.assertIn("过拟合", comparison.message)

    def test_search_done_when_the_neighbours_were_tried(self):
        agent = TransferLearning(repo_root=self.repo)
        ev = self.evidence([trial("t001", 0, 0.20, phase="baseline"), trial("t002", 1, 0.18),
                            trial("t003", 2, 0.15), trial("t004", 3, 0.17)])
        comparison = next(f for f in agent.diagnose(ev) if f.kind == "freeze_comparison")
        self.assertEqual(comparison.severity, "info")
        self.assertEqual(agent.propose([comparison], ev), [])


class CodeChangeReportTest(unittest.TestCase):
    def test_attempts_launched_with_other_code_are_listed(self):
        from apps.autopilot.reporter import code_changes

        trials = [{"trial_id": "t001", "code_hash": "aaa", "attempts": [{"attempt": 1, "code_hash": "aaa"}]},
                  {"trial_id": "t002", "code_hash": "aaa", "attempts": [{"attempt": 1, "code_hash": "bbb"},
                                                                        {"attempt": 2, "code_hash": None}]}]
        ev = Evidence(spec=default_spec("dev"), state={}, trials=trials)
        self.assertEqual(code_changes(ev, "aaa"), ["t002#1"])


class TransferCampaignTest(unittest.TestCase):
    def test_the_freeze_stage_search_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(Path(tmp), {"objective": {"freeze": {"0": 0.04, "1": 0.02, "2": 0.01, "3": 0.0, "4": 0.03}}},
                        start=False)
            converted(h.repo)
            h.start()
            try:
                campaign = h.run(campaign_id="m6-campaign", timeout=240,
                                 fixed={"max_samples": 40, "epochs": 1, "val_size": 10, "pretrained": "imagenet",
                                        "arch": "resnet18"},
                                 space={"lr": [0.001], "batch": [2], "freeze_stages": [0, 1, 2, 3, 4]},
                                 strategy={"max_candidates": 2, "seed": 0},
                                 specialists=["benchmark", "transfer"], budget={"max_trials": 12})
                self.assertEqual(campaign["status"], "completed", campaign.get("stop_reason"))
                trials = h.service.trials("m6-campaign")
                levels = {t["params"].get("freeze_stages", 0) for t in trials if t.get("params")}
                self.assertGreaterEqual(len(levels), 3)
                derived = [t for t in trials if (t.get("origin") or {}).get("agent") == "transfer"]
                self.assertTrue(derived, "the specialist proposed no freeze level")
                self.assertTrue(all("freeze_stages" in (t.get("resolved_args") or {}) for t in derived))
                best = min((t for t in trials if (t.get("verdict") or {}).get("status") == "valid"),
                           key=lambda t: t["result"]["metrics"]["val_angle_mae"])
                self.assertEqual(best["params"].get("freeze_stages"), 3)
                report = h.service.report("m6-campaign")["markdown"]
                self.assertIn("### Transfer Learning", report)
                self.assertIn("冻结阶段 |", report)
            finally:
                h.close()


if __name__ == "__main__":
    unittest.main()
