import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apps.autopilot.agents.operator_opt import OperatorOptimization, classify_kernel
from apps.autopilot.agents.transfer import TransferLearning, transfer_prior
from apps.autopilot.agents.transfer_evidence import comparison_record, paired_estimates
from apps.autopilot.evidence import Evidence
from apps.autopilot.executor import Executor
from apps.autopilot.planner import Planner
from apps.autopilot.reporter import failure_table
from apps.autopilot.spec import baseline_params, default_spec, resolve_args
from apps.train.common.outcome import categorized_failure
from apps.train.transfer_probe import probe_indices
from tests.system.test_autopilot_m4 import write_matmul


class StageThreeSpecialists(unittest.TestCase):
    def test_failures_group_by_cause_without_changing_retry_kinds(self):
        for kind, category in (("thermal_throttle", "platform"), ("oom", "platform"), ("lock_timeout", "platform"),
                               ("paging", "platform"), ("reboot", "platform"), ("error", "code"), ("crash", "code"),
                               ("nan", "training"), ("diverged", "training"), ("invalid_args", "config")):
            self.assertEqual(categorized_failure({"kind": kind}), {"kind": kind, "category": category})
        missing = Executor._classify(None, None, None, False)["failure"]
        self.assertEqual((missing["kind"], missing["category"]), ("crash", "platform"))
        invalid = Executor._classify({"returncode": 2, "status": "failed"}, None, None, False)["failure"]
        attempts = [{"attempt": i+1, "failure": failure} for i, failure in enumerate(
                    (missing, {"kind": "error"}, {"kind": "nan"}, invalid))]
        row = failure_table([{"trial_id": "t1", "attempts": attempts}])[-1]
        self.assertEqual(row, "| t1/1 (crash) | t1/2 (error) | t1/3 (nan) | t1/4 (crash) |")

    def test_kernel_diagnosis_uses_both_percentages_and_surfaces_missing_data(self):
        for sm, mem, bound in ((25, 86, "访存受限"), (95, 65, "计算受限"), (90, 85, "两项接近"),
                               (30, 19, "两项利用率均低"), (None, 40, "缺测")):
            self.assertEqual(classify_kernel({"sm_throughput_pct": sm, "memory_throughput_pct": mem})[0], bound)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_matmul(root)
            agent = OperatorOptimization(repo_root=root)
            ev = Evidence(default_spec(), {}, [])
            found = {f.kind: f for f in agent.diagnose(ev)}
            self.assertEqual(found["kernel_limits"].data["rows"][0]["bound"], "访存受限")
            report = agent.report_section(ev, [f.to_dict() for f in found.values()])
            for label in ("合并访问", "占用率", "替代算法"):
                self.assertIn(label, report)

    def test_four_priors_and_candidate_budget(self):
        space = {"freeze_stages": [0, 2, 4, 5], "backbone_lr_mult": [.1, 1.]}
        for count, score, level, mult in ((100, .7, 5, .1), (100, .2, 2, .1),
                                          (5000, .7, 0, .1), (5000, .2, 0, 1.)):
            prior = transfer_prior({"train_samples": count, "score": score}, space)
            self.assertEqual(prior["candidate"], {"freeze_stages": level, "backbone_lr_mult": mult})
        self.assertIsNone(transfer_prior({"train_samples": 100, "score": None}, space))
        self.assertEqual(transfer_prior({"train_samples": 100, "score": .7}, {"freeze_stages": [1, 3]})["candidate"],
                         {"freeze_stages": 3})
        spec = default_spec()
        spec["specialists"] = ["transfer"]
        spec["fixed"]["pretrained"] = "imagenet"
        spec["space"].update(space)
        spec["strategy"]["max_candidates"] = 2
        params = baseline_params(spec, 1)
        self.assertTrue(resolve_args(spec, params, check_split=False)["transfer_probe"])
        self.assertNotIn("transfer_probe", resolve_args(spec, baseline_params(spec, 2), check_split=False))
        trial = {"trial_id": "t001", "phase": "baseline", "status": "done", "verdict": {"status": "valid"},
                 "params": params, "result": {"metrics": {"val_angle_mae": .1}}}
        ev = Evidence(spec, {"phase": "explore"}, [trial],
                      {"t001": {"transfer": {"probe": {"train_samples": 200, "score": .75}}}})
        with tempfile.TemporaryDirectory() as folder:
            findings = TransferLearning(folder).diagnose(ev)
        options = Planner(spec)._explore(ev, findings)
        self.assertEqual(len(options), 2)
        self.assertEqual((options[0].strategy, options[0].params["freeze_stages"]), ("transfer_prior", 5))
        self.assertEqual(trial["params"], params)

    def test_probe_draws_disjoint_temporal_blocks_from_supplied_training_rows(self):
        samples = [(f"{i:05d}_x.jpg", .1, 0.) for i in range(1000, 1400)]
        fit, hold = probe_indices(samples)
        self.assertEqual((len(fit), len(hold)), (96, 32))
        self.assertGreater(min(abs(i-j) for i in fit for j in hold), 2)
        self.assertEqual((fit, hold), tuple(probe_indices(samples)))

    def test_negative_transfer_requires_paired_seeds_same_budget_and_effect_outside_noise(self):
        def run(seed, pretrained, value, split="a", epochs=3):
            cfg = {"seed": seed, "pretrained": "imagenet" if pretrained else None, "compat": "torchvision",
                   "input_norm": "imagenet", "epochs": epochs, "lr": .001, "batch": 2}
            raw = {"samples": {"train": 16, "trained": 16*epochs, "expected_trained": 16*epochs},
                   "data": {"data_hash": "data", "val_hash": "val"}, "epochs": [{}]*epochs,
                   "metrics": {"val_angle_mae_final": value}, "transfer": {"load_report": {"npz_sha256": "w"}}}
            return comparison_record(raw, cfg, split_hash=split, code="same", source=f"s{seed}-{pretrained}")
        control = [run(s, False, .1) for s in (1, 2, 3)]
        worse = [run(s, True, v) for s, v in enumerate((.2, .21, .19), 1)]
        pair, = paired_estimates(control+worse)
        self.assertTrue(pair["negative"])
        noisy, = paired_estimates(control+[run(s, True, v) for s, v in enumerate((.01, .16, .2), 1)])
        self.assertFalse(noisy["negative"])
        self.assertEqual(paired_estimates(control + worse[:2] + [run(4, True, .2), run(3, True, .3, split="b"),
                                                               run(3, True, .3, epochs=2)]), [])
        with patch("apps.autopilot.agents.transfer_evidence.matching_records", return_value=control+worse):
            findings = TransferLearning().diagnose(Evidence(default_spec(), {}, []))
        self.assertEqual(findings[0].kind, "negative_transfer")
        self.assertEqual(len(findings[0].evidence), 6)
