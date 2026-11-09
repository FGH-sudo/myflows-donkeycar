"""Closed-loop metrics and termination rules run without Unity or a trained model."""

import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from apps.eval.closed_loop import EpisodeMetrics, aggregate, episode, model_manifest
from apps.eval.sim_probe import summarize


RULES = {"cte_threshold": 2., "rearm_threshold": 1., "consecutive_frames": 3, "rearm_frames": 2, "route_signal": None}


class FakeStream:
    def __init__(self, ctes):
        self.ctes = iter(ctes)
        self.actions = []
        self.braked = False

    def reset(self, *, seed):
        self.seed = seed
        return np.zeros((2, 2, 3), np.uint8), {"pos": [0., 0., 0.], "lap_count": 0}

    def step(self, action):
        self.actions.append(action)
        cte, hit = next(self.ctes)
        i = len(self.actions)
        return np.zeros((2, 2, 3), np.uint8), dict(frame_id=i, monotonic_s=i, unix_s=100+i, pos=[3*i, 4*i, 0],
                                                   cte=cte, speed=5, hit=hit, lap_count=0, last_lap_time=0, line_events=0)

    def brake(self):
        self.braked = True


class ClosedLoopTest(unittest.TestCase):
    def test_offline_score_follows_the_weights_when_a_deployment_model_is_replaced(self):
        weights = {"path": "current.onnx", "offline_angle_mae": .032, "offline_weights_sha256": "archived"}
        with patch("apps.eval.closed_loop._fingerprints", return_value={"current.onnx":"archived"}):
            self.assertEqual(model_manifest(weights)["offline_angle_mae"], .032)
        with patch("apps.eval.closed_loop._fingerprints", return_value={"current.onnx":"different"}):
            self.assertIsNone(model_manifest(weights)["offline_angle_mae"])
    def test_episode_debounces_excursions_brakes_on_collision_and_logs_applied_actions(self):
        ctes = [2.1, 2.2, 2.3, 2.4, 1.5, 2.3, .5, .5, -2.1, -2.2, -2.3]
        stream = FakeStream([(v, "none") for v in ctes[:-1]] + [(ctes[-1], "wall")])
        log = io.StringIO()
        result = episode(stream, lambda rgb: 4., {"id": "s1", "seed": 17, "duration_s": 10, "start": "simulator_default"},
                         RULES, log, model_id="test")
        self.assertEqual((result["termination"], result["excursions"], result["distance_units"]), ("collision", 2, 55.))
        self.assertEqual(result["max_abs_cte"], 2.4)
        self.assertAlmostEqual(result["mean_abs_cte"], np.mean(np.abs(ctes)))
        self.assertEqual(stream.actions, [(1., .2)] * 11)
        self.assertTrue(stream.braked)
        self.assertEqual(stream.seed, 17)
        self.assertEqual(len([json.loads(line) for line in log.getvalue().splitlines()]), 11)

    def test_missing_frame_breaks_streak_and_missing_positions_do_not_bridge_distance(self):
        metrics = EpisodeMetrics(RULES)
        for cte, pos in [(3., [0, 0, 0]), (3., [1, 0, 0]), (None, None), (3., [100, 0, 0])]:
            metrics.add({"pos": pos, "cte": cte, "inference_ms": 2})
        result = metrics.summary(1., "time_limit")
        self.assertEqual((result["excursions"], result["distance_units"], result["cte_frames"]), (0, 1., 3))
        empty = EpisodeMetrics(RULES).summary(0., "time_limit")
        self.assertIsNone(empty["mean_abs_cte"])
        self.assertIsNone(empty["distance_units"])

    def test_paired_deltas_use_the_scenario_id_and_worst_direction(self):
        def row(model, scenario, cte, survival):
            metrics = EpisodeMetrics(RULES)
            metrics.add({"pos": [0, 0, 0], "cte": cte, "inference_ms": 10})
            return dict(model=model, scenario=scenario, **metrics.summary(survival, "time_limit"))
        rows = [row("a", "one", .4, 4), row("b", "two", 2, 2), row("a", "two", 1, 3), row("b", "one", .2, 5)]
        summary = aggregate(rows, "a")["b"]
        self.assertAlmostEqual(summary["paired_difference_from_baseline"]["mean_abs_cte"]["mean"], .4)
        self.assertEqual(summary["metrics"]["survival_s"]["worst"], 2)
        self.assertEqual(summary["metrics"]["mean_abs_cte"]["worst"], 2)
        with self.assertRaises(ValueError):
            aggregate(rows + [rows[0]], "a")

    def test_probe_distinguishes_missing_wire_fields_from_observed_zero(self):
        rows = [dict(monotonic_s=0, wire_fields=["speed", "image"], speed=0, cte=None, line_events=0),
                dict(monotonic_s=.2, wire_fields=["speed", "cte", "image"], speed=0, cte=0, line_events=0)]
        result = summarize(rows)
        self.assertEqual(result["wire_field_counts"]["cte"], 1)
        self.assertEqual(result["fields"]["speed"]["min"], 0)
        self.assertEqual(result["fields"]["cte"]["available_share"], .5)
        self.assertNotIn("min", result["fields"]["pos"])
