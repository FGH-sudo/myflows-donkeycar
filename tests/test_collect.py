"""Unattended collection without Unity: the data set registry, the tub writer, the expert and the episode loop."""

import io
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from apps.collect.donkey_auto import DonkeyPilot, RoadPerEpisode
from apps.collect.policy import CteExpert, Perturbation, SpeedHold
from apps.collect.recorder import EXTRA_FIELDS, Limits, collect, record_episode, summarize
from apps.common.datasets import (COLLECTED_ROOT, DEFAULT_DATA, DatasetError, TubWriter, dataset_key, describe,
                                  label_summary, list_datasets, normalize)
from apps.common.donkey_data import load_donkey_index
from apps.common.splits import frame_numbers


def picture(value=90):
    return np.full((12, 16, 3), value, np.uint8)


class Clock:
    def __init__(self, step=.05):
        self.now, self.step = 0., step

    def __call__(self):
        return self.now


class LaneStream:
    """A car whose cross-track error grows with the applied steering, as DonkeySim reports it."""

    def __init__(self, clock, *, gain=.3, hit_at=None, nodes=None, drift=0.):
        self.clock, self.gain, self.hit_at, self.nodes, self.drift = clock, gain, hit_at, nodes, drift
        self.cte = 0.
        self.actions, self.seeds, self.roads = [], [], []
        self.braked = 0

    def _frame(self):
        frame = len(self.actions)
        info = {"cte": self.cte, "speed": 5., "hit": "wall" if self.hit_at is not None and frame >= self.hit_at else "none"}
        if self.nodes:
            info.update(active_node=frame, total_nodes=self.nodes)
        return picture(40 + frame % 100), info

    def reset(self, *, seed):
        self.seeds.append(seed)
        self.cte = 0.
        self.actions = []
        return self._frame()

    def step(self, action):
        self.actions.append(action)
        self.cte += self.gain * action[0] + self.drift
        self.clock.now += self.clock.step
        return self._frame()

    def brake(self):
        self.braked += 1

    def regenerate_road(self, seed):
        self.roads.append(seed)


class DatasetRegistryTest(unittest.TestCase):
    def test_only_the_default_set_and_collected_ids_are_accepted(self):
        self.assertEqual(normalize(None), DEFAULT_DATA)
        self.assertEqual(normalize("mycar/data/"), DEFAULT_DATA)
        self.assertEqual(normalize("udacity-lake-001"), f"{COLLECTED_ROOT}/udacity-lake-001")
        self.assertEqual(normalize("runs\\datasets\\udacity-lake-001"), f"{COLLECTED_ROOT}/udacity-lake-001")
        self.assertEqual((dataset_key(None), dataset_key("runs/datasets/lake_2")), ("donkey", "lake_2"))
        for bad in ("../secrets", "runs/datasets/../../x", "C:/data", "/etc", "runs/other/abc", "Upper-Case", "ab",
                    "runs/datasets/a/b", "mycar/data/../x", 5, "a b"):
            with self.assertRaises(DatasetError, msg=bad):
                normalize(bad)

    def test_the_writer_produces_a_tub_the_training_loader_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / COLLECTED_ROOT / "set-one"
            with TubWriter(directory, meta={"id": "set-one", "environment": "test-env", "simulator": "none"},
                           extra_fields=EXTRA_FIELDS) as writer:
                names = [writer.add(picture(60 + i), angle, .2, **{"sim/error": .1 * i, "collect/episode": 0})
                         for i, angle in enumerate((-.25, 0., .5))]
                with self.assertRaises(DatasetError):
                    writer.add(picture(), float("nan"), .2)
                with self.assertRaises(DatasetError):
                    writer.add(np.zeros((12, 16), np.uint8), 0., .2)
                with self.assertRaises(DatasetError):
                    writer.add(np.zeros((6, 16, 3), np.uint8), 0., .2)
                with self.assertRaises(DatasetError):
                    writer.add(picture(), 0., .2, undeclared=1)
                meta = writer.close(summary={"accepted": True})
            self.assertEqual(names, ["0_-0.2500.jpg", "1_0.0000.jpg", "2_0.5000.jpg"])
            index = load_donkey_index(directory, fixed_throttle=.5, angle_scale=1.)
            self.assertEqual([(rel.as_posix(), angle, throttle) for rel, angle, throttle in index],
                             [("images/0_-0.2500.jpg", -.25, .2), ("images/1_0.0000.jpg", 0., .2), ("images/2_0.5000.jpg", .5, .2)])
            self.assertEqual(frame_numbers([rel for rel, *_ in index]), [0, 1, 2])
            self.assertEqual((meta["frames"], meta["image"], meta["labels"]["frames"]), (3, {"width": 16, "height": 12}, 3))
            manifest = [json.loads(line) for line in (directory / "manifest.json").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(manifest[0][:4], ["cam/image_array", "user/angle", "user/throttle", "user/mode"])
            self.assertEqual(manifest[4]["current_index"], 3)
            # A second collector must not write into the same set.
            with self.assertRaises(DatasetError):
                TubWriter(directory, meta={})
            listed = list_datasets(root)
            self.assertEqual([row["data"] for row in listed], [DEFAULT_DATA, f"{COLLECTED_ROOT}/set-one"])
            self.assertEqual((listed[0]["ready"], listed[1]["ready"], listed[1]["environment"]), (False, True, "test-env"))
            self.assertEqual(describe(root, "set-one")["frames"], 3)

    def test_label_summary_shows_a_one_sided_set(self):
        summary = label_summary([-.5, -.4, -.3, 0., 0.])
        self.assertEqual((summary["left_share"], summary["right_share"], summary["straight_share"]), (.6, 0., .4))
        self.assertEqual(sum(summary["histogram"]["counts"]), 5)
        self.assertEqual(label_summary([]), {"frames": 0})


class PolicyTest(unittest.TestCase):
    def test_the_expert_steers_against_the_error_and_its_rate(self):
        expert = CteExpert(.5, .2, rate_smoothing=1.)
        self.assertAlmostEqual(expert(1., 0.), -.5)
        self.assertAlmostEqual(expert(1.5, .5), -(.5 * 1.5 + .2 * 1.))  # the error grows by 1 per second
        self.assertEqual(expert(10., 1.), -1.)
        expert.reset()
        self.assertAlmostEqual(expert(-1., 5.), .5)
        with self.assertRaises(ValueError):
            CteExpert(0., .1)

    def test_perturbations_are_reproducible_bounded_pulses(self):
        def run(seed):
            pulses = Perturbation(random.Random(seed), interval_s=(1., 2.), frames=(2, 4), magnitude=(.3, .6))
            pulses.reset(0.)
            return [pulses.offset(i * .05) for i in range(600)], pulses.pulses
        values, count = run(4)
        self.assertEqual(run(4), (values, count))
        self.assertNotEqual(run(5)[0], values)
        active = [v for v in values if v]
        self.assertTrue(all(.3 <= abs(v) <= .6 for v in active))
        self.assertTrue(any(v > 0 for v in active) and any(v < 0 for v in active))
        self.assertTrue(10 <= count <= 30)  # 30 s with a pulse every 1 to 2 s
        self.assertTrue(2 * count <= len(active) <= 4 * count)
        off = Perturbation(random.Random(1), enabled=False)
        self.assertEqual({off.offset(i * 1.) for i in range(50)}, {0.})

    def test_a_held_perturbation_ends_its_pulse_and_waits_the_shortest_interval(self):
        pulses = Perturbation(random.Random(3), interval_s=(1., 2.), frames=(6, 6), magnitude=(.3, .6))
        pulses.reset(0.)
        now = 0.
        while not pulses.offset(now):  # run to the first pulse
            now += .05
        pulses.hold(now)
        self.assertEqual(pulses.offset(now + .05), 0.)  # the pulse under way ended
        quiet = [pulses.offset(now + .05 * i) for i in range(2, 20)]  # less than the shortest interval later
        self.assertEqual(set(quiet), {0.})
        self.assertTrue(any(pulses.offset(now + 1. + .05 * i) for i in range(60)))  # and pulses resume

    def test_speed_hold_pushes_towards_the_target_without_windup(self):
        hold = SpeedHold(10.)
        self.assertEqual(hold(0., 0.), 1.)
        for i in range(1, 400):  # standing still for a long time
            hold(0., i * .05)
        self.assertLessEqual(hold.integral, 20.)
        self.assertLess(hold(14., 20.05), 1.)
        self.assertLess(SpeedHold(10.)(20., 0.), 0.)


class EpisodeLoopTest(unittest.TestCase):
    def pilot(self, **overrides):
        settings = dict(throttle=.2, fail_cte=4., fail_frames=3, label_cte=3., warmup_frames=2)
        settings.update(overrides)
        perturbation = settings.pop("perturbation", Perturbation(random.Random(0), enabled=False))
        return DonkeyPilot(CteExpert(.5, 0.), perturbation, **settings)

    def collect(self, stream, pilot, clock, *, frames=1000, episode_s=1., mirror=False):
        with tempfile.TemporaryDirectory() as tmp:
            log = io.StringIO()
            with TubWriter(Path(tmp) / "set", meta={}, extra_fields=EXTRA_FIELDS) as writer:
                row = record_episode(stream, writer, pilot, index=3, seed=17, log=log, clock=clock, mirror=mirror,
                                     limits=Limits(frames=frames, episode_s=episode_s, max_episodes=1, wall_s=60.))
                writer.close()
                index = load_donkey_index(Path(tmp) / "set", fixed_throttle=.5, angle_scale=1.) if writer.count else []
                catalog = [json.loads(line) for line in (Path(tmp) / "set" / "catalog_generated.catalog").read_text().splitlines()]
            return row, index, catalog, [json.loads(line) for line in log.getvalue().splitlines()]

    def test_labels_are_the_experts_while_the_disturbed_steering_is_applied(self):
        clock = Clock()

        class Always:  # a disturbance on every frame
            pulses = 0

            def reset(self, now):
                pass

            def offset(self, now):
                return .3
        stream = LaneStream(clock)
        row, index, catalog, log = self.collect(stream, self.pilot(perturbation=Always()), clock)
        self.assertEqual(row["termination"], "time_limit")
        self.assertEqual((row["seed"], stream.seeds, stream.braked), (17, [17], 1))
        # The first two frames after the reset are warm-up: driven without a disturbance and not stored.
        self.assertEqual([a[0] for a in stream.actions[:2]], [0., 0.])
        self.assertEqual(row["frames_stored"], row["frames_seen"] - 2)
        for record, frame in zip(catalog, log):
            self.assertAlmostEqual(record["collect/applied"], min(1., record["user/angle"] + .3), places=5)
            self.assertAlmostEqual(record["user/angle"], -.5 * record["sim/error"], places=4)  # the expert's own answer
            self.assertEqual((record["collect/episode"], record["user/throttle"], record["user/mode"]), (3, .2, "expert"))
            self.assertEqual(frame["image"], record["cam/image_array"])
        self.assertEqual(row["perturbed_frames"], row["frames_stored"])
        self.assertTrue(all(action[1] == .2 for action in stream.actions))

    def test_an_episode_ends_off_the_track_on_a_hit_at_the_road_end_and_at_the_frame_target(self):
        clock = Clock()
        # A side slip the expert cannot hold: the error passes the label limit, then the failure limit.
        row, index, *_ = self.collect(LaneStream(clock, drift=.5), self.pilot(fail_cte=2., label_cte=1.), clock, episode_s=60.)
        self.assertEqual(row["termination"], "off_track")
        self.assertTrue(row["max_abs_error"] > 2.)
        self.assertTrue(len(index) < row["frames_seen"] - 2)  # frames beyond the label limit were dropped
        row, *_ = self.collect(LaneStream(clock, hit_at=5), self.pilot(), clock, episode_s=60.)
        self.assertEqual((row["termination"], row["frames_seen"]), ("collision", 5))
        row, *_ = self.collect(LaneStream(clock, nodes=20), self.pilot(), clock, episode_s=60.)
        self.assertEqual((row["termination"], row["frames_seen"]), ("road_end", 18))
        row, index, *_ = self.collect(LaneStream(clock), self.pilot(), clock, frames=7, episode_s=60.)
        self.assertEqual((row["termination"], len(index)), ("frame_target", 7))
        missing = LaneStream(clock)
        missing._frame = lambda: (picture(), {"cte": None, "speed": 1., "hit": "none"})
        row, *_ = self.collect(missing, self.pilot(), clock)
        self.assertEqual(row["termination"], "telemetry_missing")

    def test_mirrored_twins_negate_every_signed_field(self):
        clock = Clock()
        stream = LaneStream(clock)
        stream.cte = 1.

        def reset(*, seed):
            stream.actions = []
            return stream._frame()
        stream.reset = reset
        row, index, catalog, _ = self.collect(stream, self.pilot(warmup_frames=0), clock, frames=6, episode_s=60., mirror=True)
        self.assertEqual(len(catalog), 6)
        for original, twin in zip(catalog[0::2], catalog[1::2]):
            self.assertEqual((original["collect/mirrored"], twin["collect/mirrored"]), (False, True))
            for key in ("user/angle", "sim/error", "collect/applied"):
                self.assertEqual(original[key], -twin[key])
            self.assertEqual(original["collect/t"], twin["collect/t"])

    def test_an_odd_frame_target_stops_before_another_control_is_sent(self):
        clock = Clock()
        stream = LaneStream(clock)
        row, index, *_ = self.collect(stream, self.pilot(warmup_frames=0), clock, frames=3,
                                      episode_s=60., mirror=True)
        self.assertEqual((row["termination"], len(index)), ("frame_target", 3))
        self.assertEqual(len(stream.actions), 1)  # only advance to the second picture

    def test_the_collection_deadline_also_limits_a_running_episode_and_its_reset(self):
        for reset_s in (0., 2.):
            with self.subTest(reset_s=reset_s):
                clock = Clock()
                stream = LaneStream(clock)
                reset = stream.reset

                def slow_reset(*, seed):
                    clock.now += reset_s
                    return reset(seed=seed)

                stream.reset = slow_reset
                writer = TubWriterStub()
                rows, ended_early = collect(stream, writer, self.pilot(), seed=1, clock=clock,
                                            limits=Limits(frames=1000, episode_s=60., max_episodes=10, wall_s=.5))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["termination"], "wall_limit")
                self.assertLessEqual(len(stream.actions), 11)
                self.assertIn("wall", ended_early)
                self.assertEqual(stream.braked, 1)

    def test_collect_runs_episodes_with_a_new_road_and_seed_each_until_the_target(self):
        clock = Clock()
        stream = LaneStream(clock)
        seen = []
        with tempfile.TemporaryDirectory() as tmp, TubWriter(Path(tmp) / "set", meta={}, extra_fields=EXTRA_FIELDS) as writer:
            episodes, ended_early = collect(RoadPerEpisode(stream), writer, self.pilot(), seed=40, clock=clock,
                                            on_episode=seen.append,
                                            limits=Limits(frames=45, episode_s=1., max_episodes=10, wall_s=600.))
            self.assertEqual(writer.count, 45)
        self.assertIsNone(ended_early)
        self.assertEqual(stream.roads, stream.seeds)
        self.assertEqual(stream.seeds, [40, 41, 42][:len(episodes)])
        self.assertEqual([row["episode"] for row in episodes], list(range(len(episodes))))
        self.assertEqual(seen, episodes)
        self.assertEqual(episodes[-1]["termination"], "frame_target")
        capped, _ = collect(stream, TubWriterStub(), self.pilot(), seed=1, clock=clock,
                            limits=Limits(frames=10 ** 6, episode_s=.5, max_episodes=2, wall_s=600.))
        self.assertEqual(len(capped), 2)

    def test_a_simulator_failure_ends_collection_and_keeps_the_frames(self):
        clock = Clock()
        limits = Limits(frames=10 ** 6, episode_s=1., max_episodes=10, wall_s=600.)
        # The scene cannot be restarted for the second episode (on the real simulator: a minimised window).
        stream = LaneStream(clock)
        first_reset = stream.reset

        def reset(*, seed):
            if stream.seeds:
                raise RuntimeError("the simulator window has no client area")
            return first_reset(seed=seed)
        stream.reset = reset
        writer = TubWriterStub()
        episodes, ended_early = collect(stream, writer, self.pilot(), seed=5, clock=clock, limits=limits)
        self.assertEqual([row["termination"] for row in episodes], ["time_limit"])
        self.assertGreater(writer.count, 0)
        self.assertIn("episode 1 could not start: RuntimeError: the simulator window has no client area", ended_early)
        # The simulator stops answering in the middle of an episode: the episode ends there, its frames stay.
        stream = LaneStream(clock)
        first_step = stream.step

        def step(action):
            if len(stream.actions) >= 8:
                raise TimeoutError("no frame for 10 s")
            return first_step(action)
        stream.step = step
        writer = TubWriterStub()
        episodes, ended_early = collect(stream, writer, self.pilot(), seed=5, clock=clock, limits=limits)
        self.assertEqual(len(episodes), 1)
        self.assertEqual((episodes[0]["termination"], episodes[0]["error"]), ("simulator_error", "TimeoutError: no frame for 10 s"))
        self.assertEqual(episodes[0]["frames_stored"], writer.count)
        self.assertEqual(stream.braked, 1)
        summary = summarize(episodes, label_summary([.1] * writer.count), target_frames=100, ended_early=ended_early)
        self.assertEqual(summary["ended_early"], "episode 0: TimeoutError: no frame for 10 s")
        self.assertFalse(summary["accepted"])
        # A failed episode where nobody can restart the scene ends collection too.
        stream = LaneStream(clock, hit_at=3)
        episodes, ended_early = collect(stream, TubWriterStub(), self.pilot(), seed=5, clock=clock, limits=limits,
                                        restartable=False)
        self.assertEqual((len(episodes), episodes[0]["termination"]), (1, "collision"))
        self.assertIn("cannot be restarted", ended_early)

    def test_summary_checks_flag_a_short_one_sided_or_failing_collection(self):
        good = [{"frames_stored": 60, "frames_seen": 66, "perturbed_frames": 6, "seconds": 3., "termination": "time_limit",
                 "mean_abs_error": .2, "max_abs_error": .9}] * 2
        summary = summarize(good, label_summary([-.3, .3, 0., .2, -.2] * 24), target_frames=120)
        self.assertTrue(summary["accepted"])
        self.assertEqual((summary["frames_stored"], summary["terminations"]), (120, {"time_limit": 2}))
        self.assertAlmostEqual(summary["perturbed_share"], 12 / 132)
        bad = [dict(good[0], termination="off_track"), dict(good[0], termination="collision")]
        failed = {check["name"]: check["ok"] for check in
                  summarize(bad, label_summary([-.3] * 100), target_frames=500)["checks"]}
        self.assertEqual(failed, {"frames": False, "left_and_right": False, "episodes_survive": False, "not_saturated": True})
        self.assertFalse(summarize([], label_summary([]), target_frames=1)["accepted"])


class TubWriterStub:
    """Counts frames without touching the disk."""

    def __init__(self):
        self.count = 0

    def add(self, *args, **kwargs):
        self.count += 1
        return f"{self.count}.jpg"


if __name__ == "__main__":
    unittest.main()
