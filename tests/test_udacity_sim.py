"""The Udacity simulator link without the Unity build: wire protocol, session, camera expert and collector."""

import csv
import json
import math
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from apps.collect.policy import Perturbation, SpeedHold
from apps.collect.udacity_auto import IMAGE_SIZE, VisionPilot, main as collect_main
from apps.collect.vision import RoadFinder, VisionExpert
from apps.common.datasets import COLLECTED_ROOT, list_datasets
from apps.common.donkey_data import load_donkey_index
from apps.sim.udacity import (SimulatorGone, UdacityServer, UdacityStream, decode_image, event_packet, number,
                              parse_event)
from apps.sim.udacity_fake import HEIGHT, WIDTH, FakeCar, FakeUdacitySim, StadiumTrack, render


class WireFormatTest(unittest.TestCase):
    def test_event_packets_round_trip_and_other_packets_are_ignored(self):
        packet = event_packet("steer", {"steering_angle": "0.2500", "throttle": "-1.0000"})
        self.assertEqual(packet, '42["steer",{"steering_angle":"0.2500","throttle":"-1.0000"}]')
        self.assertEqual(parse_event(packet), ("steer", {"steering_angle": "0.2500", "throttle": "-1.0000"}))
        self.assertEqual(parse_event('4213["telemetry",{}]'), ("telemetry", {}))  # with an ack id
        self.assertEqual(parse_event('42["manual"]'), ("manual", None))
        for other in ("40", "2", "3probe", '42{"not":"a list"}', "42[", "42[5]", '43["ack"]', "42x[\"a\"]", ""):
            self.assertIsNone(parse_event(other), other)

    def test_numbers_formatted_with_a_decimal_comma_are_read(self):
        self.assertEqual((number("12.5000"), number("12,5000"), number("-0,2500")), (12.5, 12.5, -.25))
        self.assertIsNone(number("fast"))
        self.assertIsNone(number(None))
        for bad in ("nan", "inf", "-inf", "1e999", float("nan")):
            self.assertIsNone(number(bad), str(bad))

    def test_a_broken_camera_image_is_an_error_not_a_crash(self):
        for bad in ("", "!!!!", "aGVsbG8="):
            with self.assertRaises(ValueError):
                decode_image(bad)

    def test_camera_pixel_limit_is_checked_before_decoding(self):
        import base64
        import io
        from PIL import Image

        encoded = io.BytesIO()
        Image.new("RGB", (1025, 1025)).save(encoded, format="PNG")
        with mock.patch("cv2.imdecode") as decoder, self.assertRaisesRegex(ValueError, "pixel limit"):
            decode_image(base64.b64encode(encoded.getvalue()).decode("ascii"))
        decoder.assert_not_called()


class ServerProtocolTest(unittest.TestCase):
    def test_a_dead_or_unresponsive_owned_simulator_is_relaunched(self):
        from apps.sim import udacity
        from apps.sim.udacity_launcher import LauncherError

        for dead in (True, False):
            with self.subTest(dead=dead), mock.patch.object(udacity, "UdacityServer") as server_type, \
                    mock.patch("apps.sim.udacity_launcher.UdacityLauncher") as launcher_type:
                server = server_type.return_value.start.return_value
                server.connects, server.port, server.connected = 0, 4567, False
                server.wait_connected.return_value = True
                launcher = launcher_type.return_value
                launcher.describe.return_value = {}
                launcher.process.poll.return_value = 1 if dead else None
                if not dead:
                    launcher.restart.side_effect = LauncherError("menu did not respond")
                with udacity.session() as (stream, _metadata):
                    stream.restart()
                self.assertEqual(launcher.start.call_count, 2)
                self.assertEqual(launcher.stop.call_count, 2)  # replacement, then final cleanup
                server.close.assert_called_once()

    def test_a_browser_origin_and_a_second_simulator_cannot_replace_the_driver(self):
        from websockets.sync.client import connect
        from websockets.exceptions import ConnectionClosed, InvalidStatus

        with UdacityServer(port=0) as server:
            url = f"ws://127.0.0.1:{server.port}/socket.io/?EIO=4&transport=websocket"
            with self.assertRaises(InvalidStatus):
                connect(url, origin="https://example.invalid")
            with connect(url, compression=None) as first:
                first.recv(timeout=5)
                first.recv(timeout=5)
                self.assertTrue(server.wait_connected(5))
                with connect(url, compression=None) as second, self.assertRaises(ConnectionClosed):
                    second.recv(timeout=5)
                self.assertEqual(server.connects, 1)
                server.send_control(.25, .2)
                self.assertEqual(parse_event(first.recv(timeout=5))[0], "steer")
                for bad in (float("nan"), float("inf"), 1.1):
                    with self.assertRaises(ValueError):
                        server.send_control(bad, .2)

    def test_handshake_heartbeat_manual_mode_and_string_valued_commands(self):
        from websockets.sync.client import connect

        with UdacityServer(port=0) as server:
            with self.assertRaises(SimulatorGone):
                server.send_control(0., 0.)
            with connect(f"ws://127.0.0.1:{server.port}/socket.io/?EIO=4&transport=websocket", compression=None) as client:
                opened = client.recv(timeout=5)
                self.assertEqual(opened[0], "0")
                self.assertEqual(json.loads(opened[1:])["upgrades"], [])
                self.assertEqual(client.recv(timeout=5), "40")
                self.assertTrue(server.wait_connected(5))
                client.send("2")
                self.assertEqual(client.recv(timeout=5), "3")
                client.send("2probe")
                self.assertEqual(client.recv(timeout=5), "3probe")
                # A hand-driven simulator reports an empty telemetry and waits for "manual".
                client.send('42["telemetry",{}]')
                self.assertEqual(client.recv(timeout=5), '42["manual",{}]')
                self.assertEqual((server.seq, server.manual_frames), (0, 1))
                with self.assertRaises(TimeoutError):
                    server.next_frame(0, .05)
                client.send(event_packet("telemetry", {"steering_angle": "5,0000", "throttle": "0.1000", "speed": "9,5000",
                                                       "image": "not checked here"}))
                seq, fields, _monotonic, _unix = server.next_frame(0, 5)
                self.assertEqual((seq, fields["speed"]), (1, "9,5000"))
                server.send_control(.123456, -1)
                # The simulator reads the values with GetField(...).str: numbers would be dropped.
                self.assertEqual(client.recv(timeout=5), '42["steer",{"steering_angle":"0.1235","throttle":"-1.0000"}]')
                client.send("41")
            self.assertFalse(server.wait_connected(.2, after=1))
            with self.assertRaises(SimulatorGone):
                server.next_frame(1, .05)

    def test_stream_gives_frames_brakes_and_follows_a_scene_restart(self):
        with UdacityServer(port=0) as server:
            stream = UdacityStream(server, timeout_s=5., connect_timeout_s=.1)
            with self.assertRaisesRegex(SimulatorGone, "Autonomous Mode"):
                stream.reset(seed=1)
            with FakeUdacitySim(server.port, ping_every=5) as sim:
                stream = UdacityStream(server, timeout_s=5., restart=sim.restart, connect_timeout_s=10.)
                image, info = stream.reset(seed=1)
                self.assertEqual((image.shape, image.dtype), ((HEIGHT, WIDTH, 3), np.uint8))
                self.assertEqual(info["wire_fields"], ["image", "speed", "steering_angle", "throttle"])
                self.assertIsNone(info["cte"])
                for _ in range(40):
                    image, info = stream.step((.4, 1.))
                self.assertGreater(info["speed"], 5.)
                self.assertAlmostEqual(info["observed_steering"], .4, places=3)  # degrees on the wire, [-1, 1] here
                self.assertAlmostEqual(info["observed_throttle"], 1.)
                self.assertGreater(sim.history[-1]["offset"], .5)  # a right turn leaves the anticlockwise track outwards
                self.assertGreater(sim.pongs, 3)  # the server answers the client's Engine.IO pings
                self.assertGreater(info["frame_id"], 40)
                stream.brake()
                self.assertLess(sim.history[-1]["speed_mph"], .5)
                _image, info = stream.reset(seed=2)  # the scene restarts: a new connection, the car on the start line
                self.assertEqual((sim.connections, server.connects), (2, 2))
                self.assertEqual((info["speed"], sim.history[-1]["offset"], sim.history[-1]["x"]), (0., 0., 5.))
                self.assertIsNone(sim.error)
            UdacityStream(server, timeout_s=.2).brake(limit_s=.2)  # a vanished simulator must not raise here


class LearnerCollectionTest(unittest.TestCase):
    def test_the_expert_keeps_labelling_when_a_learner_drives_or_needs_intervention(self):
        from apps.collect.vision import RoadView

        class Finder:
            ready = steady = True
            lateral = 0.

            def reset(self):
                pass

            def __call__(self, _picture):
                return RoadView(found=True, near=0., far=0., lateral=self.lateral)

        finder = Finder()
        learner = mock.Mock(return_value=.28)
        pilot = VisionPilot(finder, lambda view: .25, SpeedHold(12.),
                            Perturbation(random.Random(1), enabled=False), learner=learner, warmup_frames=0)
        decision = pilot.step(np.zeros((12, 16, 3), np.uint8), {"speed": 12.}, 0.)
        self.assertEqual((decision.label, decision.steering, decision.learner, decision.intervened), (.25, .28, .28, False))
        learner.return_value = .8
        decision = pilot.step(None, {"speed": 12.}, .05)
        self.assertEqual((decision.label, decision.steering, decision.intervened), (.25, .25, True))
        learner.return_value = .28
        finder.lateral = .4
        self.assertTrue(pilot.step(None, {"speed": 12.}, .1).intervened)
        learner.return_value = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            pilot.step(None, {"speed": 12.}, .15)


class CameraExpertTest(unittest.TestCase):
    def picture(self, offset, heading=0.):
        track = StadiumTrack()
        car = FakeCar(track)
        car.x, car.y, car.heading = 40., -track.radius - offset, heading  # the lower straight, driven towards +x
        return render(track, car)

    def began(self, offset=0.):
        """A finder that has seen the first pictures of an episode, the car standing at ``offset``."""
        finder = RoadFinder()
        for _ in range(finder.settings.start_skip + finder.settings.start_pictures):
            seen = finder(self.picture(offset))
        self.assertTrue(finder.ready)
        return finder, seen

    def finder_at(self, offset):
        """A finder that began its episode in the middle of the lane and has followed the car to ``offset``."""
        finder, seen = self.began()
        for step in np.linspace(0., offset, 1 + int(abs(offset) / .4))[1:]:
            seen = finder(self.picture(float(step)))
        return finder, seen

    def test_the_road_model_recovers_position_and_heading(self):
        surface = StadiumTrack().half_width - .35  # the grey between the edge lines, in metres either side
        for offset in (-3., -1.6, 0., 1.6, 3.):
            _finder, seen = self.finder_at(offset)
            self.assertTrue(seen.found, offset)
            self.assertAlmostEqual(seen.lateral, offset / surface, delta=.04)
            self.assertAlmostEqual(seen.heading, 0., delta=.02)
            self.assertAlmostEqual(seen.curvature, 0., delta=1.)  # a bend of the track shows as 2 and more
        (_, centre), (_, left), (_, right) = self.finder_at(0.), self.finder_at(-1.6), self.finder_at(1.6)
        # Left of its line the car sees the line to the right, and the expert steers right (positive).
        self.assertGreater(left.near, .1)
        self.assertGreater(left.near, left.far)  # the same offset is more pixels close to the car
        self.assertGreater(VisionExpert()(left), .1)
        self.assertLess(VisionExpert()(right), -.1)
        self.assertAlmostEqual(VisionExpert()(centre), 0., delta=.02)
        # Pointing 0.15 rad to the left of the road: the vanishing point moves right by f * tan(0.15) pixels and
        # the position across the lane does not change.
        finder, _ = self.finder_at(0.)
        for heading in (.05, .1, .15):
            turned = finder(self.picture(0., heading))
        self.assertAlmostEqual(turned.heading, 200. * math.tan(.15) / (WIDTH / 2), delta=.02)
        self.assertAlmostEqual(turned.lateral, 0., delta=.04)
        self.assertGreater(turned.far, .1)
        with self.assertRaises(ValueError):
            VisionExpert(0., 0.)

    def test_the_line_to_follow_is_set_where_the_episode_begins(self):
        # The simulator starts the car in its lane: wherever it stands in the first pictures is "on the line".
        finder, first = self.began(1.)
        self.assertAlmostEqual(first.lateral, 0., delta=.01)
        self.assertAlmostEqual(finder(self.picture(1.4)).lateral, .4 / (StadiumTrack().half_width - .35 - 1.), delta=.04)
        finder.reset()
        self.assertAlmostEqual(finder(self.picture(0.)).lateral, 0., delta=.01)

    def test_settling_and_misread_start_pictures_do_not_move_the_line(self):
        # On the real simulator the car is dropped onto the road: the first pictures put the right edge a third
        # closer than the later ones, and a car that took its distance from them drove that much to the right
        # for the whole episode. Those pictures are left out, and the distance is the middle value of the next
        # ones, so a single misread picture among them cannot set it either.
        finder = RoadFinder()
        settings = finder.settings
        for _ in range(settings.start_skip):
            self.assertEqual(finder(self.picture(1.5)).lateral, 0.)  # stand for the settling pictures
        self.assertFalse(finder.ready)
        self.assertEqual(finder(self.picture(1.5)).lateral, 0.)  # one misread picture inside the window
        for _ in range(settings.start_pictures - 1):
            self.assertEqual(finder(self.picture(0.)).lateral, 0.)  # still taking the reference: on the line
        self.assertTrue(finder.ready)
        alone, _ = self.began()
        self.assertAlmostEqual(finder.reference, alone.reference, delta=.02)
        self.assertAlmostEqual(finder(self.picture(0.)).lateral, 0., delta=.02)
        self.assertAlmostEqual(finder(self.picture(.8)).lateral, .8 / (StadiumTrack().half_width - .35), delta=.06)

    def test_the_view_is_steady_only_after_a_run_of_readable_pictures(self):
        finder, _ = self.began()
        need = finder.settings.steady_pictures
        for _ in range(need - 1):
            finder(self.picture(0.))
        self.assertFalse(finder.steady)
        finder(self.picture(0.))
        self.assertTrue(finder.steady)
        self.assertFalse(finder(self.picture(9.)).found)  # one picture of the verge
        self.assertFalse(finder.steady)
        for _ in range(need):
            self.assertTrue(finder(self.picture(.2)).found)
        self.assertTrue(finder.steady)

    def test_the_verge_is_not_taken_for_a_new_road_surface(self):
        finder, _ = self.finder_at(0.)
        for _ in range(3):
            self.assertFalse(finder(self.picture(9.)).found)  # on the grass beside the road
        self.assertTrue(finder(self.picture(.3)).found)  # back on the road

    def test_a_hidden_right_edge_is_placed_by_the_distance_between_the_edges(self):
        finder, _ = self.finder_at(1.)
        picture = self.picture(1.)
        truth = finder(picture.copy())
        hidden = picture.copy()
        hidden[:, WIDTH // 2 + 30:] = hidden[140, WIDTH // 2]  # road colour: no edge to find on the right
        blind = finder(hidden)
        self.assertTrue(blind.found)
        self.assertAlmostEqual(blind.lateral, truth.lateral, delta=.15)  # one edge alone is a coarser measurement
        # Without ever having seen both edges there is nothing to place the right one by.
        fresh = RoadFinder()
        self.assertFalse(fresh(hidden).found)

    def drive(self, steps, *, perturb):
        with UdacityServer(port=0) as server, FakeUdacitySim(server.port) as sim:
            stream = UdacityStream(server, timeout_s=5., restart=sim.restart, connect_timeout_s=10.)
            pilot = VisionPilot(RoadFinder(), VisionExpert(), SpeedHold(12.), Perturbation(random.Random(2), enabled=perturb))
            image, info = stream.reset(seed=1)
            pilot.reset(0.)
            stop = None
            for _ in range(steps):
                step = pilot.step(image, info, sim.elapsed)
                if step.stop:
                    stop = step.stop
                    break
                image, info = stream.step((step.steering, step.throttle))
            self.assertIsNone(sim.error)
            return stop, sim.history, pilot

    def test_the_camera_expert_takes_the_straight_and_the_bend_with_and_without_disturbances(self):
        for perturb in (False, True):
            stop, history, pilot = self.drive(800, perturb=perturb)
            offsets = np.abs([row["offset"] for row in history])
            self.assertIsNone(stop)
            self.assertGreater(history[-1]["distance"], 180.)  # past the first bend of the 348 m track
            self.assertLess(offsets.max(), 1.2, perturb)  # the road is 4 m to either side
            self.assertAlmostEqual(history[-1]["speed_mph"], 12., delta=1.)
            self.assertEqual(pilot.perturbation.pulses > 3, perturb)

    def test_a_car_that_leaves_the_road_is_reported_as_lost_or_stuck(self):
        class Astray:
            far_gain = near_gain = 1.

            def __call__(self, view):
                return 1.  # full lock to the right, whatever the picture shows
        with UdacityServer(port=0) as server, FakeUdacitySim(server.port) as sim:
            stream = UdacityStream(server, timeout_s=5., connect_timeout_s=10.)
            pilot = VisionPilot(RoadFinder(), Astray(), SpeedHold(12.), Perturbation(random.Random(0), enabled=False))
            image, info = stream.reset(seed=1)
            pilot.reset(0.)
            stop = None
            for _ in range(600):
                step = pilot.step(image, info, sim.elapsed)
                if step.stop:
                    stop = step.stop
                    break
                image, info = stream.step((step.steering, step.throttle))
            # The pilot gives up as soon as the picture no longer shows a road with both edges, which is before
            # the car has left the surface.
            self.assertIn(stop, ("road_lost", "stuck"))
            self.assertGreater(sim.history[-1]["offset"], 1.)
            self.assertLess(len(sim.history), 200)


class UdacityCollectorTest(unittest.TestCase):
    def test_learner_fingerprint_is_taken_after_waiting_for_the_gpu(self):
        import contextlib
        import hashlib
        from apps.common.gpu_lock import GpuLock as DatasetLock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "learner.onnx"
            model.write_bytes(b"before training finishes")
            dataset = root / COLLECTED_ROOT / "learner-wait"
            (dataset / "logs").mkdir(parents=True)

            @contextlib.contextmanager
            def queued():
                model.write_bytes(b"updated while collection was queued")
                yield

            def queued_lock(*args, **kwargs):
                return DatasetLock(*args, **kwargs) if args else queued()

            with mock.patch("apps.common.gpu_lock.GpuLock", queued_lock), \
                    mock.patch("apps.eval.closed_loop.model_predictor", return_value=lambda _rgb: 0.):
                code = collect_main(["--name", "learner-wait", "--learner", str(model), "--frames", "40",
                                     "--episode-s", "3", "--max-episodes", "1", "--no-perturb", "--sim", "fake",
                                     "--worker"], root=root)
            self.assertEqual(code, 2)
            meta = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["learner"]["sha256"][model.name], hashlib.sha256(model.read_bytes()).hexdigest())

    def test_bad_settings_fail_before_creating_a_dataset_or_starting_unity(self):
        import contextlib
        import io

        from apps.collect.donkey_auto import main as donkey_main
        from apps.sim.udacity_drive import main as drive_main

        with tempfile.TemporaryDirectory() as tmp:
            for main, argv in (
                (collect_main, ["--name", "invalid-set", "--speed", "nan"]),
                (collect_main, ["--name", "invalid-set", "--wall-s", "inf"]),
                (collect_main, ["--name", "invalid-set", "--horizon", "159"]),
                (donkey_main, ["--name", "invalid-set", "--kp", "inf"]),
            ):
                with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failed:
                    main(argv, root=Path(tmp))
                self.assertEqual(failed.exception.code, 2)
            self.assertFalse((Path(tmp) / COLLECTED_ROOT).exists())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failed:
                drive_main(["--model", "unused.onnx", "--seconds", "inf"])
            self.assertEqual(failed.exception.code, 2)

    def test_rejected_collection_returns_nonzero_and_can_resume_without_losing_frames(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"MYFLOWS_GPU_LOCK": "off"}):
            root = Path(tmp)
            out = root / COLLECTED_ROOT / "resume-lake"
            (out / "logs").mkdir(parents=True)
            argv = ["--name", "resume-lake", "--frames", "200", "--episode-s", "4", "--max-episodes", "1",
                    "--sim", "fake", "--no-perturb", "--worker"]
            self.assertEqual(collect_main(argv, root=root), 2)
            original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "images").glob("*.jpg")}
            self.assertTrue(original)
            result = json.loads((out / "logs/result.json").read_text(encoding="utf-8"))
            self.assertEqual((result["status"], result["exit_code"]), ("rejected", 2))
            argv = ["--name", "resume-lake", "--frames", "1200", "--episode-s", "60", "--max-episodes", "5",
                    "--sim", "fake", "--no-perturb", "--worker", "--resume"]
            self.assertEqual(collect_main(argv, root=root), 0)
            for name, digest in original.items():
                self.assertEqual(hashlib.sha256((out / "images" / name).read_bytes()).hexdigest(), digest)
            meta = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
            catalog = [json.loads(line) for line in (out / "catalog_generated.catalog").read_text(encoding="utf-8").splitlines()]
            frames = [json.loads(line) for line in (out / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
            episodes = [json.loads(line) for line in (out / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual((meta["frames"], len(catalog), len(frames)), (1200, 1200, 1200))
            self.assertEqual(sum(r["frames_stored"] for r in episodes), 1200)
            self.assertEqual([r["episode"] for r in episodes], [0, 1])
            self.assertEqual([r["cam/image_array"] for r in catalog], [r["image"] for r in frames])
            with mock.patch("apps.collect.udacity_auto.open_simulator") as unexpected:
                self.assertEqual(collect_main(argv, root=root), 0)
            unexpected.assert_not_called()

    def test_a_dry_run_against_the_stand_in_leaves_a_trainable_set(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"MYFLOWS_GPU_LOCK": "off"}):
            root = Path(tmp)
            out = root / COLLECTED_ROOT / "fake-lake"
            (out / "logs").mkdir(parents=True)
            code = collect_main(["--name", "fake-lake", "--frames", "400", "--episode-s", "8", "--sim", "fake",
                                 "--worker"], root=root)
            self.assertEqual(code, 2)  # this short, mostly straight set does not pass directional coverage checks
            meta = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["frames"], meta["image"], meta["image_size"]), (400, {"width": WIDTH, "height": HEIGHT}, IMAGE_SIZE))
            self.assertEqual((meta["environment"], meta["simulator"]), ("udacity-fake-stadium", "udacity-fake"))
            self.assertEqual(meta["summary"]["terminations"].get("time_limit", 0) + meta["summary"]["terminations"].get("frame_target", 0),
                             meta["summary"]["episodes"])
            self.assertGreaterEqual(meta["summary"]["episodes"], 2)  # the scene was restarted between episodes
            index = load_donkey_index(out, fixed_throttle=.5, angle_scale=1.)
            self.assertEqual(len(index), 400)
            angles = np.array([angle for _rel, angle, _throttle in index])
            np.testing.assert_allclose(angles[0::2], -angles[1::2])  # every frame is followed by its mirrored twin
            self.assertAlmostEqual(meta["labels"]["left_share"], meta["labels"]["right_share"])
            with (out / "driving_log.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["center", "left", "right", "steering", "throttle", "brake", "speed"])
            self.assertEqual(len(rows), 401)
            self.assertTrue((out / rows[1][0]).is_file())
            self.assertEqual((rows[1][1], rows[1][2]), ("", ""))  # the link carries no side cameras
            self.assertIn("自检", (out / "collection_report.md").read_text(encoding="utf-8"))
            listed = {row["id"]: row for row in list_datasets(root)}
            self.assertEqual((listed["fake-lake"]["image_size"], listed["fake-lake"]["frames"]), (IMAGE_SIZE, 400))


if __name__ == "__main__":
    unittest.main()


def camera_expert():
    """A steering function with a model's call shape, for runs that need a driver but no trained weights."""
    finder, expert = RoadFinder(), VisionExpert()

    def predict(rgb):
        view = finder(rgb)
        return expert(view) if view.found else 0.
    return predict


class UdacityClosedLoopTest(unittest.TestCase):
    PROTOCOL = Path(__file__).resolve().parents[1] / "apps" / "eval" / "scenarios" / "udacity_lake.json"

    def test_drive_failures_write_a_result_and_camera_frames_match_the_logged_rows(self):
        from apps.sim import udacity_drive

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "failure"
            with mock.patch.object(udacity_drive, "model_predictor", side_effect=ValueError("broken model")):
                code = udacity_drive.main(["--model", "missing.onnx", "--sim", "fake", "--out", str(out)])
            self.assertEqual(code, 1)
            result = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual((result["status"], result["passed"], result["error"]), ("failed", False, "ValueError: broken model"))
            out = Path(tmp) / "camera"
            with mock.patch.object(udacity_drive, "model_predictor", return_value=camera_expert()):
                code = udacity_drive.main(["--model", "expert.onnx", "--sim", "fake", "--out", str(out),
                                           "--seconds", "3", "--camera-every", "10"])
            self.assertEqual(code, 0)
            result = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (out / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
            photos = [r["camera"] for r in rows if r.get("camera")]
            self.assertEqual(len(photos), result["camera_frames"])
            self.assertTrue(all((out / name).is_file() for name in photos))

    def test_a_paired_run_requires_a_simulator_that_can_restart_the_scene(self):
        from apps.eval import closed_loop

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            models = tmp / "models.json"
            models.write_text(json.dumps([{"id": "pilot", "endpoint": "http://127.0.0.1:8000"}]), encoding="utf-8")
            with mock.patch.object(closed_loop, "guarded_worker"), self.assertRaises(SystemExit) as refused:
                closed_loop.main(["--models", str(models), "--scenarios", str(self.PROTOCOL),
                                  "--sim", "remote", "--out", str(tmp / "out")])
            self.assertEqual(refused.exception.code, 2)
            self.assertFalse((tmp / "out").exists())

    def test_calibration_pictures_and_missing_speed_are_not_valid_measurements(self):
        from apps.eval.environments import UdacityMonitor

        monitor = UdacityMonitor(speed_mph=12.)
        monitor.reset({}, 0.)
        picture = render(StadiumTrack(), FakeCar(StadiumTrack()))
        for frame in range(1, 23):
            measured = monitor.inspect(picture, {"speed": 10.}, frame * .05)
            self.assertIsNone(measured["cte"])
        self.assertIsNotNone(monitor.inspect(picture, {"speed": 10.}, 1.15)["cte"])
        for speed in (None, "nan", float("inf")):
            with self.subTest(speed=speed), self.assertRaisesRegex(ValueError, "speed"):
                monitor.throttle({"speed": speed}, 2.)

    def test_paired_evaluation_without_position_telemetry(self):
        from apps.eval import closed_loop

        protocol = json.loads(self.PROTOCOL.read_text(encoding="utf-8"))
        protocol["environment"] = "udacity-fake-stadium"
        protocol["scenarios"] = [dict(scenario, duration_s=12.) for scenario in protocol["scenarios"][:2]]
        drivers = {"expert": camera_expert, "astray": lambda: (lambda rgb: 1.)}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"MYFLOWS_GPU_LOCK": "off"}), \
                mock.patch.object(closed_loop, "model_predictor", lambda item, device: drivers[item["id"]]()), \
                mock.patch.object(closed_loop, "_fingerprints", return_value={}):
            tmp = Path(tmp)
            (tmp / "models.json").write_text(json.dumps([{"id": "expert", "path": "expert.onnx"},
                                                         {"id": "astray", "path": "astray.onnx"}]), encoding="utf-8")
            (tmp / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            out = tmp / "out"
            out.mkdir()
            closed_loop.main(["--models", str(tmp / "models.json"), "--scenarios", str(tmp / "protocol.json"),
                              "--out", str(out), "--device", "cpu", "--worker"])
            result = json.loads((out / "result.json").read_text(encoding="utf-8"))
            episodes = [json.loads(line) for line in (out / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
            frames = [json.loads(line) for line in (out / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
            report = (out / "report.md").read_text(encoding="utf-8")
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((result["environment"], result["cte_source"], result["fixed_throttle"], result["target_speed_mph"]),
                         ("udacity-fake-stadium", "camera_estimate", None, 12.))
        self.assertEqual(manifest["session"]["simulator"], "udacity-fake")
        good, bad = result["models"]["expert"], result["models"]["astray"]
        self.assertEqual((good["episodes"], good["failures"], good["time_limits"]), (2, 0, 2))
        self.assertEqual((bad["failures"], bad["collisions"]), (2, 0))
        self.assertTrue({row["termination"] for row in episodes if row["model"] == "astray"} <= {"road_lost", "stuck"})
        by_scenario = {(row["model"], row["scenario"]): row for row in episodes}
        slow, fast = by_scenario[("expert", "start-09mph")], by_scenario[("expert", "start-12mph-a")]
        # The distance comes from the reported speed: about 12 s at the target, less the pull-away.
        self.assertTrue(25. < slow["distance_units"] < 9. * .44704 * 12.5)
        self.assertGreater(fast["distance_units"], slow["distance_units"])
        self.assertLess(slow["mean_abs_cte"], .2)  # the camera estimate, in half road widths
        self.assertEqual(slow["excursions"], 0)
        self.assertLess(bad["metrics"]["survival_s"]["mean"], 6.)
        self.assertLess(bad["paired_difference_from_baseline"]["distance_units"]["mean"], 0.)
        expert_frames = [row for row in frames if row["model"] == "expert"]
        self.assertTrue(all(row["pos"] is None and row["hit"] is None for row in expert_frames))
        self.assertTrue(all(set(row["missing"]) >= {"pos", "hit"} for row in expert_frames))
        self.assertTrue(any(row["throttle"] != expert_frames[0]["throttle"] for row in expert_frames))  # a speed controller
        self.assertIn("Udacity 模拟器配对闭环评估", report)
        self.assertIn("失败次数", report)
        self.assertIn("不是模拟器真值", report)

    def test_the_drive_command_runs_a_model_and_a_deployed_service(self):
        import contextlib
        import io
        import threading

        import uvicorn

        from apps.serve import serve_fastapi
        from apps.sim import udacity_drive

        class ServedExpert:
            """The camera expert behind the inference service, in the place of an ONNX model."""

            device, image_w, image_h = "cpu", WIDTH, HEIGHT

            def __init__(self, *args, **kwargs):
                self.steer = camera_expert()
                self.calls = 0

            def info(self):
                return {"model_path": "expert", "image_w": WIDTH, "image_h": HEIGHT, "providers": [], "device": "cpu"}

            def predict(self, rgb):
                self.calls += 1
                return np.asarray([[self.steer(rgb), 0.]], dtype=np.float32)

            def format_prediction(self, out):
                return {"model_type": "regression", "outputs": [float(v) for v in out.reshape(-1)],
                        "angle": float(out.reshape(-1)[0]), "throttle": 0.}

        def drive(*argv):
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                code = udacity_drive.main([*argv, "--sim", "fake", "--seconds", "10", "--speed", "10"])
            return code, json.loads(printed.getvalue().strip().splitlines()[-1])

        with mock.patch.object(udacity_drive, "model_predictor", lambda item, device: camera_expert()):
            code, summary = drive("--model", "anything.onnx")
        self.assertEqual((code, summary["termination"], summary["cte_source"]), (0, "time_limit", "camera_estimate"))
        self.assertGreater(summary["distance_units"], 25.)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(serve_fastapi, "OnnxPredictor", ServedExpert):
            app = serve_fastapi.create_app(Path("expert.onnx"), image_w=WIDTH, image_h=HEIGHT, log_file=Path(tmp) / "serve.jsonl")
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            try:
                for _ in range(200):
                    if server.started:
                        break
                    threading.Event().wait(.05)
                self.assertTrue(server.started)
                port = server.servers[0].sockets[0].getsockname()[1]
                code, summary = drive("--endpoint", f"http://127.0.0.1:{port}")
            finally:
                server.should_exit = True
                thread.join(timeout=10)
        # The frames went to the service as they were and its answers kept the car on the road.
        self.assertEqual((code, summary["termination"]), (0, "time_limit"))
        self.assertLess(summary["max_abs_cte"], .5)
        self.assertGreater(summary["frames"], 100)
        with self.assertRaises(ValueError):
            from apps.serve.remote import remote_predictor

            remote_predictor("ftp://example.invalid")

    def test_drive_delegates_to_guarded_worker_for_owned_simulator_binary(self):
        from apps.sim import udacity_drive

        with tempfile.TemporaryDirectory() as tmp:
            fake_bin = Path(tmp) / "unity.exe"
            fake_bin.write_text("stub", encoding="utf-8")
            with mock.patch.object(udacity_drive, "guarded_worker", return_value=0) as guarded:
                code = udacity_drive.main(["--model", "x.onnx", "--sim", str(fake_bin), "--seconds", "10"])
            self.assertEqual(code, 0)
            self.assertEqual(guarded.call_count, 1)
            self.assertEqual(guarded.call_args.args[0], "apps.sim.udacity_drive")
            self.assertIn("--guard-dir", guarded.call_args.args[1])
