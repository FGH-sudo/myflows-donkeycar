"""Collect a data set from the Udacity simulator with nobody at the wheel.

The simulator's own way to get data is a person driving in training mode. In autonomous mode it only sends the
centre camera and the speed, so the expert here drives by the picture (``apps.collect.vision``): it follows the
road it sees, holds a target speed, and its steering is the label of each frame. Steering disturbances make the
car drift so that the set also shows the way back to the centre.

    python -m apps.collect.udacity_auto --name udacity-lake-001 --frames 6000

starts the simulator, enters autonomous mode by itself and restarts the scene between episodes (Windows; see
``apps.sim.udacity_launcher``). ``--sim remote`` uses a simulator that is already in autonomous mode, ``--sim
fake`` the protocol stand-in (``apps.sim.udacity_fake``) for a dry run without the Unity build.

The set is a tub under ``runs/datasets/<name>`` that ``python -m apps.train --data runs/datasets/<name>
--image-size 80x160`` reads. ``driving_log.csv`` is written next to it in the simulator's own column order
(centre image, left, right, steering, throttle, brake, speed); the side cameras are empty because the
autonomous-mode link does not carry them.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import random
import sys

from apps.common.datasets import COLLECTED_ROOT, DATASET_ID, TubWriter, label_summary
from apps.eval.simulator import ROOT, guarded_worker, write_json

from .policy import Perturbation, SpeedHold, clip, finite, finite_float
from .recorder import EXTRA_FIELDS, Limits, Step, collect, recover_logs, summarize, write_report
from .vision import RoadFinder, RoadFinderSettings, VisionExpert

IMAGE_SIZE = "80x160"  # H x W the trainer should resize the 160 x 320 camera to: same aspect ratio, a quarter of the pixels


class VisionPilot:
    """The camera expert, its disturbances, the speed controller and the rules that end an episode."""

    def __init__(self, finder: RoadFinder, expert: VisionExpert, speed: SpeedHold, perturbation: Perturbation, *,
                 lost_frames: int = 20, stuck_s: float = 4., warmup_frames: int = 15, label_lateral: float = .9,
                 perturb_lateral: float = .4, on_unread=None, learner=None, learner_limit: float = .12,
                 learner_lateral: float = .3):
        self.finder, self.expert, self.speed, self.perturbation = finder, expert, speed, perturbation
        self.lost_frames, self.stuck_s, self.warmup_frames = int(lost_frames), float(stuck_s), int(warmup_frames)
        self.label_lateral, self.perturb_lateral = float(label_lateral), float(perturb_lateral)
        self.on_unread = on_unread  # called with (episode, frame, picture) for every picture the finder cannot read
        if not 0 < learner_limit <= 2 or not 0 < learner_lateral < label_lateral:
            raise ValueError("learner_limit must be in (0, 2] and learner_lateral in (0, label_lateral)")
        self.learner, self.learner_limit, self.learner_lateral = learner, float(learner_limit), float(learner_lateral)
        self.episodes = -1
        self.reset(0.)

    def reset(self, now: float) -> None:
        self.episodes += 1
        self.finder.reset()
        self.speed.reset()
        self.perturbation.reset(now)
        self.frame = self.lost = 0
        self.slow_since = None
        self.last_steering = 0.

    def step(self, observation, info, now: float) -> Step:
        self.frame += 1
        speed = finite(info.get("speed"))
        if speed is None:
            return Step(stop="telemetry_missing")
        if self.frame > self.warmup_frames and speed < .5:
            self.slow_since = now if self.slow_since is None else self.slow_since
            if now - self.slow_since >= self.stuck_s:
                return Step(stop="stuck")
        else:
            self.slow_since = None
        view = self.finder(observation)
        if not view.found:
            self.lost += 1
            if self.on_unread is not None:
                self.on_unread(self.episodes, self.frame, observation)
            if self.lost >= self.lost_frames:
                return Step(stop="road_lost")
            # Coast on the last command; a frame the expert cannot read gets no label.
            return Step(label=None, steering=self.last_steering, throttle=0., speed=speed)
        self.lost = 0
        label = self.expert(view)
        # Frames count once the finder has taken the distance to keep; before that every position reads as zero.
        warm = self.frame > self.warmup_frames and self.finder.ready
        if warm and self.finder.steady and abs(view.lateral) < self.perturb_lateral:
            offset = self.perturbation.offset(now)
        else:
            # Disturbances only where the road has been read without a doubt for a while: not across the bridge,
            # not while an edge changes, not right after a picture the finder could not read. A push at the bridge
            # exit once cost the road altogether.
            self.perturbation.hold(now)
            offset = 0.
        learned = None
        intervened = False
        command = label
        if self.learner is not None:
            learned = finite(self.learner(observation))
            if learned is None:
                raise ValueError("learner returned nonfinite steering")
            learned = clip(learned)
            intervened = not (warm and abs(view.lateral) < self.learner_lateral
                              and abs(learned - label) <= self.learner_limit)
            command = label if intervened else learned
        self.last_steering = clip(command + offset)
        store = warm and abs(view.lateral) <= self.label_lateral
        return Step(label=label if store else None, steering=self.last_steering, throttle=self.speed(speed, now),
                    error=view.lateral, speed=speed, offset=offset, learner=learned, intervened=intervened)


def write_driving_log(directory: Path) -> int:
    """The simulator's own log format for the frames of the tub, centre camera only."""
    rows = 0
    with (directory / "catalog_generated.catalog").open(encoding="utf-8") as catalog, \
            (directory / "driving_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["center", "left", "right", "steering", "throttle", "brake", "speed"])
        for line in catalog:
            record = json.loads(line)
            throttle = float(record["user/throttle"])
            writer.writerow([f"images/{record['cam/image_array']}", "", "", f"{record['user/angle']:.6f}",
                             f"{max(throttle, 0.):.6f}", f"{max(-throttle, 0.):.6f}",
                             "" if record.get("sim/speed") is None else f"{record['sim/speed']:.4f}"])
            rows += 1
    return rows


@contextmanager
def open_simulator(args, logs: Path):
    """``(stream, description, clock)`` for the chosen simulator: the real one, one already running, or the
    stand-in. The stand-in advances its own time by a fixed step per command, so the collector reads that clock
    instead of the wall clock and a dry run takes seconds."""
    import time

    from apps.sim.udacity import UdacityServer, UdacityStream, session

    if args.sim != "fake":
        with session(logs, executable=args.sim, track=args.track) as (stream, described):
            yield stream, described, time.perf_counter
        return
    from apps.sim.udacity_fake import FakeUdacitySim

    with UdacityServer(port=0) as server, FakeUdacitySim(server.port) as fake:
        yield (UdacityStream(server, timeout_s=10., restart=fake.restart, connect_timeout_s=20.),
               {"environment": "udacity-fake-stadium", "simulator": "udacity-fake", "port": server.port,
                "owned_simulator": True, "start_position_control": "scene restart"}, lambda: fake.elapsed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help=f"data set id; the set is written to {COLLECTED_ROOT}/<name>")
    parser.add_argument("--frames", type=int, default=6000, help="frames to store")
    parser.add_argument("--episode-s", type=finite_float, default=240., help="longest episode before the scene is restarted")
    parser.add_argument("--max-episodes", type=int, default=60)
    parser.add_argument("--max-restarts", type=int, default=2, help="simulator error retries before ending collection")
    parser.add_argument("--resume", action="store_true", help="recover and append to this existing data set")
    parser.add_argument("--wall-s", type=finite_float, default=3600., help="stop collecting after this many seconds")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--speed", type=finite_float, default=12., help="target speed in mph")
    parser.add_argument("--learner", help="model to drive while the visual expert labels and intervenes on disagreement")
    parser.add_argument("--learner-config", help="training config.json if --learner is a checkpoint")
    parser.add_argument("--learner-limit", type=finite_float, default=.12, help="largest learner/expert steering disagreement before intervention")
    parser.add_argument("--learner-lateral", type=finite_float, default=.3, help="largest lateral estimate at which the learner may drive")
    parser.add_argument("--track", choices=("lake", "jungle"), default="lake")
    parser.add_argument("--far-gain", type=finite_float, default=.8, help="steering per unit of lane-centre offset ahead")
    parser.add_argument("--near-gain", type=finite_float, default=.3, help="steering per unit of lane-centre offset at the car")
    parser.add_argument("--chroma-tolerance", type=finite_float, default=RoadFinderSettings.chroma_tolerance,
                        help="colour distance from the surface in front of the car at which the lane ends")
    parser.add_argument("--horizon", type=int, default=RoadFinderSettings.horizon)
    parser.add_argument("--hood-rows", type=int, default=RoadFinderSettings.hood_rows)
    parser.add_argument("--no-perturb", action="store_true", help="record the undisturbed expert only")
    parser.add_argument("--perturb-interval-s", type=finite_float, nargs=2, default=(2., 5.), metavar=("MIN", "MAX"))
    parser.add_argument("--perturb-frames", type=int, nargs=2, default=(3, 8), metavar=("MIN", "MAX"))
    parser.add_argument("--perturb-magnitude", type=finite_float, nargs=2, default=(.2, .6), metavar=("MIN", "MAX"))
    parser.add_argument("--no-mirror", action="store_true",
                        help="do not store the left-right mirrored twin of every frame (the tracks turn mostly one way; "
                             "the twins, with the steering negated, balance the set and count towards --frames)")
    parser.add_argument("--keep-unread", type=int, default=400, metavar="N",
                        help="keep up to N pictures the expert could not read in logs/unread, the evidence of a lost road")
    parser.add_argument("--sim", help="simulator executable, remote, or fake")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv=None, *, root=ROOT):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not DATASET_ID.fullmatch(args.name):
        parser.error("--name: lowercase letters, digits, - and _, 3 to 64 characters")
    if args.frames < 1 or args.episode_s <= 0 or args.max_episodes < 1 or args.wall_s <= 0 or args.speed <= 0:
        parser.error("--frames, --episode-s, --max-episodes, --wall-s and --speed must be positive")
    if not 0 < args.learner_limit <= 2 or not 0 < args.learner_lateral < .9:
        parser.error("--learner-limit must be in (0, 2] and --learner-lateral in (0, 0.9)")
    if args.learner_config and not args.learner:
        parser.error("--learner-config requires --learner")
    if args.max_restarts < 0:
        parser.error("--max-restarts must be non-negative")
    if args.chroma_tolerance <= 0 or not 0 <= args.horizon < 160 - args.hood_rows - 5 or not 0 < args.hood_rows < 80:
        parser.error("road finder settings do not fit the 160-row camera")
    try:
        VisionExpert(args.far_gain, args.near_gain)
        Perturbation(random.Random(0), interval_s=args.perturb_interval_s, frames=args.perturb_frames,
                     magnitude=args.perturb_magnitude)
    except ValueError as exc:
        parser.error(str(exc))
    out = Path(root) / COLLECTED_ROOT / args.name
    logs = out / "logs"
    if not args.worker:
        if args.resume and not out.is_dir():
            parser.error("--resume requires an existing data set")
        out.mkdir(parents=True, exist_ok=args.resume)
        logs.mkdir(exist_ok=args.resume)
        guarded_worker("apps.collect.udacity_auto", argv, 240 + args.wall_s + 60 * args.max_episodes, logs,
                       accepted_codes=(0, 2), append_log=args.resume)
        meta = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
        print(f"{meta['frames']} frames in {COLLECTED_ROOT}/{args.name}; accepted={meta['summary']['accepted']}")
        return 0 if meta["summary"]["accepted"] else 2

    from apps.autopilot.provenance import code_state
    from apps.common.gpu_lock import GpuLock

    rng = random.Random(args.seed)
    settings = RoadFinderSettings(chroma_tolerance=args.chroma_tolerance, horizon=args.horizon, hood_rows=args.hood_rows)
    unread = logs / "unread"
    kept = []

    def keep_unread(episode, frame, picture):
        if len(kept) >= args.keep_unread:
            return
        import cv2

        unread.mkdir(exist_ok=True)
        name = f"e{episode}-f{frame:05d}.jpg"
        cv2.imwrite(str(unread / name), cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
        kept.append(name)

    pilot = VisionPilot(RoadFinder(settings), VisionExpert(args.far_gain, args.near_gain), SpeedHold(args.speed),
                        Perturbation(rng, interval_s=args.perturb_interval_s, frames=args.perturb_frames,
                                     magnitude=args.perturb_magnitude, enabled=not args.no_perturb),
                        on_unread=keep_unread if args.keep_unread > 0 else None,
                        learner_limit=args.learner_limit, learner_lateral=args.learner_lateral)
    limits = Limits(frames=args.frames, episode_s=args.episode_s, max_episodes=args.max_episodes, wall_s=args.wall_s)
    collection = {key: value for key, value in vars(args).items()
                  if key not in ("worker", "sim", "name", "keep_unread", "learner", "learner_config", "resume")}
    meta = {"id": args.name, "simulator": "udacity-fake" if args.sim == "fake" else "udacity-term1",
            "environment": "udacity-fake-stadium" if args.sim == "fake" else f"udacity-{args.track}",
            "source": "自动采集：模型驾驶、视觉专家标注与接管" if args.learner else "自动采集：前视图像巡线控制器",
            "image_size": IMAGE_SIZE,
            "expert": {"kind": "vision_road_following", "far_gain": args.far_gain, "near_gain": args.near_gain},
            "collection": collection, "code": code_state(ROOT)}
    item = None
    if args.learner:
        from apps.eval.closed_loop import model_manifest, model_predictor

        item = {"id": "learner", "path": args.learner}
        if args.learner_config:
            item["config"] = args.learner_config

    def collection_meta():
        # Waiting for another training can replace its best model. Pin provenance after obtaining the lock.
        if item is not None:
            described_learner = model_manifest(item)
            meta["learner"] = {"sha256": described_learner["sha256"], "model": Path(args.learner).name,
                               "steering_limit": args.learner_limit, "lateral_limit": args.learner_lateral}
        return meta

    with GpuLock(label="data collection", run_dir=logs), \
            TubWriter(out, meta=collection_meta(), extra_fields=EXTRA_FIELDS, resume=args.resume) as writer:
        episodes = recover_logs(out, writer) if args.resume else []
        mode = "a" if args.resume else "w"
        with (out / "frames.jsonl").open(mode, encoding="utf-8", buffering=1) as frames, \
                (out / "episodes.jsonl").open(mode, encoding="utf-8", buffering=1) as log:

            def report(row):
                log.write(json.dumps(row, allow_nan=False) + "\n")
                print(f"episode {row['episode']} {row['termination']} {row['seconds']:.1f}s stored={row['frames_stored']} "
                      f"total={writer.count}", flush=True)

            ended_early = None
            if writer.count < limits.frames:
                with open_simulator(args, logs) as (stream, described, clock):
                    writer.meta.update(environment=described["environment"], session=described)
                    if item is not None:
                        pilot.learner = model_predictor(item, "auto")
                    episodes, ended_early = collect(stream, writer, pilot, limits=limits, seed=args.seed, log=frames,
                                                    clock=clock, mirror=not args.no_mirror, on_episode=report,
                                                    restartable=stream.restart is not None, max_restarts=args.max_restarts,
                                                    initial_episodes=episodes)
        if ended_early:
            print(f"collection ended early: {ended_early}", flush=True)
        summary = summarize(episodes, label_summary(writer.angles), target_frames=args.frames, ended_early=ended_early)
        summary["perturbation_pulses"] = pilot.perturbation.pulses
        summary["unread_pictures_kept"] = len(kept)
        meta = writer.close(summary=summary)
    write_driving_log(out)
    write_report(out, title="Udacity 模拟器自动采集", error_name="横向位置估计", meta=meta, episodes=episodes,
                 intro=f"环境 {meta['environment']}，目标速度 {args.speed} mph。专家只看前视图像：以车前路面颜色为参考，在各扫描行找出路面两侧的边缘点，"
                       "用平地车道模型拟合出两条边缘、航向和弯度；行驶线与右边缘平行，间距取车辆落稳后头 12 帧的中位数。"
                       "扰动只在连续读路稳定时施加（过桥、边缘切换和读不出路面之后不加）。"
                       f"转向按行驶线在前方与近处的偏移给出（far_gain={args.far_gain}，near_gain={args.near_gain}），标签是它在该帧的转向输出。"
                       "横向位置是图像估计值（0 在行驶线上，+1 为右边缘，−1 为向左同样远），模拟器不提供真值。")
    code = 0 if summary["accepted"] else 2
    write_json(logs / "result.json", {"status": "succeeded" if not code else "rejected", "frames": meta["frames"],
                                     "accepted": summary["accepted"], "exit_code": code})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
