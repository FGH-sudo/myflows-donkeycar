"""Collect a DonkeyCar data set with nobody at the wheel.

DonkeySim reports the cross-track error of the car in every telemetry message. A PD controller on that error is
the expert: its steering is the label of the frame. Steering disturbances push the car off the lane centre every
few seconds, so the set also shows how to come back. The result is a tub (``images/`` and a catalog) under
``runs/datasets/<name>`` that ``python -m apps.train --data runs/datasets/<name>`` reads as it is.

    python -m apps.collect.donkey_auto --name donkey-auto-001 --frames 3000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

from apps.common.datasets import COLLECTED_ROOT, DATASET_ID, TubWriter
from apps.eval.simulator import FIXED_THROTTLE, ROOT, guarded_worker, session, write_json

from .policy import CteExpert, Perturbation, clip, finite
from .recorder import EXTRA_FIELDS, Limits, Step, collect, summarize, write_report

ENVIRONMENT = "donkey-generated-roads-v0"


class DonkeyPilot:
    """The expert, its disturbances and the rules that end an episode, for one telemetry frame at a time."""

    def __init__(self, expert: CteExpert, perturbation: Perturbation, *, throttle: float, fail_cte: float,
                 fail_frames: int, label_cte: float, warmup_frames: int, end_margin: int = 2):
        if not 0 < label_cte <= fail_cte:
            raise ValueError("label_cte must be positive and not above fail_cte")
        self.expert, self.perturbation = expert, perturbation
        self.throttle, self.fail_cte, self.fail_frames = float(throttle), float(fail_cte), int(fail_frames)
        self.label_cte, self.warmup_frames, self.end_margin = float(label_cte), int(warmup_frames), int(end_margin)
        self.frame = self.outside = 0

    def reset(self, now: float) -> None:
        self.expert.reset()
        self.perturbation.reset(now)
        self.frame = self.outside = 0

    def step(self, observation, info, now: float) -> Step:
        cte = finite(info.get("cte"))
        if cte is None:
            return Step(stop="telemetry_missing")
        if str(info.get("hit") or "none").strip().lower() != "none":
            return Step(stop="collision")
        self.outside = self.outside + 1 if abs(cte) > self.fail_cte else 0
        if self.outside >= self.fail_frames:
            return Step(stop="off_track")
        active, total = info.get("active_node"), info.get("total_nodes")
        if isinstance(active, int) and isinstance(total, int) and total > 0 and active >= total - self.end_margin:
            return Step(stop="road_end")
        label = self.expert(cte, now)
        self.frame += 1
        warm = self.frame > self.warmup_frames
        # No disturbance while the car is still pulling away, and none on top of a large error.
        offset = self.perturbation.offset(now) if warm and abs(cte) < .5 * self.fail_cte else 0.
        store = warm and abs(cte) <= self.label_cte
        return Step(label=label if store else None, steering=clip(label + offset), throttle=self.throttle, error=cte,
                    speed=finite(info.get("speed")), offset=offset)


class RoadPerEpisode:
    """The session with a new generated road before every reset: resets alone replay the same first bends."""

    def __init__(self, stream):
        self.stream = stream
        self.step, self.brake = stream.step, stream.brake

    def reset(self, *, seed):
        self.stream.regenerate_road(seed)
        return self.stream.reset(seed=seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help=f"data set id; the set is written to {COLLECTED_ROOT}/<name>")
    parser.add_argument("--frames", type=int, default=3000, help="frames to store")
    parser.add_argument("--episode-s", type=float, default=60., help="longest episode before the car is reset")
    parser.add_argument("--same-road", action="store_true", help="keep the first road instead of a new one per episode")
    parser.add_argument("--max-episodes", type=int, default=400)
    parser.add_argument("--wall-s", type=float, default=3600., help="stop collecting after this many seconds")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--throttle", type=float, default=FIXED_THROTTLE)
    parser.add_argument("--kp", type=float, default=.5, help="steering per unit of cross-track error")
    parser.add_argument("--kd", type=float, default=.35, help="steering per unit of cross-track error rate")
    parser.add_argument("--no-perturb", action="store_true", help="record the undisturbed expert only")
    parser.add_argument("--perturb-interval-s", type=float, nargs=2, default=(2., 5.), metavar=("MIN", "MAX"))
    parser.add_argument("--perturb-frames", type=int, nargs=2, default=(3, 8), metavar=("MIN", "MAX"))
    parser.add_argument("--perturb-magnitude", type=float, nargs=2, default=(.3, .8), metavar=("MIN", "MAX"))
    parser.add_argument("--fail-cte", type=float, default=4., help="|CTE| that ends an episode as off the track")
    parser.add_argument("--label-cte", type=float, default=3., help="frames beyond this |CTE| are not stored")
    parser.add_argument("--warmup-frames", type=int, default=10, help="frames after a reset that are not stored")
    parser.add_argument("--sim", help="simulator executable or remote")
    parser.add_argument("--port", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv=None, *, root=ROOT):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not DATASET_ID.fullmatch(args.name):
        parser.error("--name: lowercase letters, digits, - and _, 3 to 64 characters")
    if args.frames < 1 or args.episode_s <= 0 or args.max_episodes < 1 or args.wall_s <= 0:
        parser.error("--frames, --episode-s, --max-episodes and --wall-s must be positive")
    if not 0 < args.throttle <= 1:
        parser.error("--throttle must be in (0, 1]")
    out = Path(root) / COLLECTED_ROOT / args.name
    logs = out / "logs"
    if not args.worker:
        out.mkdir(parents=True, exist_ok=False)
        logs.mkdir()
        guarded_worker("apps.collect.donkey_auto", argv, 180 + args.wall_s + 20 * args.max_episodes, logs)
        meta = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
        print(f"{meta['frames']} frames in {COLLECTED_ROOT}/{args.name}; accepted={meta['summary']['accepted']}")
        return 0 if meta["frames"] else 1

    from apps.autopilot.provenance import code_state
    from apps.common.gpu_lock import GpuLock

    rng = random.Random(args.seed)
    pilot = DonkeyPilot(CteExpert(args.kp, args.kd),
                        Perturbation(rng, interval_s=args.perturb_interval_s, frames=args.perturb_frames,
                                     magnitude=args.perturb_magnitude, enabled=not args.no_perturb),
                        throttle=args.throttle, fail_cte=args.fail_cte, fail_frames=3, label_cte=args.label_cte,
                        warmup_frames=args.warmup_frames)
    settings = {key: value for key, value in vars(args).items() if key not in ("worker", "sim", "port", "name")}
    meta = {"id": args.name, "environment": ENVIRONMENT, "simulator": "donkey",
            "source": "自动采集：CTE 的 PD 控制器", "expert": {"kind": "cte_pd", "kp": args.kp, "kd": args.kd},
            "collection": settings, "code": code_state(ROOT)}
    limits = Limits(frames=args.frames, episode_s=args.episode_s, max_episodes=args.max_episodes, wall_s=args.wall_s)
    with GpuLock(label="data collection", run_dir=logs), session(logs, executable=args.sim, port=args.port) as (stream, described), \
            TubWriter(out, meta=meta, extra_fields=EXTRA_FIELDS) as writer, \
            (out / "frames.jsonl").open("w", encoding="utf-8", buffering=1) as frames, \
            (out / "episodes.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        writer.meta["session"] = described

        def report(row):
            log.write(json.dumps(row, allow_nan=False) + "\n")
            print(f"episode {row['episode']} {row['termination']} {row['seconds']:.1f}s stored={row['frames_stored']} "
                  f"total={writer.count}", flush=True)

        episodes, ended_early = collect(stream if args.same_road else RoadPerEpisode(stream), writer, pilot,
                                        limits=limits, seed=args.seed, log=frames, on_episode=report)
        if ended_early:
            print(f"collection ended early: {ended_early}", flush=True)
        from apps.common.datasets import label_summary

        summary = summarize(episodes, label_summary(writer.angles), target_frames=args.frames, ended_early=ended_early)
        summary["perturbation_pulses"] = pilot.perturbation.pulses
        meta = writer.close(summary=summary)
    write_report(out, title="DonkeyCar 自动采集", error_name="绝对 CTE", meta=meta, episodes=episodes,
                 intro=f"环境 {ENVIRONMENT}，固定油门 {args.throttle}，"
                       f"{'各段使用同一条道路' if args.same_road else '每段重新生成一条道路'}。专家是横向偏差 CTE 的 PD 控制器"
                       f"（kp={args.kp}，kd={args.kd}），标签是它在该帧的转向输出。")
    write_json(logs / "result.json", {"status": "succeeded", "frames": meta["frames"], "accepted": summary["accepted"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
