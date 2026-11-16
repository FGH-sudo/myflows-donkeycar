"""The episode loop of a collector and the report it leaves next to the data.

A collector supplies a *pilot*: ``reset(now)`` and ``step(observation, info, now) -> Step``. The loop asks the
pilot what the expert would do in the frame it has, stores the frame with that label, applies the (possibly
disturbed) steering and waits for the next frame. The simulator sessions are the ones evaluation uses
(``reset`` / ``step`` / ``brake``), so collection sees exactly the frames a deployed model will see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import statistics
import time

# Catalog fields every collector writes besides DonkeyCar's own.
EXTRA_FIELDS = {"sim/error": "float", "sim/speed": "float", "collect/episode": "int", "collect/applied": "float",
                "collect/offset": "float", "collect/t": "float", "collect/mirrored": "boolean"}


@dataclass
class Step:
    """One decision of the expert. ``label`` is None for a frame that must not be stored."""

    label: float | None = None
    steering: float = 0.
    throttle: float = 0.
    error: float | None = None
    speed: float | None = None
    offset: float = 0.
    stop: str | None = None
    extra: dict = field(default_factory=dict)


class SimulatorFailure(RuntimeError):
    """The simulator could not start an episode; collection ends with the frames it already has."""


SIMULATOR_ERROR = "simulator_error"  # termination of an episode during which the simulator stopped answering


@dataclass
class Limits:
    frames: int
    episode_s: float
    max_episodes: int
    wall_s: float


def record_episode(stream, writer, pilot, *, index: int, seed: int, limits: Limits, log=None,
                   clock=time.perf_counter, mirror: bool = False, deadline: float | None = None) -> dict:
    """One episode. With ``mirror`` every stored frame is followed by its left-right mirrored twin with the
    steering negated: only for roads that look the same mirrored (a single lane), never for a lane of a two-way
    road. A simulator that stops answering ends the episode (``simulator_error``) with its frames kept; one that
    cannot start the episode raises ``SimulatorFailure``."""
    if deadline is None:
        deadline = clock() + limits.wall_s
    try:
        observation, info = stream.reset(seed=seed)
    except Exception as error:
        raise SimulatorFailure(f"{type(error).__name__}: {error}") from error
    start = clock()
    pilot.reset(start)
    reason = "time_limit"
    failure = None
    stored = seen = perturbed = 0
    errors, speeds = [], []
    try:
        while True:
            now = clock()
            if writer.count >= limits.frames:
                reason = "frame_target"
                break
            if now >= deadline:
                reason = "wall_limit"
                break
            if now - start >= limits.episode_s:
                break
            step = pilot.step(observation, info, now)
            if step.stop:
                reason = step.stop
                break
            seen += 1
            if step.error is not None:
                errors.append(abs(step.error))
            if step.speed is not None:
                speeds.append(step.speed)
            if step.offset:
                perturbed += 1
            if step.label is not None:
                for flipped in (False, True) if mirror else (False,):
                    if writer.count >= limits.frames:
                        break
                    side = -1. if flipped else 1.
                    extra = {"sim/error": None if step.error is None else round(side * step.error, 5),
                             "sim/speed": None if step.speed is None else round(step.speed, 4),
                             "collect/episode": index, "collect/applied": round(side * step.steering, 6),
                             "collect/offset": round(side * step.offset, 6) + 0., "collect/t": round(now - start, 4),
                             "collect/mirrored": flipped, **step.extra}
                    name = writer.add(observation[:, ::-1] if flipped else observation, side * step.label,
                                      step.throttle, **extra)
                    stored += 1
                    if log is not None:
                        log.write(json.dumps({"image": name, "episode": index, "label": side * step.label,
                                              "applied": side * step.steering, "offset": side * step.offset,
                                              "error": None if step.error is None else side * step.error,
                                              "speed": step.speed, "t": now - start, "mirrored": flipped},
                                             allow_nan=False) + "\n")
            if writer.count >= limits.frames:
                reason = "frame_target"
                break
            try:
                observation, info = stream.step((step.steering, step.throttle))
            except Exception as error:  # the frames stored so far stay; the episode ends here
                reason, failure = SIMULATOR_ERROR, f"{type(error).__name__}: {error}"
                break
    finally:
        stream.brake()
    elapsed = clock() - start
    return {"episode": index, "seed": seed, "termination": reason, "error": failure, "seconds": elapsed,
            "frames_seen": seen,
            "frames_stored": stored, "perturbed_frames": perturbed,
            "frame_rate": seen / elapsed if elapsed > 0 else None,
            "mean_abs_error": statistics.fmean(errors) if errors else None, "max_abs_error": max(errors, default=None),
            "mean_speed": statistics.fmean(speeds) if speeds else None}


def collect(stream, writer, pilot, *, limits: Limits, seed: int, log=None, clock=time.perf_counter,
            mirror: bool = False, on_episode=None, restartable: bool = True) -> tuple[list[dict], str | None]:
    """Episodes until the frame target, the episode cap or the wall-clock limit is reached.

    Returns the episodes and, when collection had to stop early, why: the simulator could not restart the scene
    or stopped answering, or an episode failed where the scene cannot be restarted (``restartable`` False). The
    frames stored until then still make a data set; nobody has to be there to rescue them."""
    episodes, ended_early = [], None
    began = clock()
    while writer.count < limits.frames and len(episodes) < limits.max_episodes and clock() - began < limits.wall_s:
        try:
            row = record_episode(stream, writer, pilot, index=len(episodes), seed=seed + len(episodes),
                                 limits=limits, log=log, clock=clock, mirror=mirror, deadline=began + limits.wall_s)
        except SimulatorFailure as error:
            ended_early = f"episode {len(episodes)} could not start: {error}"
            break
        episodes.append(row)
        if on_episode is not None:
            on_episode(row)
        if row["termination"] == "wall_limit":
            ended_early = f"collection reached its wall limit ({limits.wall_s:g} s)"
            break
        if row["termination"] == SIMULATOR_ERROR:
            ended_early = f"episode {row['episode']}: {row['error']}"
            break
        if row["termination"] in FAILURES and not restartable:
            ended_early = f"episode {row['episode']} ended in {row['termination']} and the scene cannot be restarted"
            break
    return episodes, ended_early


FAILURES = ("collision", "off_track", "stuck", "telemetry_missing", "road_lost")


def summarize(episodes: list[dict], labels: dict, *, target_frames: int, min_side_share: float = .08,
              max_failure_share: float = .5, ended_early: str | None = None) -> dict:
    """Totals and the collector's own acceptance checks; a human reads them, nobody has to watch the run."""
    stored = sum(row["frames_stored"] for row in episodes)
    seen = sum(row["frames_seen"] for row in episodes)
    failed = [row for row in episodes if row["termination"] in FAILURES]
    errors = [row["mean_abs_error"] for row in episodes if row["mean_abs_error"] is not None]
    checks = [
        {"name": "frames", "ok": stored >= target_frames, "detail": f"{stored} / {target_frames}"},
        {"name": "left_and_right", "ok": min(labels.get("left_share", 0.), labels.get("right_share", 0.)) >= min_side_share,
         "detail": f"left {labels.get('left_share', 0.):.3f}, right {labels.get('right_share', 0.):.3f}, "
                   f"minimum {min_side_share}"},
        {"name": "episodes_survive", "ok": bool(episodes) and len(failed) / len(episodes) <= max_failure_share,
         "detail": f"{len(failed)} of {len(episodes)} episodes ended in " + "/".join(FAILURES)},
        {"name": "not_saturated", "ok": labels.get("saturated_share", 0.) <= .2,
         "detail": f"saturated labels {labels.get('saturated_share', 0.):.3f}"},
    ]
    return {"episodes": len(episodes), "frames_stored": stored, "frames_seen": seen, "ended_early": ended_early,
            "perturbed_share": sum(row["perturbed_frames"] for row in episodes) / seen if seen else 0.,
            "seconds": sum(row["seconds"] for row in episodes),
            "terminations": {reason: sum(row["termination"] == reason for row in episodes)
                             for reason in sorted({row["termination"] for row in episodes})},
            "mean_abs_error": statistics.fmean(errors) if errors else None,
            "max_abs_error": max((row["max_abs_error"] for row in episodes if row["max_abs_error"] is not None),
                                 default=None),
            "checks": checks, "accepted": all(check["ok"] for check in checks)}


def write_report(directory, *, title: str, intro: str, meta: dict, episodes: list[dict], error_name: str) -> None:
    def show(value, digits=3):
        return "缺测" if value is None else f"{value:.{digits}f}"

    labels, summary = meta["labels"], meta["summary"]
    lines = [f"# {title}", "", intro, "",
             f"共 {summary['episodes']} 段、{summary['frames_stored']} 帧（经过 {summary['frames_seen']} 帧），"
             f"其中 {summary['perturbed_share'] * 100:.1f}% 的帧施加了转向扰动。标签始终是专家控制器在该帧状态下的输出，"
             "扰动只改变实际执行的转向。", ""]
    if summary.get("ended_early"):
        lines += [f"采集提前结束：{summary['ended_early']}。已保存的帧仍组成数据集。", ""]
    lines += ["## 标签分布", "",
             f"均值 {show(labels.get('mean'), 4)}，标准差 {show(labels.get('std'), 4)}，平均绝对值 {show(labels.get('mean_abs'), 4)}；"
             f"直行（|angle| < 0.05）{labels.get('straight_share', 0) * 100:.1f}%，左转 {labels.get('left_share', 0) * 100:.1f}%，"
             f"右转 {labels.get('right_share', 0) * 100:.1f}%，饱和 {labels.get('saturated_share', 0) * 100:.1f}%。", "",
             "## 自检", "", "| 检查 | 结果 | 说明 |", "| --- | --- | --- |"]
    lines += [f"| {check['name']} | {'通过' if check['ok'] else '未通过'} | {check['detail']} |" for check in summary["checks"]]
    lines += ["", "## 各段", "", f"| 段 | 种子 | 结束原因 | 秒 | 保存帧 | 帧率 | 平均 {error_name} | 最大 {error_name} | 平均速度 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in episodes:
        lines.append(f"| {row['episode']} | {row['seed']} | {row['termination']} | {row['seconds']:.1f} | "
                     f"{row['frames_stored']} | {show(row['frame_rate'], 1)} | {show(row['mean_abs_error'])} | "
                     f"{show(row['max_abs_error'])} | {show(row['mean_speed'], 2)} |")
    lines += ["", "dataset.json 记录采集参数、标签分布与自检结果；frames.jsonl 是逐帧的标签、执行转向与偏差；"
              "episodes.jsonl 是各段的起止与结束原因。"]
    Path(directory, "collection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
