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
                "collect/offset": "float", "collect/t": "float", "collect/mirrored": "boolean",
                "collect/learner": "float", "collect/intervened": "boolean", "collect/seed": "int"}


@dataclass
class Step:
    """One decision of the expert. ``label`` is None for a frame that must not be stored."""

    label: float | None = None
    steering: float = 0.
    throttle: float = 0.
    error: float | None = None
    speed: float | None = None
    offset: float = 0.
    learner: float | None = None
    intervened: bool = False
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
    stored = seen = perturbed = learner_frames = interventions = 0
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
            if step.learner is not None:
                learner_frames += 1
                interventions += int(step.intervened)
            if step.label is not None:
                for flipped in (False, True) if mirror else (False,):
                    if writer.count >= limits.frames:
                        break
                    side = -1. if flipped else 1.
                    extra = {"sim/error": None if step.error is None else round(side * step.error, 5),
                             "sim/speed": None if step.speed is None else round(step.speed, 4),
                             "collect/episode": index, "collect/seed": seed, "collect/applied": round(side * step.steering, 6),
                             "collect/offset": round(side * step.offset, 6) + 0., "collect/t": round(now - start, 4),
                             "collect/learner": None if step.learner is None else round(side * step.learner, 6),
                             "collect/intervened": step.intervened,
                             "collect/mirrored": flipped, **step.extra}
                    name = writer.add(observation[:, ::-1] if flipped else observation, side * step.label,
                                      step.throttle, **extra)
                    stored += 1
                    if log is not None:
                        log.write(json.dumps({"image": name, "episode": index, "label": side * step.label,
                                              "applied": side * step.steering, "offset": side * step.offset,
                                              "error": None if step.error is None else side * step.error,
                                              "learner": None if step.learner is None else side * step.learner,
                                              "intervened": step.intervened,
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
        try:
            stream.brake()
        except Exception as error:
            reason, failure = SIMULATOR_ERROR, f"brake: {type(error).__name__}: {error}"
    elapsed = clock() - start
    return {"episode": index, "seed": seed, "termination": reason, "error": failure, "seconds": elapsed,
            "frames_seen": seen,
            "frames_stored": stored, "perturbed_frames": perturbed,
            "learner_frames": learner_frames, "interventions": interventions,
            "frame_rate": seen / elapsed if elapsed > 0 else None,
            "mean_abs_error": statistics.fmean(errors) if errors else None, "max_abs_error": max(errors, default=None),
            "mean_speed": statistics.fmean(speeds) if speeds else None}


def collect(stream, writer, pilot, *, limits: Limits, seed: int, log=None, clock=time.perf_counter,
            mirror: bool = False, on_episode=None, restartable: bool = True, max_restarts: int = 0,
            initial_episodes: list[dict] | None = None) -> tuple[list[dict], str | None]:
    """Episodes until the frame target, the episode cap or the wall-clock limit is reached.

    Returns the episodes and, when collection had to stop early, why: the simulator could not restart the scene
    or stopped answering, or an episode failed where the scene cannot be restarted (``restartable`` False). The
    frames stored until then still make a data set; nobody has to be there to rescue them."""
    episodes, ended_early = list(initial_episodes or []), None
    began = clock()
    restarts = 0
    next_index = max((r["episode"] for r in episodes), default=-1) + 1
    while writer.count < limits.frames and next_index < limits.max_episodes and clock() - began < limits.wall_s:
        try:
            row = record_episode(stream, writer, pilot, index=next_index, seed=seed + next_index,
                                 limits=limits, log=log, clock=clock, mirror=mirror, deadline=began + limits.wall_s)
        except SimulatorFailure as error:
            if not restartable or restarts >= max_restarts:
                ended_early = f"episode {next_index} could not start: {error}"
                break
            row = {"episode": next_index, "seed": seed + next_index, "termination": SIMULATOR_ERROR,
                   "error": str(error), "seconds": 0., "frames_seen": 0, "frames_stored": 0,
                   "perturbed_frames": 0, "frame_rate": None, "mean_abs_error": None, "max_abs_error": None,
                   "mean_speed": None, "learner_frames": 0, "interventions": 0}
        episodes.append(row)
        next_index += 1
        if on_episode is not None:
            on_episode(row)
        if row["termination"] == "wall_limit":
            ended_early = f"collection reached its wall limit ({limits.wall_s:g} s)"
            break
        if row["termination"] == SIMULATOR_ERROR:
            if restartable and restarts < max_restarts:
                restarts += 1
                continue
            ended_early = f"episode {row['episode']}: {row['error']}"
            break
        if row["termination"] in FAILURES and not restartable:
            ended_early = f"episode {row['episode']} ended in {row['termination']} and the scene cannot be restarted"
            break
    if writer.count < limits.frames and ended_early is None:
        ended_early = ("collection reached its episode limit" if next_index >= limits.max_episodes
                       else f"collection reached its wall limit ({limits.wall_s:g} s)")
    return episodes, ended_early


FAILURES = ("collision", "off_track", "stuck", "telemetry_missing", "road_lost")


def summarize(episodes: list[dict], labels: dict, *, target_frames: int, min_side_share: float = .08,
              max_failure_share: float = .5, ended_early: str | None = None) -> dict:
    """Totals and the collector's own acceptance checks; a human reads them, nobody has to watch the run."""
    stored = sum(row["frames_stored"] for row in episodes)
    seen = sum(row["frames_seen"] for row in episodes)
    failed = [row for row in episodes if row["termination"] in (*FAILURES, SIMULATOR_ERROR)]
    errors = [row["mean_abs_error"] for row in episodes if row["mean_abs_error"] is not None]
    checks = [
        {"name": "frames", "ok": stored >= target_frames and ended_early is None,
         "detail": f"{stored} / {target_frames}" + (f"; {ended_early}" if ended_early else "")},
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
            "learner_frames": sum(row.get("learner_frames", 0) for row in episodes),
            "interventions": sum(row.get("interventions", 0) for row in episodes),
            "seconds": sum(row["seconds"] for row in episodes),
            "terminations": {reason: sum(row["termination"] == reason for row in episodes)
                             for reason in sorted({row["termination"] for row in episodes})},
            "mean_abs_error": statistics.fmean(errors) if errors else None,
            "max_abs_error": max((row["max_abs_error"] for row in episodes if row["max_abs_error"] is not None),
                                 default=None),
            "checks": checks, "accepted": all(check["ok"] for check in checks)}


def recover_logs(directory: Path, writer) -> list[dict]:
    """Rebuild frame logs from the authoritative catalog and recover the final episode after an interruption."""
    from apps.common.datasets import DatasetError

    old = {}
    path = directory / "episodes.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict) and isinstance(row.get("episode"), int):
                    old[row["episode"]] = row
            except ValueError:
                continue
    groups = {}
    frames = []
    for record in writer.records:
        index = record.get("collect/episode", 0)
        if not isinstance(index, int) or index < 0:
            raise DatasetError("catalog 的 episode 无效，不能恢复采集日志")
        groups.setdefault(index, []).append(record)
        frames.append({"image": record["cam/image_array"], "episode": index, "label": record["user/angle"],
                       "applied": record.get("collect/applied"), "offset": record.get("collect/offset"),
                       "error": record.get("sim/error"), "speed": record.get("sim/speed"),
                       "learner": record.get("collect/learner"), "intervened": record.get("collect/intervened", False),
                       "t": record.get("collect/t"), "mirrored": record.get("collect/mirrored", False)})
    for index, records in groups.items():
        if index in old and old[index].get("frames_stored") == len(records):
            continue
        originals = [r for r in records if not r.get("collect/mirrored")]
        errors = [abs(r["sim/error"]) for r in originals if r.get("sim/error") is not None]
        speeds = [r["sim/speed"] for r in originals if r.get("sim/speed") is not None]
        elapsed = max((r.get("collect/t") or 0. for r in records), default=0.)
        old[index] = {"episode": index, "seed": records[0].get("collect/seed"), "termination": "interrupted",
                      "error": "recovered from catalog; warm-up and discarded frames are unknown", "recovered": True,
                      "seconds": elapsed, "frames_seen": len(originals), "frames_stored": len(records),
                      "perturbed_frames": sum(bool(r.get("collect/offset")) for r in originals),
                      "learner_frames": sum(r.get("collect/learner") is not None for r in originals),
                      "interventions": sum(bool(r.get("collect/intervened")) for r in originals),
                      "frame_rate": None, "mean_abs_error": statistics.fmean(errors) if errors else None,
                      "max_abs_error": max(errors, default=None), "mean_speed": statistics.fmean(speeds) if speeds else None}
    rows = [row for _, row in sorted(old.items()) if row.get("frames_stored") == 0 or row["episode"] in groups]
    if sum(row["frames_stored"] for row in rows) != writer.count:
        raise DatasetError("episode 日志与 catalog 帧数不一致")
    for name, values in (("frames.jsonl", frames), ("episodes.jsonl", rows)):
        writer._atomic(directory / name, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in values))
    return rows


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
    if summary.get("learner_frames"):
        lines += [f"模型提供了 {summary['learner_frames']} 帧建议，视觉专家接管 {summary['interventions']} 帧"
                  f"（{100 * summary['interventions'] / summary['learner_frames']:.1f}%）。"
                  "未接管时由模型驾驶；所有训练标签仍来自视觉专家。", ""]
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
