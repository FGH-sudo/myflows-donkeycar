"""Attribute Nsight Systems GPU idle intervals to NVTX steps, phases and source locations."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
import csv
import json
from pathlib import Path
import re
import sqlite3

PHASES = {"data", "data_prepare", "forward", "backward", "optimizer", "sync", "validation", "checkpoint"}


def union(intervals):
    """Union overlapping streams: concurrent GPU activity must not count twice."""
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap_ns(intervals, start, end):
    return sum(max(0, min(end, b) - max(start, a)) for a, b in intervals if b > start and a < end)


def load_capture(path: Path, *, device: int = 0):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        strings = dict(connection.execute("SELECT id, value FROM StringIds")) if "StringIds" in tables else {}
        activities = []
        for suffix, kind in (("KERNEL", "kernel"), ("MEMCPY", "memcpy"), ("MEMSET", "memset")):
            table = f"CUPTI_ACTIVITY_KIND_{suffix}"
            if table not in tables:
                continue
            for item in connection.execute(f"SELECT * FROM {table}"):
                row = dict(item)
                if row.get("deviceId", device) != device or row["end"] <= row["start"]:
                    continue
                activities.append({"start": row["start"], "end": row["end"], "kind": kind,
                                   "name": strings.get(row.get("shortName"), str(row.get("shortName", kind)))})
        if "NVTX_EVENTS" not in tables:
            raise ValueError("capture has no NVTX_EVENTS; enable --nvtx")
        ranges = []
        for item in connection.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL AND end > start"):
            row = dict(item)
            name = row.get("text") or strings.get(row.get("textId"), "")
            ranges.append({"start": row["start"], "end": row["end"], "name": name, "tid": row.get("globalTid")})
    return activities, ranges


def analyze(path: Path, nvtx_map=None, *, threshold_ms=.1, top=20, device=0, thread=None):
    if threshold_ms < 0 or top < 1:
        raise ValueError("threshold must be nonnegative and top positive")
    activities, ranges = load_capture(Path(path), device=device)
    source_map = (nvtx_map or {}).get("ranges", nvtx_map or {})
    for item in ranges:
        item["phase"] = item["name"].split(" @", 1)[0]
    steps = [r for r in ranges if re.fullmatch(r"step \d+", r["name"]) and (thread is None or r["tid"] == thread)]
    if not steps:
        raise ValueError("no complete 'step <n>' NVTX ranges found")
    tids = {r["tid"] for r in steps}
    if len(tids) > 1 and thread is None:
        raise ValueError("multiple training threads captured; select --thread (globalTid) to avoid mixing workers")
    steps.sort(key=lambda r: r["start"])
    if any(a["end"] > b["start"] for a, b in zip(steps, steps[1:])):
        raise ValueError("overlapping step ranges cannot share a wall-time denominator")
    busy = union((a["start"], a["end"]) for a in activities)
    phase_totals = defaultdict(lambda: {"wall_ns": 0, "gpu_idle_ns": 0})
    gap_rows, phase_segments, step_rows = [], [], []
    total_ns = sum(s["end"] - s["start"] for s in steps)
    for step in steps:
        start, end = step["start"], step["end"]
        number = int(step["name"].split()[1])
        local = [r for r in ranges if r["tid"] == step["tid"] and r["end"] > start and r["start"] < end
                 and (r["phase"] in PHASES or r["name"].startswith("node:"))]
        boundaries = sorted({start, end, *(max(start, r["start"]) for r in local), *(min(end, r["end"]) for r in local)})
        # Partition the step by CPU scopes, including explicit overhead where no phase is open.
        partitions = []
        for left, right in zip(boundaries, boundaries[1:]):
            containing = [r for r in local if r["start"] <= left and r["end"] >= right]
            phases = [r for r in containing if r["phase"] in PHASES]
            phase_range = min(phases, key=lambda r: r["end"] - r["start"]) if phases else step
            phase = phase_range["phase"] if phases else "overhead"
            if phase == "data_prepare":
                phase = "data"
            nodes = [r for r in containing if r["name"].startswith("node:")]
            owner = min(nodes, key=lambda r: r["end"] - r["start"]) if nodes else phase_range
            source = source_map.get(owner["name"]) or source_map.get(phase_range["name"]) or source_map.get(step["name"]) or {}
            partition = {"start": left, "end": right, "step": number, "phase": phase,
                         "node": owner["name"] if nodes else None, "range": owner["name"],
                         "file": source.get("file"), "line": source.get("line")}
            partitions.append(partition)
            idle_ns = right - left - overlap_ns(busy, left, right)
            phase_totals[phase]["wall_ns"] += right - left
            phase_totals[phase]["gpu_idle_ns"] += idle_ns
            phase_segments.append(partition)
        cursor = start
        idle_intervals = []
        for left, right in busy:
            if right <= start or left >= end:
                continue
            if left > cursor:
                idle_intervals.append((cursor, min(left, end)))
            cursor = max(cursor, min(right, end))
        if cursor < end:
            idle_intervals.append((cursor, end))
        # Apply threshold before splitting at CPU scopes; short fragments of a long gap remain visible.
        for gap_index, (left, right) in enumerate(idle_intervals):
            if right - left < threshold_ms * 1e6:
                continue
            for part in partitions:
                a, b = max(left, part["start"]), min(right, part["end"])
                if b <= a:
                    continue
                gap_rows.append({**part, "start": a, "end": b, "duration_ms": (b - a) / 1e6,
                                 "gap_id": f"{number}:{gap_index}", "whole_gap_ms": (right - left) / 1e6})
        kernels = [a for a in activities if a["kind"] == "kernel" and start <= a["start"] < end]
        lengths = [a["end"] - a["start"] for a in kernels]
        step_rows.append({"step": number, "start": start, "end": end, "wall_ms": (end - start) / 1e6,
                          "gpu_idle_ms": sum(b - a for a, b in idle_intervals) / 1e6,
                          "kernel_count": len(kernels), "kernel_mean_us": sum(lengths) / len(lengths) / 1000 if lengths else None})
    groups = defaultdict(list)
    for activity in activities:
        if activity["kind"] == "kernel" and any(s["start"] <= activity["start"] < s["end"] for s in steps):
            groups[activity["name"]].append((activity["end"] - activity["start"]) / 1000)
    kernels = [{"name": name, "count": len(lengths), "total_ms": sum(lengths) / 1000,
                "mean_us": sum(lengths) / len(lengths), "max_us": max(lengths)} for name, lengths in groups.items()]
    kernels.sort(key=lambda row: row["total_ms"], reverse=True)
    phases = [{"phase": phase, "wall_ms": values["wall_ns"] / 1e6, "gpu_idle_ms": values["gpu_idle_ns"] / 1e6,
               "wall_share_of_steps": values["wall_ns"] / total_ns,
               "gpu_idle_share_of_phase": values["gpu_idle_ns"] / values["wall_ns"],
               "gpu_idle_share_of_steps": values["gpu_idle_ns"] / total_ns} for phase, values in phase_totals.items()]
    return {"schema_version": 1, "device": device, "thread": next(iter(tids)), "threshold_ms": threshold_ms,
            "step_wall_ms": total_ns / 1e6, "phases": phases, "steps": step_rows, "gaps": gap_rows,
            "top_kernels": kernels[:top], "unmapped_gaps": sum(not r["file"] or not r["line"] for r in gap_rows),
            "activities": activities, "phase_segments": phase_segments,
            "scope": "device activity union inside complete steps; CPU NVTX scopes label host-side gaps, not GPU kernel causality"}


def plot_timeline(report, path: Path, *, max_steps=5):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    steps = report["steps"][:max_steps]
    origin, end = steps[0]["start"], steps[-1]["end"]
    fig, ax = plt.subplots(figsize=(15, 3.8), constrained_layout=True)
    colors = {phase: plt.get_cmap("tab10")(i) for i, phase in enumerate(("data", "forward", "backward", "optimizer", "sync", "checkpoint", "overhead", "validation"))}
    for step in steps:
        x, width = (step["start"] - origin) / 1e6, (step["end"] - step["start"]) / 1e6
        ax.broken_barh([(x, width)], (2.1, .65), facecolors="#dbeafe", edgecolors="white")
        ax.text(x + width / 2, 2.42, f"step {step['step']}", ha="center", va="center", fontsize=9)
    for item in report["phase_segments"]:
        if item["start"] >= end or item["end"] <= origin:
            continue
        ax.broken_barh([((item["start"] - origin) / 1e6, (item["end"] - item["start"]) / 1e6)], (1.1, .65),
                      facecolors=colors[item["phase"]])
    for item in report["activities"]:
        a, b = max(origin, item["start"]), min(end, item["end"])
        if a < b:
            ax.broken_barh([((a - origin) / 1e6, (b - a) / 1e6)], (.1, .65),
                          facecolors="#334155" if item["kind"] == "kernel" else "#f97316")
    ax.set(yticks=[.42, 1.42, 2.42], yticklabels=["GPU activity", "CPU phase", "Training step"],
           xlabel="Time from first captured step (ms)", xlim=(0, (end - origin) / 1e6), ylim=(-.1, 3))
    ax.legend(handles=[Patch(color=color, label=phase) for phase, color in colors.items() if any(p["phase"] == phase for p in report["phases"])],
              loc="upper center", bbox_to_anchor=(.5, 1.2), ncol=8, frameon=False)
    ax.grid(axis="x", alpha=.15)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_outputs(report, out: Path, *, plot=True):
    out.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in report.items() if k not in ("activities", "phase_segments")}
    (out / "timeline.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for kind in ("gaps", "phases", "steps", "top_kernels"):
        rows = summary[kind]
        with (out / f"{kind}.csv").open("w", encoding="utf-8", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    if plot:
        plot_timeline(report, out / "timeline.png")
    lines = ["# 训练时间线分析", "", f"完整 step 总时间 {report['step_wall_ms']:.3f} ms；"
             f"空闲阈值 {report['threshold_ms']:.3f} ms；未映射来源的空闲片段 {report['unmapped_gaps']}。",
             "", "GPU 忙区间是 kernel、memcpy、memset 在所有 stream 上的并集。各阶段是 CPU NVTX 分区，"
             "没有阶段标记的部分明确记为 overhead；阶段墙钟总和等于 step 墙钟总和。"
             "GPU 与 CPU 异步，空闲段的主机源码关联用于定位发射/等待位置，不声称为某个 kernel 的因果归属。",
             "", "| 阶段 | wall ms | GPU idle ms | 占全部 step 时间 | 阶段内 idle 占比 |",
             "| --- | --- | --- | --- | --- |"]
    for row in report["phases"]:
        lines.append(f"| {row['phase']} | {row['wall_ms']:.3f} | {row['gpu_idle_ms']:.3f} | {row['wall_share_of_steps']:.2%} | {row['gpu_idle_share_of_phase']:.2%} |")
    lines += ["", "| step | kernel 数 | 平均 kernel µs | GPU idle ms |", "| --- | --- | --- | --- |"]
    for row in report["steps"]:
        mean = "—" if row["kernel_mean_us"] is None else f"{row['kernel_mean_us']:.3f}"
        lines.append(f"| {row['step']} | {row['kernel_count']} | {mean} | {row['gpu_idle_ms']:.3f} |")
    lines += ["", "最长空闲片段（完整表见 gaps.csv；分割后的片段共享 gap_id）：", "",
              "| step | 阶段 | 时长 ms | 节点 | 源码 |", "| --- | --- | --- | --- | --- |"]
    for row in sorted(report["gaps"], key=lambda r: r["duration_ms"], reverse=True)[:20]:
        lines.append(f"| {row['step']} | {row['phase']} | {row['duration_ms']:.3f} | {row['node'] or '—'} | {row['file']}:{row['line']} |")
    lines += ["", "最长累计 kernel（完整表见 top_kernels.csv）：", "", "| kernel | 次数 | 总 ms | 平均 µs |", "| --- | --- | --- | --- |"]
    for row in report["top_kernels"]:
        lines.append(f"| {row['name']} | {row['count']} | {row['total_ms']:.3f} | {row['mean_us']:.3f} |")
    if plot:
        lines += ["", "本地可复现图：`timeline.png`（不纳入 Git，与 capture.sqlite 一同交付）。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--map", type=Path, dest="map_path")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--threshold-ms", type=float, default=.1)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--thread", type=int)
    args = parser.parse_args(argv)
    map_path = args.map_path or next((path for path in (args.capture.parent / "nvtx_map.json",
                                                        args.capture.parent / "workload" / "nvtx_map.json")
                                      if path.is_file()), None)
    mapping = json.loads(map_path.read_text(encoding="utf-8")) if map_path else {}
    report = analyze(args.capture, mapping, threshold_ms=args.threshold_ms, top=args.top, device=args.device, thread=args.thread)
    write_outputs(report, args.out or args.capture.parent / "timeline")
    print(json.dumps({k: report[k] for k in ("step_wall_ms", "unmapped_gaps", "phases")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
