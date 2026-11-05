"""Probe generated-road telemetry, reset positions and steering excursions before locking assessment rules."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

from .simulator import FIXED_THROTTLE, ROOT, guarded_worker, session, write_json


def summarize(rows):
    elapsed = rows[-1]["monotonic_s"] - rows[0]["monotonic_s"] if len(rows) > 1 else 0
    fields = {}
    names = ("cte", "speed", "pos", "hit", "gyro", "accel", "vel", "car", "lidar_points", "lap_count", "last_lap_time",
             "forward_vel", "active_node", "total_nodes", "simulation_time", "observed_steering", "observed_throttle")
    for key in names:
        values = [row[key] for row in rows if row.get(key) is not None]
        summary = {"samples": len(values), "available_share": len(values) / len(rows) if rows else 0,
                   "observations_per_s": (len(values) - 1) / elapsed if len(values) > 1 and elapsed > 0 else None}
        if values and key == "hit":
            summary["counts"] = dict(Counter(str(value) for value in values))
        elif values:
            data = np.asarray(values, dtype=float)
            summary.update(min=data.min(axis=0).tolist(), max=data.max(axis=0).tolist())
        fields[key] = summary
    wire = Counter(key for row in rows for key in row["wire_fields"])
    return {"frames": len(rows), "elapsed_s": elapsed, "fields": fields, "wire_field_counts": dict(wire),
            "line_events": max((row["line_events"] for row in rows), default=0)}


def probe(stream, out, duration_s, metadata):
    from PIL import Image

    artifacts = out / "artifacts"
    artifacts.mkdir(exist_ok=True)
    episodes = []
    with (out / "frames.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for index, (seed, steering) in enumerate(((17, 0.), (17, 0.), (23, .65))):
            image, initial = stream.reset(seed=seed)
            rows, snapshots = [], {}
            start = time.perf_counter()
            while time.perf_counter() - start < duration_s:
                image, frame = stream.step((steering, FIXED_THROTTLE))
                row = {"schema_version": 1, "episode": index, "seed": seed, "steering": steering,
                       "throttle": FIXED_THROTTLE, **frame}
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                cte = frame.get("cte")
                key = "missing" if cte is None else str(min(12, int(abs(float(cte)))))
                if key not in snapshots:
                    path = artifacts / f"episode-{index}-cte-{key}.jpg"
                    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
                    snapshots[key] = {"image": path.relative_to(out).as_posix(), "cte": cte, "frame_id": frame["frame_id"]}
                if str(frame.get("hit") or "none").lower() != "none":
                    break
            stream.brake()
            episodes.append({"seed": seed, "steering": steering, "start": initial, "snapshots": snapshots,
                             **summarize(rows)})
            write_json(out / "probe.json", {"schema_version": 1, "session": metadata, "episodes": episodes})
    return episodes


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/semester_2026_fall/closed_loop/probe-001"))
    parser.add_argument("--duration-s", type=float, default=10.)
    parser.add_argument("--sim", help="Unity executable or remote")
    parser.add_argument("--port", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 0 < args.duration_s <= 120:
        parser.error("--duration-s must be in (0, 120]")
    out = (ROOT / args.out).resolve()
    if not args.worker:
        out.mkdir(parents=True, exist_ok=False)
        guarded_worker("apps.eval.sim_probe", argv, 100 + 3 * (args.duration_s + 10), out)
        return
    with session(out, executable=args.sim, port=args.port) as (stream, metadata):
        episodes = probe(stream, out, args.duration_s, metadata)
    lines = ["# generated-road 遥测探测", "", "固定油门 0.2。前两次使用相同客户端种子和直行动作，第三次保持转向 0.65。", "",
             "| reset | seed | 转向 | 帧数 | 起点 | CTE 范围 | 速度范围 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for i, row in enumerate(episodes):
        cte, speed = row["fields"]["cte"], row["fields"]["speed"]
        lines.append(f"| {i+1} | {row['seed']} | {row['steering']} | {row['frames']} | {row['start']['pos']} | "
                     f"{cte.get('min')} 至 {cte.get('max')} | {speed.get('min')} 至 {speed.get('max')} |")
    lines += ["", "Gymnasium 的 reset(seed) 设置客户端 RNG。SDK 向 Unity 发送 reset_car，消息使用模拟器默认起点。"
              "配对清单须保存请求的种子和各次实际起点。", "",
              "probe.json 包含各字段观测频率、范围和原始消息字段计数。frames.jsonl 与 artifacts/ 保存逐帧证据与按 CTE 取样的画面。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"captured {sum(r['frames'] for r in episodes)} frames", flush=True)


if __name__ == "__main__":
    main()
