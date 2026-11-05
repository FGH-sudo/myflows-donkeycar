"""Evaluate paired fixed-throttle scenarios using the driving stack's checkpoint/ONNX predictors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from .simulator import FIXED_THROTTLE, ROOT, guarded_worker, session, write_json


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class EpisodeMetrics:
    def __init__(self, rules):
        self.threshold = float(rules["cte_threshold"])
        self.rearm = float(rules["rearm_threshold"])
        self.required = int(rules["consecutive_frames"])
        self.rearm_frames = int(rules["rearm_frames"])
        if not (0 <= self.rearm < self.threshold and min(self.required, self.rearm_frames) >= 1):
            raise ValueError("invalid excursion thresholds or debounce frame counts")
        self.above = self.inside = self.events = self.frames = 0
        self.latched = False
        self.ctes, self.latencies = [], []
        self.distance = 0.
        self.position_intervals = 0
        self.previous_position = None
        self.laps = 0

    def position(self, coordinates):
        point = None
        if isinstance(coordinates, (list, tuple)) and len(coordinates) == 3:
            values = tuple(finite(v) for v in coordinates)
            if all(v is not None for v in values):
                point = values
        if point is not None and self.previous_position is not None:
            self.distance += math.dist(point, self.previous_position)
            self.position_intervals += 1
        self.previous_position = point

    def add(self, row):
        self.frames += 1
        self.position(row.get("pos"))
        latency = finite(row.get("inference_ms"))
        if latency is not None:
            self.latencies.append(latency)
        self.laps = max(self.laps, int(row.get("lap_count") or 0))
        cte = finite(row.get("cte"))
        if cte is None:
            self.above = self.inside = 0
            return
        magnitude = abs(cte)
        self.ctes.append(magnitude)
        self.above = self.above + 1 if magnitude > self.threshold else 0
        self.inside = self.inside + 1 if magnitude <= self.rearm else 0
        if not self.latched and self.above >= self.required:
            self.events += 1
            self.latched = True
        elif self.latched and self.inside >= self.rearm_frames:
            self.latched = False

    def summary(self, elapsed, reason):
        return {"frames": self.frames, "cte_frames": len(self.ctes),
                "mean_abs_cte": statistics.fmean(self.ctes) if self.ctes else None,
                "max_abs_cte": max(self.ctes, default=None), "excursions": self.events if self.ctes else None,
                "survival_s": elapsed, "distance_units": self.distance if self.position_intervals else None,
                "position_intervals": self.position_intervals, "lap_count": self.laps, "termination": reason,
                "inference_p50_ms": float(np.percentile(self.latencies, 50)) if self.latencies else None,
                "inference_p95_ms": float(np.percentile(self.latencies, 95)) if self.latencies else None}


def episode(stream, predict, scenario, rules, log, *, model_id, clock=time.perf_counter):
    if scenario.get("start") != "simulator_default":
        raise ValueError("this simulator interface supports its default reset position only")
    metrics = EpisodeMetrics(rules)
    observation, initial = stream.reset(seed=int(scenario["seed"]))
    metrics.position(initial.get("pos"))
    start = clock()
    reason = "time_limit"
    try:
        while clock() - start < scenario["duration_s"]:
            begin = clock()
            steering = finite(predict(observation))
            latency_ms = (clock() - begin) * 1000
            if steering is None:
                raise ValueError("model returned nonfinite steering")
            steering = float(np.clip(steering, -1., 1.))
            observation, frame = stream.step((steering, FIXED_THROTTLE))
            row = {key: frame.get(key) for key in ("unix_s", "monotonic_s", "frame_id", "pos", "cte", "speed",
                                                    "hit", "lap_count", "last_lap_time", "line_events", "active_node",
                                                    "total_nodes", "simulation_time", "observed_steering", "observed_throttle")}
            row.update(schema_version=1, model=model_id, scenario=scenario["id"], seed=scenario["seed"],
                       elapsed_s=clock() - start, steering=steering, throttle=FIXED_THROTTLE, inference_ms=latency_ms)
            row["missing"] = {key: "telemetry field absent" for key in ("pos", "cte", "speed", "hit") if row[key] is None}
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            metrics.add(row)
            if row["hit"] is not None and str(row["hit"]).strip().lower() not in ("", "none"):
                reason = "collision"
                break
            if rules.get("route_signal") == "lap" and metrics.laps > int(initial.get("lap_count") or 0):
                reason = "lap_completed"
                break
        elapsed = clock() - start
    finally:
        stream.brake()
    return {"schema_version": 1, "model": model_id, "scenario": scenario["id"], "seed": scenario["seed"],
            "start": initial.get("pos"), **metrics.summary(elapsed, reason)}


def model_predictor(item, device):
    path = (ROOT / item["path"]).resolve()
    config = item.get("config")
    config = json.loads((ROOT / config).read_text(encoding="utf-8")) if config else None
    width, height = (config["input_w"], config["input_h"]) if config else (160, 120)
    if path.suffix.lower() == ".onnx":
        from apps.serve.onnx_predictor import OnnxPredictor

        predictor = OnnxPredictor(path, width, height, device=device)
        # Camera dimensions are declared by fixed-shape exports; dynamic exports use the training config.
        shape = predictor.input_shape
        if len(shape) == 4 and all(isinstance(dim, int) and dim > 0 for dim in shape[2:]):
            predictor.image_h, predictor.image_w = shape[2:]
        return lambda frame: float(predictor.predict(frame)[0])
    from mycar.myflows_pilot import MyFlowsResNet18Pilot

    pilot = MyFlowsResNet18Pilot(path, image_w=width, image_h=height, device=device, fixed_throttle=FIXED_THROTTLE,
                                 max_throttle=FIXED_THROTTLE, training_config=config, prefer_onnx=False)
    return lambda frame: pilot.run(frame)[0]


# Worst direction is specified per metric: e.g. high CTE and short survival are adverse.
WORST = {"mean_abs_cte": max, "max_abs_cte": max, "excursions": max, "survival_s": min,
         "distance_units": min, "lap_count": min, "inference_p50_ms": max, "inference_p95_ms": max}


def aggregate(episodes, baseline):
    grouped = {}
    for row in episodes:
        by_scenario = grouped.setdefault(row["model"], {})
        if row["scenario"] in by_scenario:
            raise ValueError("duplicate model/scenario episode")
        by_scenario[row["scenario"]] = row
    out = {}
    for model, group in grouped.items():
        metrics, pairs = {}, {}
        for key, worst in WORST.items():
            vals = [r[key] for r in group.values() if r[key] is not None]
            metrics[key] = {"mean": statistics.fmean(vals) if vals else None, "worst": worst(vals) if vals else None, "n": len(vals)}
            deltas = [row[key] - grouped[baseline][s][key] for s, row in group.items()
                      if s in grouped.get(baseline, {}) and row[key] is not None and grouped[baseline][s][key] is not None]
            pairs[key] = {"mean": statistics.fmean(deltas) if deltas else None,
                          "std": statistics.stdev(deltas) if len(deltas) > 1 else None, "n": len(deltas)}
        out[model] = {"episodes": len(group), "collisions": sum(r["termination"] == "collision" for r in group.values()),
                      "time_limits": sum(r["termination"] == "time_limit" for r in group.values()),
                      "metrics": metrics, "paired_difference_from_baseline": pairs}
    return out


def _fingerprints(item):
    path = ROOT / item["path"]
    files = [path] if path.suffix == ".onnx" else [path.with_suffix(".json"), path.with_suffix(".npz")]
    if item.get("config"):
        files.append(ROOT / item["config"])
    digests = {}
    for file in files:
        with file.open("rb") as handle:
            digests[file.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return digests


def frame_statistics(path):
    groups = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ctes, delays = groups.setdefault(row["model"], ([], []))
            if (cte := finite(row.get("cte"))) is not None:
                ctes.append(abs(cte))
            delays.append(row["inference_ms"])
    return {name: {"frames": len(delays), "cte_frames": len(ctes),
                   "mean_abs_cte": statistics.fmean(ctes) if ctes else None, "max_abs_cte": max(ctes, default=None),
                   "inference_p50_ms": float(np.percentile(delays, 50)), "inference_p95_ms": float(np.percentile(delays, 95))}
            for name, (ctes, delays) in groups.items()}


def _report(out, models, result):
    lines = ["# DonkeyCar 配对闭环评估", "", "每个场景使用相同的默认 reset、客户端种子、终止规则、时限和 0.2 油门。"
             "模型顺序按场景轮换。位置距离使用连续有效 3D 遥测点之间的欧氏距离。", "",
             "| 模型 | 次数 | 碰撞次数 | 平均 / 最差存活秒 | 平均 / 最差距离 | 平均 / 最差绝对 CTE | "
             "平均 / 最差最大绝对 CTE | 平均 / 最差出界数 | 全部帧 P50 / P95 推理 ms |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    def display(value, digits=3):
        return "缺测" if value is None else f"{value:.{digits}f}"
    for model in models:
        name = model["id"]
        row = result["models"][name]
        m = row["metrics"]
        cells = [f"{display(m[k]['mean'])} / {display(m[k]['worst'])}" for k in
                 ("survival_s", "distance_units", "mean_abs_cte", "max_abs_cte", "excursions")]
        lines.append(f"| {name} | {row['episodes']} | {row['collisions']} | " + " | ".join(cells)
                     + f" | {display(row['frame_statistics']['inference_p50_ms'])} / {display(row['frame_statistics']['inference_p95_ms'])} |")
    lines += ["", f"配对差值采用模型值减去 {result['baseline']} 在同一场景的值。", "",
              "| 模型 | 离线角度 MAE | 平均差 Δ存活秒 | Δ距离 | Δ平均绝对 CTE | Δ出界数 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for model in models:
        name = model["id"]
        pairs = result["models"][name]["paired_difference_from_baseline"]
        lines.append(f"| {name} | {display(model.get('offline_angle_mae'), 5)} | " +
                     " | ".join(display(pairs[k]["mean"]) for k in ("survival_s", "distance_units", "mean_abs_cte", "excursions")) + " |")
    lines += ["", "manifest.json 记录规则、场景、模型来源和 SHA-256。frames.jsonl 保存逐帧遥测，episodes.jsonl 保存各次 reset 起点与结局，"
              "result.json 保存平均、最差和配对差异。Unity 场景随机种子和任意起点坐标无法通过当前接口指定。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--baseline", help="model ID used to calculate paired differences, defaults to the first model")
    parser.add_argument("--sim", help="simulator executable or remote")
    parser.add_argument("--port", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    models = json.loads((ROOT / args.models).read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / args.scenarios).read_text(encoding="utf-8"))
    scenarios, rules = protocol["scenarios"], protocol["rules"]
    names = [m["id"] for m in models]
    if not names or len(set(names)) != len(names) or not scenarios or len({s["id"] for s in scenarios}) != len(scenarios):
        parser.error("models and scenarios must have unique, nonempty IDs")
    baseline = args.baseline or names[0]
    if baseline not in names:
        parser.error("--baseline must name a model in the manifest")
    if any(not 0 < s["duration_s"] <= 600 for s in scenarios):
        parser.error("scenario duration must be in (0, 600] seconds")
    EpisodeMetrics(rules)
    out = (ROOT / args.out).resolve()
    if not args.worker:
        out.mkdir(parents=True, exist_ok=False)
        timeout = 150 + len(models) * sum(15 + s["duration_s"] for s in scenarios)
        guarded_worker("apps.eval.closed_loop", argv, timeout, out)
        return
    from apps.common.gpu_lock import GpuLock
    from apps.autopilot.provenance import code_state

    manifest = {"schema_version": 1, "protocol": protocol, "models": [dict(m, sha256=_fingerprints(m)) for m in models],
                "device": args.device, "code": code_state(ROOT)}
    write_json(out / "manifest.json", manifest)
    rows = []
    with GpuLock(label="closed-loop evaluation", run_dir=out), session(out, executable=args.sim, port=args.port) as (stream, metadata):
        manifest["session"] = metadata
        write_json(out / "manifest.json", manifest)
        predictors = [model_predictor(m, args.device) for m in models]
        with (out / "frames.jsonl").open("w", encoding="utf-8", buffering=1) as frames, (out / "episodes.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            for index, scenario in enumerate(scenarios):
                for offset in range(len(models)):
                    model_index = (index + offset) % len(models)
                    name = names[model_index]
                    row = episode(stream, predictors[model_index], scenario, rules, frames, model_id=name)
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    rows.append(row)
                    print(f"{name} {scenario['id']} {row['termination']} {row['survival_s']:.2f}s", flush=True)
    result = {"schema_version": 1, "status": "succeeded", "baseline": baseline, "models": aggregate(rows, baseline),
              "route_signal": rules.get("route_signal"), "fixed_throttle": FIXED_THROTTLE}
    for name, frame_summary in frame_statistics(out / "frames.jsonl").items():
        result["models"][name]["frame_statistics"] = frame_summary
    write_json(out / "result.json", result)
    _report(out, models, result)


if __name__ == "__main__":
    main()
