"""Read archived closed-loop evaluations (``apps.eval.closed_loop`` outputs) for the console.

One evaluation is a directory with ``manifest.json`` (protocol, models), ``result.json`` (per-model aggregates
and paired differences from the baseline), ``episodes.jsonl`` (one row per model and scenario) and
``frames.jsonl`` (telemetry of every frame). Nothing here starts the simulator or touches the GPU.
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections import OrderedDict
from pathlib import Path

from .readers import downsample_indices, read_json

ARCHIVE = Path("docs") / "experiments" / "semester_2026_fall" / "closed_loop"
EVALUATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
METRICS = ("mean_abs_cte", "max_abs_cte", "excursions", "survival_s", "distance_units", "lap_count",
           "inference_p50_ms", "inference_p95_ms")
EPISODE_FIELDS = ("model", "scenario", "seed", "frames", "cte_frames", "termination", "target_speed_mph", *METRICS)
# Metrics where a larger value is the better one; every other metric is better when smaller.
HIGHER_IS_BETTER = ("survival_s", "distance_units", "lap_count")
TRACE_FIELDS = ("elapsed_s", "cte", "steering", "speed", "inference_ms")
MAX_TRACE_POINTS = 5000


class ClosedLoopError(Exception):
    def __init__(self, message: str, status: int = 404):
        super().__init__(message)
        self.status = status


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _rows(path: Path) -> list[dict]:
    """Complete JSON object lines of a JSONL file; a damaged or unfinished line is skipped."""
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.endswith("\n"):
                    break
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows


def _stat(path: Path) -> tuple | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


class ClosedLoopArchive:
    """Evaluations under one directory. Frame logs (10–15 MB each) are parsed once and kept for a few runs."""

    def __init__(self, root: Path, *, cache_size: int = 2):
        self.root = Path(root)
        self._cache: OrderedDict[str, tuple[tuple, dict]] = OrderedDict()
        self._cache_size = int(cache_size)
        self._lock = threading.Lock()

    # ---- discovery -----------------------------------------------------------------------------
    def _dir(self, evaluation_id: str) -> Path:
        if not isinstance(evaluation_id, str) or EVALUATION_ID.fullmatch(evaluation_id) is None:
            raise ClosedLoopError("闭环评估不存在")
        path = self.root / evaluation_id
        if path.resolve().parent != self.root.resolve() or not (path / "result.json").is_file():
            raise ClosedLoopError(f"闭环评估不存在：{evaluation_id}")
        return path

    def evaluations(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        out = []
        for path in sorted(self.root.iterdir(), reverse=True):
            if not path.is_dir() or EVALUATION_ID.fullmatch(path.name) is None or not (path / "result.json").is_file():
                continue
            try:
                out.append(self._summary(path))
            except Exception as exc:  # one damaged archive must not hide the others
                out.append({"id": path.name, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                            "models": [], "scenarios": 0})
        return out

    # ---- summaries -----------------------------------------------------------------------------
    @staticmethod
    def _offline(directory: Path, manifest_models: list) -> dict:
        """Offline steering MAE per model: recorded in the manifest, or in a ``<model>-offline.json`` file."""
        out = {}
        for item in manifest_models:
            if isinstance(item, dict) and _number(item.get("offline_angle_mae")) is not None:
                out[str(item.get("id"))] = item["offline_angle_mae"]
        for path in directory.glob("*-offline.json"):
            results = (read_json(path) or {}).get("results")
            first = results[0] if isinstance(results, list) and results and isinstance(results[0], dict) else {}
            if _number(first.get("angle_mae")) is not None:
                out.setdefault(path.name[:-len("-offline.json")], first["angle_mae"])
        return out

    def _summary(self, directory: Path) -> dict:
        result = read_json(directory / "result.json") or {}
        manifest = read_json(directory / "manifest.json") or {}
        protocol = manifest.get("protocol") if isinstance(manifest.get("protocol"), dict) else {}
        manifest_models = manifest.get("models") if isinstance(manifest.get("models"), list) else []
        offline = self._offline(directory, manifest_models)
        paths = {str(m.get("id")): m.get("path") for m in manifest_models if isinstance(m, dict)}
        baseline = result.get("baseline")
        models = []
        for name, data in (result.get("models") or {}).items():
            if not isinstance(data, dict):
                continue
            metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
            paired = data.get("paired_difference_from_baseline")
            paired = paired if isinstance(paired, dict) else {}
            models.append({
                "id": name, "path": paths.get(name), "baseline": name == baseline,
                "episodes": data.get("episodes"), "collisions": data.get("collisions"),
                "failures": data.get("failures"),
                "time_limits": data.get("time_limits"), "offline_angle_mae": offline.get(name),
                "metrics": {k: {"mean": _number((metrics.get(k) or {}).get("mean")),
                                "worst": _number((metrics.get(k) or {}).get("worst"))} for k in METRICS},
                "paired": None if name == baseline else {
                    k: {"mean": _number((paired.get(k) or {}).get("mean")),
                        "std": _number((paired.get(k) or {}).get("std")), "n": (paired.get(k) or {}).get("n")}
                    for k in METRICS if isinstance(paired.get(k), dict)},
            })
        scenarios = protocol.get("scenarios") if isinstance(protocol.get("scenarios"), list) else []
        session = manifest.get("session") if isinstance(manifest.get("session"), dict) else {}
        return {
            "id": directory.name, "status": result.get("status"), "baseline": baseline,
            "environment": protocol.get("environment") or session.get("environment"),
            "fixed_throttle": _number(result.get("fixed_throttle")), "device": manifest.get("device"),
            # Udacity runs hold a target speed, estimate CTE from the camera and measure distance in metres.
            "target_speed_mph": _number(result.get("target_speed_mph")),
            "cte_source": result.get("cte_source") if isinstance(result.get("cte_source"), str) else None,
            "distance_unit": result.get("distance_unit") if isinstance(result.get("distance_unit"), str) else None,
            "rules": protocol.get("rules") if isinstance(protocol.get("rules"), dict) else {},
            "scenarios": len(scenarios), "scenario_ids": [s.get("id") for s in scenarios if isinstance(s, dict)],
            "duration_s": next((_number(s.get("duration_s")) for s in scenarios if isinstance(s, dict)), None),
            "start_control": session.get("start_position_control"), "models": models,
            "finished_unix_s": _mtime(directory / "result.json"),
            "has_frames": (directory / "frames.jsonl").is_file(),
        }

    def detail(self, evaluation_id: str) -> dict:
        directory = self._dir(evaluation_id)
        summary = self._summary(directory)
        episodes = [{k: row.get(k) for k in EPISODE_FIELDS} for row in _rows(directory / "episodes.jsonl")]
        manifest = read_json(directory / "manifest.json") or {}
        protocol = manifest.get("protocol")
        protocol = protocol if isinstance(protocol, dict) else {}
        scenarios = protocol.get("scenarios")
        speeds = {s.get("id"): _number(s.get("speed_mph"))
                  for s in scenarios if isinstance(s, dict)} if isinstance(scenarios, list) else {}
        baseline = summary["baseline"]
        reference = {row["scenario"]: row for row in episodes if row["model"] == baseline}
        for row in episodes:
            if summary["cte_source"] == "camera_estimate" and _number(row.get("target_speed_mph")) is None:
                row["target_speed_mph"] = speeds.get(row["scenario"]) or summary["target_speed_mph"]
            base = reference.get(row["scenario"])
            row["difference"] = None
            if base is not None and row["model"] != baseline:
                row["difference"] = {
                    k: row[k] - base[k] for k in METRICS
                    if _number(row.get(k)) is not None and _number(base.get(k)) is not None}
        return {**summary, "episodes": episodes, "higher_is_better": list(HIGHER_IS_BETTER)}

    # ---- frame traces --------------------------------------------------------------------------
    def _frames(self, directory: Path) -> dict:
        path = directory / "frames.jsonl"
        stamp = _stat(path)
        if stamp is None:
            raise ClosedLoopError("这次评估没有保存逐帧记录")
        key = str(directory)
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached[0] == stamp:
                self._cache.move_to_end(key)
                return cached[1]
        groups: dict[tuple[str, str], dict] = {}
        for row in _rows(path):
            model, scenario = row.get("model"), row.get("scenario")
            if not isinstance(model, str) or not isinstance(scenario, str):
                continue
            columns = groups.setdefault((model, scenario), {**{f: [] for f in TRACE_FIELDS}, "x": [], "z": [], "hit": []})
            for field in TRACE_FIELDS:
                columns[field].append(_number(row.get(field)))
            pos = row.get("pos")
            ok = isinstance(pos, list) and len(pos) == 3
            columns["x"].append(_number(pos[0]) if ok else None)
            columns["z"].append(_number(pos[2]) if ok else None)
            hit = row.get("hit")
            columns["hit"].append(bool(hit) and str(hit).strip().lower() not in ("", "none"))
        with self._lock:
            self._cache[key] = (stamp, groups)
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return groups

    def trace(self, evaluation_id: str, model: str, scenario: str, max_points: int = 1500) -> dict:
        """One episode's telemetry, thinned to ``max_points`` frames; frames beyond the excursion threshold and
        collisions are always kept, so the markers do not depend on the thinning."""
        directory = self._dir(evaluation_id)
        columns = self._frames(directory).get((model, scenario))
        if columns is None:
            raise ClosedLoopError(f"没有 {model} 在 {scenario} 的逐帧记录")
        rules = self._summary(directory)["rules"]
        threshold = _number(rules.get("cte_threshold"))
        total = len(columns["elapsed_s"])
        keep = set(downsample_indices(total, max(2, min(int(max_points), MAX_TRACE_POINTS))))
        outside = [i for i, cte in enumerate(columns["cte"])
                   if threshold is not None and cte is not None and abs(cte) > threshold]
        hits = [i for i, hit in enumerate(columns["hit"]) if hit]
        # A long excursion has many frames: keep its first and last frame, plus every collision frame.
        edges = [i for n, i in enumerate(outside) if n == 0 or outside[n - 1] != i - 1
                 or n == len(outside) - 1 or outside[n + 1] != i + 1]
        index = sorted(keep | set(edges) | set(hits))
        series = {field: [columns[field][i] for i in index] for field in (*TRACE_FIELDS, "x", "z")}
        series["outside"] = [threshold is not None and columns["cte"][i] is not None
                             and abs(columns["cte"][i]) > threshold for i in index]
        series["hit"] = [columns["hit"][i] for i in index]
        return {"evaluation": evaluation_id, "model": model, "scenario": scenario, "frames": total,
                "cte_frames": sum(cte is not None for cte in columns["cte"]),
                "points": len(index), "cte_threshold": threshold, "outside_frames": len(outside),
                "collision_frames": len(hits), "has_positions": any(x is not None for x in columns["x"]),
                "series": series}
