"""Evidence: per-attempt summaries of metrics.jsonl / resources.jsonl / result.json, and the campaign view
the specialists and the Planner read. Every number keeps the file it came from, so a Finding can cite it."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from apps.common.gpu_lock import acquired_unix_s

from .files import read_json, read_jsonl

WARMUP_STEPS = 3
THERMAL_LIMIT_C = 83.0
# A sample is at the power cap when it draws this share of the limit enforced at that moment; the enforced
# limit moves with the platform's power management (60-80 W on the RTX 4060 laptop), so never a fixed value.
POWER_LIMIT_SHARE = 0.97


def _median(values):
    values = [v for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
    return statistics.median(values) if values else None


def summarize_ranks(run_dir: Path) -> dict | None:
    """Data-parallel runs (``apps.train --distributed``): per-rank step times from ``rank-*/steps.jsonl``.

    Ranks run concurrently, so the aggregate throughput is the sum of the per-rank rates. Synchronous
    training makes every rank's wall time equal (the fast ones wait), so stragglers show in the compute time
    (step wall time minus gradient synchronisation and update confirmation).
    """
    files = sorted(Path(run_dir).glob("rank-*/steps.jsonl"))
    if not files:
        return None
    ranks, rate = [], 0.0
    wall_total = sync_total = data_total = 0.0
    for path in files:
        rows = read_jsonl(path)
        steady = rows[WARMUP_STEPS:] if len(rows) > 2 * WARMUP_STEPS else rows
        if not steady:
            continue
        wall = [float(r.get("step_wall_s") or 0.0) for r in steady]
        sync = [float(r.get("gradient_sync_s") or 0.0) + float(r.get("update_confirm_s") or 0.0) for r in steady]
        data = [float(r.get("data_s") or 0.0) for r in steady]
        samples = sum(int(r.get("n_samples") or 0) for r in steady)
        if sum(wall) > 0:
            rate += samples / sum(wall)
        ranks.append({"rank": steady[0].get("rank"), "steps": len(steady),
                      "step_ms_median": _median(wall) * 1000.0,
                      "sync_ms_median": _median(sync) * 1000.0,
                      "compute_ms_median": _median([w - s for w, s in zip(wall, sync)]) * 1000.0})
        wall_total, sync_total, data_total = wall_total + sum(wall), sync_total + sum(sync), data_total + sum(data)
    if not ranks or wall_total <= 0:
        return None
    compute = [r["compute_ms_median"] for r in ranks]
    mean = statistics.fmean(compute)
    return {"workers": len(ranks), "ranks": ranks, "throughput_sps": rate,
            "sync_share": sync_total / wall_total, "data_share": data_total / wall_total,
            "rank_step_cv": statistics.pstdev(compute) / mean if len(compute) > 1 and mean > 0 else 0.0}


def summarize_attempt(run_dir: Path) -> dict:
    """Throughput, data-loading share, GPU utilisation, memory and phase timings of one attempt."""
    run_dir = Path(run_dir)
    result = read_json(run_dir / "result.json") or {}
    rows = read_jsonl(run_dir / "metrics.jsonl")
    steps = [r for r in rows if r.get("kind") == "step"]
    steady = steps[WARMUP_STEPS:] if len(steps) > 2 * WARMUP_STEPS else steps
    out: dict = {"run_dir": str(run_dir), "steps": len(steps), "sources": {
        "metrics": "metrics.jsonl", "resources": "resources.jsonl", "result": "result.json"}}
    if steady:
        data_ms = sum(float(r.get("data_load_ms") or 0.0) for r in steady)
        step_ms = sum(float(r.get("step_time_ms") or 0.0) for r in steady)
        samples = sum(int(r.get("batch_size") or 0) for r in steady)
        out["data_share"] = data_ms / (data_ms + step_ms) if data_ms + step_ms > 0 else None
        out["throughput_sps"] = samples / ((data_ms + step_ms) / 1000.0) if data_ms + step_ms > 0 else None
        out["compute_sps"] = samples / (step_ms / 1000.0) if step_ms > 0 else None
        out["step_ms_median"] = _median([r.get("step_time_ms") for r in steady])
        out["data_ms_median"] = _median([r.get("data_load_ms") for r in steady])
        timed = [r for r in steady if "forward_ms" in r]
        if timed:
            out["phase_ms"] = {phase: _median([r.get(f"{phase}_ms") for r in timed])
                               for phase in ("forward", "backward", "optimizer")}
            overhead = []
            for r in timed:
                gpu_ms = sum(float(r.get(f"{p}_ms") or 0.0) for p in ("forward", "backward", "optimizer"))
                if r.get("step_time_ms"):
                    overhead.append(max(0.0, 1.0 - gpu_ms / float(r["step_time_ms"])))
            out["host_overhead_share"] = _median(overhead)
        start, end = steady[0].get("unix_s"), steady[-1].get("unix_s")
    else:
        start = end = None

    resources = read_jsonl(run_dir / "resources.jsonl")
    window = [r for r in resources if start is None or (start - 1.0 <= (r.get("unix_s") or 0) <= end + 1.0)]
    out["gpu_util_median"] = _median([r.get("gpu_utilization_pct") for r in window])
    out["gpu_util_samples"] = sum(1 for r in window if r.get("gpu_utilization_pct") is not None)
    temps = [r.get("gpu_temperature_c") for r in window if r.get("gpu_temperature_c") is not None]
    powers = [r.get("gpu_power_w") for r in window if r.get("gpu_power_w") is not None]
    pairs = [(gpu.get("power_w"), gpu.get("power_limit_w")) for r in window for gpu in (r.get("gpus") or [])[:1]]
    pairs = [(p, lim) for p, lim in pairs if p is not None and lim]
    out["temp_max_c"] = max(temps) if temps else None
    out["power_max_w"] = max(powers) if powers else None
    out["power_limit_w"] = max(lim for _, lim in pairs) if pairs else None
    out["power_limit_min_w"] = min(lim for _, lim in pairs) if pairs else None
    hot = sum(1 for t in temps if t >= THERMAL_LIMIT_C)
    capped = sum(1 for p, lim in pairs if p >= POWER_LIMIT_SHARE * lim)
    out["power_capped_share"] = capped / len(pairs) if pairs else None
    out["thermal_limited"] = bool(temps and hot / len(temps) > 0.2)
    out["power_limited"] = bool(pairs and capped / len(pairs) > 0.2)
    out["throttled"] = out["thermal_limited"] or out["power_limited"]

    memory = result.get("memory") or {}
    peak, total = memory.get("peak_pool_mb"), memory.get("device_total_mb")
    used = [r.get("gpu_memory_mib") for r in window if r.get("gpu_memory_mib") is not None]
    # Device-wide use (other processes included): what is left for more processes on the same GPU.
    out["device_mem_max_mb"] = max(used) if used else None
    if peak is None:
        peak = max(used) if used else None
        for r in window:
            for gpu in r.get("gpus") or []:
                total = gpu.get("memory_total_mib") or total
        out["mem_source"] = "nvml_device" if peak is not None else None
    else:
        out["mem_source"] = memory.get("source")
    out["mem_peak_mb"], out["mem_total_mb"] = peak, total
    out["mem_share"] = peak / total if peak is not None and total else None
    device = out["device_mem_max_mb"]
    out["device_mem_share"] = device / total if device is not None and total else None
    ranks = summarize_ranks(run_dir)
    if ranks is not None:
        # The lead rank's metrics.jsonl counts only its own shard; the rank files describe the whole run.
        out["distributed"] = ranks
        out["throughput_sps"], out["data_share"] = ranks["throughput_sps"], ranks["data_share"]
        out["sources"]["ranks"] = "rank-*/steps.jsonl"
    out["lock_acquired_unix_s"] = acquired_unix_s(run_dir)
    out["samples_per_s_reported"] = (result.get("throughput") or {}).get("samples_per_s")
    transfer = result.get("transfer")
    if transfer:
        report, freeze = transfer.get("load_report") or {}, transfer.get("freeze") or {}
        out["transfer"] = {
            "arch": transfer.get("arch"), "freeze_stages": transfer.get("freeze_stages"),
            "frozen": freeze.get("frozen"), "trainable_values": freeze.get("trainable_values"),
            "frozen_values": freeze.get("frozen_values"), "frozen_unchanged": transfer.get("frozen_unchanged"),
            "frozen_sha256": (transfer.get("frozen_after") or {}).get("sha256"),
            "loaded": report.get("loaded"), "expected": report.get("expected"),
            "success_rate": report.get("success_rate"), "missing": len(report.get("missing") or []),
            "replaced_head": [r.get("source") for r in report.get("replaced_head") or []],
            "source_sha256": (report.get("source") or {}).get("source_sha256"), "npz_sha256": report.get("npz_sha256"),
            "input_norm": transfer.get("input_norm")}
        out["sources"]["transfer"] = "result.json#transfer"
    return out


@dataclass
class Evidence:
    """What the specialists and the Planner see: the spec, working memory and every trial with its summary."""

    spec: dict
    state: dict
    trials: list[dict]
    summaries: dict[str, dict] = field(default_factory=dict)

    @property
    def metric(self) -> str:
        return self.spec["objective"]["metric"]

    @property
    def minimize(self) -> bool:
        return self.spec["objective"]["direction"] == "minimize"

    def objective(self, trial: dict) -> float | None:
        value = ((trial.get("result") or {}).get("metrics") or {}).get(self.metric)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None

    def valid(self, phase: str | None = None) -> list[dict]:
        return [t for t in self.trials if (t.get("verdict") or {}).get("status") == "valid"
                and (phase is None or t.get("phase") == phase) and self.objective(t) is not None]

    def finished(self) -> list[dict]:
        return [t for t in self.trials if t.get("status") == "done"]

    def baseline(self) -> list[dict]:
        return [t for t in self.trials if t.get("phase") == "baseline"]

    def ranked(self) -> list[dict]:
        """Valid trials, best first; throughput breaks ties."""
        sign = 1.0 if self.minimize else -1.0
        return sorted(self.valid(), key=lambda t: (sign * self.objective(t),
                                                   -((self.summaries.get(t["trial_id"]) or {}).get("throughput_sps")
                                                     or 0.0)))

    def best(self) -> dict | None:
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def summary(self, trial: dict) -> dict:
        return self.summaries.get(trial["trial_id"]) or {}

    def accuracy_key(self, trial: dict) -> tuple:
        """Trials with the same accuracy-relevant parameters (everything except throughput knobs and seed)."""
        from .spec import THROUGHPUT_KEYS

        params = trial.get("params") or {}
        return tuple(sorted((k, v) for k, v in params.items() if k not in THROUGHPUT_KEYS and k != "seed"))
