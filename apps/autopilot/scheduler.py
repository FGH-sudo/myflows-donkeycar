"""Scheduler (spec §6.2): priority queue of trials, budget accounting and duration estimates.

All trials go to the console's JobManager, which runs one job at a time; the training entry additionally
takes the machine-wide GPU lock, so trials, console jobs and terminal runs never share the GPU.
"""

from __future__ import annotations

import math
import statistics
import time

PRIORITY = {"baseline": 0, "confirm": 1, "exploit": 2, "explore": 3}
STARTUP_S = 20.0


def budget_state(trials: list[dict]) -> dict:
    started = [t for t in trials if t.get("attempts")]
    run_s = sum(float(a.get("run_s") or 0.0) for t in trials for a in t.get("attempts") or [])
    committed = [t for t in trials if t.get("status") in ("queued", "running") or t.get("attempts")]
    return {"trials_started": len(started), "trials_committed": len(committed),
            "attempts": sum(len(t.get("attempts") or []) for t in trials), "run_s": run_s}


def budget_fraction(spec: dict, trials: list[dict]) -> float:
    state = budget_state(trials)
    budget = spec["budget"]
    return max(state["trials_started"] / budget["max_trials"], state["run_s"] / budget["wall_time_s"])


def estimate_s(trial: dict, trials: list[dict], split: dict) -> float | None:
    """Expected run time from the per-sample cost of finished attempts (closest batch size first)."""
    args = trial.get("resolved_args") or {}
    batch = int(args.get("batch", 1))
    epochs = int(args.get("epochs", 1))
    train = int(split.get("train_count") or 0)
    if not train:
        return None
    rates = []
    for other in trials:
        summary = (other.get("result") or {}).get("summary") or {}
        sps = summary.get("throughput_sps")
        if not sps:
            continue
        other_batch = int((other.get("resolved_args") or {}).get("batch", 1))
        rates.append((abs(math.log2(other_batch) - math.log2(batch)), sps))
    if not rates:
        return None
    closest = min(r[0] for r in rates)
    sps = statistics.median([r[1] for r in rates if r[0] == closest])
    steps = math.ceil(train / batch)
    return STARTUP_S + epochs * (steps * batch / sps) * 1.15


class Scheduler:
    def __init__(self, store, spec: dict):
        self.store = store
        self.spec = spec

    def enqueue(self, *, phase: str, params: dict, args: dict, fingerprint: str, origin: dict, precheck: dict,
                status: str = "queued") -> dict:
        trial_id = self.store.next_trial_id()
        record = {
            "schema_version": 1, "trial_id": trial_id, "campaign_id": self.store.campaign_id, "phase": phase,
            "priority": PRIORITY.get(phase, 9), "params": params, "resolved_args": args,
            "fingerprint": fingerprint, "origin": origin, "precheck": precheck, "status": status,
            "attempts": [], "verdict": None, "result": None, "created_unix_s": time.time(),
        }
        self.store.save_trial(record)
        return record

    def dispatch(self, trials: list[dict], split: dict) -> dict | None:
        """Next queued trial that fits the remaining budget; trials that do not fit are skipped with a reason."""
        queued = sorted((t for t in trials if t.get("status") == "queued"),
                        key=lambda t: (t.get("priority", 9), t["trial_id"]))
        budget = self.spec["budget"]
        for trial in queued:
            state = budget_state(trials)
            reason = None
            if state["trials_started"] >= budget["max_trials"]:
                reason = f"trial 数已达上限 {budget['max_trials']}"
            else:
                remaining = budget["wall_time_s"] - state["run_s"]
                expected = estimate_s(trial, trials, split)
                if remaining <= 0:
                    reason = "总时长预算已用完"
                elif expected is not None and expected > remaining:
                    reason = f"预计用时 {expected:.0f} s 超过剩余预算 {remaining:.0f} s"
                elif expected is not None and expected > budget["trial_timeout_s"]:
                    reason = f"预计用时 {expected:.0f} s 超过单 trial 上限 {budget['trial_timeout_s']:.0f} s"
            if reason is None:
                return trial
            trial.update(status="skipped", skip_reason=reason, finished_unix_s=time.time())
            self.store.save_trial(trial)
            self.store.event("skip", trial_id=trial["trial_id"], reason=reason)
        return None
