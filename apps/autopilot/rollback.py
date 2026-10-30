"""Exact rollback (spec §6.5, E3): continue a failed trial from its last healthy TrainingState.

A trial that ends ``nan`` or ``diverged`` while the campaign has ``rollback.enabled`` gets a new attempt that
resumes, bit for bit, from the newest healthy state any of its attempts saved (``apps.train --resume-state``),
with the learning rate multiplied by ``rollback.lr_factor``. A state is healthy when it was saved before the
anomaly began (the online monitor records that step: the NaN row, the trough before a rising loss, the first
window of an explosion), its losses are finite and its running loss is at most ``divergence_factor`` times the
run's first loss. Without a recorded start, a divergence keeps one detection span
(``online.window_steps × online.divergence_windows``) before the stop. A fresh run also saves its initial
weights as step 0, so a loss that explodes on the first updates can still be rolled back. At most
``rollback.max_rollbacks`` rollbacks per trial; distributed trials have no TrainingState and are not rolled
back.

An OOM is not rolled back: the trial stays failed and a new trial with half the batch is derived instead.
"""

from __future__ import annotations

import math
from pathlib import Path

from .files import read_json, read_jsonl
from .state import trial_states, verify

ROLLBACK_KINDS = ("nan", "diverged")
DEFAULTS = {"enabled": False, "max_rollbacks": 2, "lr_factor": 0.5, "state_every": 20, "oom_half_batch": True}


def settings(spec: dict) -> dict:
    return {**DEFAULTS, **(spec.get("rollback") or {})}


def failure_step(attempt: dict) -> int | None:
    """The global step at which the attempt failed: from result.json, else the last step row it wrote."""
    run_dir = Path(attempt["run_dir"])
    failure = (read_json(run_dir / "result.json") or {}).get("failure") or attempt.get("failure") or {}
    if isinstance(failure.get("step"), int):
        return failure["step"]
    steps = [r.get("step") for r in read_jsonl(run_dir / "metrics.jsonl") if r.get("kind") == "step"]
    steps = [s for s in steps if isinstance(s, int)]
    return max(steps) if steps else None


def _finite(*values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values if v is not None)


def healthy_state(trial: dict, attempt: dict, spec: dict) -> tuple[dict | None, str]:
    """The newest healthy state of the trial (``{path, step, state, attempt}``) and why, or ``(None, reason)``."""
    failure = attempt.get("failure") or {}
    kind = failure.get("kind")
    fail = failure_step(attempt)
    online = spec.get("online") or {}
    margin = int(online.get("window_steps", 10)) * int(online.get("divergence_windows", 3)) if kind == "diverged" else 0
    factor = float(online.get("divergence_factor", 3.0))
    since = failure.get("since_step")
    cutoff = since - 1 if isinstance(since, int) else (fail - 1 - margin if fail is not None else None)
    candidates = trial_states(trial)
    if not candidates:
        return None, "没有可用的 TrainingState（trial 未开启 --state-every，或失败前还没保存过）"
    healthy = []
    for c in candidates:
        s = c["state"]
        if cutoff is not None and c["step"] > cutoff:
            continue
        if not _finite(s.get("loss"), s.get("running_loss"), s.get("last_mean_loss")):
            continue
        # A state whose loss had already blown up is finite but not healthy (for NaN as well as divergence).
        if s.get("first_loss") and s.get("running_loss") is not None \
                and s["running_loss"] > factor * float(s["first_loss"]):
            continue
        healthy.append(c)
    if not healthy:
        return None, f"失败步 {fail} 之前没有健康的 TrainingState（已检查 {len(candidates)} 个）"
    best = max(healthy, key=lambda c: (c["step"], c["attempt"]))
    began = f"，异常始于第 {since} 步" if isinstance(since, int) else ""
    target = "初始权重（第 0 步）" if best["step"] == 0 else f"第 {best['step']} 步的健康状态"
    return best, f"失败于第 {fail} 步{began}；回到{target}（attempt {best['attempt']}）"


def plan(trial: dict, attempt: dict, spec: dict) -> tuple[dict | None, str]:
    """Overrides for the rollback attempt (``resume_state`` and the reduced ``lr``), or ``(None, reason)``."""
    cfg = settings(spec)
    kind = (attempt.get("failure") or {}).get("kind")
    if not cfg["enabled"] or kind not in ROLLBACK_KINDS:
        return None, "未开启回滚" if not cfg["enabled"] else f"{kind} 不回滚"
    if (trial.get("params") or {}).get("distributed", "none") != "none":
        return None, "数据并行训练没有 TrainingState，不能精确回滚"
    used = len(trial.get("rollbacks") or [])
    if used >= int(cfg["max_rollbacks"]):
        return None, f"已回滚 {used} 次，达到上限 {cfg['max_rollbacks']}"
    state, why = healthy_state(trial, attempt, spec)
    if state is None:
        return None, why
    if not verify(state["path"]):
        return None, f"第 {state['step']} 步的 TrainingState 校验失败，放弃回滚"
    lr_before = float((attempt.get("overrides") or {}).get("lr")
                      or state["state"].get("lr") or trial["resolved_args"]["lr"])
    lr = lr_before * float(cfg["lr_factor"])
    return {"resume_state": str(state["path"]), "lr": lr, "from_step": state["step"],
            "from_attempt": state["attempt"], "lr_before": lr_before, "kind": kind, "reason": why}, why


def oom_proposal(trial: dict, spec: dict) -> dict | None:
    """Delta of the trial derived after an OOM (half the batch), or None when it cannot shrink further."""
    cfg = settings(spec)
    if not cfg["enabled"] or not cfg["oom_half_batch"]:
        return None
    batch = int((trial.get("params") or {}).get("batch", 1))
    if batch < 2:
        return None
    return {"batch": batch // 2}
