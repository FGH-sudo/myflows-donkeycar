"""Planner (spec §6.1): phase management and the merge of proposals from every source.

Phases: baseline (manual baseline × seeds) → explore (≤ N grid candidates, until 70 % of the budget) →
exploit (specialist / LLM refinements backed by evidence) → confirm (top 1–2 rerun with fresh seeds, paired
with the baseline on the same seeds) → done.
"""

from __future__ import annotations

import hashlib
import json

from .agents.base import Proposal
from .scheduler import budget_fraction
from .spec import ACCURACY_KEYS, THROUGHPUT_KEYS, baseline_params
from .strategies.candidate_list import CandidateList
from .strategies.rule_refine import RuleRefine

EXPLORE_BUDGET_SHARE = 0.7
PHASES = ("baseline", "explore", "exploit", "confirm", "done")


def proposal_key(phase: str, params: dict) -> str:
    return hashlib.sha256(json.dumps([phase, params], sort_keys=True, default=str).encode()).hexdigest()[:16]


def confirm_seeds(spec: dict) -> list[int]:
    """Seeds for confirmation: never the explore seed (the candidate was selected on it), as many as the
    baseline has, reusing the remaining baseline seeds first so the pairing needs few extra baseline runs."""
    if spec["confirm"].get("seeds"):
        return list(spec["confirm"]["seeds"])
    seeds = spec["baseline"]["seeds"]
    count = len(seeds)
    out = [s for s in seeds[1:]]
    nxt = max(seeds) + 1
    while len(out) < count:
        out.append(nxt)
        nxt += 1
    return out[:count]


class Planner:
    def __init__(self, spec: dict, *, specialists=(), tpe=None):
        self.spec = spec
        self.specialists = list(specialists)
        self.candidates = CandidateList()
        self.refine = RuleRefine()
        self.tpe = tpe

    # ---- helpers ------------------------------------------------------------------------
    def base_params(self, ev, seed: int) -> dict:
        params = baseline_params(self.spec, seed)
        params.update(ev.state.get("adopted") or {})
        return params

    def _new(self, ev, proposals: list[Proposal]) -> list[Proposal]:
        proposed = ev.state.setdefault("proposed", [])
        out = []
        for p in proposals:
            key = proposal_key(p.phase, p.params)
            if key in proposed:
                continue
            proposed.append(key)
            out.append(p)
        return out

    @staticmethod
    def _open(ev, phase: str) -> list[dict]:
        return [t for t in ev.trials if t.get("phase") == phase and t.get("status") in ("queued", "running")]

    def _phase(self, ev, phase: str, note: str | None = None) -> None:
        ev.state["phase"] = phase
        ev.state.setdefault("phase_log", []).append({"phase": phase, "note": note})

    # ---- main entry -----------------------------------------------------------------------
    def next(self, ev, findings: list) -> list[Proposal]:
        """Proposals for the current phase (new ones only); advances the phase when it is finished."""
        for _ in range(len(PHASES)):
            phase = ev.state.get("phase", "baseline")
            handler = getattr(self, f"_{phase}", None)
            if handler is None:
                return []
            proposals = handler(ev, findings)
            if proposals or ev.state.get("phase") == phase:
                return self._new(ev, proposals)
        return []

    def _baseline(self, ev, findings) -> list[Proposal]:
        seeds = self.spec["baseline"]["seeds"]
        have = {t["params"].get("seed") for t in ev.baseline()}
        proposed = ev.state.get("proposed", [])
        missing = [s for s in seeds if s not in have
                   and proposal_key("baseline", baseline_params(self.spec, s)) not in proposed]
        if missing:
            return [Proposal(delta={}, params=baseline_params(self.spec, s), phase="baseline", strategy="baseline",
                             agent="planner", rationale=f"人工基线（lr={self.spec['baseline']['lr']:g}、batch="
                             f"{self.spec['baseline']['batch']}、num_workers={self.spec['baseline']['num_workers']}）"
                             f" seed {s}", expected="基线与噪声水平") for s in missing]
        if self._open(ev, "baseline") or any(t.get("status") == "pending_confirmation" for t in ev.baseline()):
            return []
        if not ev.valid("baseline"):
            ev.state["stop_reason"] = "人工基线没有有效结果，campaign 无法比较"
            self._phase(ev, "done", ev.state["stop_reason"])
            return []
        self._phase(ev, "explore", "基线完成")
        return []

    def _explore(self, ev, findings) -> list[Proposal]:
        if budget_fraction(self.spec, ev.trials) >= EXPLORE_BUDGET_SHARE:
            self._phase(ev, "exploit", f"预算已用 {EXPLORE_BUDGET_SHARE:.0%}，探索结束")
            return []
        seed = self.spec["baseline"]["seeds"][0]
        tried = [t["params"] for t in ev.trials if t.get("phase") in ("baseline", "explore")]
        base = baseline_params(self.spec, seed)
        if self.spec["strategy"]["name"] == "tpe" and self.tpe is not None:
            if self._open(ev, "explore"):
                return []
            done = [t for t in ev.trials if t.get("phase") == "explore" and t.get("status") in ("done", "skipped")]
            if len(done) >= self.spec["strategy"]["max_candidates"]:
                self._phase(ev, "exploit", "TPE 候选用完")
                return []
            proposal = self.tpe.ask(ev)
            if proposal is None:
                self._phase(ev, "exploit", "TPE 没有新的候选")
                return []
            proposal.params = {**self.base_params(ev, seed), **proposal.delta}
            return [proposal]
        if not ev.state.get("explore_proposed"):
            ev.state["explore_proposed"] = True
            proposals = self.candidates.proposals(self.spec, exclude=[base] + tried)
            for p in proposals:
                p.params = {**self.base_params(ev, seed), **p.delta}
            if proposals:
                return proposals
        if self._open(ev, "explore"):
            return []
        self._phase(ev, "exploit", "候选已全部运行")
        return []

    def _exploit(self, ev, findings) -> list[Proposal]:
        self.refine.evaluate(ev)
        proposals = []
        for specialist in self.specialists:
            ok, _why = specialist.available(ev)
            if ok:
                proposals += specialist.propose([f for f in findings if f.agent == specialist.name], ev)
        proposals = [self.refine.materialize(p, ev) for p in proposals]
        proposals = [p for p in proposals if p is not None]
        fresh = [p for p in proposals if proposal_key(p.phase, p.params) not in ev.state.get("proposed", [])]
        if fresh:
            return fresh
        if self._open(ev, "exploit") or self._open(ev, "explore"):
            return []
        self._phase(ev, "confirm", "没有新的、有证据支撑的建议")
        return []

    def _confirm(self, ev, findings) -> list[Proposal]:
        self.refine.evaluate(ev)
        plan = ev.state.get("confirm_plan")
        if plan is None:
            plan = self._confirm_plan(ev)
            ev.state["confirm_plan"] = plan
            proposals = []
            for item in plan["candidates"]:
                for seed in plan["seeds"]:
                    params = {**item["params"], "seed": seed}
                    proposals.append(Proposal(delta={"seed": seed}, params=params, phase="confirm",
                                              strategy="confirm", agent="planner", base_trial=item["trial_id"],
                                              rationale=f"确认 {item['trial_id']}（排名 {item['rank']}）：换新 seed {seed} 独立重跑",
                                              expected="与基线同 seed 配对比较"))
            if plan["candidates"]:
                for seed in plan["seeds"]:
                    if not any(t["params"].get("seed") == seed and self._is_baseline(t) and
                               (t.get("verdict") or {}).get("status") == "valid" for t in ev.trials):
                        proposals.append(Proposal(delta={"seed": seed}, params=baseline_params(self.spec, seed),
                                                  phase="confirm", strategy="confirm", agent="planner",
                                                  rationale=f"人工基线补跑 seed {seed}，与确认运行配对",
                                                  expected="配对差值"))
            if proposals:
                return proposals
        if self._open(ev, "confirm"):
            return []
        self._phase(ev, "done", "确认完成" if plan["candidates"] else plan.get("note"))
        return []

    def _done(self, ev, findings) -> list[Proposal]:
        return []

    # ---- confirmation plan -------------------------------------------------------------
    def _is_baseline(self, trial: dict) -> bool:
        base = self.spec["baseline"]
        params = trial.get("params") or {}
        return all(params.get(k) == base[k] for k in ACCURACY_KEYS) and not any(
            k not in (*ACCURACY_KEYS, *THROUGHPUT_KEYS, "seed") for k in params)

    def _confirm_plan(self, ev) -> dict:
        seeds = confirm_seeds(self.spec)
        top_k = int(self.spec["confirm"]["top_k"])
        baseline_values = [ev.objective(t) for t in ev.valid("baseline")]
        reference = (sum(baseline_values) / len(baseline_values)) if baseline_values else None
        candidates, seen = [], set()
        for rank, trial in enumerate(ev.ranked(), start=1):
            if trial.get("phase") == "confirm" or self._is_baseline(trial):
                continue
            key = ev.accuracy_key(trial)
            if key in seen:
                continue
            seen.add(key)
            value = ev.objective(trial)
            better = reference is None or (value < reference if ev.minimize else value > reference)
            if not better:
                continue
            params = {k: v for k, v in trial["params"].items() if k != "seed"}
            params.update(ev.state.get("adopted") or {})
            candidates.append({"trial_id": trial["trial_id"], "rank": rank, "value": value, "params": params})
        # Fit the confirmation into the remaining trial budget: top-2 only when both fit.
        started = sum(1 for t in ev.trials if t.get("attempts"))
        remaining = self.spec["budget"]["max_trials"] - started
        have_baseline = {t["params"].get("seed") for t in ev.trials if self._is_baseline(t)
                         and (t.get("verdict") or {}).get("status") == "valid"}
        extra_baseline = sum(1 for s in seeds if s not in have_baseline)
        chosen = []
        for item in candidates[:top_k]:
            need = len(seeds) * (len(chosen) + 1) + extra_baseline
            if need <= remaining or not chosen:
                chosen.append(item)
        note = None
        if not candidates:
            note = "没有候选在筛选中优于人工基线均值：人工基线未被超越，不需要确认"
        elif len(seeds) + extra_baseline > remaining:
            note = f"剩余 trial 预算 {remaining} 不足以完整确认（需要 {len(seeds) + extra_baseline}），报告中标注为未确认"
        return {"seeds": seeds, "candidates": chosen, "note": note, "reference": reference}
