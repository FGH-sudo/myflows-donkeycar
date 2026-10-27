"""RuleRefine: specialist proposals become trial parameters; finished refinements become diagnosis chains.

A chain links finding → proposal → trial → before/after → decision. Throughput refinements (num_workers,
conv backend, parallel mode) are adopted for later trials when the paired run is faster beyond the noise and
the accuracy did not move by more than the baseline's seed-to-seed spread.
"""

from __future__ import annotations

from ..agents.base import Proposal

MIN_THROUGHPUT_GAIN = 0.05


def _side(ev, trial: dict) -> dict:
    summary = ev.summary(trial)
    return {"trial_id": trial["trial_id"], "objective": ev.objective(trial),
            "throughput_sps": summary.get("throughput_sps"), "data_share": summary.get("data_share"),
            "gpu_util_median": summary.get("gpu_util_median"), "mem_share": summary.get("mem_share"),
            "step_ms_median": summary.get("step_ms_median"), "params": trial.get("params")}


class RuleRefine:
    name = "rule_refine"

    def materialize(self, proposal: Proposal, ev) -> Proposal | None:
        base = next((t for t in ev.trials if t["trial_id"] == proposal.base_trial), None)
        if base is None:
            return None
        params = {**(base.get("params") or {}), **(ev.state.get("adopted") or {}), **proposal.delta}
        proposal.params = params
        proposal.phase = "exploit"
        proposal.strategy = self.name
        return proposal

    def evaluate(self, ev) -> list[dict]:
        """Close the diagnosis chain of every finished exploit trial (once)."""
        chains = ev.state.setdefault("chains", [])
        closed = {c["trial_id"] for c in chains}
        noise = ev.state.get("noise") or {}
        threshold = max(MIN_THROUGHPUT_GAIN, 2.0 * (noise.get("throughput_cv") or 0.0))
        new = []
        for trial in ev.trials:
            origin = trial.get("origin") or {}
            if trial.get("phase") != "exploit" or trial.get("status") != "done" or trial["trial_id"] in closed:
                continue
            if origin.get("strategy") != self.name:
                continue
            base = next((t for t in ev.trials if t["trial_id"] == origin.get("base_trial")), None)
            if base is None:
                continue
            findings = ev.state.get("findings") or {}
            chain = {"trial_id": trial["trial_id"], "base_trial": base["trial_id"],
                     "finding_ids": origin.get("finding_ids") or [],
                     "findings": [findings.get(fid, {}).get("message") for fid in origin.get("finding_ids") or []],
                     "proposal": {"delta": origin.get("delta"), "rationale": origin.get("rationale"),
                                  "agent": origin.get("agent"), "kind": origin.get("kind")},
                     "before": _side(ev, base), "after": _side(ev, trial)}
            verdict = (trial.get("verdict") or {}).get("status")
            base_verdict = base.get("verdict") or {}
            if verdict == "valid" and base_verdict.get("status") == "failed":
                b_ms, a_ms = chain["before"].get("step_ms_median"), chain["after"].get("step_ms_median")
                steps = (f"；步时中位数 {b_ms:.0f} → {a_ms:.0f} ms" if b_ms and a_ms else "")
                chain.update(decision="recovered", reason=f"基准 {base['trial_id']} 失败（{base_verdict.get('kind')}），"
                             f"派生 trial 正常完成，{ev.metric} = {chain['after']['objective']}{steps}")
            elif verdict != "valid":
                chain.update(decision="failed", reason=f"派生 trial 结果为 {verdict}："
                             f"{'；'.join((trial.get('verdict') or {}).get('reasons') or [])}")
            elif origin.get("kind") == "throughput":
                before, after = chain["before"]["throughput_sps"], chain["after"]["throughput_sps"]
                gain = (after / before - 1.0) if before and after else None
                same = chain["before"]["objective"] == chain["after"]["objective"]
                chain["throughput_gain"] = gain
                chain["accuracy_unchanged"] = same
                # Data loading does not change the numbers; a parallel mode or a conv backend may. Those are
                # adopted only while the objective moves less than the baseline's seed-to-seed spread.
                shift = (abs(chain["after"]["objective"] - chain["before"]["objective"])
                         if not same and chain["after"]["objective"] is not None
                         and chain["before"]["objective"] is not None else None)
                spread = noise.get("std")
                if not same and (shift is None or spread is None or shift > spread):
                    chain.update(decision="rejected", reason=(
                        f"吞吐变化 {gain:+.1%}，但 {ev.metric} 变化 {shift:.5f} 超过基线噪声 {spread:.5f}，不沿用"
                        if shift is not None and spread is not None and gain is not None
                        else "精度有变化且没有基线噪声可比，不沿用"))
                elif gain is not None and gain >= threshold:
                    adopted = ev.state.setdefault("adopted", {})
                    for key, value in (origin.get("delta") or {}).items():
                        adopted[key] = value
                    chain.update(decision="adopted", reason=f"吞吐提升 {gain:+.1%}（阈值 {threshold:.0%}），"
                                 f"{'精度逐位不变' if same else '精度有变化（见指标）'}；后续 trial 沿用 "
                                 + "、".join(f"{k}={v}" for k, v in (origin.get('delta') or {}).items()))
                else:
                    chain.update(decision="rejected", reason=f"吞吐变化 {gain:+.1%} 未超过阈值 {threshold:.0%}"
                                 if gain is not None else "缺少吞吐数据")
            else:
                b, a = chain["before"]["objective"], chain["after"]["objective"]
                better = a is not None and b is not None and (a < b if ev.minimize else a > b)
                chain.update(decision="better" if better else "not_better",
                             reason=f"{ev.metric} {b} → {a}" + ("（优于基准 trial，进入排名）" if better else ""))
            chains.append(chain)
            new.append(chain)
        return new
