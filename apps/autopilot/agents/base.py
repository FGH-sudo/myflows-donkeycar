"""Specialist interface: ``available()``, ``diagnose(evidence) -> [Finding]``, ``propose(findings, evidence)``.

Specialists are rule packages, not chat processes (spec §7). A Finding cites the files and values that
triggered it; a Proposal is a parameter change relative to a trial, with the findings it answers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Finding:
    id: str
    agent: str
    kind: str
    message: str
    trial_id: str | None = None
    severity: str = "info"          # info | warning | action
    evidence: list[dict] = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Proposal:
    delta: dict
    rationale: str
    agent: str
    strategy: str = "rule_refine"
    base_trial: str | None = None
    finding_ids: list[str] = field(default_factory=list)
    expected: str = ""
    kind: str = "accuracy"           # accuracy | throughput
    phase: str = "exploit"
    params: dict | None = None       # full trial parameters, filled by the Planner

    def to_dict(self) -> dict:
        return asdict(self)


def cite(trial: dict, summary: dict, metric: str, value, threshold=None, source: str = "metrics") -> dict:
    """One evidence item: which attempt file produced which value."""
    run_dir = summary.get("run_dir") or ""
    file = summary.get("sources", {}).get(source, source)
    item = {"trial_id": trial["trial_id"], "file": f"{run_dir}/{file}" if run_dir else file, "metric": metric,
            "value": value}
    if threshold is not None:
        item["threshold"] = threshold
    return item


class Specialist:
    name = ""
    label = ""

    def available(self, evidence) -> tuple[bool, str | None]:
        return True, None

    def diagnose(self, evidence) -> list[Finding]:
        return []

    def propose(self, findings: list[Finding], evidence) -> list[Proposal]:
        return []
