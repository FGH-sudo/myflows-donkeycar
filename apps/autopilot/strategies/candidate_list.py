"""CandidateList: stratified sample of the discrete lr × batch (× freeze_stages) grid (spec §6.1), not the full
product."""

from __future__ import annotations

import itertools
import math
import random

from ..agents.base import Proposal
from ..spec import ACCURACY_KEYS, TRANSFER_DEFAULTS, TRANSFER_KEYS, grid


def stratified(space: dict, count: int, seed: int, exclude: list[dict] | None = None) -> list[dict]:
    """Pick ``count`` grid points so that every value of every key is covered as evenly as possible.

    Greedy: each pick takes the remaining point whose values have been used least; ties follow a shuffle
    seeded by ``seed``, so the sample is reproducible.
    """
    keys = [k for k in (*ACCURACY_KEYS, *TRANSFER_KEYS) if k in space]
    excluded = [{k: e.get(k, TRANSFER_DEFAULTS.get(k)) for k in keys} for e in exclude or []]
    remaining = [p for p in grid(space, keys) if p not in excluded]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    count = min(count, len(remaining))
    if math.comb(len(remaining), count) <= 20000:
        # Small grids: the most balanced subset (spread of per-value counts, summed over keys); ties keep the
        # seeded shuffle order.
        def imbalance(subset):
            total = 0
            for k in keys:
                available = [v for v in space[k] if any(p[k] == v for p in remaining)]
                counts = [sum(1 for p in subset if p[k] == v) for v in available]
                total += max(counts) - min(counts)
            return total

        best = min(itertools.combinations(range(len(remaining)), count),
                   key=lambda idx: imbalance([remaining[i] for i in idx]))
        return [remaining[i] for i in best]
    used = {k: {} for k in keys}
    picked = []
    while remaining and len(picked) < count:
        point = min(remaining, key=lambda p: sum(used[k].get(p[k], 0) for k in keys))
        remaining.remove(point)
        picked.append(point)
        for k in keys:
            used[k][point[k]] = used[k].get(point[k], 0) + 1
    return picked


class CandidateList:
    name = "candidate_list"

    def proposals(self, spec: dict, *, exclude: list[dict]) -> list[Proposal]:
        strategy = spec["strategy"]
        points = stratified(spec["space"], int(strategy["max_candidates"]), int(strategy["seed"]), exclude)
        return [Proposal(delta=dict(point), kind="accuracy", agent="planner", strategy=self.name, phase="explore",
                         rationale=f"分层抽样候选 {i + 1}/{len(points)}：" +
                                   "、".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in point.items()),
                         expected=f"与基线比较 {spec['objective']['metric']}")
                for i, point in enumerate(points)]
