"""Specialist sub-agents (spec §7)."""

from __future__ import annotations

from .base import Finding, Proposal, Specialist
from .benchmark import Benchmark
from .gpu_monitor import GpuMonitor
from .transfer import TransferLearning


def build_specialists(names: list[str], **context) -> list[Specialist]:
    """Instances for the names a campaign enables; unknown names are ignored (the spec validator rejects them)."""
    from .operator_opt import OperatorOptimization
    from .ps_training import PsTraining

    registry = {"gpu_monitor": GpuMonitor, "benchmark": Benchmark, "transfer": TransferLearning,
                "operator_opt": OperatorOptimization, "ps_training": PsTraining}
    out = []
    for name in names:
        cls = registry.get(name)
        if cls is None:
            continue
        try:
            out.append(cls(**context))
        except TypeError:
            out.append(cls())
    return out


__all__ = ["Benchmark", "Finding", "GpuMonitor", "Proposal", "Specialist", "TransferLearning", "build_specialists"]
