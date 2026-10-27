"""Benchmark: noise level of the baseline seeds and the fixed comparison protocol (spec §7)."""

from __future__ import annotations

import statistics

from ..evidence import WARMUP_STEPS
from .base import Finding, Specialist, cite

PROTOCOL = (f"同一数据划分（campaign 固定的 split 文件）；吞吐去掉前 {WARMUP_STEPS} 步预热；"
            "前向/反向/优化器用 CUDA event 计时；受温度或功耗限制的 trial 不比较计时")


def noise(evidence) -> dict | None:
    """Spread of the objective and of throughput across the valid baseline seeds."""
    trials = [t for t in evidence.valid("baseline")]
    values = [evidence.objective(t) for t in trials]
    if len(values) < 2:
        return None
    throughputs = [evidence.summary(t).get("throughput_sps") for t in trials]
    throughputs = [v for v in throughputs if v]
    mean_tp = statistics.fmean(throughputs) if throughputs else None
    return {
        "metric": evidence.metric, "seeds": [t["params"]["seed"] for t in trials], "values": values,
        "mean": statistics.fmean(values), "std": statistics.stdev(values),
        "throughput_mean": mean_tp,
        "throughput_cv": statistics.stdev(throughputs) / mean_tp if len(throughputs) > 1 and mean_tp else None,
        "trial_ids": [t["trial_id"] for t in trials],
    }


class Benchmark(Specialist):
    name = "benchmark"
    label = "Benchmark"

    def available(self, evidence):
        return True, None

    def diagnose(self, evidence) -> list[Finding]:
        level = noise(evidence)
        if level is None:
            return []
        trials = {t["trial_id"]: t for t in evidence.valid("baseline")}
        cv = level["throughput_cv"]
        message = (f"人工基线 {len(level['values'])} 个 seed 的 {level['metric']} = {level['mean']:.5f} ± "
                   f"{level['std']:.5f}（标准差即噪声水平：小于它的提升不宣称）")
        if cv is not None:
            message += f"；吞吐变异系数 {cv:.1%}"
        findings = [Finding(
            id=f"{self.name}:noise_level", agent=self.name, kind="noise_level", severity="info", message=message,
            evidence=[cite(trials[tid], evidence.summary(trials[tid]), level["metric"], value, source="result")
                      for tid, value in zip(level["trial_ids"], level["values"])],
            data={**level, "protocol": PROTOCOL})]
        hashes = {((t.get("result") or {}).get("data") or {}).get("val_hash") for t in trials.values()}
        if len(hashes) > 1:
            findings.append(Finding(id=f"{self.name}:protocol_split", agent=self.name, kind="protocol_violation",
                                    severity="warning", message="基线各 seed 的验证集指纹不一致，比较协议被破坏",
                                    data={"val_hashes": sorted(h or "" for h in hashes)}))
        return findings
