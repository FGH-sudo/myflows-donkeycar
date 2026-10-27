"""GPU Monitor: data-loading share, GPU utilisation, memory peak, host overhead and throttling (spec §7)."""

from __future__ import annotations

from .base import Finding, Proposal, Specialist, cite

DATA_SHARE = 0.3
DATA_BOUND_UTIL = 50.0
LOW_UTIL = 40.0
LOW_MEMORY = 0.5
HIGH_MEMORY = 0.9
# A hang or timeout with the memory this full is put down to WDDM paging.
PAGING_MEMORY = 0.95
HOST_OVERHEAD = 0.2
WORKER_LADDER = (0, 2, 4)


def _next_workers(current: int) -> int | None:
    for value in WORKER_LADDER:
        if value > current:
            return value
    return None


class GpuMonitor(Specialist):
    name = "gpu_monitor"
    label = "GPU Monitor"

    def diagnose(self, evidence) -> list[Finding]:
        findings = []
        for trial in evidence.finished():
            verdict = trial.get("verdict") or {}
            if verdict.get("status") == "failed" and verdict.get("kind") in ("hang", "timeout"):
                findings += self._paging(trial, evidence.summary(trial))
                continue
            if verdict.get("status") not in ("valid", "invalid"):
                continue
            summary = evidence.summary(trial)
            if not summary.get("steps"):
                continue
            tid = trial["trial_id"]
            params = trial.get("params") or {}
            findings += self._paging(trial, summary)  # finished, but slowed down by paging
            share, util = summary.get("data_share"), summary.get("gpu_util_median")
            mem = summary.get("mem_share")
            if share is not None and util is not None and share > DATA_SHARE and util < DATA_BOUND_UTIL:
                findings.append(Finding(
                    id=f"{self.name}:data_bound:{tid}", agent=self.name, kind="data_bound", trial_id=tid,
                    severity="action",
                    message=f"数据读取占步时 {share:.0%}（阈值 {DATA_SHARE:.0%}），GPU 利用率中位数 {util:.0f}%"
                            f"（阈值 {DATA_BOUND_UTIL:.0f}%）：训练在等数据，num_workers={params.get('num_workers', 0)}",
                    evidence=[cite(trial, summary, "data_share", round(share, 4), DATA_SHARE),
                              cite(trial, summary, "gpu_util_median", util, DATA_BOUND_UTIL, source="resources")],
                    data={"data_share": share, "gpu_util_median": util, "num_workers": params.get("num_workers", 0)}))
            elif util is not None and mem is not None and util < LOW_UTIL and mem < LOW_MEMORY:
                findings.append(Finding(
                    id=f"{self.name}:underutilized:{tid}", agent=self.name, kind="gpu_underutilized", trial_id=tid,
                    severity="action",
                    message=f"GPU 利用率中位数 {util:.0f}%（< {LOW_UTIL:.0f}%）且显存峰值只占 {mem:.0%}"
                            f"（< {LOW_MEMORY:.0%}）：可以加大 batch",
                    evidence=[cite(trial, summary, "gpu_util_median", util, LOW_UTIL, source="resources"),
                              cite(trial, summary, "mem_share", round(mem, 4), LOW_MEMORY, source="result")],
                    data={"gpu_util_median": util, "mem_share": mem}))
            # Paging and OOM depend on the whole GPU: other processes (the desktop takes ~1-2 GB) count too.
            device = summary.get("device_mem_share")
            peak_share = max(v for v in (mem, device) if v is not None) if mem is not None or device is not None else None
            if peak_share is not None and peak_share > HIGH_MEMORY:
                own = f"，本进程显存池占 {mem:.0%}" if mem is not None and device is not None else ""
                findings.append(Finding(
                    id=f"{self.name}:memory_pressure:{tid}", agent=self.name, kind="memory_pressure", trial_id=tid,
                    severity="action",
                    message=f"显存峰值占整张 GPU 的 {peak_share:.0%}（> {HIGH_MEMORY:.0%}{own}）："
                            "有显存不足或 WDDM 换页变慢的风险",
                    evidence=[cite(trial, summary, "device_mem_share" if device is not None and device >= (mem or 0)
                                   else "mem_share", round(peak_share, 4), HIGH_MEMORY,
                                   source="resources" if device is not None and device >= (mem or 0) else "result")],
                    data={"mem_share": mem, "device_mem_share": device}))
            overhead = summary.get("host_overhead_share")
            if overhead is not None and overhead > HOST_OVERHEAD:
                findings.append(Finding(
                    id=f"{self.name}:host_overhead:{tid}", agent=self.name, kind="host_sync", trial_id=tid,
                    severity="warning",
                    message=f"步时中有 {overhead:.0%} 不在 GPU 前向/反向/优化器 kernel 里（阈值 {HOST_OVERHEAD:.0%}）："
                            "主机端调度与同步开销偏高",
                    evidence=[cite(trial, summary, "host_overhead_share", round(overhead, 4), HOST_OVERHEAD)],
                    data={"host_overhead_share": overhead}))
            if summary.get("throttled"):
                findings.append(Finding(
                    id=f"{self.name}:throttled:{tid}", agent=self.name, kind="thermal_power_limited", trial_id=tid,
                    severity="warning",
                    message=(f"GPU 温度最高 {summary.get('temp_max_c')} °C（≥ 83 °C 的采样超过 20%）"
                             if summary.get("thermal_limited") else
                             f"{summary.get('power_capped_share'):.0%} 的采样功耗达到当时的功耗上限"
                             f"（最高 {summary.get('power_max_w'):g} W，上限 {summary.get('power_limit_min_w'):g}～"
                             f"{summary.get('power_limit_w'):g} W）")
                            + "：受温度或功耗限制，本 trial 的计时不参与横向比较",
                    evidence=[cite(trial, summary, "temp_max_c", summary.get("temp_max_c"), source="resources"),
                              cite(trial, summary, "power_capped_share", summary.get("power_capped_share"), 0.2,
                                   source="resources")],
                    data={k: summary.get(k) for k in ("temp_max_c", "power_max_w", "power_limit_w", "power_limit_min_w",
                                                      "power_capped_share", "thermal_limited", "power_limited")}))
        return findings

    def _paging(self, trial: dict, summary: dict) -> list[Finding]:
        """GPU memory beyond what is free: on Windows WDDM pages VRAM to system RAM instead of raising OOM, so the
        run slows down by an order of magnitude (it finishes slowly, or the online check calls it a hang)."""
        shares = [v for v in (summary.get("mem_share"), summary.get("device_mem_share")) if v is not None]
        if not shares or max(shares) <= PAGING_MEMORY or not summary.get("steps"):
            return []
        share, tid = max(shares), trial["trial_id"]
        step_ms = summary.get("step_ms_median")
        slow = f"，步时中位数 {step_ms / 1000:.1f} s" if step_ms else ""
        verdict = trial.get("verdict") or {}
        outcome = f"被判为 {verdict.get('kind')}" if verdict.get("status") == "failed" else "虽然跑完了"
        return [Finding(
            id=f"{self.name}:paging:{tid}", agent=self.name, kind="memory_paging", trial_id=tid, severity="action",
            message=f"{tid} {outcome}，但显存占整张 GPU 的 {share:.0%}{slow}："
                    f"更可能是显存不够、驱动把显存换页到系统内存（不报 OOM 而是变慢），batch={trial['params'].get('batch')} "
                    "在这台机器上放不下",
            evidence=[cite(trial, summary, "device_mem_share" if summary.get("device_mem_share") == share
                           else "mem_share", round(share, 4), PAGING_MEMORY, source="resources"),
                      cite(trial, summary, "step_ms_median", step_ms and round(step_ms, 1))],
            data={"mem_share": share, "step_ms_median": step_ms, "batch": trial["params"].get("batch")})]

    def propose(self, findings: list[Finding], evidence) -> list[Proposal]:
        """The current best configuration is refined from findings on that trial; a configuration slowed down or
        hung by memory paging gets one retry with half the batch (its numbers say nothing about the setting)."""
        proposals = []
        for finding in findings:
            if finding.kind != "memory_paging":
                continue
            batch = int(finding.data.get("batch") or 1)
            if batch >= 2:
                proposals.append(Proposal(
                    delta={"batch": batch // 2}, kind="accuracy", agent=self.name, base_trial=finding.trial_id,
                    finding_ids=[finding.id], rationale=f"{finding.message}；batch {batch}→{batch // 2}",
                    expected="不再换页：步时回到正常水平，trial 能跑完；精度需重新评估"))
        best = evidence.best()
        if best is None:
            return proposals
        params = best.get("params") or {}
        for finding in findings:
            if finding.trial_id != best["trial_id"] or finding.severity != "action":
                continue
            if finding.kind == "data_bound":
                target = _next_workers(int(params.get("num_workers", 0)))
                if target is None:
                    continue
                proposals.append(Proposal(
                    delta={"num_workers": target}, kind="throughput", agent=self.name, base_trial=best["trial_id"],
                    finding_ids=[finding.id],
                    rationale=f"{finding.message}；用 {target} 个数据读取进程预取",
                    expected="samples/sec 上升、数据读取占比下降；精度不变（样本顺序与增强不受 num_workers 影响）"))
            elif finding.kind == "gpu_underutilized":
                batch, lr = int(params.get("batch", 1)), float(params.get("lr", 1e-3))
                proposals.append(Proposal(
                    delta={"batch": batch * 2, "lr": lr * 2}, kind="accuracy", agent=self.name,
                    base_trial=best["trial_id"], finding_ids=[finding.id],
                    rationale=f"{finding.message}；batch {batch}→{batch * 2}，lr 按线性缩放 {lr:g}→{lr * 2:g}",
                    expected="GPU 利用率与 samples/sec 上升；精度需重新评估"))
            elif finding.kind == "memory_pressure":
                batch = int(params.get("batch", 1))
                if batch > 1:
                    proposals.append(Proposal(
                        delta={"batch": batch // 2}, kind="accuracy", agent=self.name, base_trial=best["trial_id"],
                        finding_ids=[finding.id], rationale=f"{finding.message}；batch {batch}→{batch // 2}",
                        expected="显存峰值下降到 90% 以下"))
        return proposals
