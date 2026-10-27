"""PS Training (spec §7, lecture page 8): data-parallel throughput against the single-process run.

Once the best single-process configuration is known, it proposes short data-parallel runs of the same
configuration (``apps.train --distributed``) over workers {2, 4} × PS / Ring × transport, within the GPU
memory limit and the global batch (every worker needs at least one sample per step). From each run's
``rank-*/steps.jsonl`` it reads the aggregate throughput, the share of the step spent waiting for gradient
synchronisation and the spread of the ranks' compute times, and concludes:

- synchronisation wait above 50 % of the step: fewer workers or another scheme;
- coefficient of variation of the ranks' compute times above 0.2: a straggler;
- every parallel configuration slower than the single process: keep the single process.
"""

from __future__ import annotations

from .base import Finding, Proposal, Specialist, cite

SYNC_BOUND = 0.5
STRAGGLER_CV = 0.2
WORKERS = (2, 4)
# Parallel mode -> transports, as in apps.console.presets.TRAINING_PARALLEL.
SCHEMES = {"ps": ("grpc_proto", "socket_json"), "ring": ("grpc_proto",)}
SOCKET_JSON_MAX_WIDTH = 32
# apps.train caps each worker's CuPy pool at 90 % of the free GPU memory divided by the workers.
MEMORY_LIMIT = 0.9
WORKER_OVERHEAD_MB = 300  # CUDA context of every extra worker process, outside the CuPy pool
MEMORY_MARGIN = 1.1


def memory_fit(summary: dict, workers: int) -> tuple[bool, float | None, float | None]:
    """Whether ``workers`` processes of the reference configuration fit on its GPU: ``(fits, need, cap)`` in MiB.

    Activations dominate the pool, so a worker holding 1/``workers`` of the global batch needs about the single
    process's pool peak / ``workers``; its cap is apps.train's share of the memory the other processes leave.
    """
    peak, total = summary.get("mem_peak_mb"), summary.get("mem_total_mb")
    if not peak or not total:
        return True, None, None  # unknown: the per-worker cap still fails fast with OOM instead of paging
    device = summary.get("device_mem_max_mb")
    others = max(0.0, device - peak) if device else 0.0  # other processes plus this process's CUDA context
    free = total - others - WORKER_OVERHEAD_MB * (workers - 1)
    need, cap = peak / workers * MEMORY_MARGIN, MEMORY_LIMIT * free / workers
    return need <= cap, need, cap


def _parallel(trial: dict) -> tuple[str, int, str]:
    params = trial.get("params") or {}
    mode = params.get("distributed") or "none"
    return mode, int(params.get("workers") or 1) if mode != "none" else 1, params.get("transport") or "none"


def _label(mode: str, workers: int, transport: str) -> str:
    return "单进程" if mode == "none" else f"{mode.upper()} × {workers} worker（{transport}）"


class PsTraining(Specialist):
    name = "ps_training"
    label = "PS Training"

    def available(self, evidence):
        if evidence is None:  # the defaults page asks without a campaign
            return True, None
        if evidence.spec["fixed"].get("device") != "cuda":
            return False, "数据并行对比只在 CUDA 上进行"
        if evidence.spec["fixed"].get("dtype", "float32") != "float32":
            return False, "分布式训练按 FP32 交换梯度，只支持 float32"
        return True, None

    def single_best(self, evidence) -> dict | None:
        """The best valid single-process trial: the reference every parallel run is compared with."""
        return next((t for t in evidence.ranked() if _parallel(t)[0] == "none"), None)

    def matrix(self, evidence) -> list[dict]:
        """One row per measured configuration (the single-process reference first)."""
        reference = self.single_best(evidence)
        if reference is None:
            return []
        ref_tps = evidence.summary(reference).get("throughput_sps")
        ref_backend = (reference.get("params") or {}).get("conv_backend")
        rows = [{"trial_id": reference["trial_id"], "mode": "none", "workers": 1, "transport": "none",
                 "throughput_sps": ref_tps, "speedup": 1.0 if ref_tps else None, "sync_share": 0.0,
                 "rank_step_cv": None, "verdict": "valid", "objective": evidence.objective(reference)}]
        for trial in evidence.finished():
            mode, workers, transport = _parallel(trial)
            if mode == "none":
                continue
            summary = evidence.summary(trial)
            dist = summary.get("distributed") or {}
            tps = summary.get("throughput_sps")
            backend = (trial.get("params") or {}).get("conv_backend")
            rows.append({"trial_id": trial["trial_id"], "mode": mode, "workers": workers, "transport": transport,
                         "note": backend if backend != ref_backend else None, "throughput_sps": tps, "speedup": tps / ref_tps if tps and ref_tps else None,
                         "sync_share": dist.get("sync_share"), "rank_step_cv": dist.get("rank_step_cv"),
                         "verdict": (trial.get("verdict") or {}).get("status"), "objective": evidence.objective(trial)})
        return rows

    # ---- diagnosis ------------------------------------------------------------------------------------
    def diagnose(self, evidence) -> list[Finding]:
        findings = []
        for trial in evidence.finished():
            mode, workers, transport = _parallel(trial)
            dist = evidence.summary(trial).get("distributed")
            if mode == "none" or not dist:
                continue
            summary, tid, label = evidence.summary(trial), trial["trial_id"], _label(mode, workers, transport)
            if dist["sync_share"] > SYNC_BOUND:
                findings.append(Finding(
                    id=f"{self.name}:sync_bound:{tid}", agent=self.name, kind="sync_bound", trial_id=tid,
                    severity="warning",
                    message=f"{tid} {label}：梯度同步等待占步时 {dist['sync_share']:.0%}（阈值 {SYNC_BOUND:.0%}），"
                            "通信压过计算；减少 worker 或换方案（PS ↔ Ring）",
                    evidence=[cite(trial, summary, "sync_share", round(dist["sync_share"], 4), SYNC_BOUND, source="ranks")],
                    data={"sync_share": dist["sync_share"], "mode": mode, "workers": workers}))
            if dist["rank_step_cv"] > STRAGGLER_CV:
                slow = max(dist["ranks"], key=lambda r: r["compute_ms_median"])
                findings.append(Finding(
                    id=f"{self.name}:straggler:{tid}", agent=self.name, kind="straggler", trial_id=tid,
                    severity="warning",
                    message=f"{tid} {label}：各 rank 计算时间的变异系数 {dist['rank_step_cv']:.2f}（阈值 {STRAGGLER_CV}），"
                            f"rank {slow['rank']} 最慢（中位 {slow['compute_ms_median']:.1f} ms），其余 rank 在等它",
                    evidence=[cite(trial, summary, "rank_step_cv", round(dist["rank_step_cv"], 4), STRAGGLER_CV,
                                   source="ranks")],
                    data={"ranks": dist["ranks"]}))
        rows = self.matrix(evidence)
        parallel = [r for r in rows[1:] if r["verdict"] == "valid" and r["throughput_sps"]]
        reference = self.single_best(evidence)
        if rows and len(rows) == 1 and reference is not None:
            # Nothing ran in parallel: conclude when no worker count fits the GPU memory at all.
            batch = int((reference.get("params") or {}).get("batch", 1))
            summary = evidence.summary(reference)
            fits = [(w, *memory_fit(summary, w)) for w in WORKERS if w <= batch]
            if fits and not any(ok for _, ok, _, _ in fits) and fits[0][2] is not None:
                w, _, need, cap = fits[0]
                findings.append(Finding(
                    id=f"{self.name}:conclusion", agent=self.name, kind="ps_conclusion", severity="info",
                    message=f"显存上限内放不下数据并行：单进程 {reference['trial_id']} 的显存池峰值 "
                            f"{summary['mem_peak_mb']:.0f} MiB，{w} 个 worker 每个约需 {need:.0f} MiB，"
                            f"而每个 worker 的上限约 {cap:.0f} MiB（其余进程用掉后空闲显存的 90% 平分）：结论为保留单进程",
                    evidence=[cite(reference, summary, "mem_peak_mb", round(summary["mem_peak_mb"], 1), source="result"),
                              cite(reference, summary, "device_mem_max_mb", summary.get("device_mem_max_mb"),
                                   source="resources")],
                    data={"rows": rows, "keep_single_process": True, "memory_bound": True,
                          "workers": w, "need_mb": need, "cap_mb": cap}))
        if rows and parallel:
            ref = rows[0]
            fastest = max(parallel, key=lambda r: r["throughput_sps"])
            if fastest["throughput_sps"] <= (ref["throughput_sps"] or 0):
                message = (f"已测的 {len(parallel)} 种数据并行配置吞吐都低于单进程 {ref['trial_id']}"
                           f"（{ref['throughput_sps']:.1f} samples/s；最快的 {_label(fastest['mode'], fastest['workers'], fastest['transport'])} "
                           f"为 {fastest['throughput_sps']:.1f}，{fastest['speedup']:.2f}×）：结论为保留单进程")
                keep = True
            else:
                message = (f"{_label(fastest['mode'], fastest['workers'], fastest['transport'])} 最快："
                           f"{fastest['throughput_sps']:.1f} samples/s，是单进程的 {fastest['speedup']:.2f}×")
                keep = False
            findings.append(Finding(
                id=f"{self.name}:conclusion", agent=self.name, kind="ps_conclusion", severity="info", message=message,
                evidence=[{"trial_id": r["trial_id"], "metric": "throughput_sps", "value": r["throughput_sps"]}
                          for r in rows],
                data={"rows": rows, "keep_single_process": keep}))
        return findings

    # ---- proposals ------------------------------------------------------------------------------------
    def propose(self, findings: list[Finding], evidence) -> list[Proposal]:
        reference = self.single_best(evidence)
        if reference is None or reference is not evidence.best():
            return []
        params = reference.get("params") or {}
        batch = int(params.get("batch", 1))
        width = int(evidence.spec["fixed"].get("base_width", 64))
        summary = evidence.summary(reference)
        # A parallel layout is measured once per campaign: an adopted throughput knob (conv backend, num_workers)
        # changes the reference, but it acts on the single process and the workers alike.
        tried = {_parallel(t) for t in evidence.trials if t.get("status") != "skipped"}
        out = []
        for workers in WORKERS:
            if workers > batch:
                continue  # every worker needs at least one sample of the global batch
            if not memory_fit(summary, workers)[0]:
                continue  # the workers share one GPU: each would hit its pool cap
            for mode, transports in SCHEMES.items():
                for transport in transports:
                    if transport == "socket_json" and width > SOCKET_JSON_MAX_WIDTH:
                        continue
                    if (mode, workers, transport) in tried:
                        continue
                    out.append(Proposal(
                        delta={"distributed": mode, "workers": workers, "transport": transport}, kind="throughput",
                        agent=self.name, base_trial=reference["trial_id"],
                        rationale=f"与单进程 {reference['trial_id']} 同配置（全局 batch {batch}）的数据并行短跑："
                                  f"{_label(mode, workers, transport)}",
                        expected="比较吞吐与同步占比；比单进程慢则保留单进程"))
        return out

    # ---- report ------------------------------------------------------------------------------------------
    def report_section(self, evidence, findings: list[dict]) -> str | None:
        # Diagnose again: the stored conclusion may predate the last parallel runs.
        fresh = [f.to_dict() for f in self.diagnose(evidence)]
        conclusion = next((f for f in fresh if f.get("kind") == "ps_conclusion"), None)
        findings = fresh
        rows = conclusion["data"]["rows"] if conclusion else self.matrix(evidence)
        if conclusion and (conclusion.get("data") or {}).get("memory_bound"):
            return f"### PS Training：数据并行对比\n\n- {conclusion['message']}\n"
        if len(rows) < 2:
            reference = self.single_best(evidence)
            why = ("全局 batch 小于 2，无法分给多个 worker" if reference is not None and
                   int((reference.get("params") or {}).get("batch", 1)) < 2 else "还没有完成的数据并行短跑")
            return f"### PS Training：数据并行对比\n\n- 未得出结论：{why}\n"
        lines = ["### PS Training：数据并行对比", "",
                 "| trial | 方案 | 吞吐 samples/s | 相对单进程 | 同步等待占比 | rank 计算时间 CV | 结果 |", "|---|---|---|---|---|---|---|"]
        def fmt(value, spec, suffix=""):
            return "—" if value is None else format(value, spec) + suffix

        for r in rows:
            note = f"，{r['note']}" if r.get("note") else ""
            lines.append(f"| {r['trial_id']} | {_label(r['mode'], r['workers'], r['transport'])}{note} | "
                         f"{fmt(r['throughput_sps'], '.1f')} | {fmt(r['speedup'], '.2f', '×')} | "
                         f"{fmt(r['sync_share'], '.0%')} | {fmt(r['rank_step_cv'], '.2f')} | {r['verdict']} |")
        lines.append("")
        for f in findings:
            if f.get("agent") == self.name and f.get("kind") in ("ps_conclusion", "sync_bound", "straggler"):
                lines.append(f"- {f['message']}")
        lines.append("")
        return "\n".join(lines)
