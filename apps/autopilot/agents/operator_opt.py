"""Operator Optimization (spec §7, lecture page 9): operator comparison tables, kernel hot spots, and a
throughput recheck of the conv backend. It reads existing benchmark results and only suggests; it never
changes source code.

Inputs (the newest of each, under ``runs/benchmarks`` then ``docs/experiments``):
- ``matmul-*/results.json`` from ``python -m benchmark.matmul_compare``: NumPy / CuPy / CUDA C time, speedup,
  error and scale (the lecture's data table) plus Nsight Compute metrics of the kernels;
- ``capture.sqlite`` from an Nsight Systems capture: the kernel time ranking.

Kernels are compared on ``sm__cycles_elapsed.max`` when the Nsight Compute rows carry it, otherwise on
``gpu__time_duration.sum`` (the report names the metric used).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .base import Finding, Proposal, Specialist, cite

SEARCH_DIRS = ("runs/benchmarks", "docs/experiments")
LECTURE_SIZES = (1024, 2048, 4096)
LECTURE_ROWS = (("numpy", "NumPy（基线）"), ("cupy", "CuPy"), ("basic-fp32", "CUDA C 基础版"),
                ("tiled16-fp32", "CUDA C 分块版"), ("shared16-fp32", "CUDA C 共享内存版"),
                ("register-fp32", "CUDA C 寄存器分块版（优化）"))
LECTURE_KERNELS = ("basic-fp32", "tiled16-fp32", "shared16-fp32")
# The lecture's NumPy agreement criterion.
AGREEMENT = 1e-5
# Lecture kernels slower than this share of the register-tiled kernel are called memory-bound.
MEMORY_BOUND_SHARE = 0.5
NATIVE_BACKENDS = ("cuda_native_register", "cuda_native_cublas")
HOTSPOTS = 10


def newest(repo_root: Path, pattern: str) -> Path | None:
    found = []
    for folder in SEARCH_DIRS:
        root = repo_root / folder
        if root.is_dir():
            found += [p for p in root.glob(pattern) if p.is_file()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def lecture_table(results: dict) -> list[dict]:
    """The lecture's data-record table: implementation × size, mean time, speedup over NumPy, error vs NumPy."""
    by = {(r["implementation"], r["size"]): r for r in results.get("rows") or []}
    rows = []
    for n in LECTURE_SIZES:
        base = by.get(("numpy", n))
        if base is None:
            continue
        for name, label in LECTURE_ROWS:
            r = by.get((name, n))
            if r is None:
                continue
            ms = r.get("stable_mean_ms") or r.get("median_ms")
            base_ms = base.get("stable_mean_ms") or base.get("median_ms")
            rows.append({"size": n, "implementation": name, "label": label, "time_ms": ms,
                         "speedup": base_ms / ms if ms and base_ms else None,
                         "max_abs_vs_numpy": r.get("max_abs_vs_numpy"), "rel_vs_numpy": r.get("rel_vs_numpy"),
                         "rel_rms_vs_fp64": r.get("rel_rms"), "gflops": r.get("gflops")})
    return rows


def kernel_metrics(results: dict) -> tuple[str | None, list[dict]]:
    """Nsight Compute rows of the matmul kernels, ranked by the comparison metric (largest first)."""
    rows = results.get("ncu") or []
    if not rows:
        return None, []
    metric = "sm__cycles_elapsed.max" if any(r.get("sm__cycles_elapsed.max") is not None for r in rows) \
        else "gpu__time_duration.sum"
    ranked = sorted((r for r in rows if r.get(metric) is not None), key=lambda r: -float(r[metric]))
    return metric, [{"implementation": r.get("implementation"), "kernel": r.get("kernel"), "value": float(r[metric]),
                     "sm_throughput_pct": r.get("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                     "memory_throughput_pct": r.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
                     "bank_conflicts": r.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")}
                    for r in ranked]


def kernel_hotspots(path: Path, top: int = HOTSPOTS) -> dict | None:
    """Kernel time ranking of an Nsight Systems capture (read-only)."""
    try:
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        rows = con.execute(
            "SELECT s.value, COUNT(*), SUM(k.end - k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id = COALESCE(k.shortName, k.demangledName) GROUP BY s.value "
            "ORDER BY 3 DESC").fetchall()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    total = sum(r[2] or 0 for r in rows)
    if not total:
        return None
    return {"path": str(path), "captured": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
            "total_ms": total / 1e6, "kernel_count": len(rows),
            "kernels": [{"name": name, "calls": calls, "total_ms": ns / 1e6, "share": ns / total}
                        for name, calls, ns in rows[:top]]}


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _fmt(value, spec: str) -> str:
    return "—" if value is None else format(value, spec)


class OperatorOptimization(Specialist):
    name = "operator_opt"
    label = "Operator Optimization"

    def __init__(self, repo_root: Path | None = None, **_context):
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        self.matmul_path = newest(self.repo_root, "**/matmul-*/results.json")
        self.capture_path = newest(self.repo_root, "**/capture.sqlite")

    MISSING = ("没有找到 benchmark 结果：先运行 python -m benchmark.matmul_compare --out-dir "
               "runs/benchmarks/matmul-001（或保存一次 Nsight Systems 采样），对比表与 kernel 热点才会出现")

    def available(self, evidence):
        # The conv-backend recheck needs no benchmark files; the tables appear once the files exist.
        return True, None

    def _results(self) -> dict:
        import json

        if self.matmul_path is None:
            return {}
        try:
            return json.loads(self.matmul_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    # ---- diagnosis -------------------------------------------------------------------------------
    def diagnose(self, evidence) -> list[Finding]:
        findings = []
        results = self._results()
        table = lecture_table(results)
        if table:
            source = _rel(self.repo_root, self.matmul_path)
            largest = max(r["size"] for r in table)
            at = {r["implementation"]: r for r in table if r["size"] == largest}
            parts = [f"{at[name]['label']} {at[name]['speedup']:.2f}×" for name, _ in LECTURE_ROWS[1:]
                     if name in at and at[name]["speedup"]]
            findings.append(Finding(
                id=f"{self.name}:operator_table", agent=self.name, kind="operator_table", severity="info",
                message=f"矩阵乘对比表（{source}）：n={largest} 时相对 NumPy 的加速比为 " + "，".join(parts),
                evidence=[{"file": source, "metric": "stable_mean_ms", "value": len(table)}],
                data={"source": source, "rows": table}))
            register = at.get("register-fp32")
            lecture = [at[name] for name in LECTURE_KERNELS if name in at and at[name]["speedup"]]
            if register and register.get("speedup") and lecture:
                fastest = max(lecture, key=lambda r: r["speedup"])
                if fastest["speedup"] < MEMORY_BOUND_SHARE * register["speedup"]:
                    metric, kernels = kernel_metrics(results)
                    by_impl = {k["implementation"]: k for k in kernels}
                    detail = ""
                    if fastest["implementation"] in by_impl and "register-fp32" in by_impl:
                        a, b = by_impl[fastest["implementation"]], by_impl["register-fp32"]
                        detail = (f"；Nsight Compute（n={results.get('ncu_size') or '采样规模'}）SM 吞吐 "
                                  f"{_fmt(a['sm_throughput_pct'], '.0f')}% 对 {_fmt(b['sm_throughput_pct'], '.0f')}%，"
                                  f"访存吞吐 {_fmt(a['memory_throughput_pct'], '.0f')}% 对 "
                                  f"{_fmt(b['memory_throughput_pct'], '.0f')}%")
                    findings.append(Finding(
                        id=f"{self.name}:lecture_kernels_memory_bound", agent=self.name, kind="memory_bound_kernels",
                        severity="info",
                        message=f"课件三版每个线程只算一个输出、每次 FMA 都要读内存：最快的{fastest['label']}在 n={largest} "
                                f"只有 {fastest['speedup']:.2f}×，寄存器分块版 {register['speedup']:.2f}×{detail}。"
                                "建议：卷积与 Dense 的 GEMM 保持寄存器分块引擎（当前默认），课件三版只作教学对照",
                        evidence=[{"file": source, "metric": f"speedup[{fastest['implementation']}@{largest}]",
                                   "value": round(fastest["speedup"], 3)},
                                  {"file": source, "metric": f"speedup[register-fp32@{largest}]",
                                   "value": round(register["speedup"], 3)}],
                        data={"fastest_lecture": fastest, "register": register}))
            bad = [r for r in table if r["implementation"] != "numpy" and r.get("rel_vs_numpy") is not None
                   and r["rel_vs_numpy"] > AGREEMENT]
            if bad:
                worst = max(bad, key=lambda r: r["rel_vs_numpy"])
                findings.append(Finding(
                    id=f"{self.name}:numpy_agreement", agent=self.name, kind="numpy_agreement", severity="warning",
                    message=f"{worst['label']} 在 n={worst['size']} 与 NumPy 的相对误差 {worst['rel_vs_numpy']:.2e}"
                            f" 超过课件的 {AGREEMENT:g}（大规模下 float32 累加顺序不同，应改看相对 FP64 的误差）",
                    evidence=[{"file": source, "metric": "rel_vs_numpy", "value": worst["rel_vs_numpy"],
                               "threshold": AGREEMENT}], data={"rows": bad}))
        if self.capture_path is not None:
            hot = kernel_hotspots(self.capture_path)
            if hot:
                hot["path"] = _rel(self.repo_root, self.capture_path)
                lead = hot["kernels"][0]
                findings.append(Finding(
                    id=f"{self.name}:kernel_hotspots", agent=self.name, kind="kernel_hotspots", severity="info",
                    message=f"最新的 Nsight Systems 采样（{hot['captured']}，{hot['path']}）共 {hot['kernel_count']} 种 kernel、"
                            f"{hot['total_ms']:.1f} ms；最耗时的是 {lead['name']}（{lead['share']:.0%}，{lead['calls']} 次）",
                    evidence=[{"file": hot["path"], "metric": "kernel_time_share", "value": round(lead["share"], 4)}],
                    data=hot))
        findings += self._backend_recheck(evidence)
        return findings

    def _backend_recheck(self, evidence) -> list[Finding]:
        """Once the best trial has run on a native backend, recheck the other GEMM engine by throughput."""
        best = evidence.best()
        if best is None:
            return []
        args = best.get("resolved_args") or {}
        backend = (best.get("params") or {}).get("conv_backend") or args.get("conv_backend") \
            or evidence.spec["fixed"].get("conv_backend")
        if backend not in NATIVE_BACKENDS or args.get("device", evidence.spec["fixed"].get("device")) != "cuda":
            return []
        other = NATIVE_BACKENDS[1 - NATIVE_BACKENDS.index(backend)]
        if any(((t.get("params") or {}).get("conv_backend") or (t.get("resolved_args") or {}).get("conv_backend")) == other
               for t in evidence.trials):
            return []
        summary = evidence.summary(best)
        tps = summary.get("throughput_sps")
        return [Finding(
            id=f"{self.name}:backend_recheck:{best['trial_id']}", agent=self.name, kind="conv_backend_recheck",
            trial_id=best["trial_id"], severity="action",
            message=f"当前最好的 {best['trial_id']} 用 {backend}（{_fmt(tps, '.1f')} samples/s）；"
                    f"按吞吐复核另一个 GEMM 引擎 {other}",
            evidence=[cite(best, summary, "throughput_sps", round(tps, 3) if tps else None)],
            data={"backend": backend, "other": other})]

    def propose(self, findings: list[Finding], evidence) -> list[Proposal]:
        best = evidence.best()
        out = []
        for f in findings:
            if f.kind != "conv_backend_recheck" or best is None or f.trial_id != best["trial_id"]:
                continue
            out.append(Proposal(
                delta={"conv_backend": f.data["other"]}, kind="throughput", agent=self.name,
                base_trial=best["trial_id"], finding_ids=[f.id],
                rationale=f"{f.message}：同一配置只换卷积 GEMM 引擎",
                expected="两个引擎的梯度误差在同一量级（engine-001），吞吐差异超过噪声才沿用"))
        return out

    # ---- report ----------------------------------------------------------------------------------------
    def report_section(self, evidence, findings: list[dict]) -> str | None:
        table = next((f["data"] for f in findings if f.get("kind") == "operator_table"), None)
        hot = next((f["data"] for f in findings if f.get("kind") == "kernel_hotspots"), None)
        lines = ["### Operator Optimization：算子对比（课件第 9 页）", ""]
        if table is None and hot is None:
            return "\n".join(lines + [f"- {self.MISSING}", ""])
        if table:
            lines += [f"来源：`{table['source']}`。GPU 只计 kernel（CUDA event），加速比 = NumPy 时间 / 该实现时间，"
                      "误差以 NumPy float32 结果为基线；相对 FP64 一列是该实现自身的舍入误差。", "",
                      "| 规模 | 实现 | 平均时间 ms | 加速比 | 最大绝对误差（对 NumPy） | 相对误差（对 NumPy） | 相对 FP64 RMS |",
                      "|---|---|---|---|---|---|---|"]
            for r in table["rows"]:
                lines.append(f"| {r['size']} | {r['label']} | {_fmt(r['time_ms'], '.3f')} | "
                             f"{_fmt(r['speedup'], '.2f')}× | {_fmt(r['max_abs_vs_numpy'], '.2e')} | "
                             f"{_fmt(r['rel_vs_numpy'], '.2e')} | {_fmt(r['rel_rms_vs_fp64'], '.2e')} |")
            lines.append("")
            metric, kernels = kernel_metrics(self._results())
            if kernels:
                lines += [f"Nsight Compute 的 kernel 比较（指标 `{metric}`，越小越快）：", "",
                          "| 实现 | kernel | 指标值 | SM 吞吐 % | 访存吞吐 % | 共享内存 bank 冲突 |", "|---|---|---|---|---|---|"]
                lines += [f"| {k['implementation']} | `{k['kernel']}` | {k['value']:.0f} | {_fmt(k['sm_throughput_pct'], '.1f')} | "
                          f"{_fmt(k['memory_throughput_pct'], '.1f')} | {_fmt(k['bank_conflicts'], '.0f')} |"
                          for k in sorted(kernels, key=lambda k: k["value"]) if k["implementation"].endswith("fp32")
                          or k["implementation"] == "cublas"]
                lines.append("")
        if hot:
            lines += [f"Nsight Systems kernel 热点（{hot['captured']}，`{hot['path']}`，合计 {hot['total_ms']:.1f} ms）：", "",
                      "| kernel | 次数 | 合计 ms | 占比 |", "|---|---|---|---|"]
            lines += [f"| `{k['name'][:80]}` | {k['calls']} | {k['total_ms']:.2f} | {k['share']:.1%} |" for k in hot["kernels"]]
            lines.append("")
        advice = [f["message"] for f in findings if f.get("kind") in ("memory_bound_kernels", "numpy_agreement")]
        if advice:
            lines += ["建议（只提建议，不修改源码）：", ""] + [f"- {a}" for a in advice] + [""]
        return "\n".join(lines)
