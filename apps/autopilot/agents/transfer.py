"""Transfer Learning (spec §7, E2): pretrained-weight provenance, frozen-stage integrity and the freeze_stages search.

Available when the campaign fixes ``pretrained`` and the converted ImageNet weights exist (the same capability
check that lets ``freeze_stages`` into the search space). Rules:

- load report: how many backbone tensors were loaded, which head was replaced, the weights' checksums;
- integrity: a trial whose frozen parameters or BN buffers changed during training is flagged (the Verifier
  also marks it invalid);
- comparison: valid trials grouped by the number of frozen stages; around the best trial, the nearest frozen
  levels of the search space not yet tried with its lr/batch are proposed on the same seed. When the best
  trial's validation loss rose after its best epoch (over-fitting), freezing more is tried first.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from ..spec import TRANSFER_DEFAULTS, transfer_status
from .base import Finding, Proposal, Specialist, cite

ROOT = Path(__file__).resolve().parents[3]
ARCHS = ("resnet18", "mobilenet_v2", "alexnet")
OVERFIT_RISE = 0.10  # final val loss this much above the best epoch's: the run over-fitted


def freeze_level(trial: dict, evidence) -> int:
    transfer = evidence.summary(trial).get("transfer") or {}
    if transfer.get("freeze_stages") is not None:
        return int(transfer["freeze_stages"])
    params = trial.get("params") or {}
    fixed = evidence.spec.get("fixed") or {}
    return int(params.get("freeze_stages", fixed.get("freeze_stages", TRANSFER_DEFAULTS["freeze_stages"])))


def _overfit(trial: dict) -> float | None:
    """Relative rise of the validation loss from its best epoch to the last one."""
    epochs = [e for e in ((trial.get("result") or {}).get("epochs") or []) if e.get("val_loss") is not None]
    if len(epochs) < 2:
        return None
    best = min(float(e["val_loss"]) for e in epochs)
    last = float(epochs[-1]["val_loss"])
    return (last - best) / best if best > 0 else None


def comparison(evidence) -> list[dict]:
    """Valid transfer trials grouped by the number of frozen stages, fewest frozen first."""
    groups: dict[int, list[dict]] = {}
    for trial in evidence.valid():
        if evidence.summary(trial).get("transfer"):
            groups.setdefault(freeze_level(trial, evidence), []).append(trial)
    rows = []
    for level in sorted(groups):
        trials = groups[level]
        values = [evidence.objective(t) for t in trials]
        steps = [evidence.summary(t).get("step_ms_median") for t in trials]
        steps = [s for s in steps if s is not None]
        transfer = [evidence.summary(t)["transfer"] for t in trials]
        rows.append({
            "freeze_stages": level, "trials": [t["trial_id"] for t in trials], "n": len(trials),
            "mean": statistics.fmean(values), "best": (min if evidence.minimize else max)(values),
            "trainable_values": transfer[0].get("trainable_values"), "frozen": transfer[0].get("frozen"),
            "step_ms_median": statistics.fmean(steps) if steps else None,
            "frozen_unchanged": all(t.get("frozen_unchanged") is not False for t in transfer),
        })
    return rows


class TransferLearning(Specialist):
    name = "transfer"
    label = "Transfer Learning"

    def __init__(self, repo_root=None):
        self.repo_root = Path(repo_root or ROOT)

    def available(self, evidence):
        if evidence is None:  # the console's capability list: are any converted weights there at all?
            converted = [a for a in ARCHS if (self.repo_root / "runs" / "pretrained" / "converted" / f"{a}.npz").is_file()]
            return (True, None) if converted else (
                False, "没有转换后的 ImageNet 预训练权重（先运行 tools/convert_torchvision_weights.py）")
        return transfer_status(evidence.spec.get("fixed"), self.repo_root)

    # ---- diagnose ---------------------------------------------------------------------------------
    def diagnose(self, evidence) -> list[Finding]:
        findings = []
        with_transfer = [t for t in evidence.finished() if evidence.summary(t).get("transfer")]
        if with_transfer:
            first = with_transfer[0]
            info = evidence.summary(first)["transfer"]
            complete = info.get("missing") == 0 and info.get("loaded") == info.get("expected")
            findings.append(Finding(
                id=f"{self.name}:load_report", agent=self.name, kind="pretrained_load",
                severity="info" if complete else "warning", trial_id=first["trial_id"],
                message=(f"{info.get('arch')} 预训练主干载入 {info.get('loaded')}/{info.get('expected')} 个张量"
                         f"（替换任务头 {', '.join(info.get('replaced_head') or []) or '无'}；来源 sha256 "
                         f"{(info.get('source_sha256') or '')[:12]}…；输入归一化 {info.get('input_norm')}）"),
                evidence=[cite(first, evidence.summary(first), "loaded_tensors", info.get("loaded"), source="transfer")],
                data=dict(info)))
        for trial in with_transfer:
            info = evidence.summary(trial)["transfer"]
            if info.get("frozen_unchanged") is False:
                findings.append(Finding(
                    id=f"{self.name}:frozen_changed:{trial['trial_id']}", agent=self.name, kind="frozen_changed",
                    severity="warning", trial_id=trial["trial_id"],
                    message=f"{trial['trial_id']} 冻结 {info.get('freeze_stages')} 个阶段，但冻结张量在训练后改变",
                    evidence=[cite(trial, evidence.summary(trial), "frozen_unchanged", False, source="transfer")]))
        rows = comparison(evidence)
        best = evidence.best()
        if rows and best is not None and evidence.summary(best).get("transfer"):
            level = freeze_level(best, evidence)
            space = sorted(int(v) for v in (evidence.spec.get("space") or {}).get("freeze_stages") or [])
            tried = self._tried_levels(evidence, best)
            # Only the adjacent levels of the space: past a neighbour already run (and worse), the search stops
            above = next((v for v in space if v > level), None)
            below = next((v for v in reversed(space) if v < level), None)
            rise = _overfit(best)
            neighbours = [v for v in (above, below) if v is not None and v not in tried]
            if rise is None or rise <= OVERFIT_RISE:
                neighbours.reverse()  # not over-fitted: training more of the network is tried first
            table = "；".join(f"冻结 {r['freeze_stages']}：{r['mean']:.5f}（{r['n']} 个）" for r in rows)
            message = f"按冻结阶段比较 {evidence.metric}：{table}；最佳 {best['trial_id']} 冻结 {level} 个阶段"
            if rise is not None and rise > OVERFIT_RISE:
                message += f"，其验证 loss 在最佳 epoch 后上升 {rise:.0%}（过拟合）"
            findings.append(Finding(
                id=f"{self.name}:freeze_comparison", agent=self.name, kind="freeze_comparison",
                severity="action" if neighbours else "info", trial_id=best["trial_id"],
                message=message + ("" if neighbours else "；搜索空间内相邻的冻结阶段都已比较"),
                evidence=[cite(best, evidence.summary(best), evidence.metric, evidence.objective(best), source="result")],
                data={"rows": rows, "best_level": level, "neighbours": neighbours, "overfit_rise": rise,
                      "space": space}))
        return findings

    def _tried_levels(self, evidence, best: dict) -> set[int]:
        """Frozen levels already run (or queued) with the best trial's lr and batch."""
        params = best.get("params") or {}
        out = set()
        for trial in evidence.trials:
            other = trial.get("params") or {}
            if trial.get("status") == "skipped":
                continue
            if other.get("lr") == params.get("lr") and other.get("batch") == params.get("batch"):
                fixed = evidence.spec.get("fixed") or {}
                out.add(int(other.get("freeze_stages", fixed.get("freeze_stages", 0))))
        return out

    # ---- propose ----------------------------------------------------------------------------------
    def propose(self, findings: list[Finding], evidence) -> list[Proposal]:
        proposals = []
        for finding in findings:
            if finding.kind != "freeze_comparison" or finding.severity != "action":
                continue
            level = finding.data["best_level"]
            rows = {r["freeze_stages"]: r for r in finding.data["rows"]}
            for target in finding.data["neighbours"]:
                direction = "多冻结" if target > level else "少冻结"
                why = ("减少可训练参数、抑制过拟合" if target > level else "让更多层适应 DonkeyCar 图像")
                proposals.append(Proposal(
                    delta={"freeze_stages": target}, kind="accuracy", agent=self.name, base_trial=finding.trial_id,
                    finding_ids=[finding.id],
                    rationale=f"{finding.message}；{direction}：freeze_stages {level}→{target}（{why}），lr/batch/seed 不变",
                    expected=f"与 {finding.trial_id} 同 seed 配对比较 {evidence.metric}"
                             + (f"；冻结 {target} 已有结果 {rows[target]['mean']:.5f}" if target in rows else "")))
        return proposals

    # ---- report -----------------------------------------------------------------------------------
    def report_section(self, evidence, findings: list[dict]) -> str:
        lines = [f"### {self.label}", ""]
        with_transfer = [t for t in evidence.finished() if evidence.summary(t).get("transfer")]
        if not with_transfer:
            return "\n".join(lines + ["- 还没有带迁移训练记录的 trial。", ""])
        info = evidence.summary(with_transfer[0])["transfer"]
        lines += [f"- 预训练权重：{info.get('arch')}，来源 sha256 `{(info.get('source_sha256') or '')[:16]}`，"
                  f"转换文件 sha256 `{(info.get('npz_sha256') or '')[:16]}`",
                  f"- 主干载入 {info.get('loaded')}/{info.get('expected')} 个张量（成功率 "
                  f"{(info.get('success_rate') or 0):.0%}），替换的任务头：{', '.join(info.get('replaced_head') or []) or '无'}；"
                  f"输入归一化 {info.get('input_norm')}"]
        changed = [t["trial_id"] for t in with_transfer if evidence.summary(t)["transfer"].get("frozen_unchanged") is False]
        lines.append("- 冻结检查：" + ("所有 trial 的冻结张量训练前后逐字节相同" if not changed
                                   else f"{', '.join(changed)} 的冻结张量被改动（已判为无效）"))
        rows = comparison(evidence)
        if rows:
            lines += ["", f"| 冻结阶段 | 冻结的阶段 | 可训练参数 | trial 数 | {evidence.metric} 均值 | 最好 | 步时中位数 ms |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for r in rows:
                step = "—" if r["step_ms_median"] is None else f"{r['step_ms_median']:,.0f}"
                lines.append(f"| {r['freeze_stages']} | {', '.join(r['frozen'] or []) or '—'} | "
                             f"{r['trainable_values'] or 0:,} | {r['n']} | {r['mean']:.5f} | {r['best']:.5f} | {step} |")
        for finding in findings:
            if finding.get("kind") == "freeze_comparison":
                lines += ["", f"- 结论：{finding.get('message')}"]
        return "\n".join(lines + [""])
