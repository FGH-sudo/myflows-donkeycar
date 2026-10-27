"""Reporter (spec §6.6): report.md, leaderboard.csv and figures for one campaign."""

from __future__ import annotations

import csv
import math
import statistics
import time
from pathlib import Path

from .files import write_json
from .planner import confirm_seeds
from .spec import MODES, SPECIALISTS

PHASE_NAMES = {"baseline": "基线", "explore": "探索", "exploit": "利用", "confirm": "确认", "done": "完成"}
LEADERBOARD_FIELDS = ("rank", "trial_id", "phase", "origin", "lr", "lr_effective", "batch", "freeze_stages", "num_workers",
                      "seed", "objective",
                      "val_loss", "throughput_sps", "data_share", "gpu_util_median", "mem_peak_mb", "verdict",
                      "reason", "run_dir")


def _fmt(value, digits: int = 5) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if abs(value) >= 1000 or (value != 0 and abs(value) < 1e-3):
            return f"{value:.3g}"
        return f"{value:.{digits}g}"
    return str(value)


def _pct(value) -> str:
    return "—" if value is None else f"{value:.0%}"


def _ms(value) -> str:
    return "—" if value is None else f"{value:,.0f}"


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def _transfer(ev) -> bool:
    """A transfer-learning campaign: pretrained weights fixed (the leaderboard then shows freeze_stages)."""
    return bool((ev.spec.get("fixed") or {}).get("pretrained"))


def code_changes(ev, campaign_hash: str | None) -> list[str]:
    """Attempts started with other code than the campaign's (each attempt records the code at its launch)."""
    return [f"{t['trial_id']}#{a['attempt']}" for t in ev.trials for a in t.get("attempts") or []
            if a.get("code_hash") and a["code_hash"] != (t.get("code_hash") or campaign_hash)]


def leaderboard(ev) -> list[dict]:
    rows = []
    ranked = {t["trial_id"]: i for i, t in enumerate(ev.ranked(), start=1)}
    for trial in ev.trials:
        verdict = trial.get("verdict") or {}
        summary = ev.summary(trial)
        params = trial.get("params") or {}
        metrics = (trial.get("result") or {}).get("metrics") or {}
        status = verdict.get("status") or trial.get("status")
        if verdict.get("kind"):
            status = f"{status}:{verdict['kind']}"
        reason = "；".join(verdict.get("reasons") or []) or trial.get("skip_reason") or ""
        rows.append({
            "rank": ranked.get(trial["trial_id"]), "trial_id": trial["trial_id"], "phase": trial.get("phase"),
            "origin": (trial.get("origin") or {}).get("strategy"), "lr": params.get("lr"),
            "lr_effective": _effective_lr(trial), "batch": params.get("batch"),
            "freeze_stages": params.get("freeze_stages", (ev.spec.get("fixed") or {}).get("freeze_stages", 0))
            if _transfer(ev) else None,
            "num_workers": params.get("num_workers"), "seed": params.get("seed"), "objective": ev.objective(trial),
            "val_loss": metrics.get("val_loss"), "throughput_sps": summary.get("throughput_sps"),
            "data_share": summary.get("data_share"), "gpu_util_median": summary.get("gpu_util_median"),
            "mem_peak_mb": summary.get("mem_peak_mb"), "verdict": status, "reason": reason,
            "run_dir": (trial.get("attempts") or [{}])[-1].get("run_dir"),
        })
    rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0, r["trial_id"]))
    return rows


def confirmation(ev) -> dict:
    """Confirmation statistics: candidate vs baseline on the confirmation seeds (paired by seed)."""
    plan = ev.state.get("confirm_plan") or {}
    seeds = plan.get("seeds") or confirm_seeds(ev.spec)
    noise = ev.state.get("noise") or {}
    base = ev.spec["baseline"]
    baseline_by_seed = {}
    for t in ev.valid():
        p = t.get("params") or {}
        if all(p.get(k) == base[k] for k in ("lr", "batch")) and p.get("seed") in seeds and \
                set(p) <= {"lr", "batch", "num_workers", "seed"}:
            baseline_by_seed.setdefault(p["seed"], ev.objective(t))
    out = {"seeds": seeds, "noise_std": noise.get("std"), "baseline": {}, "candidates": [], "note": plan.get("note")}
    out["baseline"]["values"] = baseline_by_seed
    out["baseline"]["mean"], out["baseline"]["std"] = _mean_std(list(baseline_by_seed.values()))
    for item in plan.get("candidates") or []:
        values = {}
        for t in ev.trials:
            if t.get("phase") == "confirm" and (t.get("origin") or {}).get("base_trial") == item["trial_id"]:
                if (t.get("verdict") or {}).get("status") == "valid":
                    values[t["params"]["seed"]] = ev.objective(t)
        diffs = [values[s] - baseline_by_seed[s] for s in seeds if s in values and s in baseline_by_seed]
        mean, std = _mean_std(list(values.values()))
        dmean, dstd = _mean_std(diffs)
        complete = len(values) == len(seeds) and len(diffs) == len(seeds)
        improvement = None if dmean is None else (-dmean if ev.minimize else dmean)
        if not complete:
            verdict = "未确认（确认运行不完整）"
        elif noise.get("std") is None:
            verdict = "无法判断（缺少基线噪声水平）"
        elif improvement is not None and improvement > noise["std"]:
            verdict = "显著优于人工基线（提升大于基线 seed 间标准差）"
        elif improvement is not None and improvement > 0:
            verdict = "与基线无显著差别（提升小于基线 seed 间标准差）"
        else:
            verdict = "未超过人工基线"
        out["candidates"].append({**item, "values": values, "mean": mean, "std": std, "diffs": diffs,
                                  "diff_mean": dmean, "diff_std": dstd, "improvement": improvement,
                                  "complete": complete, "verdict": verdict})
    return out


def conclusion(ev, confirm: dict) -> str:
    if not ev.valid("baseline"):
        return "人工基线没有有效结果，本 campaign 不能给出结论。"
    significant = [c for c in confirm["candidates"] if c["verdict"].startswith("显著")]
    if significant:
        best = min(significant, key=lambda c: c["mean"] if ev.minimize else -c["mean"])
        return (f"{best['trial_id']} 的配置（{_params_text(best['params'])}）在确认运行中优于人工基线："
                f"{ev.metric} {_fmt(best['mean'])} ± {_fmt(best['std'])}，配对差值 {_fmt(best['diff_mean'])} ± "
                f"{_fmt(best['diff_std'])}，大于基线噪声 {_fmt(confirm['noise_std'])}。")
    if confirm["candidates"]:
        parts = "；".join(f"{c['trial_id']}：{c['verdict']}" for c in confirm["candidates"])
        return f"人工基线未被超越。确认结果：{parts}。"
    baseline = [ev.objective(t) for t in ev.valid("baseline")]
    others = [t for t in ev.ranked() if t.get("phase") != "baseline"]
    mean = _fmt(sum(baseline) / len(baseline))
    status = ev.state.get("status")
    if not others:
        ended = (f"campaign 在{PHASE_NAMES.get(ev.state.get('phase'), ev.state.get('phase'))}阶段结束（{status}），"
                 if status != "completed" else "")
        return (f"{ended}没有候选 trial 得到有效结果，无法判断能否超越人工基线；人工基线 {len(baseline)} 个 seed 的均值 "
                f"{mean}。")
    text = (f"人工基线未被超越：{len(others)} 个候选在筛选中都没有优于人工基线 {len(baseline)} 个 seed 的均值 "
            f"{mean}，因此不需要确认运行。")
    if status not in (None, "completed"):
        text += f"（campaign 提前结束（{status}），只比较了已跑完的候选。）"
    if others:
        best = others[0]
        rolled = "" if _effective_lr(best) is None else f"，回滚后实际 lr={_fmt(_effective_lr(best))}"
        text += (f"最好的候选是 {best['trial_id']}（{_params_text({k: v for k, v in best['params'].items() if k != 'seed'})}"
                 f"{rolled}），{ev.metric} = {_fmt(ev.objective(best))}。")
    return text


def _effective_lr(trial: dict) -> float | None:
    """The lr the trial finished with when a rollback reduced it, else None."""
    rollbacks = trial.get("rollbacks") or []
    return rollbacks[-1]["lr"] if rollbacks else None


def _params_text(params: dict) -> str:
    return "、".join(f"{k}={_fmt(v)}" for k, v in sorted(params.items()))


def _figures(ev, out_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    from .files import read_jsonl

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    valid = [t for t in ev.trials if ev.objective(t) is not None and (t.get("verdict") or {}).get("status") == "valid"]
    if valid:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        colors = {"baseline": "#71717a", "explore": "#3b6fd8", "exploit": "#c47a12", "confirm": "#2f9e62"}
        ax.bar([t["trial_id"] for t in valid], [ev.objective(t) for t in valid],
               color=[colors.get(t.get("phase"), "#999") for t in valid])
        ax.set_ylabel(ev.metric)
        ax.set_title("objective of valid trials (grey=baseline, blue=explore, orange=exploit, green=confirm)", fontsize=9)
        ax.tick_params(axis="x", labelrotation=60, labelsize=7)
        fig.tight_layout()
        path = out_dir / "objective.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path.name)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for t in valid[:12]:
            run_dir = (t.get("attempts") or [{}])[-1].get("run_dir")
            rows = [r for r in read_jsonl(Path(run_dir) / "metrics.jsonl") if r.get("kind") == "epoch"] if run_dir else []
            if rows:
                ax.plot([r["epoch"] for r in rows], [r.get("val_angle_mae") or r.get("val_loss") for r in rows],
                        marker="o", label=t["trial_id"])
        ax.set_xlabel("epoch")
        ax.set_ylabel(ev.metric)
        ax.legend(fontsize=7, ncol=4)
        fig.tight_layout()
        path = out_dir / "curves.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path.name)
        tps = [(t["trial_id"], ev.summary(t).get("throughput_sps")) for t in valid]
        tps = [(tid, v) for tid, v in tps if v]
        if tps:
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.bar([tid for tid, _ in tps], [v for _, v in tps], color="#3b6fd8")
            ax.set_ylabel("samples/sec incl. data loading")
            ax.tick_params(axis="x", labelrotation=60, labelsize=7)
            fig.tight_layout()
            path = out_dir / "throughput.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            paths.append(path.name)
    return paths


def write(ev, store, *, llm_events: list[dict] | None = None, specialists_status: dict | None = None,
          sections: list[str] | None = None) -> Path:
    """Write report/report.md, report/leaderboard.csv and report/figures/*.png; returns the report path."""
    report_dir = store.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    spec, state = ev.spec, ev.state
    rows = leaderboard(ev)
    with (report_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in LEADERBOARD_FIELDS})
    figures = _figures(ev, report_dir / "figures")
    confirm = confirmation(ev)
    write_json(report_dir / "confirmation.json", confirm)
    from .scheduler import budget_state

    used = budget_state(ev.trials)
    fixed = spec["fixed"]
    split = (state.get("provenance") or {}).get("split") or {}
    code = (state.get("provenance") or {}).get("code") or {}
    env = (state.get("provenance") or {}).get("environment") or {}
    lines = [f"# AutoPilot 报告：{spec['campaign_id']}", ""]
    if spec.get("title"):
        lines += [spec["title"], ""]
    lines += [f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
              f"- 状态：{state.get('status')} · 阶段：{state.get('phase')}"
              + (f" · 结束原因：{state['stop_reason']}" if state.get("stop_reason") else ""),
              f"- 模式：{MODES.get(spec['mode'], spec['mode'])}（seed {len(spec['baseline']['seeds'])} 个）· 创建者：{spec['created_by']}",
              "", "## 结论", "", conclusion(ev, confirm), ""]
    lines += ["## 1. 目标、搜索空间与设置", "",
              f"- 任务：{spec['task']}；目标：{spec['objective']['metric']}（{'越低越好' if ev.minimize else '越高越好'}）",
              f"- 人工基线：lr={_fmt(spec['baseline']['lr'])}、batch={spec['baseline']['batch']}、num_workers="
              f"{spec['baseline']['num_workers']}，seed {spec['baseline']['seeds']}",
              f"- 搜索空间：" + "；".join(f"{k} ∈ {v}" for k, v in spec["space"].items())
              + f"（策略 {spec['strategy']['name']}，最多 {spec['strategy']['max_candidates']} 个候选）",
              f"- 固定参数：" + "、".join(f"{k}={v}" for k, v in fixed.items() if k not in ("split_file",)),
              f"- 数据划分：`{split.get('split_file')}`（{split.get('kind', 'random')}；训练 {split.get('train_count')}"
              f" 条、验证 {split.get('val_count')} 条、测试 {split.get('test_count', 0)} 条；验证集指纹 "
              f"`{(split.get('val_hash') or '')[:12]}`）",
              "- 子 Agent：" + "；".join(f"{SPECIALISTS.get(n, n)}：{'可用' if ok else '不可用（' + (why or '') + '）'}"
                                        for n, (ok, why) in (specialists_status or {}).items()),
              f"- 吞吐参数沿用：{state.get('adopted') or '无'}", ""]
    lines += ["## 2. 预算与实际用量", "",
              f"| 项目 | 预算 | 实际 |", "| --- | --- | --- |",
              f"| trial 数 | {spec['budget']['max_trials']} | {used['trials_started']}（尝试 {used['attempts']} 次） |",
              f"| 总运行时长（不含排队） | {spec['budget']['wall_time_s']:.0f} s | {used['run_s']:.0f} s |",
              f"| 单 trial 上限 | {spec['budget']['trial_timeout_s']:.0f} s | — |",
              f"| 基础设施失败重试 | {spec['budget']['max_retries']} 次 | "
              f"{sum(max(0, len(t.get('attempts') or []) - 1) for t in ev.trials)} 次 |", ""]
    skipped = [t for t in ev.trials if t.get("status") == "skipped"]
    if skipped:
        lines += ["跳过的 trial：", ""] + [f"- {t['trial_id']}：{t.get('skip_reason')}" for t in skipped] + [""]
    lines += ["## 3. 代码版本与环境", ""]
    for name, repo in (code.get("repos") or {}).items():
        dirty = f"，未提交改动 {len(repo.get('dirty_files') or [])} 个文件（哈希 `{(repo.get('dirty_hash') or '')[:12]}`）" \
            if repo.get("dirty_hash") else "，无未提交改动"
        lines.append(f"- {name}：`{(repo.get('head') or '')[:10]}`{dirty}")
    lines += [f"- code_hash：`{(code.get('code_hash') or '')[:16]}`",
              f"- Python {env.get('python')} · {env.get('platform')} · GPU {env.get('gpu')}", ""]
    code_hashes = {t.get("code_hash") for t in ev.trials if t.get("code_hash")}
    if len(code_hashes) > 1:
        lines += [f"> 注意：本 campaign 的 trial 来自 {len(code_hashes)} 个不同的代码版本。", ""]
    changed = code_changes(ev, code.get("code_hash"))
    if changed:
        lines += [f"> 注意：{len(changed)} 次运行启动时的代码与 campaign 开始时不同（{', '.join(changed[:12])}"
                  + ("…" if len(changed) > 12 else "") + "）：期间代码被修改过，请核对这些改动是否影响结果。", ""]
    transfer = _transfer(ev)
    lines += ["## 4. 排行榜", "", "只有 Verdict 为 valid 的 trial 参与排名；失败、无效和跳过的 trial 也列出原因。", "",
              "| 名次 | trial | 阶段 | 来源 | lr | batch | " + ("冻结阶段 | " if transfer else "") + "workers | seed | "
              + spec["objective"]["metric"] + " | samples/s | 数据读取占比 | GPU 利用率 | Verdict | 原因 |",
              "| --- " * (15 if transfer else 14) + "|"]
    for r in rows:
        lr = _fmt(r["lr"]) if r.get("lr_effective") is None else f"{_fmt(r['lr'])}（回滚后 {_fmt(r['lr_effective'])}）"
        freeze = f"{r['freeze_stages']} | " if transfer else ""
        lines.append(f"| {r['rank'] or ''} | {r['trial_id']} | {r['phase']} | {r['origin']} | {lr} | "
                     f"{r['batch']} | {freeze}{r['num_workers']} | {r['seed']} | {_fmt(r['objective'])} | "
                     f"{_fmt(r['throughput_sps'], 4)} | {_pct(r['data_share'])} | "
                     f"{_fmt(r['gpu_util_median'], 3)} | {r['verdict']} | {r['reason'][:80]} |")
    lines += ["", "## 5. 诊断链", ""]
    chains = state.get("chains") or []
    if not chains:
        lines += ["本 campaign 没有由诊断派生的 trial。", ""]
    by_id = {t["trial_id"]: t for t in ev.trials}
    for i, chain in enumerate(chains, start=1):
        before, after = dict(chain["before"]), dict(chain["after"])
        for side, tid in ((before, chain["base_trial"]), (after, chain["trial_id"])):
            if tid in by_id:  # chains recorded before a field existed take it from the trial's own evidence
                summary = ev.summary(by_id[tid])
                for key in ("mem_share", "step_ms_median"):
                    side.setdefault(key, summary.get(key))
        lines += [f"### 5.{i} {chain['proposal'].get('agent')} → {chain['trial_id']}（{chain['decision']}）", ""]
        for fid in chain.get("finding_ids") or []:
            finding = (state.get("findings") or {}).get(fid) or {}
            lines.append(f"- 发现 `{fid}`：{finding.get('message')}")
            for item in finding.get("evidence") or []:
                lines.append(f"  - 证据：{item.get('metric')} = {_fmt(item.get('value'))}"
                             + (f"（阈值 {item['threshold']}）" if item.get("threshold") is not None else "")
                             + f"，来自 `{item.get('file')}`")
        lines += [f"- 建议：{chain['proposal'].get('rationale')}（参数变化 {chain['proposal'].get('delta')}）",
                  f"- 派生 trial：{chain['trial_id']}（基准 {chain['base_trial']}，同 seed 配对）", "",
                  "| | 基准 " + chain["base_trial"] + " | 派生 " + chain["trial_id"] + " |", "| --- | --- | --- |",
                  f"| {ev.metric} | {_fmt(before['objective'])} | {_fmt(after['objective'])} |",
                  f"| samples/s（含数据读取） | {_fmt(before['throughput_sps'], 4)} | {_fmt(after['throughput_sps'], 4)} |",
                  f"| 数据读取占比 | {_pct(before['data_share'])} | {_pct(after['data_share'])} |",
                  f"| GPU 利用率中位数 | {_fmt(before['gpu_util_median'], 3)} | {_fmt(after['gpu_util_median'], 3)} |",
                  f"| 显存峰值占比 | {_pct(before.get('mem_share'))} | {_pct(after.get('mem_share'))} |",
                  f"| 步时中位数 ms | {_ms(before.get('step_ms_median'))} | {_ms(after.get('step_ms_median'))} |",
                  "", f"- 结论：{chain.get('reason')}", ""]
    rolled = [t for t in ev.trials if t.get("rollbacks")]
    if rolled:
        lines += ["### 异常与精确回滚", "",
                  "| trial | 失败 | 检测到的异常 | 回到 | lr | 结果 |", "|---|---|---|---|---|---|"]
        for t in rolled:
            verdict = (t.get("verdict") or {}).get("status")
            attempts = {a["attempt"]: a for a in t.get("attempts") or []}
            for i, r in enumerate(t["rollbacks"], start=1):
                last = i == len(t["rollbacks"])
                failure = (attempts.get(r["failed_attempt"]) or {}).get("failure") or {}
                target = "初始权重（第 0 步）" if r["from_step"] == 0 else f"第 {r['from_step']} 步"
                lines.append(f"| {t['trial_id']} | attempt {r['failed_attempt']}：{r['kind']} | "
                             f"{failure.get('message') or '—'} | {target}"
                             f"（attempt {r['from_attempt']} 的 TrainingState） | {_fmt(r['lr_before'])} → {_fmt(r['lr'])} | "
                             f"{(verdict or '—') if last else '再次失败'} |")
        lines += ["", "回滚从失败前最近的健康 TrainingState 逐位精确续训（随机数状态、数据游标与优化器状态都恢复），"
                  "只把学习率按 rollback.lr_factor 缩小。", ""]
    grouped: dict[tuple, list[dict]] = {}
    withdrawn: dict[tuple, list[dict]] = {}
    for f in (state.get("findings") or {}).values():
        if f.get("severity") in ("warning", "info") and f.get("kind") not in ("noise_level",):
            target = withdrawn if f.get("retracted") else grouped
            target.setdefault((f.get("agent"), f.get("kind")), []).append(f)

    def where(items):
        trials = [f.get("trial_id") for f in items if f.get("trial_id")]
        return f"（{len(trials)} 个 trial：{'、'.join(trials[:8])}{' 等' if len(trials) > 8 else ''}）" if trials else ""

    if grouped:
        lines += ["其他发现：", ""]
        for (agent, _kind), items in grouped.items():
            lines.append(f"- [{agent}] {items[0].get('message')}{where(items)}")
        lines.append("")
    if withdrawn:
        lines += ["已撤回的发现（运行时记录，重新生成报告时按当前规则重放后不再成立）：", ""]
        for (agent, _kind), items in withdrawn.items():
            lines.append(f"- [{agent}] ~~{items[0].get('message')}~~{where(items)}：{items[0]['retracted']}")
        lines.append("")
    for section in sections or []:
        lines += [section.rstrip(), ""]
    lines += ["## 6. 最佳配置与确认", ""]
    noise = state.get("noise") or {}
    if noise:
        lines += [f"- 噪声水平：人工基线 seed {noise.get('seeds')} 的 {ev.metric} = {_fmt(noise.get('mean'))} ± "
                  f"{_fmt(noise.get('std'))}" + (f"，吞吐变异系数 {noise['throughput_cv']:.1%}"
                                                 if noise.get("throughput_cv") is not None else "")]
    best = ev.best()
    if best is not None:
        lines.append(f"- 筛选排名第一：{best['trial_id']}（{_params_text(best['params'])}），{ev.metric} = "
                     f"{_fmt(ev.objective(best))}；选择理由：有效 trial 中 {ev.metric} "
                     f"{'最低' if ev.minimize else '最高'}，吞吐打破平局")
    lines.append(f"- 确认 seed：{confirm['seeds']}；人工基线在这些 seed 上：{_fmt(confirm['baseline']['mean'])} ± "
                 f"{_fmt(confirm['baseline']['std'])}")
    for c in confirm["candidates"]:
        lines += [f"- {c['trial_id']}（筛选名次 {c['rank']}，{_params_text(c['params'])}）：确认运行 "
                  f"{_fmt(c['mean'])} ± {_fmt(c['std'])}（{c['values']}）；与基线的配对差值 "
                  f"{_fmt(c['diff_mean'])} ± {_fmt(c['diff_std'])} → **{c['verdict']}**"]
    if confirm.get("note"):
        lines.append(f"- 说明：{confirm['note']}")
    lines.append("")
    test = state.get("test_evaluation")
    if test:
        lines += ["### 独立测试集", "",
                  f"测试集（{split.get('test_count')} 条，{split.get('kind', 'random')} 划分）不参与排名与确认，"
                  "只在 campaign 结束时用各 trial 的最佳模型评估一次。", "",
                  "| 配置 | 参数 | trial | 测试集 angle MAE | 均值 |", "|---|---|---|---|---|"]
        labels = {"baseline": "人工基线", "best": "最好的候选"}
        for group, item in test["groups"].items():
            values = "、".join(f"{tid} {_fmt(v)}" for tid, v in item["angle_mae"].items()) or "—"
            lines.append(f"| {labels.get(group, group)} | {_params_text(item.get('params') or {})} | "
                         f"{'、'.join(item['trials'])} | {values} | {_fmt(item.get('mean'))} |")
        for failure in test.get("failures") or []:
            lines.append(f"- 评估失败：{failure.get('key')}：{failure.get('error')}")
        lines.append("")
    if figures:
        lines += ["## 7. 图表", ""] + [f"![{name}](figures/{name})" for name in figures] + [""]
    lines += ["## 8. LLM", ""]
    llm = state.get("llm") or {}
    if not spec["llm"]["enabled"]:
        lines.append("- 本 campaign 未启用 LLM，全部决策来自规则策略")
    else:
        lines.append(f"- 后台调用 {llm.get('calls', 0)} 次（上限 {spec['llm']['max_calls']}），token "
                     f"{llm.get('tokens', 0)}" + (f"；已关闭：{llm['disabled_reason']}" if llm.get("disabled_reason")
                                                 else ""))
    for event in llm_events or []:
        kind = event.get("kind")
        if kind == "llm_call" and event.get("status") != "started":
            usage = event.get("usage") or {}
            tools = ", ".join(str(c.get("tool", "")).removeprefix("mcp__autopilot__") for c in event.get("tool_calls") or [])
            lines.append(f"- 顾问调用（{event.get('reason')}）：{'成功' if event.get('status') == 'ok' else '失败'}，"
                         f"{usage.get('total_tokens', 0)} token，{_fmt(event.get('duration_s'))} 秒"
                         + (f"，工具 {tools}" if tools else "") + (f"；{event['error']}" if event.get("error") else ""))
        elif kind == "llm_action":
            lines.append(f"- [{event.get('source')}] `{event.get('tool')}` {event.get('args') or ''}"
                         + ("" if event.get("ok") else f" → 被拒绝：{event.get('error')}"))
        elif kind in ("llm_skip", "llm_disabled", "llm_error"):
            lines.append(f"- {'跳过' if kind == 'llm_skip' else '关闭' if kind == 'llm_disabled' else '出错'}"
                         f"（{event.get('reason') or ''}）：{event.get('why') or event.get('error')}")
    for note in state.get("notes") or []:
        lines += ["", f"> **LLM 生成**（{note.get('source', 'llm')}，{note.get('title') or '说明'}）", ">",
                  *[f"> {line}" for line in str(note.get("text", "")).splitlines()]]
    lines.append("")
    path = report_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
