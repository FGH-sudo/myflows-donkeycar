"""Task prompts of the headless advisor runs (spec §6.1 LLMAdvisor, §8.5).

The persona and the tool rules live in the dsh profile (bundle/cordis.patch.yml); these prompts only name
the campaign, the decision point and what to produce. Campaign data is not pasted in: the model reads it
through the read-only tools, so the prompt stays small and the data the model sees is the console's.
"""

from __future__ import annotations


def describe(reason: str) -> str:
    if reason.startswith("phase:"):
        before, _, after = reason[len("phase:"):].partition("->")
        if before == "baseline":
            return "基线（baseline）的各个 seed 已经跑完，规则策略即将开始探索"
        if before == "explore":
            return "一批探索（explore）trial 已经结束，规则策略即将进入细化（exploit）"
        if after == "confirm":
            return "细化阶段结束，即将用新 seed 做确认重跑"
        return f"campaign 从 {before} 阶段进入 {after} 阶段"
    if reason.startswith("failure:"):
        _, kind, trial = (reason.split(":") + ["", ""])[:3]
        names = {"oom": "显存不足（OOM）", "nan": "出现 NaN/Inf", "diverged": "损失发散"}
        return f"trial {trial} 失败：{names.get(kind, kind)}"
    return reason


def decision_prompt(campaign_id: str, reason: str) -> str:
    return (
        f"campaign：{campaign_id}\n"
        f"决策点：{describe(reason)}。\n\n"
        "请按顺序做：\n"
        "1. 用 get_campaign、list_trials、get_findings（需要时 get_trial_summary、get_search_space）了解目标、预算、"
        "排行榜、噪声水平和诊断。\n"
        "2. 如果有规则策略还没覆盖、且有证据支持的配置，用 request_trials 提出最多 3 个候选：只写相对基准 trial 的改动，"
        "参数必须在搜索空间内，每个都写理由（引用具体指标和 trial id）和预期。没有把握就不要提。"
        "失败的决策点优先考虑避免同类失败（例如 OOM 时减小 batch）。\n"
        "3. 用 write_note 写一段不超过 300 字的说明，标题以“决策点”开头：现状、你的判断、提出或不提出候选的原因。\n"
        "不要调用其他工具，也不要重复提出已经跑过的配置。"
    )


def report_prompt(campaign_id: str) -> str:
    return (
        f"campaign：{campaign_id}\n"
        "campaign 已经结束，报告即将生成。请用 get_campaign、list_trials、get_findings 读取结果，"
        "然后用 write_note 写一段不超过 500 字的解读，标题为“LLM 解读”，内容包括：\n"
        "- 最好的配置与人工基线相比，差异是否超过基线 seed 间的标准差（不超过就明确说“未超越基线”）；\n"
        "- 诊断链的结论（例如数据读取瓶颈、吞吐变化）；\n"
        "- 下一步值得做的实验。\n"
        "不要提出新的 trial。"
    )


def build(campaign_id: str, reason: str) -> str:
    return report_prompt(campaign_id) if reason == "report" else decision_prompt(campaign_id, reason)
