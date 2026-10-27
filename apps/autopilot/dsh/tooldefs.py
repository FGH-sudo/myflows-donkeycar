"""The AutoPilot tool surface the dsh model sees (spec §8.3), shared by the stdio server and the console.

Standard library only and free of package-relative imports: ``tools_server.py`` loads this file by path,
so the MCP server starts without importing the console or MyFlows. The categories must match
``bundle/policy.js`` (a test checks that they do).
"""

from __future__ import annotations

PREFIX = "mcp__autopilot__"
SERVER_NAME = "autopilot"

CATEGORIES = {
    "read": ("list_campaigns", "get_campaign", "list_trials", "get_trial_summary", "get_findings",
             "get_search_space", "list_runs"),
    "config": ("create_campaign", "update_campaign", "pause_campaign", "stop_trial"),
    "start": ("start_campaign", "submit_trials"),
    "request": ("request_campaign_start", "request_trials"),
    "note": ("write_note",),
}
HEADLESS_TOOLS = (*CATEGORIES["read"], "request_trials", "write_note")

_CAMPAIGN = {"type": "string", "description": "campaign id；省略时使用当前上下文中的 campaign（助手页打开的或后台运行所属的）"}
_PROPOSALS = {
    "type": "array", "minItems": 1, "maxItems": 6,
    "description": "候选配置。每项只写相对基准 trial 的改动；参数必须在搜索空间内",
    "items": {
        "type": "object",
        "properties": {
            "delta": {"type": "object", "description": "参数改动，例如 {\"lr\": 0.0005}；键必须是搜索空间中的参数",
                      "additionalProperties": True},
            "base_trial": {"type": "string", "description": "基准 trial id；省略时用当前最好的 trial"},
            "rationale": {"type": "string", "description": "理由：引用具体指标、trial id 和诊断"},
            "expected": {"type": "string", "description": "预期效果，例如“val_angle_mae 降低 5%”"},
        },
        "required": ["delta", "rationale"],
        "additionalProperties": False,
    },
}


def _schema(properties: dict | None = None, required: tuple[str, ...] = ()) -> dict:
    return {"type": "object", "properties": properties or {}, "required": list(required),
            "additionalProperties": False}


TOOLS: dict[str, dict] = {
    "list_campaigns": {
        "description": "列出所有 campaign 的摘要：任务、状态、阶段、预算使用、最好结果、待确认数",
        "inputSchema": _schema(),
    },
    "get_campaign": {
        "description": "读取一个 campaign：目标指标、搜索空间、预算、阶段记录、噪声水平、已采纳的改动、待确认项和说明",
        "inputSchema": _schema({"campaign_id": _CAMPAIGN}),
    },
    "list_trials": {
        "description": "排行榜：每个 trial 的阶段、状态、参数、目标指标、吞吐、失败类型和来源",
        "inputSchema": _schema({"campaign_id": _CAMPAIGN,
                                "limit": {"type": "integer", "minimum": 1, "maximum": 100}}),
    },
    "get_trial_summary": {
        "description": "一个 trial 的指标摘要：结果、计时、吞吐、各阶段耗时、GPU 利用率和显存峰值、校验结论、证据文件路径",
        "inputSchema": _schema({"campaign_id": _CAMPAIGN, "trial_id": {"type": "string"}}, ("trial_id",)),
    },
    "get_findings": {
        "description": "专业子 Agent 的诊断、诊断链（诊断→派生 trial→结论）、噪声水平和不可用的子 Agent",
        "inputSchema": _schema({"campaign_id": _CAMPAIGN}),
    },
    "get_search_space": {
        "description": "搜索空间：可调参数、取值范围、基线参数和不可用参数；省略 campaign_id 时返回默认空间",
        "inputSchema": _schema({"campaign_id": _CAMPAIGN}),
    },
    "list_runs": {
        "description": "控制台里的普通训练运行（非 AutoPilot）：状态、参数和最终指标摘要",
        "inputSchema": _schema({"limit": {"type": "integer", "minimum": 1, "maximum": 50}}),
    },
    "create_campaign": {
        "description": "新建 campaign（草稿，不启动训练）。spec 字段同控制台新建页：campaign_id（可选，小写字母、数字、"
                       "- 和 _，3～64 个字符；省略时自动生成，创建后不能修改）、title、task、mode、objective、fixed、"
                       "baseline、space（搜索空间）、strategy、budget、specialists、llm 等，省略的字段用默认值",
        "inputSchema": _schema({"spec": {"type": "object", "additionalProperties": True}}, ("spec",)),
    },
    "update_campaign": {
        "description": "修改草稿或已暂停 campaign 的配置（任务、搜索空间、预算、seed 等）",
        "inputSchema": _schema({"campaign_id": _CAMPAIGN,
                                "patch": {"type": "object", "additionalProperties": True}}, ("patch",)),
    },
    "pause_campaign": {
        "description": "暂停运行中的 campaign：当前 trial 跑完后不再派发新的 trial",
        "inputSchema": _schema({"campaign_id": _CAMPAIGN}),
    },
    "stop_trial": {
        "description": "停止一个运行中的 trial",
        "inputSchema": _schema({"campaign_id": _CAMPAIGN, "trial_id": {"type": "string"}}, ("trial_id",)),
    },
    "start_campaign": {
        "description": "直接启动或恢复 campaign（会启动训练）。只有工作区写入模式下经审批才可用；其他情况请用 request_campaign_start",
        "inputSchema": _schema({"campaign_id": _CAMPAIGN}),
    },
    "submit_trials": {
        "description": "把候选配置直接放进队列（会启动训练）。只有工作区写入模式下经审批才可用；其他情况请用 request_trials",
        "inputSchema": _schema({"campaign_id": _CAMPAIGN, "proposals": _PROPOSALS}, ("proposals",)),
    },
    "request_campaign_start": {
        "description": "请求启动 campaign：在控制台生成待确认项，由人确认后才启动",
        "inputSchema": _schema({"campaign_id": _CAMPAIGN}),
    },
    "request_trials": {
        "description": "提出带理由的候选 trial：先做与规则策略相同的预检，再在控制台生成待确认项，由人确认后才运行",
        "inputSchema": _schema({"campaign_id": _CAMPAIGN, "proposals": _PROPOSALS}, ("proposals",)),
    },
    "write_note": {
        "description": "给 campaign 报告补充一段文字说明（报告中标注为“LLM 生成”）；用同一标题再写一次会替换之前那段",
        "inputSchema": _schema({"campaign_id": _CAMPAIGN, "title": {"type": "string", "maxLength": 80},
                                "text": {"type": "string", "maxLength": 8000}}, ("text",)),
    },
}


def category_of(name: str) -> str | None:
    for category, names in CATEGORIES.items():
        if name in names:
            return category
    return None


def visible_tools(source: str) -> list[str]:
    """Raw tool names offered to a dsh process: the headless profile only gets its subset."""
    headless = source.endswith("headless")
    return [name for name in TOOLS if not headless or name in HEADLESS_TOOLS]


assert set(TOOLS) == {n for names in CATEGORIES.values() for n in names}
