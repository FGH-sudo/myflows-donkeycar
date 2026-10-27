"""Layer-3 acceptance (spec §13): a small campaign with the LLM advisor on the real DeepSeek model.

Run it against a running console once the DeepSeek key is set in the AutoPilot dsh settings page (the
console's AutoPilot assistant page embeds it; the key never goes through this script)::

    python -m apps.autopilot.dsh.acceptance --console http://127.0.0.1:8790

It checks that the AutoPilot dsh is ready (pinned clone, profile composition, key), creates and starts a
small campaign with the advisor on, waits for it to finish and summarises every background call from the
campaign's ``llm_call`` events: tokens (input / output / reasoning), latency and failures, plus the cost when
``--price-in`` / ``--price-out`` (per million tokens, from the provider's current price list) are given. From
the largest call it proposes ``llm.max_total_tokens`` for later campaigns (spec §8.4: the token limit is set
after the first real acceptance). The summary is printed and written next to the campaign as layer3.json.

The other half of layer 3 — chatting in the embedded web page, configuring and starting a campaign from
there — is a manual check; the printed checklist names it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..spec import MAX_LLM_CALLS

DEFAULT_CONSOLE = "http://127.0.0.1:8790"
HEADROOM = 1.5
MANUAL_CHECKS = (
    "在控制台的 AutoPilot 助手页（内嵌 dsh 网页）里对话：让它列出 campaign、读取一个报告",
    "在助手页里让它配置并创建一个小 campaign（工作区写入模式下应先弹出审批）",
    "让它启动训练：控制台“待确认”里出现启动请求，确认后才进入队列",
)


class AcceptanceError(RuntimeError):
    pass


# ---- pure parts (unit tested) ------------------------------------------------------------------------
def readiness_problems(status: dict) -> list[str]:
    """Why the AutoPilot dsh cannot run real calls yet, from ``GET /api/autopilot/dsh/status``."""
    problems = []
    if status.get("checking"):
        problems.append("dsh 运行时仍在检查中，稍后再试")
        return problems
    if not status.get("available"):
        problems.append(f"dsh 运行时不可用：{status.get('reason') or '未知原因'}")
    if not (status.get("composition") or {}).get("ok", True):
        problems.append("profile 组合检查未通过：" + "；".join((status.get("composition") or {}).get("errors") or []))
    if not status.get("key_configured"):
        problems.append("还没有 DeepSeek Key 或账号登录：打开控制台 AutoPilot 助手页，在内嵌 dsh 里添加 API Key 或"
                        "登录 DeepSeek 账号（只保存在 AutoPilot 专用 dsh 的凭据里）")
    return problems


def campaign_spec(campaign_id: str, *, max_samples: int = 200, max_calls: int = 6) -> dict:
    """A small dev campaign whose decision points call the advisor (rules still drive every trial)."""
    return {
        "campaign_id": campaign_id, "title": "第三层验收：真实 DeepSeek 顾问（小规模）", "mode": "dev",
        "baseline": {"lr": 0.002, "batch": 8, "num_workers": 0, "seeds": [1, 2]},
        "fixed": {"max_samples": max_samples, "epochs": 1, "val_size": max(20, max_samples // 10), "base_width": 16},
        "space": {"lr": [0.001, 0.002, 0.004], "batch": [8]},
        "strategy": {"name": "candidate_list", "max_candidates": 2, "seed": 0},
        "budget": {"max_trials": 6, "trial_timeout_s": 900, "wall_time_s": 3600},
        "specialists": ["gpu_monitor", "benchmark"],
        "llm": {"enabled": True, "max_calls": max_calls},
    }


def summarize(events: list[dict], *, max_calls: int = MAX_LLM_CALLS, price_in: float | None = None,
              price_out: float | None = None, headroom: float = HEADROOM) -> dict:
    """Token, latency and cost figures of a campaign's finished ``llm_call`` events, and a proposed token cap."""
    calls = [e for e in events if e.get("kind") == "llm_call" and e.get("status") in ("ok", "failed")]
    rows = []
    for e in calls:
        usage = e.get("usage") or {}
        rows.append({"reason": e.get("reason"), "ok": e.get("status") == "ok", "duration_s": e.get("duration_s"),
                     "input_tokens": int(usage.get("input_tokens") or 0),
                     "output_tokens": int(usage.get("output_tokens") or 0),
                     "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
                     "total_tokens": int(usage.get("total_tokens") or 0),
                     "tools": [c.get("tool") for c in e.get("tool_calls") or []], "error": e.get("error")})
    totals = {k: sum(r[k] for r in rows) for k in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")}
    latency = [float(r["duration_s"]) for r in rows if isinstance(r["duration_s"], (int, float))]
    out = {"calls": len(rows), "ok": sum(r["ok"] for r in rows), "failed": sum(not r["ok"] for r in rows),
           "totals": totals, "per_call": rows,
           "latency_s": {"median": statistics.median(latency), "max": max(latency),
                         "mean": sum(latency) / len(latency)} if latency else None,
           "cost": None, "proposed_max_total_tokens": None}
    if price_in is not None and price_out is not None:
        out["cost"] = {"input": totals["input_tokens"] * price_in / 1e6, "output": totals["output_tokens"] * price_out / 1e6,
                       "price_in_per_million": price_in, "price_out_per_million": price_out,
                       "note": "reasoning token 单列；服务商若另计，按输出单价补上"}
        out["cost"]["total"] = out["cost"]["input"] + out["cost"]["output"]
    largest = max((r["total_tokens"] for r in rows), default=0)
    if largest:
        # The session is reused, so later calls carry more context: budget every allowed call at the largest one.
        out["proposed_max_total_tokens"] = int(math.ceil(largest * max_calls * headroom / 1000.0) * 1000)
    return out


def render(summary: dict, campaign_id: str) -> str:
    lines = [f"第三层验收（{campaign_id}）：{summary['calls']} 次后台调用，成功 {summary['ok']}，失败 {summary['failed']}"]
    t = summary["totals"]
    lines.append(f"- token：输入 {t['input_tokens']}，输出 {t['output_tokens']}，推理 {t['reasoning_tokens']}，"
                 f"合计 {t['total_tokens']}")
    if summary["latency_s"]:
        lat = summary["latency_s"]
        lines.append(f"- 延迟：中位 {lat['median']:.1f} s，平均 {lat['mean']:.1f} s，最长 {lat['max']:.1f} s")
    if summary["cost"]:
        lines.append(f"- 费用：约 {summary['cost']['total']:.4f}（输入 {summary['cost']['input']:.4f} + "
                     f"输出 {summary['cost']['output']:.4f}；{summary['cost']['note']}）")
    else:
        lines.append("- 费用：未给出单价（--price-in / --price-out，按每百万 token）")
    if summary["proposed_max_total_tokens"]:
        lines.append(f"- 建议 llm.max_total_tokens = {summary['proposed_max_total_tokens']}"
                     f"（单次最大调用 × 每个 campaign {MAX_LLM_CALLS} 次 × {HEADROOM}）")
    for r in summary["per_call"]:
        lines.append(f"  - {r['reason']}：{'ok' if r['ok'] else '失败'}，{r['total_tokens']} token，"
                     f"{r['duration_s'] if r['duration_s'] is not None else '—'} s，工具 {', '.join(t for t in r['tools'] if t) or '无'}"
                     + (f"，错误 {r['error']}" if r['error'] else ""))
    lines += ["", "还需手动检查（第三层的网页部分）："] + [f"  {i}. {c}" for i, c in enumerate(MANUAL_CHECKS, start=1)]
    return "\n".join(lines)


# ---- the console client --------------------------------------------------------------------------------
class Console:
    def __init__(self, url: str, timeout_s: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def call(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        request = urllib.request.Request(self.url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json; charset=utf-8"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AcceptanceError(f"{method} {path} → HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise AcceptanceError(f"连不上控制台 {self.url}：{exc.reason}") from exc


def run(console: Console, campaign_id: str, *, max_samples: int, max_calls: int, timeout_s: float, poll_s: float,
        price_in: float | None, price_out: float | None) -> dict:
    problems = readiness_problems(console.call("GET", "/api/autopilot/dsh/status"))
    if problems:
        raise AcceptanceError("AutoPilot dsh 还不能调用真实模型：\n- " + "\n- ".join(problems))
    console.call("POST", "/api/autopilot/campaigns", campaign_spec(campaign_id, max_samples=max_samples,
                                                                    max_calls=max_calls))
    console.call("POST", f"/api/autopilot/campaigns/{campaign_id}/start")
    deadline = time.time() + timeout_s
    while True:
        detail = console.call("GET", f"/api/autopilot/campaigns/{campaign_id}")
        if detail.get("status") in ("completed", "stopped", "failed"):
            break
        if time.time() > deadline:
            raise AcceptanceError(f"{campaign_id} 在 {timeout_s:.0f} s 内没有结束（状态 {detail.get('status')}）")
        time.sleep(poll_s)
    events = console.call("GET", f"/api/autopilot/campaigns/{campaign_id}/events?kinds=llm_call,llm_skip,llm_disabled")
    summary = summarize(events, price_in=price_in, price_out=price_out)
    summary.update(campaign_id=campaign_id, status=detail.get("status"), console=console.url,
                   skipped=[e.get("why") for e in events if e.get("kind") == "llm_skip"],
                   disabled=[e.get("why") for e in events if e.get("kind") == "llm_disabled"])
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AutoPilot 第三层验收：用真实 DeepSeek 跑一个开启 LLM 顾问的小 campaign")
    ap.add_argument("--console", default=DEFAULT_CONSOLE, help="运行中的控制台地址")
    ap.add_argument("--campaign-id", default=f"layer3-{time.strftime('%Y%m%d-%H%M%S')}")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--max-calls", type=int, default=6, help="本次验收的后台调用上限（≤ 30）")
    ap.add_argument("--price-in", type=float, default=None, help="输入单价（每百万 token）")
    ap.add_argument("--price-out", type=float, default=None, help="输出单价（每百万 token）")
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument("--poll-s", type=float, default=10.0)
    ap.add_argument("--out", default=None, help="结果 JSON；默认 runs/autopilot/<campaign>/layer3.json")
    ap.add_argument("--check-only", action="store_true", help="只检查 dsh 是否就绪，不创建 campaign")
    args = ap.parse_args(argv)
    console = Console(args.console)
    try:
        if args.check_only:
            problems = readiness_problems(console.call("GET", "/api/autopilot/dsh/status"))
            print("就绪：可以运行第三层验收" if not problems else "未就绪：\n- " + "\n- ".join(problems))
            return 0 if not problems else 3
        summary = run(console, args.campaign_id, max_samples=args.max_samples, max_calls=min(args.max_calls, MAX_LLM_CALLS),
                      timeout_s=args.timeout_s, poll_s=args.poll_s, price_in=args.price_in, price_out=args.price_out)
    except AcceptanceError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    out = Path(args.out) if args.out else Path("runs") / "autopilot" / args.campaign_id / "layer3.json"
    if out.parent.is_dir():
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render(summary, args.campaign_id))
    print(f"\n结果：{out if out.parent.is_dir() else '（未写文件：' + str(out.parent) + ' 不存在）'}")
    return 0 if summary["calls"] and not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
