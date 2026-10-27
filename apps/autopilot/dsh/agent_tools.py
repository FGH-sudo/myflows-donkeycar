"""What each AutoPilot tool does for the dsh model (spec §8.3), on top of AutoPilotService.

The console's ``/api/autopilot/agent/*`` router authenticates the stdio tool server and calls
:meth:`AgentTools.call`. Reads return summaries, never raw JSONL; writes use the same service methods (and so
the same pre-checks and budgets) as the pages. Every call, allowed or refused, is recorded: in the campaign's
``events.jsonl`` as ``llm_action`` when it concerns a campaign, otherwise in ``<root>/llm_actions.jsonl``.
"""

from __future__ import annotations

import json
import time

from ..files import append_jsonl
from ..service import CampaignError
from ..spec import (ACCURACY_KEYS, THROUGHPUT_KEYS, TPE_KEYS, TRANSFER_KEYS, baseline_params, default_spec,
                    unavailable_keys)
from . import tooldefs

GLOBAL_TOOLS = ("list_campaigns", "list_runs", "create_campaign")
WRITE_TOOLS = (*tooldefs.CATEGORIES["config"], *tooldefs.CATEGORIES["start"], *tooldefs.CATEGORIES["request"],
               *tooldefs.CATEGORIES["note"])


class AgentToolError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _round(value, digits: int = 4):
    return round(value, digits) if isinstance(value, float) else value


def _brief_args(args: dict) -> str:
    text = json.dumps(args, ensure_ascii=False, default=str)
    return text if len(text) <= 600 else text[:600] + "…"


class AgentTools:
    def __init__(self, service, *, registry=None, context=None):
        self.service = service
        self.registry = registry
        # Campaign the assistant page was opened for (DshHost.context_campaign); used when a call names none.
        self.context = context or (lambda: None)

    # ---- dispatch -----------------------------------------------------------------------------------
    def call(self, name: str, args: dict | None, *, source: str, campaign_hint: str | None = None) -> dict:
        args = dict(args or {})
        started = time.time()
        # The campaign a refused call named still gets the record.
        named = args.get("campaign_id") or campaign_hint or self.context()
        cid = named if (name not in GLOBAL_TOOLS and isinstance(named, str)
                        and self.service.store(named).exists()) else None
        try:
            if name not in tooldefs.TOOLS:
                raise AgentToolError(f"未知工具：{name}", 404)
            if source.endswith("headless") and name not in tooldefs.HEADLESS_TOOLS:
                raise AgentToolError(f"后台一次性运行不能使用 {name}", 403)
            if name == "get_search_space":
                cid = self._campaign_id(args, campaign_hint, required=False)
            elif name not in GLOBAL_TOOLS:
                cid = self._campaign_id(args, campaign_hint)
            result = getattr(self, f"_{name}")(cid, args, source)
            if name == "create_campaign":
                cid = result.get("campaign_id")
        except CampaignError as exc:
            self._record(name, cid, args, source, started, error=str(exc))
            raise AgentToolError(str(exc), exc.status) from exc
        except AgentToolError as exc:
            self._record(name, cid, args, source, started, error=str(exc))
            raise
        self._record(name, cid, args, source, started, error=None)
        return result

    def _campaign_id(self, args: dict, hint: str | None, *, required: bool = True) -> str | None:
        cid = args.get("campaign_id") or hint or self.context()
        if not cid and required:
            raise AgentToolError("需要 campaign_id（当前没有上下文 campaign，可先用 list_campaigns 查看）", 422)
        if cid and not self.service.store(cid).exists():
            raise AgentToolError(f"campaign 不存在：{cid}", 404)
        return cid

    def _record(self, name: str, cid: str | None, args: dict, source: str, started: float, *, error) -> None:
        fields = {"tool": name, "source": source, "ok": error is None, "args": _brief_args(args),
                  "write": name in WRITE_TOOLS, "duration_s": round(time.time() - started, 3)}
        if error is not None:
            fields["error"] = error[:500]
        if cid and self.service.store(cid).exists():
            self.service.store(cid).event("llm_action", **fields)
        else:
            append_jsonl(self.service.root / "llm_actions.jsonl", {"unix_s": time.time(), **fields})

    # ---- read ---------------------------------------------------------------------------------------
    def _list_campaigns(self, _cid, _args, _source) -> dict:
        current = self.context()
        items = []
        for c in self.service.list_campaigns():
            best = c.get("best") or {}
            items.append({"campaign_id": c["campaign_id"], "title": c.get("title"), "task": c["task"],
                          "mode": c["mode"], "status": c["status"], "phase": c["phase"],
                          "objective": c["objective"], "trials": c["trials"], "valid": c["valid"],
                          "best": {"trial_id": best.get("trial_id"), "value": _round(best.get("value"))} if best else None,
                          "budget": c["budget"], "used": c["used"], "pending": c["pending"],
                          "current": c["campaign_id"] == current})
        return {"campaigns": items, "current": current}

    def _get_campaign(self, cid, _args, _source) -> dict:
        c = self.service.get_campaign(cid)
        state = c["state"]
        return {
            "campaign_id": cid, "title": c.get("title"), "status": c["status"], "phase": c["phase"],
            "objective": c["objective"], "budget": c["budget"], "used": c["used"],
            "best": c.get("best"), "trials": c["trials"], "valid": c["valid"],
            "spec": {k: c["spec"].get(k) for k in ("task", "mode", "baseline", "fixed", "space", "strategy",
                                                    "confirm", "specialists", "llm")},
            "phase_log": state.get("phase_log") or [], "noise": state.get("noise"),
            "adopted": state.get("adopted") or {}, "confirm_plan": state.get("confirm_plan"),
            "pending": [{k: p.get(k) for k in ("id", "kind", "status", "source", "precheck")}
                        for p in state.get("pending") or []],
            "notes": [{"title": n.get("title"), "source": n.get("source"), "text": n.get("text", "")[:400]}
                      for n in state.get("notes") or []],
            "llm": {k: state.get("llm", {}).get(k) for k in ("calls", "failures", "disabled_reason")},
            "stop_reason": state.get("stop_reason"), "report_available": c.get("report_available"),
        }

    def _list_trials(self, cid, args, _source) -> dict:
        limit = int(args.get("limit") or 50)
        rows = []
        for t in self.service.trials(cid):
            result = t.get("result") or {}
            summary = result.get("summary") or {}
            origin = t.get("origin") or {}
            rows.append({
                "trial_id": t["trial_id"], "phase": t["phase"], "status": t["status"], "rank": t.get("rank"),
                "objective": _round(t.get("objective")), "params": t["params"],
                "delta": origin.get("delta") or {}, "base_trial": origin.get("base_trial"),
                "origin": f"{origin.get('source', '?')}/{origin.get('strategy', '?')}",
                "verdict": (t.get("verdict") or {}).get("status"), "failure": (t.get("verdict") or {}).get("kind"),
                "throughput_sps": _round(summary.get("throughput_sps"), 2),
                "run_s": _round((result.get("timing") or {}).get("run_s"), 1),
                "skip_reason": t.get("skip_reason"),
            })
        rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0, r["trial_id"]))
        return {"campaign_id": cid, "trials": rows[:limit], "total": len(rows)}

    def _get_trial_summary(self, cid, args, _source) -> dict:
        tid = str(args.get("trial_id") or "")
        trial = next((t for t in self.service.trials(cid) if t["trial_id"] == tid), None)
        if trial is None:
            raise AgentToolError(f"trial 不存在：{tid}", 404)
        result = trial.get("result") or {}
        summary = {k: _round(v) for k, v in (result.get("summary") or {}).items()
                   if k not in ("sources", "lock_acquired_unix_s")}
        attempts = [{"attempt": a.get("attempt"), "status": a.get("status"), "kind": a.get("kind"),
                     "run_dir": a.get("run_dir"), "returncode": a.get("returncode")}
                    for a in trial.get("attempts") or []]
        return {
            "campaign_id": cid, "trial_id": tid, "phase": trial["phase"], "status": trial["status"],
            "params": trial["params"], "origin": trial.get("origin"), "precheck": trial.get("precheck"),
            "verdict": trial.get("verdict"), "objective": _round(trial.get("objective")), "rank": trial.get("rank"),
            "metrics": {k: _round(v) for k, v in (result.get("metrics") or {}).items()},
            "timing": {k: _round(v, 2) for k, v in (result.get("timing") or {}).items() if not k.endswith("unix_s")},
            "samples": result.get("samples"), "backend": result.get("backend"), "memory": result.get("memory"),
            "evidence": summary, "epochs": [{k: _round(v) for k, v in e.items()} for e in result.get("epochs") or []][:20],
            "attempts": attempts,
        }

    def _get_findings(self, cid, _args, _source) -> dict:
        f = self.service.findings(cid)
        return {"campaign_id": cid, "findings": f["findings"][-30:], "chains": f["chains"][-20:],
                "noise": f["noise"], "unavailable": f["unavailable"]}

    def _get_search_space(self, cid, _args, _source) -> dict:
        spec = self.service.get_campaign(cid)["spec"] if cid else default_spec("dev")
        return {
            "campaign_id": cid, "space": spec["space"],
            "baseline": baseline_params(spec, spec["baseline"]["seeds"][0]), "fixed": spec["fixed"],
            "accuracy_keys": list(ACCURACY_KEYS), "throughput_keys": list(THROUGHPUT_KEYS),
            "regularization_keys": list(TPE_KEYS), "transfer_keys": list(TRANSFER_KEYS),
            "unavailable": unavailable_keys(spec, self.service.repo_root),
            "rules": ["数值参数只能在搜索空间的最小值和最大值之间取值",
                      "吞吐类参数（num_workers 等）由专业子 Agent 按吞吐决定，不参与精度排名",
                      "提议的配置还要经过 apps.console.presets.validate_single 与经验库检查"],
        }

    def _list_runs(self, _cid, args, _source) -> dict:
        if self.registry is None:
            return {"runs": [], "note": "控制台未提供运行列表"}
        limit = int(args.get("limit") or 20)
        runs = self.registry.list_runs()
        runs.sort(key=lambda r: r.get("started_unix_s") or 0, reverse=True)
        keep = ("id", "source", "label", "kind", "task", "status", "device", "global_batch", "epochs",
                "learning_rate", "elapsed_s", "final", "error")
        return {"runs": [{k: r.get(k) for k in keep} for r in runs[:limit]], "total": len(runs)}

    # ---- config ---------------------------------------------------------------------------------------
    def _create_campaign(self, _cid, args, source) -> dict:
        spec = args.get("spec")
        if not isinstance(spec, dict):
            raise AgentToolError("spec 必须是对象", 422)
        created = self.service.create_campaign(spec, created_by="llm", source=source)
        return {"campaign_id": created["campaign_id"], "status": created["status"], "spec": created["spec"]}

    def _update_campaign(self, cid, args, source) -> dict:
        patch = args.get("patch")
        if not isinstance(patch, dict):
            raise AgentToolError("patch 必须是对象", 422)
        updated = self.service.update_campaign(cid, patch, source=source)
        return {"campaign_id": cid, "status": updated["status"], "spec": updated["spec"]}

    def _pause_campaign(self, cid, _args, source) -> dict:
        return {"campaign_id": cid, "status": self.service.pause_campaign(cid, source=source)["status"]}

    def _stop_trial(self, cid, args, source) -> dict:
        trial = self.service.stop_trial(cid, str(args.get("trial_id") or ""), source=source)
        return {"campaign_id": cid, "trial_id": trial["trial_id"], "status": trial["status"]}

    # ---- start training ---------------------------------------------------------------------------------
    def _start_campaign(self, cid, _args, source) -> dict:
        return {"campaign_id": cid, "status": self.service.start_campaign(cid, source=source)["status"]}

    def _submit_trials(self, cid, args, source) -> dict:
        return {"campaign_id": cid, "results": self.service.submit_trials(cid, self._proposals(args), source=source)}

    def _request_campaign_start(self, cid, _args, source) -> dict:
        item = self.service.request_campaign_start(cid, source=source)
        return {"campaign_id": cid, "pending_id": item["id"], "status": item["status"],
                "note": "已在控制台生成待确认项，由人确认后才会启动"}

    def _request_trials(self, cid, args, source) -> dict:
        item = self.service.request_trials(cid, self._proposals(args), source=source)
        return {"campaign_id": cid, "pending_id": item["id"], "status": item["status"],
                "precheck": item["precheck"], "proposals": item["proposals"],
                "note": "已在控制台生成待确认项，由人确认后才会运行" + ("" if item["precheck"]["passed"]
                                                               else "；部分候选未通过预检，确认时会被跳过")}

    @staticmethod
    def _proposals(args: dict) -> list[dict]:
        proposals = args.get("proposals")
        if not isinstance(proposals, list) or not proposals:
            raise AgentToolError("proposals 必须是非空数组", 422)
        if len(proposals) > 6:
            raise AgentToolError("一次最多提出 6 个候选", 422)
        out = []
        for p in proposals:
            if not isinstance(p, dict) or not isinstance(p.get("delta"), dict) or not p["delta"]:
                raise AgentToolError("每个候选都需要非空的 delta 对象", 422)
            out.append({"delta": p["delta"], "base_trial": p.get("base_trial"),
                        "rationale": str(p.get("rationale") or "")[:2000], "expected": str(p.get("expected") or "")[:500],
                        "kind": "accuracy"})
        return out

    # ---- note ----------------------------------------------------------------------------------------------
    def _write_note(self, cid, args, source) -> dict:
        text = str(args.get("text") or "").strip()
        if not text:
            raise AgentToolError("text 不能为空", 422)
        note = self.service.add_note(cid, text, title=args.get("title") or "LLM 说明", source=source)
        return {"campaign_id": cid, "saved": True, "chars": len(note["text"])}
