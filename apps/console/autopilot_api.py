"""AutoPilot routes of the training console (spec §9).

``/api/autopilot/*`` serves the console pages; ``/api/autopilot/agent/*`` serves the dsh tool server only and
requires the token generated at console start (see apps/console/autopilot_agent_api.py).
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from apps.autopilot.agents import build_specialists
from apps.autopilot.service import AutoPilotService, CampaignError
from apps.autopilot.spec import MODES, SPECIALISTS, STRATEGIES, TASKS, default_spec

from .api import SafeJSONResponse, _sse


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except CampaignError as exc:
        raise HTTPException(exc.status, str(exc)) from exc


def build_autopilot_router(service: AutoPilotService, *, dsh_host=None) -> APIRouter:
    router = APIRouter(prefix="/api/autopilot", default_response_class=SafeJSONResponse)

    @router.get("/defaults")
    def defaults():
        specialists = build_specialists(list(SPECIALISTS), repo_root=service.repo_root)
        availability = {}
        for specialist in specialists:
            ok, why = specialist.available(None)
            availability[specialist.name] = {"label": SPECIALISTS[specialist.name], "available": ok, "reason": why}
        return {"spec": default_spec("dev"), "formal": default_spec("formal"),
                "tasks": [{"task": k, "label": v["label"], "metrics": v["metrics"]} for k, v in TASKS.items()],
                "modes": MODES, "strategies": list(STRATEGIES), "specialists": availability,
                "llm_available": bool(dsh_host is not None and dsh_host.status().get("llm_ready"))}

    @router.get("/campaigns")
    def campaigns():
        return service.list_campaigns()

    @router.post("/campaigns")
    def create(body: dict = Body(...)):
        return _call(service.create_campaign, body, created_by="user", source="console")

    @router.get("/campaigns/{cid}")
    def detail(cid: str):
        return _call(service.get_campaign, cid)

    @router.patch("/campaigns/{cid}")
    def update(cid: str, body: dict = Body(...)):
        return _call(service.update_campaign, cid, body, source="console")

    @router.post("/campaigns/{cid}/start")
    def start(cid: str):
        return _call(service.start_campaign, cid, source="console")

    @router.post("/campaigns/{cid}/resume")
    def resume(cid: str):
        return _call(service.start_campaign, cid, source="console")

    @router.post("/campaigns/{cid}/pause")
    def pause(cid: str):
        return _call(service.pause_campaign, cid, source="console")

    @router.post("/campaigns/{cid}/stop")
    def stop(cid: str):
        return _call(service.stop_campaign, cid, source="console")

    @router.get("/campaigns/{cid}/trials")
    def trials(cid: str):
        return _call(service.trials, cid)

    @router.post("/campaigns/{cid}/trials/{tid}/stop")
    def stop_trial(cid: str, tid: str):
        return _call(service.stop_trial, cid, tid, source="console")

    @router.get("/campaigns/{cid}/findings")
    def findings(cid: str):
        return _call(service.findings, cid)

    @router.get("/campaigns/{cid}/pending")
    def pending(cid: str):
        return _call(service.pending, cid)

    @router.post("/campaigns/{cid}/pending/{pid}/confirm")
    def confirm(cid: str, pid: str):
        return _call(service.resolve_pending, cid, pid, confirm=True, source="console")

    @router.post("/campaigns/{cid}/pending/{pid}/reject")
    def reject(cid: str, pid: str):
        return _call(service.resolve_pending, cid, pid, confirm=False, source="console")

    @router.get("/campaigns/{cid}/events")
    def events(cid: str, since: int = Query(0, ge=0), kinds: str | None = None):
        return _call(service.events, cid, since, tuple(kinds.split(",")) if kinds else None)

    @router.get("/campaigns/{cid}/report")
    def report(cid: str):
        return _call(service.report, cid)

    @router.post("/campaigns/{cid}/report")
    def regenerate_report(cid: str):
        return _call(service.regenerate_report, cid)

    @router.get("/campaigns/{cid}/report/figures/{name}")
    def figure(cid: str, name: str):
        store = _call(service._existing, cid)
        path = (store.report_dir / "figures" / name).resolve()
        if path.parent != (store.report_dir / "figures").resolve() or path.suffix != ".png" or not path.is_file():
            raise HTTPException(404, "图表不存在")
        return FileResponse(path, media_type="image/png")

    @router.get("/campaigns/{cid}/stream")
    async def stream(cid: str, request: Request, since: int = Query(0, ge=0)):
        _call(service._existing, cid)
        state = {"since": since, "status": None}

        async def gen():
            while not await request.is_disconnected():
                rows = await asyncio.to_thread(service.events, cid, state["since"])
                for row in rows:
                    state["since"] = row["seq"]
                    yield _sse("event", row)
                summary = await asyncio.to_thread(service._summary, service.store(cid))
                key = (summary["status"], summary["phase"], summary["trials"], summary["pending"])
                if key != state["status"]:
                    state["status"] = key
                    yield _sse("summary", summary)
                if summary["status"] in ("completed", "stopped", "failed", "draft", "paused"):
                    yield _sse("end", {"status": summary["status"]})
                    break
                await asyncio.sleep(1.0)
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # ---- the embedded dsh web UI --------------------------------------------------------------
    @router.get("/dsh/status")
    def dsh_status(ensure_web: bool = False, campaign: str | None = None):
        if dsh_host is None:
            return {"available": False, "llm_ready": False, "reason": "控制台未启用 dsh 接入"}
        if campaign and service.store(campaign).exists():
            dsh_host.context_campaign = campaign
        return dsh_host.status(ensure_web=ensure_web)

    @router.post("/dsh/context")
    def dsh_context(body: dict = Body(default_factory=dict)):
        """The campaign the assistant page was opened for; tools use it when a call names no campaign."""
        if dsh_host is None:
            raise HTTPException(409, "控制台未启用 dsh 接入")
        cid = body.get("campaign_id") or None
        if cid and not service.store(cid).exists():
            raise HTTPException(404, f"campaign 不存在：{cid}")
        dsh_host.context_campaign = cid
        return {"campaign_id": cid}

    @router.post("/dsh/restart")
    def dsh_restart():
        if dsh_host is None:
            raise HTTPException(409, "控制台未启用 dsh 接入")
        return dsh_host.restart_web()

    return router
