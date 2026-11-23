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
from apps.common.datasets import list_datasets

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
                "datasets": list_datasets(service.repo_root),
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
        store = _call(service._existing, cid)
        initial_since = since
        raw_last = request.headers.get("last-event-id")
        if raw_last is not None:
            try:
                parsed = int(raw_last.strip())
                if parsed >= 0:
                    initial_since = max(since, parsed)
            except ValueError:
                pass
        state = {"since": initial_since, "status": None}

        async def gen():
            loop = asyncio.get_running_loop()
            wakeup = asyncio.Event()

            def on_event(_row: dict) -> None:
                try:
                    loop.call_soon_threadsafe(wakeup.set)
                except RuntimeError:
                    pass

            store.listeners.append(on_event)
            try:
                while not await request.is_disconnected():
                    wakeup.clear()
                    rows = await asyncio.to_thread(service.events, cid, state["since"])
                    for row in rows:
                        state["since"] = row["seq"]
                        yield _sse("event", row, str(row["seq"]))
                    summary = await asyncio.to_thread(service._summary, store)
                    key = (summary["status"], summary["phase"], summary["trials"], summary["pending"])
                    if key != state["status"]:
                        state["status"] = key
                        yield _sse("summary", summary)
                    if summary["status"] in ("completed", "stopped", "failed", "draft", "paused"):
                        yield _sse("end", {"status": summary["status"]})
                        break
                    if not wakeup.is_set():
                        try:
                            await asyncio.wait_for(wakeup.wait(), timeout=5.0)
                        except asyncio.TimeoutError:
                            pass
            finally:
                try:
                    store.listeners.remove(on_event)
                except ValueError:
                    pass

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # ---- the embedded dsh web UI --------------------------------------------------------------
    @router.get("/dsh/status")
    def dsh_status(campaign: str | None = None):
        if dsh_host is None:
            return {"available": False, "llm_ready": False, "reason": "控制台未启用 dsh 接入"}
        if campaign:
            _call(service.store, campaign)
        return dsh_host.status(ensure_web=False)

    @router.post("/dsh/start")
    def dsh_start(body: dict = Body(default_factory=dict)):
        if dsh_host is None:
            raise HTTPException(409, "控制台未启用 dsh 接入")
        cid = (body or {}).get("campaign_id") or None
        if cid:
            if not _call(service.store, cid).exists():
                raise HTTPException(404, f"campaign 不存在：{cid}")
            dsh_host.context_campaign = cid
        return dsh_host.status(ensure_web=True)

    @router.post("/dsh/context")
    def dsh_context(body: dict = Body(default_factory=dict)):
        """The campaign the assistant page was opened for; tools use it when a call names no campaign."""
        if dsh_host is None:
            raise HTTPException(409, "控制台未启用 dsh 接入")
        cid = body.get("campaign_id") or None
        if cid and not _call(service.store, cid).exists():
            raise HTTPException(404, f"campaign 不存在：{cid}")
        dsh_host.context_campaign = cid
        return {"campaign_id": cid}

    @router.post("/dsh/restart")
    def dsh_restart():
        if dsh_host is None:
            raise HTTPException(409, "控制台未启用 dsh 接入")
        return dsh_host.restart_web()

    return router
