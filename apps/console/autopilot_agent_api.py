"""``/api/autopilot/agent/*``: the AutoPilot dsh tool server's view of AutoPilot (spec §8.3, §8.7).

Only the stdio tool server calls these routes. Every request carries the token DshHost generated when the
console started (``Authorization: Bearer …``); the tool server learns it from the environment of the dsh
process, never from the repository or a log. The source header (``llm:web`` / ``llm:headless``) labels the
events and limits a headless run to its tool subset.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Body, Header, HTTPException

from apps.autopilot.dsh import tooldefs
from apps.autopilot.dsh.agent_tools import AgentToolError, AgentTools

from .api import SafeJSONResponse

SOURCES = ("llm:web", "llm:headless")


def build_agent_router(service, host, *, registry=None) -> APIRouter:
    router = APIRouter(prefix="/api/autopilot/agent", default_response_class=SafeJSONResponse)
    tools = AgentTools(service, registry=registry, context=lambda: getattr(host, "context_campaign", None))

    def authorize(authorization: str | None) -> None:
        token = getattr(host, "token", None)
        offered = (authorization or "").removeprefix("Bearer ").strip()
        if not token or not offered or not secrets.compare_digest(offered.encode(), str(token).encode()):
            raise HTTPException(401, "AutoPilot 工具令牌无效")

    @router.get("/tools")
    def list_tools(authorization: str | None = Header(default=None),
                   x_autopilot_source: str = Header(default="llm:web")):
        authorize(authorization)
        source = x_autopilot_source if x_autopilot_source in SOURCES else "llm:web"
        return [{"name": n, "category": tooldefs.category_of(n), **tooldefs.TOOLS[n]}
                for n in tooldefs.visible_tools(source)]

    @router.post("/tools/{name}")
    def call_tool(name: str, arguments: dict = Body(default_factory=dict),
                  authorization: str | None = Header(default=None),
                  x_autopilot_source: str = Header(default="llm:web"),
                  x_autopilot_campaign: str | None = Header(default=None)):
        authorize(authorization)
        source = x_autopilot_source if x_autopilot_source in SOURCES else "llm:web"
        try:
            return tools.call(name, arguments, source=source, campaign_hint=x_autopilot_campaign or None)
        except AgentToolError as exc:
            raise HTTPException(exc.status, str(exc)) from exc

    return router
