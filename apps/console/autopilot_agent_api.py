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

    def parse_source(x_autopilot_source: str) -> str:
        if x_autopilot_source not in SOURCES:
            raise HTTPException(400, f"未知的 AutoPilot 来源：{x_autopilot_source}")
        return x_autopilot_source

    def authorize(authorization: str | None, source: str) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "AutoPilot 工具令牌无效")
        offered = authorization[len("Bearer "):].strip()
        if not offered:
            raise HTTPException(401, "AutoPilot 工具令牌无效")
        web_token = getattr(host, "token", None)
        headless_token = getattr(host, "headless_token", None)
        if headless_token is not None:
            if source == "llm:web":
                if secrets.compare_digest(offered.encode(), str(headless_token).encode()):
                    raise HTTPException(403, "headless 令牌不能以 llm:web 身份调用")
                if not web_token or not secrets.compare_digest(offered.encode(), str(web_token).encode()):
                    raise HTTPException(401, "AutoPilot 工具令牌无效")
            else:
                if web_token and secrets.compare_digest(offered.encode(), str(web_token).encode()):
                    raise HTTPException(403, "web 令牌不能以 llm:headless 身份调用")
                if not secrets.compare_digest(offered.encode(), str(headless_token).encode()):
                    raise HTTPException(401, "AutoPilot 工具令牌无效")
            return
        if not web_token or not secrets.compare_digest(offered.encode(), str(web_token).encode()):
            raise HTTPException(401, "AutoPilot 工具令牌无效")

    @router.get("/tools")
    def list_tools(authorization: str | None = Header(default=None),
                   x_autopilot_source: str = Header(default="llm:web")):
        source = parse_source(x_autopilot_source)
        authorize(authorization, source)
        return [{"name": n, "category": tooldefs.category_of(n), **tooldefs.TOOLS[n]}
                for n in tooldefs.visible_tools(source)]

    @router.post("/tools/{name}")
    def call_tool(name: str, arguments: dict = Body(default_factory=dict),
                  authorization: str | None = Header(default=None),
                  x_autopilot_source: str = Header(default="llm:web"),
                  x_autopilot_campaign: str | None = Header(default=None)):
        source = parse_source(x_autopilot_source)
        authorize(authorization, source)
        try:
            return tools.call(name, arguments, source=source, campaign_hint=x_autopilot_campaign or None)
        except AgentToolError as exc:
            raise HTTPException(exc.status, str(exc)) from exc

    return router
