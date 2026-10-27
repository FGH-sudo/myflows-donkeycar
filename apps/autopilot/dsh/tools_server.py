"""AutoPilot MCP tool server for the AutoPilot-only dsh (spec §8.3), standard library only.

dsh's MCP client starts it over stdio (the ``mcp-autopilot`` row of ``bundle/cordis.patch.yml``). It speaks
newline-delimited JSON-RPC 2.0 (``initialize``, ``tools/list``, ``tools/call``, ``ping``) and forwards every
call to the console's ``/api/autopilot/agent/*`` routes with the token the console generated at start, so the
model sees the same data as the pages and its writes go through the same pre-checks and budgets.

Environment (set by DshHost through the dsh profile):
    MYFLOWS_CONSOLE_URL          console base URL, e.g. http://127.0.0.1:8791
    MYFLOWS_AUTOPILOT_TOKEN      required; the server refuses to start without it
    MYFLOWS_AUTOPILOT_SOURCE     ``llm:web`` or ``llm:headless`` (the headless profile only lists its subset)
    MYFLOWS_AUTOPILOT_CAMPAIGN   default campaign of a headless run (optional)

stdout belongs to the protocol: diagnostics go to stderr, and JSON is written ASCII-escaped so the Windows
console code page cannot corrupt it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "myflows-autopilot", "version": "0.1.0"}
CALL_TIMEOUT_S = 60.0


def _load_tooldefs():
    spec = importlib.util.spec_from_file_location("myflows_autopilot_tooldefs", Path(__file__).with_name("tooldefs.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tooldefs = _load_tooldefs()


def log(message: str) -> None:
    print(f"[autopilot-tools] {message}", file=sys.stderr, flush=True)


class ToolServer:
    def __init__(self, *, console_url: str, token: str, source: str = "llm:web", campaign: str = "",
                 timeout_s: float = CALL_TIMEOUT_S):
        self.console_url = console_url.rstrip("/")
        self.token = token
        self.source = source
        self.campaign = campaign
        self.timeout_s = timeout_s
        self.names = tooldefs.visible_tools(source)

    # ---- JSON-RPC ----------------------------------------------------------------------------------
    def handle(self, message: dict) -> dict | None:
        """One request → one response; notifications (no id) get none."""
        method = message.get("method")
        mid = message.get("id")
        params = message.get("params") or {}
        if mid is None:
            return None
        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": [{"name": n, "description": tooldefs.TOOLS[n]["description"],
                                     "inputSchema": tooldefs.TOOLS[n]["inputSchema"]} for n in self.names]}
            elif method == "tools/call":
                result = self._call(params.get("name"), params.get("arguments") or {})
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
        except Exception as exc:  # protocol errors must not kill the server
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    @staticmethod
    def _initialize(params: dict) -> dict:
        asked = params.get("protocolVersion")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": "MyFlows AutoPilot 工具：读取 campaign 与 trial，提出带理由的建议；启动训练需要人确认。"}

    def _call(self, name, arguments: dict) -> dict:
        if name not in self.names:
            return self._error(f"工具不可用：{name}")
        if not isinstance(arguments, dict):
            return self._error("arguments 必须是对象")
        status, body = self.post(f"/api/autopilot/agent/tools/{name}", arguments)
        if status != 200:
            detail = body.get("detail") if isinstance(body, dict) else body
            return self._error(f"控制台拒绝了 {name}（HTTP {status}）：{detail}")
        text = json.dumps(body, ensure_ascii=False, indent=1, default=str)
        return {"content": [{"type": "text", "text": text}], "isError": False}

    @staticmethod
    def _error(text: str) -> dict:
        return {"content": [{"type": "text", "text": text}], "isError": True}

    # ---- console ------------------------------------------------------------------------------------
    def post(self, path: str, payload: dict) -> tuple[int, object]:
        data = json.dumps(payload, ensure_ascii=True).encode("ascii")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.token}",
                   "X-AutoPilot-Source": self.source}
        if self.campaign:
            headers["X-AutoPilot-Campaign"] = self.campaign
        request = urllib.request.Request(self.console_url + path, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return response.status, json.loads(response.read().decode("utf-8") or "null")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try:
                return exc.code, json.loads(raw)
            except ValueError:
                return exc.code, raw[:500]
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return 503, {"detail": f"无法连接控制台 {self.console_url}：{exc}"}


def serve(server: ToolServer, stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            if isinstance(message, list):
                replies = [r for r in (server.handle(m) for m in message if isinstance(m, dict)) if r is not None]
                reply = replies or None
            elif isinstance(message, dict):
                reply = server.handle(message)
            else:
                reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}}
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=True) + "\n")
            stdout.flush()


def main() -> int:
    token = os.environ.get("MYFLOWS_AUTOPILOT_TOKEN", "").strip()
    url = os.environ.get("MYFLOWS_CONSOLE_URL", "").strip()
    if not token:
        log("MYFLOWS_AUTOPILOT_TOKEN 未设置，拒绝启动（令牌由控制台启动 dsh 时生成）")
        return 2
    if not url:
        log("MYFLOWS_CONSOLE_URL 未设置，拒绝启动")
        return 2
    source = os.environ.get("MYFLOWS_AUTOPILOT_SOURCE", "llm:web").strip() or "llm:web"
    server = ToolServer(console_url=url, token=token, source=source,
                        campaign=os.environ.get("MYFLOWS_AUTOPILOT_CAMPAIGN", "").strip())
    log(f"serving {len(server.names)} tools for {source} via {url}")
    serve(server)
    return 0


if __name__ == "__main__":
    sys.exit(main())
