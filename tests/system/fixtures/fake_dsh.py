"""Stand-in for ``node apps/cli/lib/bin.js`` in the DshHost tests (spec §13, layer 1).

    fake_dsh.py --profile P --dump-config                  print a composed tree (FAKE_DSH_DUMP=bad adds a shell row)
    fake_dsh.py --profile P --host H --port N --no-open     serve HTTP and print the "dsh web: URL" readiness line
    fake_dsh.py --profile P --json [--session-id S] TASK    a scripted "model" calling the console's agent API

Behaviour switches (environment):
    FAKE_DSH_WEB=crash        the Web process exits with code 3 before it is ready
    FAKE_DSH_HEADLESS=fail    the headless run prints an error event and exits with code 1
    FAKE_DSH_HEADLESS=slow    the headless run sleeps for 60 s (timeouts)
    FAKE_DSH_SCRIPT=<json>    tool calls of the headless run, a list of {"tool", "arguments"}; the default reads
                              the campaign, proposes one lr inside the search space with request_trials and
                              writes a note
    FAKE_DSH_LOG=<path>       append one JSON line per invocation (argv, selected env names) for assertions
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import sys
import time
import urllib.error
import urllib.request

PREFIX = "mcp__autopilot__"


def dump(profile: str) -> str:
    mode = "headless" if profile.endswith("headless") else "web"
    rows = [
        "- id: llm\n  name: '@deepseek-ai/dsh-llm'",
        "- id: tool-bash\n  name: '@deepseek-ai/dsh-tool-bash'\n  disabled: true",
        "- id: tools\n  name: '@deepseek-ai/dsh-tools'\n  config:\n    mode: native",
        "- id: mcp-autopilot\n  name: '@deepseek-ai/dsh-mcp-client'\n  config:\n    serverName: autopilot\n"
        "    command: !!js >-\n      process.env.MYFLOWS_PYTHON",
        f"- id: autopilot-guard\n  name: '@myflows/dsh-autopilot/guard'\n  config:\n    profile: {mode}",
    ]
    if mode == "web":
        rows.append("- id: preset-standard\n  name: '@deepseek-ai/dsh-agent-preset'\n  config:\n    id: standard\n"
                    "    plugins:\n      - id: compaction\n        name: cordis:group\n        config: []")
    if os.environ.get("FAKE_DSH_DUMP") == "bad":
        rows.append("- id: tool-pwsh\n  name: '@deepseek-ai/dsh-tool-pwsh'")
    return "\n".join(rows) + "\n"


def serve_web(port: int) -> int:
    if os.environ.get("FAKE_DSH_WEB") == "crash":
        print("fake dsh web: crashing", flush=True)
        return 3

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = b"<html><body>fake dsh</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"dsh web: http://127.0.0.1:{port}/?token=fake-launch-token", flush=True)
    server.serve_forever()
    return 0


def call(tool: str, arguments: dict) -> tuple[str, object]:
    url = os.environ["MYFLOWS_CONSOLE_URL"].rstrip("/") + f"/api/autopilot/agent/tools/{tool}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['MYFLOWS_AUTOPILOT_TOKEN']}",
               "X-AutoPilot-Source": os.environ.get("MYFLOWS_AUTOPILOT_SOURCE", "llm:headless")}
    if os.environ.get("MYFLOWS_AUTOPILOT_CAMPAIGN"):
        headers["X-AutoPilot-Campaign"] = os.environ["MYFLOWS_AUTOPILOT_CAMPAIGN"]
    request = urllib.request.Request(url, data=json.dumps(arguments).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return "completed", json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return "error", exc.read().decode("utf-8", "replace")


def headless(session_id: str | None) -> int:
    def emit(event: dict) -> None:
        print(json.dumps(event, ensure_ascii=False), flush=True)

    behaviour = os.environ.get("FAKE_DSH_HEADLESS")
    if behaviour == "slow":
        time.sleep(60)
    if behaviour == "fail":
        emit({"type": "error", "message": "fake model failure"})
        return 1
    emit({"type": "session", "sessionId": session_id or f"session-fake-{int(time.time() * 1000)}", "cwd": os.getcwd()})
    script = json.loads(os.environ["FAKE_DSH_SCRIPT"]) if os.environ.get("FAKE_DSH_SCRIPT") else None
    steps = script or [{"tool": "get_search_space", "arguments": {}}]
    index, n = 0, 0
    while index < len(steps):
        step = steps[index]
        index += 1
        n += 1
        emit({"type": "tool_call", "callId": f"c{n}", "tool": PREFIX + step["tool"], "input": step["arguments"]})
        status, result = call(step["tool"], step["arguments"])
        emit({"type": "tool_result", "callId": f"c{n}", "status": status, "result": json.dumps(result, ensure_ascii=False)[:2000]})
        emit({"type": "status", "phase": "step_end", "usage": {"inputTokens": 100, "outputTokens": 20}})
        if script is None and step["tool"] == "get_search_space" and status == "completed":
            space, base = result["space"], result["baseline"]
            lr = min(space["lr"]) if min(space["lr"]) != base["lr"] else max(space["lr"])
            steps += [{"tool": "request_trials", "arguments": {"proposals": [
                          {"delta": {"lr": lr}, "rationale": "fake: 试一个更小的学习率", "expected": "fake"}]}},
                      {"tool": "write_note", "arguments": {"title": "决策点（fake）", "text": "fake advisor note"}}]
    emit({"type": "final", "text": "fake advisor done"})
    return 0


def main() -> int:
    log = os.environ.get("FAKE_DSH_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"argv": sys.argv[1:], "DSH_HOME": os.environ.get("DSH_HOME"),
                                 "has_deepseek": any(k.startswith("DEEPSEEK_") and k != "DEEPSEEK_API_KEY"
                                                     for k in os.environ),
                                 "source": os.environ.get("MYFLOWS_AUTOPILOT_SOURCE")}) + "\n")
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--patch", action="append")
    ap.add_argument("--dump-config", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--session-id")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("task", nargs="?")
    args = ap.parse_args()
    if args.dump_config:
        sys.stdout.write(dump(args.profile))
        return 0
    if args.port is not None:
        return serve_web(args.port)
    return headless(args.session_id)


if __name__ == "__main__":
    sys.exit(main())
