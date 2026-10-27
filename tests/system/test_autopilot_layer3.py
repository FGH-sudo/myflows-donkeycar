"""Layer-3 acceptance runner (apps.autopilot.dsh.acceptance): readiness, the token/latency/cost summary and
the console flow, against a stub console (the real run needs the user's DeepSeek key)."""

from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from apps.autopilot.dsh import acceptance
from apps.autopilot.spec import validate_spec

READY = {"available": True, "key_configured": True, "composition": {"ok": True, "errors": []}}


def call(reason, total, seconds, *, status="ok", inp=None, out=None):
    inp = total - 100 if inp is None else inp
    out = 100 if out is None else out
    return {"kind": "llm_call", "status": status, "reason": reason, "duration_s": seconds,
            "usage": {"input_tokens": inp, "output_tokens": out, "reasoning_tokens": 40, "total_tokens": total},
            "tool_calls": [{"tool": "mcp__autopilot__autopilot_get_campaign"}], "error": None}


class SummaryTest(unittest.TestCase):
    def test_readiness_names_what_is_missing(self):
        self.assertEqual(acceptance.readiness_problems(READY), [])
        problems = acceptance.readiness_problems({**READY, "key_configured": False})
        self.assertEqual(len(problems), 1)
        self.assertIn("DeepSeek Key", problems[0])
        self.assertIn("检查中", acceptance.readiness_problems({"checking": True})[0])

    def test_tokens_latency_cost_and_proposed_cap(self):
        events = [{"kind": "llm_call", "status": "started"}, call("phase:baseline->explore", 4000, 12.0),
                  call("phase:explore->exploit", 6000, 20.0), call("report", 1000, 5.0, status="failed")]
        summary = acceptance.summarize(events, price_in=2.0, price_out=8.0)
        self.assertEqual((summary["calls"], summary["ok"], summary["failed"]), (3, 2, 1))
        self.assertEqual(summary["totals"]["total_tokens"], 11000)
        self.assertEqual(summary["latency_s"]["median"], 12.0)
        self.assertAlmostEqual(summary["cost"]["total"], (10700 * 2.0 + 300 * 8.0) / 1e6)
        self.assertEqual(summary["proposed_max_total_tokens"], 270000)  # 6000 × 30 × 1.5
        text = acceptance.render(summary, "layer3-x")
        self.assertIn("建议 llm.max_total_tokens = 270000", text)
        self.assertIn("手动检查", text)

    def test_the_campaign_spec_is_valid(self):
        spec = validate_spec(acceptance.campaign_spec("layer3-unit"), Path("."))
        self.assertTrue(spec["llm"]["enabled"])
        self.assertEqual(spec["llm"]["max_calls"], 6)


class StubConsole(BaseHTTPRequestHandler):
    status = READY
    events: list = []
    requests: list = []

    def log_message(self, *args):
        pass

    def reply(self, payload, code=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.requests.append(("GET", self.path))
        if self.path == "/api/autopilot/dsh/status":
            return self.reply(self.status)
        if self.path.startswith("/api/autopilot/campaigns/layer3-stub/events"):
            return self.reply(self.events)
        if self.path == "/api/autopilot/campaigns/layer3-stub":
            return self.reply({"campaign_id": "layer3-stub", "status": "completed"})
        self.reply({"detail": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"null")
        self.requests.append(("POST", self.path, body))
        self.reply({"campaign_id": "layer3-stub", "status": "running"})


class ConsoleFlowTest(unittest.TestCase):
    def setUp(self):
        StubConsole.requests = []
        StubConsole.events = [call("phase:baseline->explore", 3000, 9.0)]
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), StubConsole)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = acceptance.main(["--console", self.url, *args])
        return code, out.getvalue(), err.getvalue()

    def test_without_a_key_nothing_is_created(self):
        StubConsole.status = {**READY, "key_configured": False}
        try:
            code, _, err = self.main("--campaign-id", "layer3-stub")
        finally:
            StubConsole.status = READY
        self.assertEqual(code, 3)
        self.assertIn("DeepSeek Key", err)
        self.assertFalse([r for r in StubConsole.requests if r[0] == "POST"])

    def test_creates_starts_waits_and_writes_the_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "layer3.json"
            code, text, _ = self.main("--campaign-id", "layer3-stub", "--poll-s", "0", "--out", str(out))
            self.assertEqual(code, 0, text)
            posts = [r for r in StubConsole.requests if r[0] == "POST"]
            self.assertEqual([p[1] for p in posts], ["/api/autopilot/campaigns", "/api/autopilot/campaigns/layer3-stub/start"])
            self.assertTrue(posts[0][2]["llm"]["enabled"])
            summary = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual((summary["calls"], summary["status"]), (1, "completed"))
            self.assertIn("建议 llm.max_total_tokens", text)


if __name__ == "__main__":
    unittest.main()
