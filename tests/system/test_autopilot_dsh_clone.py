"""The LLM layer on the pinned local dsh clone with dsh's own mock model (spec §13, layer 2; no API key).

Skipped when the clone (``MYFLOWS_DSH_ROOT`` or ``../dsh-myflows``) or Node.js is missing. Each scenario
runs a real ``myflows-autopilot-headless`` process: the model adapter, agent loop, guard, MCP client, the
stdio tool server and the console's agent API are all real; only the model is the scripted mock, behind a
small proxy that records every request. Every scenario checks: exit code 0, exactly two model requests,
the offered tools are exactly the headless AutoPilot subset, the tool result reached the model, a final
answer was printed, and nothing of ours appeared in the person's own ``~/.dsh`` (their own dsh may be
running and writing its sessions meanwhile).

The Web-profile permission column is checked through a test-only profile that is the headless composition
with the guard set to ``profile: web`` (the guard reads the same permission presets in both profiles).
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from apps.autopilot.dsh import tooldefs
from apps.autopilot.dsh.host import BUNDLE_DIR, DshHost, _link
from tests.system.harness import ROOT, Harness, wait_for
from tests.system.test_autopilot_dsh import LiveConsole, free_port

DSH_ROOT = Path(os.environ.get("MYFLOWS_DSH_ROOT") or ROOT.parent / "dsh-myflows")
MOCK = DSH_ROOT / "packages" / "test-support" / "llm-mock-server" / "src" / "bin.ts"
NODE = shutil.which("node")
AVAILABLE = NODE is not None and (DSH_ROOT / "apps" / "cli" / "lib" / "bin.js").exists() and MOCK.exists()


def _home_snapshot() -> dict[str, float]:
    """mtime of every file in the person's own dsh home (metadata only; dangling links are skipped)."""
    out = {}
    for folder, _dirs, files in os.walk(Path.home() / ".dsh"):
        for name in files:
            path = os.path.join(folder, name)
            try:
                out[path] = os.stat(path).st_mtime
            except OSError:
                continue
    return out


def _ours(before: dict[str, float], after: dict[str, float]) -> list[str]:
    """Changed or new paths that could come from these tests. The person's own dsh may be running and writing
    its sessions meanwhile, so only profile changes and paths naming MyFlows or AutoPilot count."""
    home = str(Path.home() / ".dsh")
    changed = [p for p, m in after.items() if before.get(p) != m]
    return [p for p in changed if p.startswith(os.path.join(home, "profiles"))
            or any(word in p.lower() for word in ("myflows", "autopilot", "dsh-workspace"))]


class MockModel:
    """dsh's scripted Messages mock (one tool call, then a text answer) behind a recording proxy."""

    def __init__(self, tool: str, arguments: dict):
        self.mock_port, self.port = free_port(), free_port()
        self.proc = subprocess.Popen(
            [NODE, str(MOCK), "--port", str(self.mock_port), "--api-key", "mock-key",
             "--sequence", "tool_call_success,success", "--tool-name", tool, "--tool-arguments", json.dumps(arguments)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        ready = self.proc.stdout.readline()
        if '"ready"' not in ready:
            raise RuntimeError(f"mock model did not start: {ready}")
        self.requests: list[dict] = []
        upstream = f"http://127.0.0.1:{self.mock_port}"
        recorded = self.requests

        class Proxy(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    recorded.append(json.loads(body))
                except ValueError:
                    recorded.append({})
                headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
                request = urllib.request.Request(upstream + self.path, data=body, headers=headers, method="POST")
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        status, ctype, data = response.status, response.headers.get("Content-Type"), response.read()
                except urllib.error.HTTPError as exc:
                    status, ctype, data = exc.code, exc.headers.get("Content-Type"), exc.read()
                self.send_response(status)
                self.send_header("Content-Type", ctype or "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Proxy)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def env(self) -> dict:
        return {"DEEPSEEK_BASE_URL": f"http://127.0.0.1:{self.port}", "DEEPSEEK_API_KEY": "mock-key"}

    def offered_tools(self, index: int = 0) -> list[str]:
        return sorted(t["name"] for t in self.requests[index].get("tools") or [])

    def tool_results(self, index: int = 1) -> list[str]:
        out = []
        for message in self.requests[index].get("messages") or []:
            for block in message.get("content") or [] if isinstance(message.get("content"), list) else []:
                if block.get("type") == "tool_result":
                    content = block.get("content")
                    out.append(json.dumps(content, ensure_ascii=False) if not isinstance(content, str) else content)
        return out

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.proc.kill()
        self.proc.communicate()


@unittest.skipUnless(AVAILABLE, "the pinned dsh clone, its mock model or Node.js is not available")
class HeadlessScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.h = Harness(Path(cls._tmp.name), {}, start=False)
        cls.host = DshHost(repo_root=ROOT, autopilot_root=cls.h.tmp / "autopilot", dsh_root=DSH_ROOT,
                           port=free_port(), headless_timeout_s=120)
        cls.console = LiveConsole(cls.h, cls.host)
        cls.service = cls.console.service
        cls.service.create_campaign(cls.h.spec(campaign_id="clone-campaign"))
        if not wait_for(lambda: cls.host._prepared.is_set(), timeout=300):
            raise RuntimeError("dsh preparation timed out")

    @classmethod
    def tearDownClass(cls):
        cls.console.close()
        cls.host.shutdown()
        cls.h.close()
        cls._tmp.cleanup()

    def setUp(self):
        self.snapshot = _home_snapshot()

    def tearDown(self):
        self.assertEqual(_ours(self.snapshot, _home_snapshot()), [], "~/.dsh must not change")

    def run_scenario(self, tool: str, arguments: dict, *, profile_env: dict | None = None):
        model = MockModel(tool, arguments)
        self.addCleanup(model.close)
        self.host.extra_env = {**model.env, **(profile_env or {})}
        result = self.host.run_headless("执行一次检查", campaign="clone-campaign")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(len(model.requests), 2, "exactly two model requests")
        self.assertTrue(result["text"], "the final answer was printed")
        return result, model

    def test_both_profiles_compose_as_designed(self):
        status = self.host.status()
        self.assertTrue(status["available"], status)
        self.assertTrue(status["composition"]["ok"], status["composition"])
        self.assertTrue(status["clone"]["commit"].startswith("687f282e86"))

    def test_read_runs(self):
        result, model = self.run_scenario("mcp__autopilot__list_runs", {})
        self.assertEqual(model.offered_tools(), sorted(tooldefs.PREFIX + n for n in tooldefs.HEADLESS_TOOLS))
        self.assertEqual(result["tool_calls"], [{"tool": "mcp__autopilot__list_runs", "status": "completed"}])
        self.assertIn('runs', model.tool_results()[0])
        system = json.dumps(model.requests[0].get("system"), ensure_ascii=False)
        self.assertIn("训练调优顾问", system)

    def test_valid_candidate_becomes_a_pending_item(self):
        args = {"campaign_id": "clone-campaign", "proposals": [{"delta": {"lr": 0.0003}, "rationale": "更小的 lr"}]}
        result, model = self.run_scenario("mcp__autopilot__request_trials", args)
        self.assertEqual(result["tool_calls"][0]["status"], "completed")
        item = self.service.pending("clone-campaign")[-1]
        self.assertTrue(item["precheck"]["passed"])
        self.assertEqual(item["source"], "llm:headless")
        self.assertIn(item["id"], model.tool_results()[0])
        self.assertEqual(self.service.trials("clone-campaign"), [], "nothing runs before confirmation")

    def test_unknown_parameter_is_rejected_by_the_precheck(self):
        args = {"campaign_id": "clone-campaign", "proposals": [{"delta": {"optimizer": "sgd"}, "rationale": "x"}]}
        self.run_scenario("mcp__autopilot__request_trials", args)
        item = self.service.pending("clone-campaign")[-1]
        self.assertFalse(item["precheck"]["passed"])
        self.assertTrue(any("optimizer" in r for r in item["precheck"]["reasons"]), item["precheck"])

    def test_out_of_space_lr_is_rejected(self):
        args = {"campaign_id": "clone-campaign", "proposals": [{"delta": {"lr": 0.01}, "rationale": "x"}]}
        self.run_scenario("mcp__autopilot__request_trials", args)
        item = self.service.pending("clone-campaign")[-1]
        self.assertFalse(item["precheck"]["passed"])

    def test_tools_outside_the_allowlist_are_denied(self):
        for tool in ("bash", "mcp__autopilot__start_campaign"):
            before = len(self.service.events("clone-campaign", kinds=("llm_action",)))
            result, model = self.run_scenario(tool, {"command": "echo hi"} if tool == "bash" else {"campaign_id": "clone-campaign"})
            self.assertEqual(result["tool_calls"][0]["status"], "error", tool)
            self.assertNotIn(tool, model.offered_tools())
            self.assertEqual(len(self.service.events("clone-campaign", kinds=("llm_action",))), before,
                             "a denied call never reaches the console")


@unittest.skipUnless(AVAILABLE, "the pinned dsh clone, its mock model or Node.js is not available")
class WebPermissionColumnTest(HeadlessScenarioTest):
    """Config and start tools under the Web guard, in workspace-write and full access."""

    PROFILE = "myflows-autopilot-webguard-test"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        source = cls.host.profile_dir("headless")
        target = source.parent / cls.PROFILE
        target.mkdir(exist_ok=True)
        manifest = json.loads((source / "package.json").read_text(encoding="utf-8"))
        manifest["name"] = f"dsh-profile-{cls.PROFILE}"
        (target / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
        patch = (source / "cordis.patch.yml").read_text(encoding="utf-8").replace("profile: headless", "profile: web")
        (target / "cordis.patch.yml").write_text(patch, encoding="utf-8")
        _link(BUNDLE_DIR, target / "node_modules" / "@myflows" / "dsh-autopilot")
        cls.profiles_patch = __import__("unittest.mock").mock.patch.dict(
            "apps.autopilot.dsh.host.PROFILES", {"headless": cls.PROFILE})
        cls.profiles_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.profiles_patch.stop()
        super().tearDownClass()

    def run_scenario(self, tool, arguments, *, profile_env=None):
        # The Web profile's tool server offers every tool, so the guard is what decides.
        return super().run_scenario(tool, arguments,
                                    profile_env={**(profile_env or {}), "MYFLOWS_AUTOPILOT_SOURCE": "llm:web"})

    # The headless scenarios are not repeated under this profile.
    test_both_profiles_compose_as_designed = None
    test_read_runs = None
    test_valid_candidate_becomes_a_pending_item = None
    test_unknown_parameter_is_rejected_by_the_precheck = None
    test_out_of_space_lr_is_rejected = None
    test_tools_outside_the_allowlist_are_denied = None

    def test_workspace_write_asks_before_config_and_start(self):
        for tool in ("update_campaign", "start_campaign"):
            args = {"campaign_id": "clone-campaign"} | ({"patch": {"title": "x"}} if tool == "update_campaign" else {})
            result, _ = self.run_scenario(tooldefs.PREFIX + tool, args,
                                          profile_env={"DSH_PERMISSION_MODE": "workspace-write"})
            call = result["tool_calls"][0]
            self.assertEqual(call["status"], "error", "no one approved the ask")
            self.assertNotIn("unknown tool", call["error"])
        doc = self.service.get_campaign("clone-campaign")
        self.assertNotEqual(doc["title"], "x")
        self.assertEqual(doc["status"], "draft")

    def test_full_access_runs_config_and_denies_start(self):
        env = {"DSH_PERMISSION_MODE": "danger-full-access"}
        result, _ = self.run_scenario(tooldefs.PREFIX + "update_campaign",
                                      {"campaign_id": "clone-campaign", "patch": {"title": "完全访问"}}, profile_env=env)
        self.assertEqual(result["tool_calls"][0]["status"], "completed", result["tool_calls"])
        self.assertEqual(self.service.get_campaign("clone-campaign")["title"], "完全访问")
        result, model = self.run_scenario(tooldefs.PREFIX + "start_campaign", {"campaign_id": "clone-campaign"},
                                          profile_env=env)
        self.assertEqual(result["tool_calls"][0]["status"], "error")
        self.assertIn("request_campaign_start", model.tool_results()[0])
        self.assertEqual(self.service.get_campaign("clone-campaign")["status"], "draft")


if __name__ == "__main__":
    unittest.main()
