"""The LLM layer without a model or the dsh clone (spec §13, layer 1).

- the guard's permission table (``node --test``) and its categories against the tool definitions;
- the stdio tool server: protocol, missing token, read-only tools leave campaign state alone;
- the agent API: token, headless subset, pending items, notes, the event record;
- the composition check on synthetic ``--dump-config`` trees;
- DshHost with ``fixtures/fake_dsh.py``: Web start, crash and restart, environment scrubbing, key detection,
  headless transcripts, and the LLM advisor's degradation rules inside a running campaign.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import uvicorn
from fastapi.testclient import TestClient

from apps.autopilot.dsh import tooldefs
from apps.autopilot.dsh.host import DshHost, check_composition
from apps.autopilot.strategies.llm_advisor import LLMAdvisor
from apps.console.server import create_app
from tests.system.harness import FAKE_MODULE, ROOT, Harness, wait_for

FAKE_DSH = Path(__file__).with_name("fixtures") / "fake_dsh.py"
TOOLS_SERVER = ROOT / "apps" / "autopilot" / "dsh" / "tools_server.py"
NODE = shutil.which("node")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Sampler:
    backend_name = "fake"
    driver_version = None
    errors = []

    def sample(self):
        return {"backend": "fake", "unix_s": time.time(), "monotonic_s": time.perf_counter(), "gpus": []}

    def close(self):
        pass


class LiveConsole:
    """The real console app served over HTTP (the tool server and fake_dsh call it), with the fake trainer."""

    def __init__(self, h: Harness, host):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.app = create_app(repo_root=h.repo, console_root=h.tmp / "console", sampler=_Sampler(),
                              web_dist=h.tmp / "no-dist", autopilot_root=h.tmp / "autopilot",
                              train_module=FAKE_MODULE, dsh_host=host)
        self.app.state.port = self.port
        self.app.state.autopilot.executor.poll_s = 0.05
        self.app.state.autopilot.poll_s = 0.05
        self.app.state.jobs.poll_interval_s = 0.05
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        if not wait_for(lambda: self.server.started, timeout=30):
            raise RuntimeError("console did not start")
        self.service = self.app.state.autopilot

    def close(self):
        for cid in self.service.campaign_ids():
            try:
                self.service.stop_campaign(cid)
            except Exception:
                pass
        self.server.should_exit = True
        self.thread.join(timeout=30)


class StubHost:
    """Just the token and context the agent API needs."""

    def __init__(self, token="test-token"):
        self.token = token
        self.context_campaign = None

    def status(self, ensure_web=False):
        return {"available": False, "llm_ready": False, "reason": "stub"}

    def start(self, console_url=None):
        pass

    def shutdown(self):
        pass

    def advisor(self, service):
        return None


@unittest.skipIf(NODE is None, "node is not installed")
class GuardPolicyTest(unittest.TestCase):
    def test_permission_table(self):
        out = subprocess.run([NODE, "--test", str(Path(__file__).with_name("dsh_policy.test.mjs"))],
                             capture_output=True, text=True, encoding="utf-8", cwd=ROOT, timeout=120)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)

    def test_categories_match_the_tool_definitions(self):
        policy = (ROOT / "apps" / "autopilot" / "dsh" / "bundle" / "policy.js").as_uri()
        out = subprocess.run([NODE, "--input-type=module", "-e",
                              f"const m = await import({json.dumps(policy)}); "
                              "console.log(JSON.stringify({c: m.CATEGORIES, h: m.HEADLESS_TOOLS}))"],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        data = json.loads(out.stdout)
        self.assertEqual({k: sorted(v) for k, v in data["c"].items()},
                         {k: sorted(v) for k, v in tooldefs.CATEGORIES.items()})
        self.assertEqual(sorted(data["h"]), sorted(tooldefs.HEADLESS_TOOLS))


class CompositionCheckTest(unittest.TestCase):
    GOOD = """
- id: tool-bash
  name: '@deepseek-ai/dsh-tool-bash'
  disabled: true
- id: tools
  name: '@deepseek-ai/dsh-tools'
  config: {mode: native}
- id: preset-standard
  name: '@deepseek-ai/dsh-agent-preset'
  config:
    id: standard
    plugins:
      - id: compaction
        name: cordis:group
        config:
          - id: compaction-basic
            name: '@deepseek-ai/dsh-compaction-basic'
- id: preset-ptc
  name: '@deepseek-ai/dsh-agent-preset'
  disabled: true
  config:
    plugins:
      - id: tool-fs
        name: '@deepseek-ai/dsh-tool-fs'
- id: mcp-autopilot
  name: '@deepseek-ai/dsh-mcp-client'
  config:
    serverName: autopilot
    command: !!js >-
      process.env.MYFLOWS_PYTHON ?? 'python'
- id: autopilot-guard
  name: '@myflows/dsh-autopilot/guard'
  config: {profile: web}
"""

    def test_designed_tree_passes(self):
        self.assertEqual(check_composition(self.GOOD, "web"), [])

    def test_problems_are_reported(self):
        bad_preset = self.GOOD.replace("      - id: compaction\n", "      - id: tool-web\n        name: '@deepseek-ai/dsh-tool-web'\n      - id: compaction\n")
        self.assertTrue(any("tool-web" in e for e in check_composition(bad_preset, "web")))
        enabled_shell = self.GOOD.replace("  name: '@deepseek-ai/dsh-tool-bash'\n  disabled: true", "  name: '@deepseek-ai/dsh-tool-bash'")
        self.assertTrue(any("tool-bash" in e for e in check_composition(enabled_shell, "web")))
        js_switch = self.GOOD.replace("  name: '@deepseek-ai/dsh-tool-bash'\n  disabled: true",
                                      "  name: '@deepseek-ai/dsh-tool-bash'\n  disabled: !!js process.platform === 'win32'")
        self.assertTrue(check_composition(js_switch, "web"), "a !!js switch counts as enabled")
        no_guard = self.GOOD.split("- id: autopilot-guard")[0]
        self.assertTrue(any("守卫" in e for e in check_composition(no_guard, "web")))
        self.assertTrue(any("profile" in e for e in check_composition(self.GOOD, "headless")))
        two_presets = self.GOOD.replace("- id: preset-ptc\n  name: '@deepseek-ai/dsh-agent-preset'\n  disabled: true",
                                        "- id: preset-ptc\n  name: '@deepseek-ai/dsh-agent-preset'")
        self.assertTrue(any("preset" in e for e in check_composition(two_presets, "web")))
        warned = check_composition(self.GOOD, "web", stderr='dsh: [@myflows/dsh-autopilot] patch: entry "x" not found\n')
        self.assertTrue(any("not found" in e for e in warned))
        self.assertTrue(check_composition("- id: [unclosed", "web"))


class ToolServerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.h = Harness(Path(self._tmp.name), {}, start=False)
        self.host = StubHost()
        self.console = LiveConsole(self.h, self.host)
        self.console.service.create_campaign(self.h.spec(campaign_id="tools-campaign"))

    def tearDown(self):
        self.console.close()
        self.h.close()
        self._tmp.cleanup()

    def _server(self, **env):
        base = {k: v for k, v in os.environ.items() if not k.startswith("MYFLOWS_AUTOPILOT")}
        base.update({"MYFLOWS_CONSOLE_URL": self.console.url, "PYTHONIOENCODING": "utf-8"})
        base.update(env)
        return subprocess.Popen([sys.executable, "-X", "utf8", str(TOOLS_SERVER)], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=base, encoding="utf-8")

    @staticmethod
    def _rpc(proc, messages):
        out = []
        for i, (method, params) in enumerate(messages, start=1):
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": params}) + "\n")
            proc.stdin.flush()
            out.append(json.loads(proc.stdout.readline()))
        return out

    def test_missing_token_refuses_to_start(self):
        proc = self._server()
        _, err = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("MYFLOWS_AUTOPILOT_TOKEN", err)

    def test_protocol_and_read_only_tools(self):
        proc = self._server(MYFLOWS_AUTOPILOT_TOKEN="test-token", MYFLOWS_AUTOPILOT_SOURCE="llm:web")
        try:
            init, tools = self._rpc(proc, [("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                                           "clientInfo": {"name": "t", "version": "0"}}),
                                           ("tools/list", {})])
            self.assertEqual(init["result"]["protocolVersion"], "2025-06-18")
            names = [t["name"] for t in tools["result"]["tools"]]
            self.assertEqual(sorted(names), sorted(tooldefs.TOOLS))
            campaign_dir = self.h.tmp / "autopilot" / "tools-campaign"
            before = {p.name: p.read_bytes() for p in campaign_dir.glob("*.json")}
            reads = [("tools/call", {"name": n, "arguments": {"campaign_id": "tools-campaign"}})
                     for n in ("get_campaign", "list_trials", "get_findings", "get_search_space")]
            reads += [("tools/call", {"name": "list_campaigns", "arguments": {}}),
                      ("tools/call", {"name": "list_runs", "arguments": {}}), ("ping", {})]
            replies = self._rpc(proc, reads)
            for reply in replies[:-1]:
                self.assertFalse(reply["result"]["isError"], reply)
                json.loads(reply["result"]["content"][0]["text"])
            listed = json.loads(replies[4]["result"]["content"][0]["text"])
            self.assertEqual([c["campaign_id"] for c in listed["campaigns"]], ["tools-campaign"])
            self.assertEqual(replies[-1]["result"], {})
            after = {p.name: p.read_bytes() for p in campaign_dir.glob("*.json")}
            self.assertEqual(before, after, "read-only tools must not change campaign state")
            actions = self.console.service.events("tools-campaign", kinds=("llm_action",))
            self.assertEqual(len(actions), 4)
            self.assertTrue(all(a["source"] == "llm:web" and a["ok"] and not a["write"] for a in actions))
            unknown, = self._rpc(proc, [("tools/call", {"name": "delete_campaign", "arguments": {}})])
            self.assertTrue(unknown["result"]["isError"])
            missing, = self._rpc(proc, [("no/such", {})])
            self.assertEqual(missing["error"]["code"], -32601)
        finally:
            proc.communicate(timeout=30)

    def test_headless_subset_and_wrong_token(self):
        proc = self._server(MYFLOWS_AUTOPILOT_TOKEN="wrong", MYFLOWS_AUTOPILOT_SOURCE="llm:headless")
        try:
            tools, call = self._rpc(proc, [("tools/list", {}),
                                           ("tools/call", {"name": "list_campaigns", "arguments": {}})])
            self.assertEqual(sorted(t["name"] for t in tools["result"]["tools"]), sorted(tooldefs.HEADLESS_TOOLS))
            self.assertTrue(call["result"]["isError"])
            self.assertIn("401", call["result"]["content"][0]["text"])
        finally:
            proc.communicate(timeout=30)


class AgentApiTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.h = Harness(Path(self._tmp.name), {}, start=False)
        self.host = StubHost()
        self.app = create_app(repo_root=self.h.repo, console_root=self.h.tmp / "console", sampler=_Sampler(),
                              web_dist=self.h.tmp / "no-dist", autopilot_root=self.h.tmp / "autopilot",
                              train_module=FAKE_MODULE, dsh_host=self.host, start_background=False)
        self.client = TestClient(self.app)
        self.service = self.app.state.autopilot
        self.service.create_campaign(self.h.spec(campaign_id="agent-campaign"))
        self.auth = {"Authorization": "Bearer test-token"}

    def tearDown(self):
        self.client.close()
        self.app.state.jobs.shutdown()
        self.h.close()
        self._tmp.cleanup()

    def call(self, tool, args=None, source="llm:web", **headers):
        return self.client.post(f"/api/autopilot/agent/tools/{tool}", json=args or {},
                                headers={**self.auth, "X-AutoPilot-Source": source, **headers})

    def test_token_is_required(self):
        url = "/api/autopilot/agent/tools/list_campaigns"
        self.assertEqual(self.client.post(url, json={}).status_code, 401)
        self.assertEqual(self.client.post(url, json={}, headers={"Authorization": "Bearer nope"}).status_code, 401)
        self.assertEqual(self.client.get("/api/autopilot/agent/tools", headers=self.auth).status_code, 200)

    def test_context_campaign_and_errors(self):
        self.assertEqual(self.call("get_campaign").status_code, 422)
        self.assertEqual(self.client.post("/api/autopilot/dsh/context", json={"campaign_id": "nope"}).status_code, 404)
        self.client.post("/api/autopilot/dsh/context", json={"campaign_id": "agent-campaign"})
        self.assertEqual(self.call("get_campaign").json()["campaign_id"], "agent-campaign")
        self.assertTrue(self.call("list_campaigns").json()["campaigns"][0]["current"])
        self.assertEqual(self.call("get_trial_summary", {"trial_id": "t999"}).status_code, 404)
        self.assertEqual(self.call("delete_campaign").status_code, 404)
        self.assertEqual(self.call("get_campaign", X_AutoPilot_Campaign="x").status_code, 200)

    def test_headless_runs_get_their_subset_only(self):
        self.assertEqual(self.call("pause_campaign", {"campaign_id": "agent-campaign"}, source="llm:headless").status_code, 403)
        self.assertEqual(self.call("start_campaign", {"campaign_id": "agent-campaign"}, source="llm:headless").status_code, 403)
        refused = self.service.events("agent-campaign", kinds=("llm_action",))
        self.assertEqual([(e["tool"], e["ok"]) for e in refused], [("pause_campaign", False), ("start_campaign", False)])

    def test_requests_create_pending_items_and_nothing_runs(self):
        cid = {"campaign_id": "agent-campaign"}
        good = self.call("request_trials", {**cid, "proposals": [{"delta": {"lr": 0.0003}, "rationale": "r"}]},
                         source="llm:headless").json()
        self.assertTrue(good["precheck"]["passed"])
        bad = self.call("request_trials", {**cid, "proposals": [{"delta": {"lr": 0.5}, "rationale": "r"}]}).json()
        self.assertFalse(bad["precheck"]["passed"])
        self.assertEqual(self.call("request_trials", {**cid, "proposals": []}).status_code, 422)
        start = self.call("request_campaign_start", cid).json()
        pending = self.service.pending("agent-campaign")
        self.assertEqual([p["id"] for p in pending], [good["pending_id"], bad["pending_id"], start["pending_id"]])
        self.assertTrue(all(p["status"] == "pending_confirmation" for p in pending))
        self.assertEqual(self.service.trials("agent-campaign"), [])
        self.assertEqual(self.service.get_campaign("agent-campaign")["status"], "draft")
        note = self.call("write_note", {**cid, "text": "说明", "title": "LLM 说明"}, source="llm:headless").json()
        self.assertTrue(note["saved"])
        self.assertEqual(self.service.get_campaign("agent-campaign")["state"]["notes"][0]["source"], "llm:headless")

    def test_rewriting_a_note_with_the_same_title_replaces_it(self):
        cid = {"campaign_id": "agent-campaign"}
        self.call("write_note", {**cid, "text": "太长的第一版", "title": "LLM 解读"}, source="llm:headless")
        self.call("write_note", {**cid, "text": "决策点说明", "title": "决策点 1"}, source="llm:headless")
        self.call("write_note", {**cid, "text": "缩短后的第二版", "title": "LLM 解读"}, source="llm:headless")
        self.call("write_note", {**cid, "text": "网页里写的", "title": "LLM 解读"}, source="llm:web")
        notes = self.service.get_campaign("agent-campaign")["state"]["notes"]
        self.assertEqual([(n["title"], n["source"], n["text"]) for n in notes],
                         [("LLM 解读", "llm:headless", "缩短后的第二版"), ("决策点 1", "llm:headless", "决策点说明"),
                          ("LLM 解读", "llm:web", "网页里写的")])
        writes = self.service.events("agent-campaign", kinds=("note",))
        self.assertEqual([e["replaced"] for e in writes], [False, False, True, False])

    def test_config_tools_and_global_log(self):
        created = self.call("create_campaign", {"spec": {**self.h.spec(campaign_id="llm-made"), "title": "LLM"}}).json()
        self.assertEqual(created["status"], "draft")
        self.assertEqual(self.service.get_campaign("llm-made")["spec"]["created_by"], "llm")
        updated = self.call("update_campaign", {"campaign_id": "llm-made", "patch": {"title": "改名"}}).json()
        self.assertEqual(updated["spec"]["title"], "改名")
        self.assertEqual(self.call("pause_campaign", {"campaign_id": "llm-made"}).status_code, 409)
        self.call("list_campaigns")
        log = (self.h.tmp / "autopilot" / "llm_actions.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(log[-1])["tool"], "list_campaigns")


class DshHostFakeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.log = self.tmp / "fake_dsh.jsonl"
        self.env = mock.patch.dict(os.environ, {"FAKE_DSH_LOG": str(self.log), "DEEPSEEK_BASE_URL": "http://evil",
                                                "DSH_HOME": str(self.tmp / "users-own-dsh")})
        self.env.start()
        self.hosts = []

    def tearDown(self):
        for host in self.hosts:
            host.shutdown()
        self.env.stop()
        self._tmp.cleanup()

    def host(self, **kw) -> DshHost:
        host = DshHost(repo_root=ROOT, autopilot_root=self.tmp / "autopilot", launcher=[sys.executable, str(FAKE_DSH)],
                       port=free_port(), web_ready_timeout_s=20, **kw)
        self.hosts.append(host)
        return host

    def test_prepare_web_lifecycle_and_isolation(self):
        host = self.host()
        self.assertTrue(host.prepare())
        status = host.status(ensure_web=True)
        self.assertTrue(status["available"])
        self.assertFalse(status["llm_ready"], "no key configured yet")
        self.assertTrue(wait_for(lambda: host.status()["web"]["state"] == "running", timeout=30))
        url = host.status()["web"]["url"]
        self.assertIn("token=fake-launch-token", url)
        self.assertNotIn("fake-launch-token", (host.home / "web.log").read_text(encoding="utf-8"))
        pid = host.status()["web"]["pid"]
        host.stop_web()
        self.assertEqual(host.status()["web"]["state"], "stopped")
        self.assertFalse(wait_for(lambda: _alive(pid), timeout=1))
        restarted = host.restart_web()
        self.assertIn(restarted["web"]["state"], ("starting", "running"))
        calls = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(c["DSH_HOME"] == str(host.home) for c in calls))
        self.assertFalse(any(c["has_deepseek"] for c in calls), "ambient DEEPSEEK_* must not reach dsh")
        self.assertTrue((host.home / "profiles" / "myflows-autopilot-web" / "node_modules" / "@myflows" /
                         "dsh-autopilot" / "guard.js").exists())
        manifest = json.loads((host.profile_dir("headless") / "package.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["dsh"]["profile"]["bundles"][-1], "@myflows/dsh-autopilot")
        self.assertIn("profile: headless", (host.profile_dir("headless") / "cordis.patch.yml").read_text(encoding="utf-8"))
        self.assertFalse((self.tmp / "users-own-dsh").exists())

    def test_web_crash_is_reported_and_restart_recovers(self):
        host = self.host()
        host.prepare()
        with mock.patch.dict(os.environ, {"FAKE_DSH_WEB": "crash"}):
            host.start_web()
            self.assertTrue(wait_for(lambda: host.status()["web"]["state"] == "failed", timeout=30))
        self.assertIn("退出码 3", host.status()["web"]["error"])
        self.assertEqual(host.status(ensure_web=True)["web"]["state"], "failed", "no automatic retry loop")
        host.restart_web()
        self.assertTrue(wait_for(lambda: host.status()["web"]["state"] == "running", timeout=30))

    def test_leftover_web_of_a_previous_console_is_replaced(self):
        port = free_port()
        first = self.host()
        first.port = port
        first.prepare()
        first.start_web()
        self.assertTrue(wait_for(lambda: first.status()["web"]["state"] == "running", timeout=30))
        leftover = first.status()["web"]["pid"]
        first._web = None  # the console died without stopping its child
        second = self.host()
        second.port = port
        second.prepare()
        second.start_web()
        self.assertTrue(wait_for(lambda: second.status()["web"]["state"] == "running", timeout=30))
        self.assertNotEqual(second.status()["web"]["pid"], leftover)
        self.assertFalse(_alive(leftover))

    def test_bad_composition_turns_the_llm_layer_off(self):
        with mock.patch.dict(os.environ, {"FAKE_DSH_DUMP": "bad"}):
            host = self.host()
            self.assertFalse(host.prepare())
        status = host.status(ensure_web=True)
        self.assertFalse(status["available"])
        self.assertIn("tool-pwsh", status["reason"])
        self.assertEqual(status["web"]["state"], "stopped")

    def test_missing_clone_is_reported(self):
        host = DshHost(repo_root=ROOT, autopilot_root=self.tmp / "autopilot", dsh_root=self.tmp / "no-clone")
        self.assertFalse(host.prepare())
        self.assertIn("找不到 dsh 克隆", host.status()["reason"])

    def test_key_detection(self):
        host = self.host()
        host.prepare()
        self.assertFalse(host.key_configured())
        (host.home / ".credentials.yaml").write_text("version: 1\nrefs:\n  DEEPSEEK_API_KEY: sk-test\n", encoding="utf-8")
        self.assertTrue(host.key_configured())
        self.assertTrue(self.host(extra_env={"DEEPSEEK_API_KEY": "mock"}).key_configured())

    def test_account_login_counts_as_configured(self):
        host = self.host()
        host.prepare()
        (host.home / ".credentials.yaml").write_text(
            "version: 1\nrecords:\n  deepseek-account-platform/default:\n    kind: grant\n    payload:\n"
            "      version: 1\n      token: tok\n      issuer: https://platform.deepseek.com\n", encoding="utf-8")
        self.assertEqual(host.credentials(), {"api_key": False, "account": True})
        self.assertTrue(host.key_configured())
        self.assertTrue(host.status()["llm_ready"])
        # headless runs move to the account route only while no API key is stored
        patch = host._route_patch()
        self.assertEqual(patch[0], "--patch")
        self.assertIn("provider: deepseek-account", Path(patch[1]).read_text(encoding="utf-8"))
        text = (host.home / ".credentials.yaml").read_text(encoding="utf-8")
        with_key = text.replace("records:", "refs:\n  DEEPSEEK_API_KEY: sk-test\nrecords:")
        (host.home / ".credentials.yaml").write_text(with_key, encoding="utf-8")
        self.assertEqual(host.credentials(), {"api_key": True, "account": True})
        self.assertEqual(host._route_patch(), [])

    def test_web_chats_default_to_the_account_route_with_only_a_login(self):
        host = self.host()
        host.prepare()
        (host.home / ".credentials.yaml").write_text(
            "version: 1\nrecords:\n  deepseek-account-platform/default:\n    kind: grant\n    payload:\n"
            "      version: 1\n      token: tok\n", encoding="utf-8")
        host.start_web()
        self.assertTrue(wait_for(lambda: host.status()["web"]["state"] == "running", timeout=30))
        web = [json.loads(line)["argv"] for line in self.log.read_text(encoding="utf-8").splitlines()
               if "--port" in json.loads(line)["argv"]]
        self.assertEqual(web[-1][web[-1].index("--patch") + 1], str(host.home / "account-route.patch.yml"))


def _alive(pid) -> bool:
    import psutil

    try:
        return psutil.Process(pid).is_running() and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class AdvisorTest(unittest.TestCase):
    """The advisor inside a running campaign, with fake_dsh as the model."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.h = Harness(Path(self._tmp.name), {"objective": {"lr_opt": -3.5}}, start=False)
        self.host = DshHost(repo_root=ROOT, autopilot_root=self.h.tmp / "autopilot", port=free_port(),
                            launcher=[sys.executable, str(FAKE_DSH)], extra_env={"DEEPSEEK_API_KEY": "mock"},
                            headless_timeout_s=30)
        self.console = LiveConsole(self.h, self.host)
        self.service = self.console.service
        self.advisor = LLMAdvisor(self.host, run_async=False)
        self.service.advisor = self.advisor
        self.assertTrue(wait_for(lambda: self.host.available, timeout=60))

    def tearDown(self):
        self.console.close()
        self.host.shutdown()
        self.h.close()
        self._tmp.cleanup()

    def spec(self, **kw):
        kw.setdefault("llm", {"enabled": True})
        return self.h.spec(**kw)

    def test_decision_points_create_pending_items_and_notes(self):
        campaign =self.service.create_campaign(self.spec(campaign_id="llm-campaign"))
        self.service.start_campaign(campaign["campaign_id"])
        final = wait_for(lambda: self.service.get_campaign("llm-campaign")["status"] in ("completed", "stopped", "failed"),
                         timeout=180)
        self.assertTrue(final)
        doc = self.service.get_campaign("llm-campaign")
        calls = self.service.events("llm-campaign", kinds=("llm_call",))
        finished = [c for c in calls if c["status"] != "started"]
        self.assertTrue(finished and all(c["status"] == "ok" for c in finished), calls)
        reasons = [c["reason"] for c in finished]
        self.assertTrue(any(r.startswith("phase:baseline") for r in reasons), reasons)
        self.assertEqual(reasons[-1], "report")
        self.assertEqual(doc["state"]["llm"]["calls"], len(finished))
        self.assertTrue(doc["state"]["llm"]["session_id"].startswith("session-fake-"))
        self.assertGreater(doc["state"]["llm"]["tokens"], 0)
        pending = self.service.pending("llm-campaign")
        self.assertTrue(pending)
        self.assertTrue(all(p["status"] in ("pending_confirmation", "expired") for p in pending))
        self.assertFalse(any(t["origin"].get("source", "").startswith("llm") for t in self.service.trials("llm-campaign")),
                         "nothing the model asked for runs without confirmation")
        notes = doc["state"]["notes"]
        self.assertTrue(any(n["source"] == "llm:headless" for n in notes))
        report = self.service.report("llm-campaign")["markdown"]
        self.assertIn("fake advisor note", report)
        actions = self.service.events("llm-campaign", kinds=("llm_action",))
        self.assertTrue(actions and all(a["source"] == "llm:headless" for a in actions))

    def test_failures_disable_after_three_and_skips_are_recorded(self):
        self.service.create_campaign(self.spec(campaign_id="fail-campaign"))
        store = self.service.store("fail-campaign")
        with mock.patch.dict(os.environ, {"FAKE_DSH_HEADLESS": "fail"}):
            for i in range(4):
                self.advisor.consult(self.service, store, f"phase:explore->exploit#{i}")
        llm = store.load()["state"]["llm"]
        self.assertEqual((llm["calls"], llm["failures"]), (3, 3))
        self.assertIn("连续 3 次", llm["disabled_reason"])
        self.assertEqual(len(self.service.events("fail-campaign", kinds=("llm_disabled",))), 1)

        self.service.create_campaign(self.spec(campaign_id="nokey-campaign"))
        nokey = DshHost(repo_root=ROOT, autopilot_root=self.h.tmp / "autopilot", launcher=[sys.executable, str(FAKE_DSH)])
        nokey.prepare()
        LLMAdvisor(nokey, run_async=False).consult(self.service, self.service.store("nokey-campaign"), "report")
        skips = self.service.events("nokey-campaign", kinds=("llm_skip",))
        self.assertIn("API Key", skips[0]["why"])

        self.service.create_campaign(self.spec(campaign_id="capped-campaign", llm={"enabled": True, "max_calls": 1}))
        capped = self.service.store("capped-campaign")
        self.advisor.consult(self.service, capped, "phase:baseline->explore")
        self.advisor.consult(self.service, capped, "phase:explore->exploit")
        state = capped.load()["state"]["llm"]
        self.assertEqual(state["calls"], 1)
        self.assertIn("上限", state["disabled_reason"])

    def test_timeout_counts_as_failure(self):
        self.host.headless_timeout_s = 2
        self.service.create_campaign(self.spec(campaign_id="slow-campaign"))
        store = self.service.store("slow-campaign")
        with mock.patch.dict(os.environ, {"FAKE_DSH_HEADLESS": "slow"}):
            started = time.time()
            self.advisor.consult(self.service, store, "failure:oom:t003")
        self.assertLess(time.time() - started, 30)
        call = self.service.events("slow-campaign", kinds=("llm_call",))[-1]
        self.assertEqual(call["status"], "failed")
        self.assertIn("超时", call["error"])


if __name__ == "__main__":
    unittest.main()
