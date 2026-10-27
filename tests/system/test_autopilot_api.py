"""Console routes of AutoPilot (spec §9) over a real app with the fake training entry."""

import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from apps.console.server import create_app
from tests.system.harness import FAKE_MODULE, Harness, wait_for


class _Sampler:
    backend_name = "fake"
    driver_version = None
    errors = []

    def sample(self):
        return {"backend": "fake", "unix_s": time.time(), "monotonic_s": time.perf_counter(), "gpus": []}

    def close(self):
        pass


class AutoPilotApiTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.h = Harness(Path(self._tmp.name), {"data_bound": True, "objective": {"lr_opt": -3.5}}, start=False)
        self.app = create_app(repo_root=self.h.repo, console_root=self.h.tmp / "console", sampler=_Sampler(),
                              web_dist=self.h.tmp / "no-dist", autopilot_root=self.h.tmp / "autopilot",
                              train_module=FAKE_MODULE, enable_dsh=False)
        self.app.state.autopilot.executor.poll_s = 0.05
        self.app.state.jobs.poll_interval_s = 0.05
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.h.close()
        self._tmp.cleanup()

    def test_create_run_inspect_and_report(self):
        c = self.client
        defaults = c.get("/api/autopilot/defaults").json()
        self.assertEqual(defaults["spec"]["baseline"]["lr"], 0.001)
        self.assertFalse(defaults["specialists"]["transfer"]["available"])
        bad = c.post("/api/autopilot/campaigns", json={"space": {"lr": [0.1], "batch": [2], "dropout": [0.1]}})
        self.assertEqual(bad.status_code, 422)
        spec = self.h.spec(campaign_id="api-campaign")
        created = c.post("/api/autopilot/campaigns", json=spec).json()
        self.assertEqual((created["status"], created["phase"]), ("draft", "baseline"))
        self.assertEqual(c.post("/api/autopilot/campaigns", json=spec).status_code, 409)
        patched = c.patch("/api/autopilot/campaigns/api-campaign", json={"title": "接口测试"}).json()
        self.assertEqual(patched["title"], "接口测试")
        pending = c.get("/api/autopilot/campaigns/api-campaign/pending").json()
        self.assertEqual(pending, [])
        self.assertEqual(c.post("/api/autopilot/campaigns/api-campaign/start").json()["status"], "running")
        self.assertEqual(c.patch("/api/autopilot/campaigns/api-campaign", json={"title": "x"}).status_code, 409)
        done = wait_for(lambda: c.get("/api/autopilot/campaigns/api-campaign").json()["status"] == "completed",
                        timeout=120)
        self.assertTrue(done)
        listing = c.get("/api/autopilot/campaigns").json()
        self.assertEqual(listing[0]["campaign_id"], "api-campaign")
        self.assertIsNotNone(listing[0]["best"])
        trials = c.get("/api/autopilot/campaigns/api-campaign/trials").json()
        self.assertTrue(all(t["verdict"]["status"] == "valid" for t in trials))
        self.assertTrue(any(t["rank"] == 1 for t in trials))
        findings = c.get("/api/autopilot/campaigns/api-campaign/findings").json()
        self.assertEqual(findings["chains"][0]["decision"], "adopted")
        report = c.get("/api/autopilot/campaigns/api-campaign/report").json()
        self.assertIn("诊断链", report["markdown"])
        self.assertIn("objective.png", report["figures"])
        figure = c.get("/api/autopilot/campaigns/api-campaign/report/figures/objective.png")
        self.assertEqual(figure.headers["content-type"], "image/png")
        self.assertEqual(c.get("/api/autopilot/campaigns/api-campaign/report/figures/..%2Fcampaign.json").status_code,
                         404)
        events = c.get("/api/autopilot/campaigns/api-campaign/events", params={"kinds": "verdict"}).json()
        self.assertEqual(len(events), len(trials))
        with c.stream("GET", "/api/autopilot/campaigns/api-campaign/stream") as response:
            body = "".join(response.iter_text())
        self.assertIn("event: summary", body)
        self.assertIn("event: end", body)
        # Trials are ordinary console jobs: the run list shows them.
        runs = c.get("/api/runs", params={"source": "console"}).json()
        self.assertGreaterEqual(len(runs), len(trials))
        self.assertTrue(any("AutoPilot" in r["label"] for r in runs))

    def test_pending_routes_and_stop(self):
        c = self.client
        c.post("/api/autopilot/campaigns", json=self.h.spec(campaign_id="pending-campaign"))
        service = self.app.state.autopilot
        item = service.request_trials("pending-campaign", [{"delta": {"lr": 0.0003}}])
        listed = c.get("/api/autopilot/campaigns/pending-campaign/pending").json()
        self.assertEqual(listed[0]["status"], "pending_confirmation")
        rejected = c.post(f"/api/autopilot/campaigns/pending-campaign/pending/{item['id']}/reject").json()
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(c.post(f"/api/autopilot/campaigns/pending-campaign/pending/{item['id']}/confirm").status_code,
                         409)
        self.assertEqual(c.post("/api/autopilot/campaigns/pending-campaign/pending/pa-999/confirm").status_code, 404)
        stopped = c.post("/api/autopilot/campaigns/pending-campaign/stop").json()
        self.assertEqual(stopped["status"], "stopped")
        self.assertEqual(c.post("/api/autopilot/campaigns/pending-campaign/start").status_code, 409)
        self.assertEqual(c.get("/api/autopilot/campaigns/missing").status_code, 404)
        status = c.get("/api/autopilot/dsh/status").json()
        self.assertFalse(status["llm_ready"])


if __name__ == "__main__":
    unittest.main()
