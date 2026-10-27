"""AutoPilot campaigns end to end with the fake training entry (spec §13 layer one, without GPU or LLM)."""

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import psutil

from apps.common.gpu_lock import read_events
from tests.system.harness import FAKE_MODULE, ROOT, Harness, wait_for


def _trials(h, cid):
    return {t["trial_id"]: t for t in h.service.trials(cid)}


class CampaignTestCase(unittest.TestCase):
    scenario: dict = {}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.h = Harness(Path(self._tmp.name), self.scenario)

    def tearDown(self):
        self.h.close()
        self._tmp.cleanup()


class DiagnosisLoopTest(CampaignTestCase):
    scenario = {"data_bound": True, "objective": {"lr_opt": -3.5, "batch": {"4": -0.03}}}

    def test_full_campaign_with_monitor_triggered_refinement(self):
        campaign = self.h.run()
        cid = campaign["campaign_id"]
        self.assertEqual((campaign["status"], campaign["phase"]), ("completed", "done"))
        trials = _trials(self.h, cid)
        phases = [t["phase"] for t in trials.values()]
        self.assertEqual(phases[:2], ["baseline", "baseline"])
        self.assertIn("explore", phases)
        self.assertIn("confirm", phases)
        # GPU Monitor: data_bound on the best trial → num_workers 0→2 on the same seed → adopted.
        findings = self.h.service.findings(cid)
        chain = findings["chains"][0]
        self.assertEqual(chain["decision"], "adopted")
        self.assertTrue(chain["finding_ids"][0].startswith("gpu_monitor:data_bound:"))
        self.assertEqual(chain["proposal"]["delta"], {"num_workers": 2})
        self.assertTrue(chain["accuracy_unchanged"])
        self.assertGreater(chain["after"]["throughput_sps"], chain["before"]["throughput_sps"])
        derived = trials[chain["trial_id"]]
        self.assertEqual(derived["params"]["seed"], trials[chain["base_trial"]]["params"]["seed"])
        # Later trials use the adopted setting; confirmation pairs candidate and baseline on fresh seeds.
        confirm = [t for t in trials.values() if t["phase"] == "confirm"]
        self.assertTrue(all(t["params"]["seed"] in (2, 3) for t in confirm))
        self.assertTrue(any(t["params"]["num_workers"] == 2 for t in confirm))
        report = self.h.service.report(cid)["markdown"]
        for text in ("## 5. 诊断链", "data_bound", "派生 trial", "数据读取占比", "显著优于人工基线", "排行榜"):
            self.assertIn(text, report)
        self.assertTrue((self.h.tmp / "autopilot" / cid / "report" / "leaderboard.csv").exists())
        noise = findings["noise"]
        self.assertEqual(noise["seeds"], [1, 2])
        # Every dispatch is an event; every trial has a verdict file next to apps.train's result.json.
        events = self.h.service.events(cid)
        self.assertEqual(sum(1 for e in events if e["kind"] == "dispatch"), len(trials))
        for t in trials.values():
            run_dir = Path(t["attempts"][-1]["run_dir"])
            self.assertTrue((run_dir / "verdict.json").exists())
            self.assertEqual(json.loads((run_dir / "trial_result.json").read_text(encoding="utf-8"))["trial_id"],
                             t["trial_id"])


class BaselineNotBeatenTest(CampaignTestCase):
    scenario = {"objective": {"lr_opt": -3.0}}

    def test_honest_conclusion_when_nothing_beats_the_baseline(self):
        campaign = self.h.run()
        report = self.h.service.report(campaign["campaign_id"])["markdown"]
        self.assertIn("人工基线未被超越", report)
        self.assertFalse(any(t["phase"] == "confirm" for t in self.h.service.trials(campaign["campaign_id"])))


class BudgetTest(CampaignTestCase):
    scenario = {"step_s": 0.02}

    def test_max_trials_stops_the_campaign(self):
        campaign = self.h.run(budget={"max_trials": 3})
        cid = campaign["campaign_id"]
        started = [t for t in self.h.service.trials(cid) if t["attempts"]]
        self.assertEqual(len(started), 3)
        self.assertEqual(campaign["status"], "completed")
        self.assertIn("预算", campaign["stop_reason"])

    def test_wall_time_counts_running_time_only(self):
        campaign = self.h.run(budget={"wall_time_s": 1.0}, strategy={"max_candidates": 6})
        cid = campaign["campaign_id"]
        trials = self.h.service.trials(cid)
        started = [t for t in trials if t["attempts"]]
        self.assertLess(len(started), 6)
        # Trials that no longer fit the remaining running time are skipped with the reason recorded.
        skipped = [t for t in trials if t["status"] == "skipped"]
        self.assertTrue(skipped)
        self.assertTrue(all("预算" in t["skip_reason"] or "总时长" in t["skip_reason"] for t in skipped))
        self.assertLessEqual(sum(a["run_s"] for t in started for a in t["attempts"][:-1] or []), 1.0)

    def test_trial_timeout_kills_the_process_tree(self):
        self.h.set_scenario({"step_s": 0.02, "spawn_child": True, "rules": [{"when": {"seed": 1}, "behavior": "hang"}]})
        campaign = self.h.run(budget={"trial_timeout_s": 2, "max_trials": 1, "max_retries": 0},
                              online={"hang_timeout_s": 30})
        trial = self.h.service.trials(campaign["campaign_id"])[0]
        self.assertEqual((trial["verdict"]["status"], trial["verdict"]["kind"]), ("failed", "timeout"))
        child = int(Path(trial["attempts"][0]["run_dir"], "child.pid").read_text())
        self.assertTrue(wait_for(lambda: not psutil.pid_exists(child) or
                                 psutil.Process(child).status() == psutil.STATUS_ZOMBIE, timeout=20))


class MemoryPagingTest(CampaignTestCase):
    # batch 8 takes 9600 MiB of an 8192 MiB GPU: on Windows the driver pages instead of raising OOM.
    scenario = {"step_s": 0.02, "mem_mib_per_sample": 1200.0, "rules": [{"when": {"batch": 8}, "behavior": "hang"}]}

    def test_a_hang_with_full_memory_is_not_retried_and_derives_half_batch(self):
        cid = "paging"
        campaign = self.h.run(campaign_id=cid, timeout=240, baseline={"batch": 2},
                              space={"lr": [0.001], "batch": [2, 8]}, strategy={"max_candidates": 1, "seed": 0},
                              budget={"max_trials": 8}, online={"hang_timeout_s": 3},
                              specialists=["gpu_monitor", "benchmark"])
        self.assertEqual(campaign["status"], "completed", campaign.get("stop_reason"))
        trials = self.h.service.trials(cid)
        hung = next(t for t in trials if t["params"]["batch"] == 8)
        self.assertEqual((hung["verdict"]["kind"], len(hung["attempts"])), ("hang", 1))
        self.assertIn("换页", self.h.service.events(cid, kinds=("retry_skipped",))[0]["reason"])
        derived = [t for t in trials if (t.get("origin") or {}).get("agent") == "gpu_monitor"]
        self.assertEqual([(t["params"]["batch"], t["verdict"]["status"]) for t in derived], [(4, "valid")])
        chain = next(c for c in self.h.service.findings(cid)["chains"] if c["trial_id"] == derived[0]["trial_id"])
        self.assertEqual(chain["decision"], "recovered")
        self.assertTrue(chain["finding_ids"][0].startswith("gpu_monitor:paging:"))
        self.assertIn("显存峰值占比", self.h.service.report(cid)["markdown"])
        # A finished report can be re-rendered under the current rules without starting anything.
        before = len(self.h.service.trials(cid))
        self.assertIn("显存峰值占比", self.h.service.regenerate_report(cid)["markdown"])
        self.assertEqual(len(self.h.service.trials(cid)), before)
        self.assertTrue(self.h.service.get_campaign(cid)["state"]["report"]["regenerated"])


class AnomalyAndRetryTest(CampaignTestCase):
    scenario = {"rules": [
        {"when": {"num_workers": 11}, "behavior": "nan"},
        {"when": {"num_workers": 12}, "behavior": "nan_online"},
        {"when": {"num_workers": 13}, "behavior": "diverge"},
        {"when": {"num_workers": 14}, "behavior": "oom"},
        {"when": {"num_workers": 15}, "behavior": "hang"},
        {"when": {"num_workers": 16}, "behavior": "crash"},
        {"when": {"num_workers": 17, "attempt": 1}, "behavior": "crash"},
    ], "step_s": 0.01}

    def test_failures_are_classified_retried_only_for_infrastructure_and_never_ranked(self):
        service = self.h.service
        cid = service.create_campaign(self.h.spec(strategy={"max_candidates": 1},
                                                  budget={"max_trials": 20}))["campaign_id"]
        queued = service.submit_trials(cid, [{"delta": {"num_workers": n}} for n in range(11, 18)], source="user")
        self.assertTrue(all(item["queued"] for item in queued), queued)
        service.start_campaign(cid)
        self.h.wait(cid, timeout=180)
        by_workers = {t["params"]["num_workers"]: t for t in service.trials(cid) if t["params"]["num_workers"] > 10}
        expected = {11: ("failed", "nan", 1), 12: ("failed", "nan", 1), 13: ("failed", "diverged", 1),
                    14: ("failed", "oom", 1), 15: ("failed", "hang", 2), 16: ("failed", "crash", 2),
                    17: ("valid", None, 2)}
        for workers, (status, kind, attempts) in expected.items():
            trial = by_workers[workers]
            with self.subTest(workers=workers):
                self.assertEqual((trial["verdict"]["status"], trial["verdict"].get("kind"), len(trial["attempts"])),
                                 (status, kind, attempts))
                if status == "failed":
                    self.assertIsNone(trial["rank"])
        self.assertEqual(by_workers[12]["attempts"][0]["failure"]["source"], "online")
        # The NaN step row may reach the online monitor before the process exits with code 3.
        self.assertIn(by_workers[11]["attempts"][0]["failure"]["source"], ("result", "online"))
        knowledge = [json.loads(line) for line in
                     (self.h.tmp / "autopilot" / "knowledge.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(f.get("origin") == "trial" and f["outcome"] == "oom" for f in knowledge))
        events = service.events(cid, kinds=("retry",))
        self.assertEqual(sorted(e["retry_kind"] for e in events), ["crash", "crash", "hang"])


class PendingConfirmationTest(CampaignTestCase):
    def test_llm_requests_wait_for_a_person(self):
        service = self.h.service
        cid = service.create_campaign(self.h.spec(strategy={"max_candidates": 1}))["campaign_id"]
        rejected = service.request_trials(cid, [{"delta": {"lr": 0.0003, "batch": 1}, "rationale": "试更小的 lr"}])
        confirmed = service.request_trials(cid, [{"delta": {"lr": 0.0001, "batch": 1}}])
        bad = service.request_trials(cid, [{"delta": {"lr": 0.01}}])
        self.assertEqual(rejected["status"], "pending_confirmation")
        self.assertFalse(bad["precheck"]["passed"])
        self.assertIn("超出搜索空间", bad["precheck"]["reasons"][0])
        self.assertEqual(service.trials(cid), [])  # nothing runs before a person decides
        service.resolve_pending(cid, rejected["id"], confirm=False)
        item = service.resolve_pending(cid, confirmed["id"], confirm=True)
        self.assertEqual(item["status"], "confirmed")
        self.assertTrue(item["outcome"]["trials"][0]["queued"])
        with self.assertRaises(Exception):
            service.resolve_pending(cid, rejected["id"], confirm=True)
        service.start_campaign(cid)
        self.h.wait(cid)
        trials = service.trials(cid)
        llm = [t for t in trials if t["origin"].get("pending_id")]
        self.assertEqual([t["origin"]["pending_id"] for t in llm], [confirmed["id"]])
        self.assertEqual(llm[0]["params"]["lr"], 0.0001)
        self.assertFalse(any(t["params"].get("lr") == 0.0003 and t["params"].get("batch") == 1
                             and t["origin"].get("source") == "llm" for t in trials))
        pending = {p["id"]: p["status"] for p in service.pending(cid)}
        self.assertEqual(pending[rejected["id"]], "rejected")
        self.assertEqual(pending[bad["id"]], "expired")

    def test_llm_created_campaign_start_needs_confirmation(self):
        service = self.h.service
        cid = service.create_campaign(self.h.spec(), created_by="llm", source="llm:web")["campaign_id"]
        item = service.request_campaign_start(cid, source="llm:web")
        self.assertEqual(service.get_campaign(cid)["status"], "draft")
        service.resolve_pending(cid, item["id"], confirm=True)
        self.assertIn(service.get_campaign(cid)["status"], ("running", "completed"))
        service.stop_campaign(cid)


class RecoveryTest(CampaignTestCase):
    scenario = {"step_s": 0.15}

    def test_restart_keeps_finished_trials_and_reattaches_the_running_one(self):
        service = self.h.service
        cid = service.create_campaign(self.h.spec(strategy={"max_candidates": 1}))["campaign_id"]
        service.start_campaign(cid)
        running = wait_for(lambda: next((t for t in service.trials(cid)
                                         if t["trial_id"] == "t002" and t["status"] == "running"), None), timeout=60)
        self.assertIsNotNone(running)
        job_id = running["attempts"][0]["job_id"]
        service = self.h.restart()
        self.assertEqual(service.get_campaign(cid)["status"], "running")
        campaign = self.h.wait(cid)
        self.assertEqual(campaign["status"], "completed")
        trials = _trials(self.h, cid)
        self.assertEqual(len(trials["t001"]["attempts"]), 1)
        self.assertEqual([a["job_id"] for a in trials["t002"]["attempts"]], [job_id])
        self.assertEqual(trials["t002"]["verdict"]["status"], "valid")
        recovered = [e for e in service.events(cid) if e["kind"] == "campaign" and e.get("status") == "recovered"]
        self.assertEqual(len(recovered), 1)
        fingerprints = [t["fingerprint"] for t in trials.values()]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))

    def test_machine_restart_pauses_instead_of_training_at_boot(self):
        service = self.h.service
        cid = service.create_campaign(self.h.spec(strategy={"max_candidates": 1}))["campaign_id"]
        service.start_campaign(cid)
        running = wait_for(lambda: next((t for t in service.trials(cid)
                                         if t["trial_id"] == "t002" and t["status"] == "running"), None), timeout=60)
        self.assertIsNotNone(running)
        job = self.h.jobs.get(running["attempts"][0]["job_id"])
        service.shutdown()
        self.h.jobs.shutdown()
        root = psutil.Process(job["pid"])  # the reboot ends every trial process
        for proc in root.children(recursive=True) + [root]:
            proc.kill()
        psutil.wait_procs([root], timeout=10)
        with mock.patch("apps.autopilot.service.machine_boot_time", return_value=time.time()):
            service = self.h.start()
        campaign = service.get_campaign(cid)
        self.assertEqual(campaign["status"], "paused")
        self.assertNotIn(cid, [c for c, t in service._drivers.items() if t.is_alive()])
        paused = [e for e in service.events(cid) if e["kind"] == "campaign" and e.get("status") == "paused"]
        self.assertEqual(paused[-1]["source"], "reboot")
        self.assertFalse(any(e.get("status") == "recovered" for e in service.events(cid) if e["kind"] == "campaign"))
        # Continuing from the console retries the trial whose process the reboot ended.
        service.start_campaign(cid)
        self.assertEqual(self.h.wait(cid)["status"], "completed")
        t002 = _trials(self.h, cid)["t002"]
        self.assertEqual(len(t002["attempts"]), 2)
        self.assertEqual(t002["attempts"][1]["cause"], "retry")
        self.assertEqual(t002["verdict"]["status"], "valid")

    def test_regenerated_report_retracts_online_findings_the_replay_does_not_reproduce(self):
        service = self.h.service
        cid = service.create_campaign(self.h.spec(budget={"max_trials": 2}))["campaign_id"]
        service.start_campaign(cid)
        self.h.wait(cid)
        stale = {"id": "verifier:gpu_idle:t001", "agent": "verifier", "kind": "gpu_idle", "trial_id": "t001",
                 "severity": "warning", "message": "step 仍在推进，但 GPU 利用率中位数只有 15%（阈值 35%）",
                 "evidence": [], "data": {}}
        service.store(cid).update_state(lambda s: s.setdefault("findings", {}).update({stale["id"]: stale}))
        markdown = service.regenerate_report(cid)["markdown"]
        finding = service.get_campaign(cid)["state"]["findings"]["verifier:gpu_idle:t001"]
        self.assertIn("重放", finding["retracted"])
        self.assertIn("已撤回的发现", markdown)
        self.assertIn("~~step 仍在推进，但 GPU 利用率中位数只有 15%（阈值 35%）~~", markdown)
        notes = [e for e in service.events(cid) if e["kind"] == "note" and e.get("title") == "撤回在线发现"]
        self.assertEqual(len(notes), 1)

    def test_interrupted_finish_only_writes_the_report(self):
        service = self.h.service
        cid = service.create_campaign(self.h.spec(budget={"max_trials": 2}))["campaign_id"]
        service.start_campaign(cid)
        self.h.wait(cid)
        service.store(cid).update_state(lambda s: s.update(status="finishing", finishing_as="completed",
                                                           report=None))
        service = self.h.restart()
        campaign = self.h.wait(cid)
        self.assertEqual(campaign["status"], "completed")
        self.assertEqual(len(service.trials(cid)), 2)
        self.assertTrue(service.report(cid)["markdown"])


class GpuLockQueueTest(CampaignTestCase):
    scenario = {"step_s": 0.01, "rules": [{"when": {"seed": 99}, "behavior": "slow"}]}

    def test_console_job_trial_and_terminal_run_never_overlap(self):
        self.h.jobs.max_running = 3  # even if the console allowed parallel jobs, the GPU lock serialises them
        split = self.h.service  # make sure the split file exists through a campaign
        cid = split.create_campaign(self.h.spec(strategy={"max_candidates": 1}, budget={"max_trials": 3},
                                                online={"hang_timeout_s": 2}))["campaign_id"]
        split_file = self.h.service.get_campaign(cid)["spec"]["fixed"]["split_file"]
        terminal_dir = self.h.tmp / "terminal-run"
        terminal = subprocess.Popen(
            [sys.executable, "-m", FAKE_MODULE, "--run-dir", str(terminal_dir), "--split-file", split_file,
             "--max-samples", "40", "--seed", "99", "--batch", "2", "--epochs", "1"], cwd=self.h.repo)
        self.assertTrue(wait_for(lambda: any(e["event"] == "acquired" for e in read_events(terminal_dir)), 30))
        console = self.h.jobs.submit(job_type="single", label="manual", task="t",
                                     command=[sys.executable, "-m", FAKE_MODULE, "--run-dir", "{run_dir}",
                                              "--split-file", split_file, "--max-samples", "40", "--seed", "5"])
        self.h.service.start_campaign(cid)
        campaign = self.h.wait(cid, timeout=120)
        terminal.wait(timeout=60)
        self.assertTrue(wait_for(lambda: self.h.jobs.get(console["id"])["status"] == "succeeded", 30))
        run_dirs = [terminal_dir, Path(console["run_dir"])] + [
            Path(a["run_dir"]) for t in self.h.service.trials(cid) for a in t["attempts"]]
        spans = []
        for run_dir in run_dirs:
            events = read_events(run_dir)
            acquired = next(e["unix_s"] for e in events if e["event"] == "acquired")
            released = next(e["unix_s"] for e in events if e["event"] == "released")
            spans.append((acquired, released, run_dir.name))
        spans.sort()
        for (a0, a1, n0), (b0, b1, n1) in zip(spans, spans[1:]):
            self.assertLessEqual(a1, b0 + 1e-3, f"{n0} and {n1} held the GPU at the same time")
        # The first trial waited for the terminal run far longer than hang_timeout_s, yet it is valid.
        first = self.h.service.trials(cid)[0]
        self.assertEqual(first["verdict"]["status"], "valid")
        self.assertGreater(first["attempts"][0]["lock_wait_s"] or 0.0, 2.0)
        self.assertEqual(campaign["status"], "completed")


if __name__ == "__main__":
    unittest.main()
