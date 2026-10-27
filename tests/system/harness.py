"""Shared setup of the AutoPilot system tests: a tiny repository, a console JobManager and the service,
with ``tests.system.fixtures.fake_train`` in place of apps.train (no GPU, no LLM)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import psutil

from apps.autopilot.service import AutoPilotService
from apps.console.jobs import JobManager

ROOT = Path(__file__).resolve().parents[2]
FAKE_MODULE = "tests.system.fixtures.fake_train"
FAKE_EVAL_MODULE = "tests.system.fixtures.fake_evaluate"
FINISHED = ("completed", "stopped", "failed")


def make_tub(repo: Path, count: int = 40) -> Path:
    images = repo / "mycar" / "data" / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for index in range(count):
        angle = float(rng.uniform(-1.0, 1.0))
        cv2.imwrite(str(images / f"{index}_{angle:.4f}.jpg"), np.full((8, 8, 3), 40 + index, np.uint8))
    return images.parent


def wait_for(predicate, timeout: float = 60.0, interval: float = 0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


class Harness:
    def __init__(self, tmp: Path, scenario: dict | None = None, *, start: bool = True, stop_grace_s: float = 1.0):
        self.tmp = Path(tmp)
        self.repo = self.tmp / "repo"
        if not (self.repo / "mycar" / "data").exists():
            make_tub(self.repo)
        self.lock_path = self.tmp / "gpu.lock"
        self.scenario_path = self.tmp / "scenario.json"
        self.set_scenario(scenario or {})
        self.env = mock.patch.dict(os.environ, {
            "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "MYFLOWS_GPU_LOCK": str(self.lock_path), "FAKE_TRAIN_SCENARIO": str(self.scenario_path)})
        self.env.start()
        self.stop_grace_s = stop_grace_s
        self.jobs = None
        self.service = None
        if start:
            self.start()

    def set_scenario(self, scenario: dict) -> None:
        self.scenario_path.write_text(json.dumps(scenario), encoding="utf-8")

    def start(self) -> AutoPilotService:
        self.jobs = JobManager(self.tmp / "console", self.repo, python=sys.executable, poll_interval_s=0.05,
                               stop_grace_s=self.stop_grace_s).start()
        self.service = AutoPilotService(self.tmp / "autopilot", jobs=self.jobs, repo_root=self.repo,
                                        python=sys.executable, module=FAKE_MODULE, poll_s=0.05,
                                        record_environment=False, eval_module=FAKE_EVAL_MODULE).start()
        return self.service

    def restart(self) -> AutoPilotService:
        """Simulate a console restart: drivers and the job scheduler stop, running processes keep running."""
        self.service.shutdown()
        self.jobs.shutdown()
        return self.start()

    def spec(self, **overrides) -> dict:
        spec = {
            "campaign_id": overrides.pop("campaign_id", "test-campaign"),
            "fixed": {"max_samples": 40, "epochs": 1, "val_size": 10},
            "baseline": {"seeds": [1, 2]},
            "strategy": {"max_candidates": 2, "seed": 0},
            "budget": {"max_trials": 12, "trial_timeout_s": 60, "wall_time_s": 600},
            "online": {"hang_timeout_s": 3, "window_steps": 2, "divergence_windows": 3},
            "llm": {"enabled": False},
        }
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(spec.get(key), dict):
                spec[key] = {**spec[key], **value}
            else:
                spec[key] = value
        return spec

    def run(self, timeout: float = 120.0, **overrides) -> dict:
        campaign = self.service.create_campaign(self.spec(**overrides))
        cid = campaign["campaign_id"]
        self.service.start_campaign(cid)
        return self.wait(cid, timeout=timeout)

    def wait(self, cid: str, statuses=FINISHED, timeout: float = 120.0) -> dict:
        done = wait_for(lambda: self.service.get_campaign(cid)["status"] in statuses, timeout=timeout)
        campaign = self.service.get_campaign(cid)
        if not done:
            raise AssertionError(f"campaign {cid} still {campaign['status']} after {timeout} s")
        return campaign

    def close(self) -> None:
        if self.service is not None:
            for cid in self.service.campaign_ids():
                try:
                    self.service.stop_campaign(cid)
                except Exception:
                    pass
            self.service.shutdown()
        if self.jobs is not None:
            for job in self.jobs.list():
                if job["status"] in ("running", "stopping") and job.get("pid"):
                    try:
                        root = psutil.Process(job["pid"])
                        for proc in [*root.children(recursive=True), root]:
                            proc.kill()
                    except psutil.NoSuchProcess:
                        pass
            self.jobs.shutdown()
        self.env.stop()
