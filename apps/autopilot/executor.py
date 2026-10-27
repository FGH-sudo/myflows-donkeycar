"""Executor (spec §6.3): submit an attempt to the console's JobManager, watch it, classify how it ended."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from apps.common.gpu_lock import acquired_unix_s
from apps.console.presets import TRAIN_MODULE, single_job, validate_single

from .files import read_json, write_json
from .provenance import code_state
from .verifier import OnlineMonitor

FINISHED = ("succeeded", "failed", "cancelled")


class Executor:
    def __init__(self, jobs, *, python: str, module: str = TRAIN_MODULE, poll_s: float = 1.0):
        self.jobs = jobs
        self.python = python
        self.module = module
        self.poll_s = float(poll_s)

    def submit(self, store, trial: dict, attempt_no: int, *, overrides: dict | None = None,
               cause: str = "first") -> dict:
        """Start one attempt. ``cause`` is ``first``, ``retry`` (infrastructure failure) or ``rollback``, whose
        ``overrides`` (``resume_state`` and the reduced ``lr``) go through the same argument whitelist."""
        run_dir = store.attempt_dir(trial["trial_id"], attempt_no)
        run_dir.mkdir(parents=True, exist_ok=True)
        suffix = {"retry": f" · 重试 {attempt_no - 1}", "rollback": " · 回滚"}.get(cause, "")
        label = f"AutoPilot · {store.campaign_id} · {trial['trial_id']}{suffix}"
        args = dict(trial["resolved_args"])
        if overrides:
            args = validate_single({**args, **overrides}, Path(getattr(self.jobs, "repo_root", Path.cwd())))
        kwargs = single_job(args, self.python, label=label, module=self.module,
                            extra={"campaign_id": store.campaign_id, "trial_id": trial["trial_id"],
                                   "attempt": attempt_no})
        repo_root = Path(getattr(self.jobs, "repo_root", Path.cwd()))
        # The training process imports the code as it is now, which may differ from the campaign's start
        launched = code_state(repo_root).get("code_hash")
        job = self.jobs.submit(**kwargs, run_dir=run_dir)
        write_json(run_dir / "command.json", {"command": job["command"], "args": args, "job_id": job["id"],
                                              "cause": cause, "overrides": overrides or {}})
        attempt = {"attempt": attempt_no, "job_id": job["id"], "run_dir": str(run_dir), "status": "running",
                   "failure": None, "submitted_unix_s": time.time(), "finished_unix_s": None, "online_stop": None,
                   "log_path": job.get("log_path"), "cause": cause, "overrides": overrides or {},
                   "code_hash": launched}
        trial.setdefault("attempts", []).append(attempt)
        trial["status"] = "running"
        store.save_trial(trial)
        store.event("dispatch", trial_id=trial["trial_id"], attempt=attempt_no, job_id=job["id"],
                    phase=trial.get("phase"), params=trial.get("params"))
        return attempt

    def wait(self, store, trial: dict, attempt: dict, spec: dict, *, should_stop: Callable[[], bool],
             on_finding: Callable[[dict], None] | None = None,
             abort: Callable[[], bool] = lambda: False) -> dict:
        """Block until the job ends; stop it early when an online rule fires or the campaign is stopped.

        ``abort`` (the service shutting down) returns at once with the attempt still ``running``: the job keeps
        going and the next service start re-attaches to it.
        """
        run_dir = Path(attempt["run_dir"])
        monitor = OnlineMonitor(run_dir, spec["online"], spec["budget"]["trial_timeout_s"],
                                log_path=Path(attempt["log_path"]) if attempt.get("log_path") else None)
        online_stop, cancelled = None, False
        reported = 0
        while True:
            if abort():
                return attempt
            job = self.jobs.get(attempt["job_id"])
            if job is None or job["status"] in FINISHED:
                break
            if online_stop is None and not cancelled:
                stop = monitor.check()
                if stop is not None:
                    online_stop = stop
                    store.event("online_stop", trial_id=trial["trial_id"], attempt=attempt["attempt"],
                                **stop.to_dict())
                    self.jobs.stop(attempt["job_id"])
            for finding in monitor.findings[reported:]:
                if on_finding is not None:
                    on_finding({**finding, "trial_id": trial["trial_id"]})
            reported = len(monitor.findings)
            if not cancelled and online_stop is None and should_stop():
                cancelled = True
                self.jobs.stop(attempt["job_id"])
            time.sleep(self.poll_s)
        if online_stop is None and not cancelled and job is not None:
            # Rows written after the last poll: a NaN or a divergence there still fails the run. The clock-based
            # checks (hang, timeout) no longer apply once the process has exited.
            stop = monitor.check()
            if stop is not None and stop.kind in ("nan", "diverged"):
                online_stop = stop
                store.event("online_stop", trial_id=trial["trial_id"], attempt=attempt["attempt"], **stop.to_dict())
        result = read_json(run_dir / "result.json")
        attempt.update(self._classify(job, result, online_stop, cancelled))
        attempt["finished_unix_s"] = time.time()
        acquired = acquired_unix_s(run_dir)
        timing = (result or {}).get("timing") or {}
        attempt["run_s"] = timing.get("run_s") or (attempt["finished_unix_s"] - acquired if acquired else 0.0)
        attempt["lock_wait_s"] = timing.get("lock_wait_s")
        attempt["returncode"] = (job or {}).get("returncode")
        store.save_trial(trial)
        return attempt

    @staticmethod
    def _classify(job: dict | None, result: dict | None, online_stop, cancelled: bool) -> dict:
        if online_stop is not None:
            stop = online_stop.to_dict()
            failure = {"kind": online_stop.kind, "message": online_stop.message, "source": "online",
                       **{k: stop[k] for k in ("step", "since_step") if k in stop}}
            return {"status": "failed", "failure": failure, "online_stop": stop}
        if cancelled:
            return {"status": "cancelled", "failure": None, "note": "campaign 停止"}
        if job is None:
            return {"status": "failed", "failure": {"kind": "crash", "message": "控制台任务记录丢失"}}
        if result is not None and result.get("failure"):
            failure = dict(result["failure"])
            failure.setdefault("source", "result")
            return {"status": "failed", "failure": failure}
        if result is not None and result.get("status") in ("succeeded", "stopped"):
            if job["status"] == "cancelled" and result.get("status") == "stopped":
                return {"status": "cancelled", "failure": None, "note": "任务在控制台被停止"}
            return {"status": result["status"], "failure": None}
        if job["status"] == "cancelled":
            return {"status": "cancelled", "failure": None, "note": job.get("error") or "任务被取消"}
        message = job.get("error") or f"退出码 {job.get('returncode')}"
        return {"status": "failed", "failure": {"kind": "crash", "message": f"训练进程异常结束且没有 result.json：{message}"}}
