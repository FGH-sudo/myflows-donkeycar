"""Launch, queue, stop and recover console training jobs.

Layout under ``root`` (same pattern as ``benchmark.distributed_suite``)::

    <job_id>.job.json     status, pid, command, timestamps
    <job_id>.config.json  distributed config handed to benchmark.distributed_experiment
    <job_id>.log          stdout + stderr
    <job_id>/             run artifacts
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import psutil

ACTIVE = ("pending", "running", "stopping")
FINISHED = ("succeeded", "failed", "cancelled")
NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def _write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def kill_tree(pid: int, timeout: float = 5.0, descendants: list[psutil.Process] | None = None) -> list[int]:
    """Terminate ``pid`` and its descendants; return pids still alive afterwards."""
    procs = list(descendants or [])
    try:
        root = psutil.Process(pid)
        procs = [*root.children(recursive=True), *procs, root]
    except psutil.NoSuchProcess:
        pass
    unique = {p.pid: p for p in procs}.values()
    for proc in unique:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(unique), timeout=timeout)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=timeout)
    return [p.pid for p in alive]


class JobManager:
    def __init__(self, root: Path, repo_root: Path, *, python: str | None = None, max_running: int = 1,
                 poll_interval_s: float = 0.5, stop_grace_s: float = 20.0,
                 resource_recorder_factory: Callable[[dict, int], object] | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(repo_root)
        self.python = python or sys.executable
        self.max_running = int(max_running)
        self.poll_interval_s = float(poll_interval_s)
        self.stop_grace_s = float(stop_grace_s)
        self.resource_recorder_factory = resource_recorder_factory
        self._jobs: dict[str, dict] = {}
        self._procs: dict[str, subprocess.Popen | psutil.Process] = {}
        self._recorders: dict[str, object] = {}
        self._descendants: dict[str, list[psutil.Process]] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="console-jobs", daemon=True)
        self._recover()

    # ---- persistence ------------------------------------------------------------
    def _job_path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.job.json"

    def _save(self, job: dict) -> None:
        _write_json(self._job_path(job["id"]), job)

    def _recover(self) -> None:
        for path in sorted(self.root.glob("*.job.json")):
            job = _read_json(path)
            if not job or "id" not in job:
                continue
            self._jobs[job["id"]] = job
            if job["status"] in ("running", "stopping"):
                proc = self._attach(job)
                if proc is not None:
                    self._procs[job["id"]] = proc
                else:
                    self._finalize(job, None, note="控制台重启时进程已不存在")

    @staticmethod
    def _attach(job: dict) -> psutil.Process | None:
        pid, created = job.get("pid"), job.get("pid_create_time")
        if not pid:
            return None
        try:
            proc = psutil.Process(pid)
            if created is not None and abs(proc.create_time() - created) > 1.0:
                return None
            if proc.status() == psutil.STATUS_ZOMBIE:
                return None
            return proc
        except psutil.NoSuchProcess:
            return None

    # ---- public API ---------------------------------------------------------------
    def start(self) -> JobManager:
        if not self._thread.is_alive():
            self._thread.start()
        return self

    def shutdown(self) -> None:
        """Stop the scheduler thread; running jobs keep running and are re-attached on restart."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        with self._lock:
            for recorder in self._recorders.values():
                recorder.stop()
            self._recorders.clear()

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(j) for j in sorted(self._jobs.values(), key=lambda j: j["created_unix_s"], reverse=True)]

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def submit(self, *, job_type: str, label: str, task: str, command: list[str], run_dir: Path | None = None,
               config: dict | None = None, extra: dict | None = None) -> dict:
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        run_dir = Path(run_dir) if run_dir else self.root / job_id
        extra = {k: v.replace("{run_dir}", str(run_dir)) if isinstance(v, str) else v for k, v in (extra or {}).items()}
        job = {
            "id": job_id, "type": job_type, "label": label, "task": task, "status": "pending",
            "created_unix_s": time.time(), "started_unix_s": None, "finished_unix_s": None,
            "pid": None, "pid_create_time": None, "returncode": None, "error": None, "stop_requested": False,
            "command": [str(c).replace("{run_dir}", str(run_dir)).replace("{job_id}", job_id) for c in command],
            "run_dir": str(run_dir), "log_path": str(self.root / f"{job_id}.log"),
            "config": config, **extra,
        }
        if config is not None and job_type == "distributed":
            config_path = self.root / f"{job_id}.config.json"
            _write_json(config_path, config)
            job["config_path"] = str(config_path)
            job["command"] = [c.replace("{config_path}", str(config_path)) for c in job["command"]]
        with self._lock:
            self._jobs[job_id] = job
            self._save(job)
        self._tick()
        return dict(job)

    def stop(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] == "pending":
                job.update(status="cancelled", stop_requested=True, finished_unix_s=time.time(), error="启动前已取消")
                self._save(job)
                return dict(job)
            if job["status"] not in ("running", "stopping"):
                return dict(job)
            job.update(stop_requested=True, status="stopping", stop_requested_unix_s=time.time())
            self._save(job)
            proc = self._procs.get(job_id)
        pid = job.get("pid")
        if pid:
            try:
                self._descendants[job_id] = psutil.Process(pid).children(recursive=True)
            except psutil.NoSuchProcess:
                self._descendants[job_id] = []
        stop_file = job.get("stop_file")
        if stop_file:
            Path(stop_file).parent.mkdir(parents=True, exist_ok=True)
            Path(stop_file).write_text("stop requested by console\n", encoding="utf-8")
        elif pid:
            delivered = False
            if isinstance(proc, subprocess.Popen) and os.name == "nt":
                try:
                    os.kill(pid, signal.CTRL_BREAK_EVENT)
                    delivered = True
                except (OSError, AttributeError):
                    delivered = False
            if not delivered:
                threading.Thread(target=kill_tree, args=(pid,), kwargs={"descendants": self._descendants.get(job_id)},
                                 daemon=True).start()
        return dict(job)

    # ---- scheduling ------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:  # keep the scheduler alive
                print(f"[console-jobs] scheduler error: {exc}", file=sys.stderr)
            self._stop.wait(self.poll_interval_s)

    def _tick(self) -> None:
        with self._lock:
            for job_id, proc in list(self._procs.items()):
                job = self._jobs[job_id]
                code = self._returncode(proc)
                if code is not None or not self._alive(proc):
                    self._procs.pop(job_id, None)
                    self._finalize(job, code)
                elif job["status"] == "stopping":
                    waited = time.time() - (job.get("stop_requested_unix_s") or time.time())
                    if waited > self.stop_grace_s:
                        kill_tree(job["pid"], descendants=self._descendants.get(job_id))
            running = sum(1 for j in self._jobs.values() if j["status"] in ("running", "stopping"))
            pending = sorted((j for j in self._jobs.values() if j["status"] == "pending"),
                             key=lambda j: j["created_unix_s"])
            for job in pending[:max(0, self.max_running - running)]:
                self._launch(job)

    @staticmethod
    def _returncode(proc) -> int | None:
        if isinstance(proc, subprocess.Popen):
            return proc.poll()
        return None

    @staticmethod
    def _alive(proc) -> bool:
        if isinstance(proc, subprocess.Popen):
            return proc.poll() is None
        try:
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    def _launch(self, job: dict) -> None:
        log = open(job["log_path"], "ab")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        try:
            proc = subprocess.Popen(job["command"], cwd=str(self.repo_root), stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env, creationflags=NEW_GROUP)
        except OSError as exc:
            log.close()
            job.update(status="failed", error=f"启动失败：{exc}", finished_unix_s=time.time())
            self._save(job)
            return
        finally:
            if not log.closed:
                log.close()
        try:
            created = psutil.Process(proc.pid).create_time()
        except psutil.NoSuchProcess:
            created = None
        job.update(status="running", pid=proc.pid, pid_create_time=created, started_unix_s=time.time())
        self._procs[job["id"]] = proc
        self._save(job)
        if self.resource_recorder_factory and job.get("record_resources"):
            recorder = self.resource_recorder_factory(job, proc.pid)
            if recorder is not None:
                self._recorders[job["id"]] = recorder

    def _finalize(self, job: dict, returncode: int | None, note: str | None = None) -> None:
        recorder = self._recorders.pop(job["id"], None)
        if recorder is not None:
            recorder.stop()
        leftovers = self._descendants.pop(job["id"], None)
        if leftovers:
            kill_tree(job["pid"] or 0, timeout=2.0, descendants=leftovers)
        run_dir = Path(job["run_dir"])
        results = _read_json(run_dir / "results.json") or {}
        failure = _read_json(run_dir / "failure.json") or {}
        job["returncode"] = returncode
        job["finished_unix_s"] = time.time()
        if job.get("stop_requested"):
            job["status"] = "cancelled"
            job["error"] = job.get("error") or "用户停止"
        elif returncode == 0 or (returncode is None and results.get("status") == "passed"):
            job["status"] = "succeeded" if results.get("status", "passed") == "passed" else "failed"
            if job["status"] == "failed":
                job["error"] = results.get("error")
        else:
            job["status"] = "failed"
            job["error"] = (failure.get("error") or results.get("error") or note
                            or (f"退出码 {returncode}" if returncode is not None else "进程已结束，退出码未知"))
        self._save(job)
