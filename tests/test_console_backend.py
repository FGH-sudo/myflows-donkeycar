import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import psutil
from fastapi.testclient import TestClient

from apps.console.jobs import JobManager
from apps.console.readers import JsonlTail
from apps.console.registry import RunRegistry
from apps.console.server import create_app


def _write_jsonl(path: Path, rows, *, newline=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r) for r in rows)
    with path.open("a", encoding="utf-8") as f:
        f.write(text + ("\n" if newline and rows else ""))


def _step(rank, i):
    return {"rank": rank, "epoch": 0, "step": i, "start_s": 100.0 + i, "end_s": 100.5 + i, "step_wall_s": 0.5,
            "gradient_sync_s": 0.1, "loss": 1.0 / (i + 1), "n_samples": 16, "request_bytes": 10, "response_bytes": 20,
            "not_whitelisted": "x"}


def make_distributed_run(root: Path, name: str, *, ranks=2, steps=3, results=True) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({
        "task": "synthetic", "mode": "ring", "transport": "grpc_proto", "train_workers": ranks, "device": "cpu",
        "optimizer": "mbgd", "global_batch": 32, "epochs": 1, "learning_rate": 0.01}), encoding="utf-8")
    for rank in range(ranks):
        rank_dir = path / f"rank-{rank}"
        _write_jsonl(rank_dir / "steps.jsonl", [_step(rank, i) for i in range(steps)])
        _write_jsonl(rank_dir / "epochs.jsonl", [{
            "rank": rank, "epoch": 0, "steps": steps, "start_s": 100.0, "end_s": 100.5 + steps, "samples": 16 * steps,
            "val": {"split": "val", "loss": 0.5, "accuracy": 0.9}}])
        (rank_dir / "prepare.json").write_text(json.dumps({"pid": 1000 + rank}), encoding="utf-8")
    _write_jsonl(path / "resources.jsonl", [
        {"monotonic_s": 99.0 + i, "unix_s": 1.7e9 + i, "cpu_utilization_pct": 10.0, "gpu_utilization_pct": 50.0,
         "process_rss_bytes": {"1000": 100 * 1024 * 1024, "1001": 200 * 1024 * 1024}} for i in range(4)])
    if results:
        (path / "results.json").write_text(json.dumps({
            "status": "passed", "elapsed_s": 4.0,
            "final": {"val": {"loss": 0.5, "accuracy": 0.9, "samples": 100, "evaluation_s": 0.1}},
            "epochs": [{"epoch": 0, "samples_per_s": 12.0, "epoch_train_wall_s": 3.0}]}), encoding="utf-8")
    return path


def make_registry(root: Path, jobs=None) -> RunRegistry:
    archive = root / "archive-a"
    archive.mkdir(parents=True, exist_ok=True)
    return RunRegistry(root, console_root=root / "console", archive_roots=[archive], tensorboard_root=None,
                       jobs_provider=(lambda: jobs or []))


class JsonlTailTest(unittest.TestCase):
    def test_partial_line_waits_for_newline_and_fields_are_projected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            tail = JsonlTail(path, fields=("a",), nested=("val",))
            self.assertEqual(tail.poll(), [])
            with path.open("w", encoding="utf-8") as f:
                f.write('{"a": 1, "b": 2, "val": {"loss": 0.5}}\n{"a": 2')
            self.assertEqual(tail.poll(), [{"a": 1, "val_loss": 0.5}])
            with path.open("a", encoding="utf-8") as f:
                f.write(', "b": 3}\n')
            self.assertEqual(tail.poll(), [{"a": 2}])
            self.assertEqual(len(tail.rows), 2)

    def test_truncated_file_is_reread(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            _write_jsonl(path, [{"a": 1}, {"a": 2}])
            tail = JsonlTail(path)
            tail.poll()
            path.write_text('{"a": 9}\n', encoding="utf-8")
            tail.poll()
            self.assertEqual(tail.rows, [{"a": 9}])


class RunRegistryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_archive_run_summary_and_detail(self):
        registry = make_registry(self.root)
        make_distributed_run(registry.archives[0].root, "ring-2")
        (registry.archives[0].root / "run_index.json").write_text(json.dumps({"Ring 2w": "ring-2"}), encoding="utf-8")
        runs = registry.list_runs("archive-archive-a")
        self.assertEqual(len(runs), 1)
        summary = runs[0]
        self.assertEqual(summary["id"], "archive-archive-a/ring-2")
        self.assertEqual((summary["status"], summary["mode"], summary["workers"]), ("passed", "ring", 2))
        self.assertEqual(summary["final"]["val"], {"loss": 0.5, "accuracy": 0.9, "samples": 100})
        self.assertEqual(summary["final"]["samples_per_s"], 12.0)

        ref = registry.resolve("archive-archive-a/ring-2")
        self.assertEqual(ref.label, "Ring 2w")
        detail = registry.detail(ref)
        self.assertEqual(detail["ranks"], [0, 1])
        self.assertEqual(detail["rank_pids"], {0: 1000, 1: 1001})
        status = {s["rank"]: s for s in detail["rank_status"]}
        self.assertEqual(status[0]["steps_done"], 3)
        self.assertAlmostEqual(status[0]["sync_fraction"], 0.2)
        self.assertIsNone(status[0]["seconds_since_last"], "finished runs must not report staleness")

    def test_resolve_rejects_paths_outside_archive(self):
        registry = make_registry(self.root)
        make_distributed_run(self.root, "outside")
        self.assertIsNone(registry.resolve("archive-archive-a/../outside"))
        self.assertIsNone(registry.resolve("unknown/x"))

    def test_steps_are_incremental_and_aligned_to_shared_origin(self):
        registry = make_registry(self.root)
        path = make_distributed_run(registry.archives[0].root, "run")
        ref = registry.resolve("archive-archive-a/run")
        first = registry.steps(ref)
        self.assertEqual(first["origin"], 99.0)
        rank0 = first["ranks"]["0"]
        self.assertEqual(rank0["x"], [0, 1, 2])
        self.assertEqual(rank0["t"], [1.5, 2.5, 3.5])
        self.assertNotIn("not_whitelisted", rank0)

        _write_jsonl(path / "rank-0" / "steps.jsonl", [_step(0, 3)])
        second = registry.steps(ref, cursor=first["cursor"])
        self.assertEqual(second["ranks"]["0"]["x"], [3])
        self.assertEqual(second["ranks"]["1"]["x"], [])
        self.assertEqual(second["cursor"], {"steps:0": 4, "steps:1": 3})

        limited = registry.steps(ref, ranks=[1], fields=["loss"], max_points=2)
        self.assertEqual(list(limited["ranks"]), ["1"])
        self.assertEqual(set(limited["ranks"]["1"]), {"loss", "t", "x"})
        self.assertEqual(limited["ranks"]["1"]["x"][-1], 2, "downsampling keeps the last point")

    def test_resources_map_process_rss_to_ranks(self):
        registry = make_registry(self.root)
        make_distributed_run(registry.archives[0].root, "run")
        res = registry.resources(registry.resolve("archive-archive-a/run"))
        self.assertEqual(res["series"]["t"], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(res["series"]["rank_rss_mib"]["0"], [100.0] * 4)
        self.assertEqual(res["series"]["rank_rss_mib"]["1"], [200.0] * 4)

    def test_live_console_run_without_results(self):
        run_dir = make_distributed_run(self.root / "console", "job1", results=False)
        job = {"id": "job1", "type": "distributed", "label": "live", "task": "synthetic", "status": "running",
               "run_dir": str(run_dir), "started_unix_s": time.time(), "config": {}}
        registry = make_registry(self.root, jobs=[job])
        ref = registry.resolve("console/job1")
        summary = registry.summary(ref)
        self.assertEqual(summary["status"], "running")
        self.assertIsNone(summary["finished_unix_s"])
        detail = registry.detail(ref)
        self.assertEqual(detail["epochs"][0]["ranks_reported"], 2)
        self.assertEqual(detail["epochs"][0]["val_accuracy"], 0.9)
        self.assertIsNotNone(detail["rank_status"][0]["seconds_since_last"])

    def test_single_process_metrics(self):
        run_dir = self.root / "console" / "single1"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({"kind": "single", "task": "donkey_resnet18_single",
                                                         "args": {"device": "cuda", "batch": 4, "epochs": 2}}),
                                             encoding="utf-8")
        _write_jsonl(run_dir / "metrics.jsonl", [
            {"kind": "step", "monotonic_s": 10.0, "epoch": 1, "step": 1, "loss": 2.0},
            {"kind": "step", "monotonic_s": 11.0, "epoch": 1, "step": 2, "loss": 1.0},
            {"kind": "epoch", "monotonic_s": 11.5, "epoch": 1, "loss": 1.5, "val_loss": 1.2},
        ])
        job = {"id": "single1", "type": "single", "label": "s", "status": "succeeded", "run_dir": str(run_dir)}
        registry = make_registry(self.root, jobs=[job])
        ref = registry.resolve("console/single1")
        steps = registry.steps(ref)["ranks"]["0"]
        self.assertEqual((steps["x"], steps["t"], steps["loss"]), ([1, 2], [0.0, 1.0], [2.0, 1.0]))
        epochs = registry.rank_epochs(ref)
        self.assertEqual(epochs["ranks"]["0"][0]["val_loss"], 1.2)
        detail = registry.detail(ref)
        self.assertEqual((detail["kind"], len(detail["epochs"])), ("single", 1))
        self.assertEqual(registry.summary(ref)["global_batch"], 4)


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _wait(predicate, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class JobManagerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            for job in manager.list():
                if job["status"] in ("running", "stopping") and job.get("pid"):
                    try:
                        psutil.Process(job["pid"]).kill()
                    except psutil.NoSuchProcess:
                        pass
            manager.shutdown()
        self._tmp.cleanup()

    def manager(self, **kwargs) -> JobManager:
        manager = JobManager(self.root / "jobs", self.root, poll_interval_s=0.05, **kwargs).start()
        self.managers.append(manager)
        return manager

    def finished(self, manager, job_id):
        return lambda: manager.get(job_id)["status"] in ("succeeded", "failed", "cancelled")

    def test_success_log_and_placeholders(self):
        manager = self.manager()
        job = manager.submit(job_type="distributed", label="ok", task="t", config={"a": 1},
                             command=_py("import sys; print('hello', sys.argv[1:])") + ["{run_dir}", "{config_path}"])
        self.assertTrue(_wait(self.finished(manager, job["id"])))
        done = manager.get(job["id"])
        self.assertEqual((done["status"], done["returncode"]), ("succeeded", 0))
        self.assertIn(done["run_dir"], done["command"])
        self.assertTrue(Path(done["config_path"]).exists())
        self.assertIn("hello", Path(done["log_path"]).read_text(encoding="utf-8"))

    def test_failure_reports_failure_json_or_exit_code(self):
        manager = self.manager()
        job = manager.submit(job_type="distributed", label="bad", task="t", command=_py("raise SystemExit(3)"))
        self.assertTrue(_wait(self.finished(manager, job["id"])))
        done = manager.get(job["id"])
        self.assertEqual(done["status"], "failed")
        self.assertIn("3", done["error"])

        code = ("import json, pathlib, sys; p = pathlib.Path(sys.argv[1]); p.mkdir(parents=True, exist_ok=True); "
                "(p / 'failure.json').write_text(json.dumps({'error': 'worker died'})); sys.exit(1)")
        job = manager.submit(job_type="distributed", label="bad2", task="t", command=_py(code) + ["{run_dir}"])
        self.assertTrue(_wait(self.finished(manager, job["id"])))
        self.assertEqual(manager.get(job["id"])["error"], "worker died")

    def test_queue_runs_one_at_a_time_and_pending_can_be_cancelled(self):
        manager = self.manager(max_running=1)
        first = manager.submit(job_type="single", label="1", task="t", command=_py("import time; time.sleep(1.0)"))
        second = manager.submit(job_type="single", label="2", task="t", command=_py("pass"))
        third = manager.submit(job_type="single", label="3", task="t", command=_py("pass"))
        self.assertEqual(manager.get(first["id"])["status"], "running")
        self.assertEqual(manager.get(second["id"])["status"], "pending")
        cancelled = manager.stop(third["id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertTrue(_wait(self.finished(manager, second["id"])))
        self.assertGreaterEqual(manager.get(second["id"])["started_unix_s"], manager.get(first["id"])["finished_unix_s"])
        self.assertIsNone(manager.get(third["id"])["pid"])

    def test_stop_signals_process_tree(self):
        manager = self.manager()
        child = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"
        job = manager.submit(job_type="distributed", label="long", task="t", command=_py(child))
        root = psutil.Process(manager.get(job["id"])["pid"])
        self.assertTrue(_wait(root.children))
        children = root.children(recursive=True)
        manager.stop(job["id"])
        self.assertTrue(_wait(self.finished(manager, job["id"])))
        done = manager.get(job["id"])
        self.assertEqual((done["status"], done["error"]), ("cancelled", "用户停止"))
        self.assertTrue(_wait(lambda: not any(p.is_running() for p in children), timeout=10))

    def test_stop_file_lets_process_exit_cleanly(self):
        manager = self.manager()
        code = ("import pathlib, sys, time; f = pathlib.Path(sys.argv[1]); "
                "[time.sleep(0.05) for _ in iter(lambda: f.exists(), True)]; print('saved checkpoint')")
        job = manager.submit(job_type="single", label="s", task="t", command=_py(code) + ["{run_dir}/STOP"],
                             extra={"stop_file": "{run_dir}/STOP"})
        self.assertTrue(_wait(lambda: manager.get(job["id"])["status"] == "running"))
        manager.stop(job["id"])
        self.assertTrue(_wait(self.finished(manager, job["id"])))
        done = manager.get(job["id"])
        self.assertEqual((done["status"], done["returncode"]), ("cancelled", 0))
        self.assertIn("saved checkpoint", Path(done["log_path"]).read_text(encoding="utf-8"))

    def test_recover_marks_vanished_process(self):
        jobs_root = self.root / "jobs"
        jobs_root.mkdir()
        (jobs_root / "old.job.json").write_text(json.dumps({
            "id": "old", "type": "distributed", "label": "old", "task": "t", "status": "running",
            "created_unix_s": 1.0, "pid": 999999, "pid_create_time": 1.0, "stop_requested": False,
            "run_dir": str(jobs_root / "old"), "log_path": str(jobs_root / "old.log"), "command": []}),
            encoding="utf-8")
        manager = self.manager()
        job = manager.get("old")
        self.assertEqual(job["status"], "failed")
        self.assertIn("控制台重启", job["error"])


class _FakeSampler:
    backend_name = "fake"
    driver_version = "1.0"

    def __init__(self):
        self.errors = {}

    def sample(self):
        return {"monotonic_s": time.perf_counter(), "unix_s": time.time(), "backend": "fake", "status": "ok",
                "cpu_utilization_pct": 5.0, "ram_used_bytes": 1, "ram_total_bytes": 2,
                "gpus": [{"index": 0, "name": "Fake", "processes": [{"pid": 1, "used_memory_mib": None}]}]}

    def close(self):
        pass


class ConsoleApiTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "mycar" / "data").mkdir(parents=True)
        self.registry = make_registry(self.root)
        make_distributed_run(self.registry.archives[0].root, "ring-2")
        (self.registry.archives[0].root / "run_index.json").write_text(json.dumps({"Ring": "ring-2"}), encoding="utf-8")
        self.app = create_app(repo_root=self.root, console_root=self.root / "console", registry=self.registry,
                              sampler=_FakeSampler(), start_background=False, web_dist=self.root / "no-dist")
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.app.state.jobs.shutdown()
        self._tmp.cleanup()

    def test_runs_detail_series_and_spa_fallback(self):
        self.assertEqual(self.client.get("/api/health").json()["gpu_backend"], "fake")
        runs = self.client.get("/api/runs", params={"mode": "ring"}).json()
        self.assertEqual([r["id"] for r in runs], ["archive-archive-a/ring-2"])
        self.assertEqual(self.client.get("/api/runs", params={"q": "nomatch"}).json(), [])
        detail = self.client.get("/api/runs/archive-archive-a/ring-2").json()
        self.assertEqual(detail["summary"]["label"], "Ring")
        steps = self.client.get("/api/runs/archive-archive-a/ring-2/steps",
                                params={"cursor": json.dumps({"steps:0": 2}), "ranks": "0"}).json()
        self.assertEqual(steps["ranks"]["0"]["x"], [2])
        self.assertEqual(self.client.get("/api/runs/archive-archive-a/ring-2/steps", params={"cursor": "[1]"}).status_code, 400)
        self.assertEqual(self.client.get("/api/runs/archive-archive-a/missing").status_code, 404)
        self.assertEqual(self.client.get("/api/nope").status_code, 404)
        page = self.client.get("/runs")
        self.assertEqual(page.status_code, 200)
        self.assertIn("前端尚未构建", page.text)

    def test_stream_of_finished_run_ends(self):
        events = []
        with self.client.stream("GET", "/api/runs/archive-archive-a/ring-2/stream") as response:
            for line in response.iter_lines():
                if line.startswith("event: "):
                    events.append(line.split(": ", 1)[1])
                    if events[-1] == "end":
                        break
        self.assertEqual(events[:3], ["steps", "epochs", "resources"])
        self.assertIn("status", events)
        self.assertEqual(events[-1], "end")

    def test_gpu_payload_marks_job_processes(self):
        self.app.state.live._history.append(_FakeSampler().sample())
        with mock.patch.object(self.app.state.jobs, "list", return_value=[]):
            payload = self.client.get("/api/gpu").json()
        self.assertEqual(payload["backend"], "fake")
        self.assertEqual(len(payload["history"]), 1)
        self.assertIsNone(payload["latest"]["gpus"][0]["processes"][0]["owner"])

    def test_presets_validation_submit_and_stop(self):
        presets = self.client.get("/api/presets").json()
        self.assertIn({"mode": "ring", "transport": "grpc_proto", "label": "Ring AllReduce · gRPC"}, presets["modes"])
        smoke = next(p for p in presets["distributed"] if p["id"] == "synthetic-smoke")
        self.assertTrue(smoke["data_ready"])

        bad = dict(smoke["config"], mode="ring", transport="socket_json", train_workers=2)
        response = self.client.post("/api/jobs", json={"type": "distributed", "config": bad})
        self.assertEqual(response.status_code, 422)
        self.assertIn("非法的模式/传输组合", response.json()["detail"])
        response = self.client.post("/api/jobs", json={"type": "distributed", "config": dict(smoke["config"], evil=1)})
        self.assertEqual(response.status_code, 422)
        response = self.client.post("/api/jobs", json={"type": "single", "args": {"rm_rf": True}})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.post("/api/jobs", json={"type": "shell"}).status_code, 422)

        good = dict(smoke["config"], mode="ps", transport="grpc_proto", train_workers=2)
        with mock.patch.object(self.app.state.jobs, "_launch"):
            job = self.client.post("/api/jobs", json={"type": "distributed", "config": good}).json()
            single = self.client.post("/api/jobs", json={"type": "single", "args": {"epochs": 1, "batch": 2}}).json()
        self.assertEqual(job["status"], "pending")
        self.assertEqual(job["label"], "synthetic · PS · gRPC · 2w")
        self.assertEqual(job["command"][2], "benchmark.distributed_experiment")
        self.assertTrue(single["stop_file"].endswith("STOP_TRAINING"))
        self.assertIn(single["run_dir"], single["stop_file"])
        self.assertIn("--stop-file", single["command"])

        listed = self.client.get("/api/runs", params={"source": "console"}).json()
        self.assertEqual({r["status"] for r in listed}, {"pending"})
        stopped = self.client.post(f"/api/jobs/{job['id']}/stop").json()
        self.assertEqual(stopped["status"], "cancelled")
        self.assertEqual(self.client.post("/api/jobs/nope/stop").status_code, 404)


if __name__ == "__main__":
    unittest.main()
