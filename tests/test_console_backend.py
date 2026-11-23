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
from apps.console.registry import RunRegistry, phase_totals
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
    def test_phase_totals_only_pair_complete_measurements_and_reconstruct_legacy_wall(self):
        sample = {"data_load_ms": 2., "forward_ms": 3., "backward_ms": 4., "optimizer_ms": 1.}
        totals = phase_totals([{**sample, "step_total_ms": 12.}, {**sample, "step_time_ms": 10.},
                               {"step_total_ms": 100., "forward_ms": 90.}, {**sample, "backward_ms": None}])
        self.assertEqual(totals, {"count": 2, "step_total_ms": 24., "data_load_ms": 4.,
                                  "forward_ms": 6., "backward_ms": 8., "optimizer_ms": 2.})

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

    def test_formal_distributed_run_reports_parallel_mode_and_rank_status(self):
        run_dir = self.root / "console" / "ring2"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({
            "kind": "single", "task": "donkey_resnet18_single", "args": {"device": "cuda", "batch": 8},
            "parallel": {"mode": "ring", "transport": "grpc_proto", "workers": 2, "shard_sizes": [4, 4]}}),
            encoding="utf-8")
        _write_jsonl(run_dir / "metrics.jsonl", [
            {"kind": "step", "monotonic_s": 10.0, "epoch": 1, "step": 1, "loss": 2.0},
            {"kind": "epoch", "monotonic_s": 11.0, "epoch": 1, "loss": 2.0, "val_loss": 1.5},
        ])
        for rank in (0, 1):
            (run_dir / f"rank-{rank}").mkdir()
            (run_dir / f"rank-{rank}" / "prepare.json").write_text(json.dumps({"pid": 4000 + rank}), encoding="utf-8")
            _write_jsonl(run_dir / f"rank-{rank}" / "steps.jsonl", [
                {"rank": rank, "epoch": 0, "step": 0, "start_s": 9.0, "end_s": 10.0, "step_wall_s": 1.0,
                 "gradient_sync_s": 0.25, "n_samples": 4, "loss": 2.0}])
        job = {"id": "ring2", "type": "single", "label": "r", "status": "succeeded", "run_dir": str(run_dir)}
        registry = make_registry(self.root, jobs=[job])
        ref = registry.resolve("console/ring2")
        summary = registry.summary(ref)
        self.assertEqual((summary["kind"], summary["mode"], summary["transport"], summary["workers"]),
                         ("single", "ring", "grpc_proto", 2))
        detail = registry.detail(ref)
        # Curves come from rank 0's metrics.jsonl; the rank-*/ records only describe each worker.
        self.assertEqual(detail["ranks"], [])
        self.assertEqual([r["rank"] for r in detail["rank_status"]], [0, 1])
        self.assertEqual(detail["rank_status"][1]["sync_fraction"], 0.25)
        self.assertEqual(detail["rank_pids"], {0: 4000, 1: 4001})
        self.assertNotIn("steps:1", detail["cursor"])
        self.assertEqual(list(registry.steps(ref)["ranks"]), ["0"])
        self.assertEqual(list(registry.rank_epochs(ref)["ranks"]), ["0"])


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

    def test_recover_keeps_the_queue_but_never_starts_it_at_boot(self):
        jobs_root = self.root / "jobs"
        jobs_root.mkdir()
        for job_id, created in (("before-boot", psutil.boot_time() - 60), ("after-boot", time.time())):
            (jobs_root / f"{job_id}.job.json").write_text(json.dumps({
                "id": job_id, "type": "single", "label": job_id, "task": "t", "status": "pending",
                "created_unix_s": created, "pid": None, "pid_create_time": None, "stop_requested": False,
                "run_dir": str(jobs_root / job_id), "log_path": str(jobs_root / f"{job_id}.log"),
                "command": _py("print('trained')")}), encoding="utf-8")
        manager = self.manager()
        self.assertTrue(_wait(self.finished(manager, "after-boot")))
        self.assertEqual(manager.get("after-boot")["status"], "succeeded")
        job = manager.get("before-boot")
        self.assertEqual((job["status"], job["pid"]), ("failed", None))
        self.assertIn("机器重启", job["error"])


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

    def test_framework_archive_exposes_measurements_and_protocol(self):
        folder = self.root / "docs/experiments/semester_2026_fall/frameworks/resnet-test"
        folder.mkdir(parents=True)
        run = {"framework": "myflows", "seed": 1, "mode": "quality", "status": "succeeded",
               "samples_per_s": 128., "epochs": [{"epoch": 1, "val_angle_mae": .05}], "environment": {"unused": 1}}
        (folder/"summary.json").write_text(json.dumps({"task": "framework_resnet18", "protocol": {"batch": 8},
                                                       "runs": [run, {**run, "status": "failed"}]}), encoding="utf-8")
        item, = self.client.get("/api/framework-comparisons").json()
        self.assertEqual((item["id"], item["protocol"]["batch"]), ("resnet-test", 8))
        row, = item["runs"]
        self.assertEqual((row["samples_per_s"], row["curve"]), (128., [[1, .05]]))
        self.assertNotIn("environment", row)

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

    def make_closed_loop(self, name="eval-001", *, frames=True):
        directory = self.root / "docs" / "experiments" / "semester_2026_fall" / "closed_loop" / name
        directory.mkdir(parents=True)
        scenarios = [{"id": f"reset-{i:02d}", "seed": 100 + i, "duration_s": 30.0} for i in (1, 2)]
        (directory / "manifest.json").write_text(json.dumps({
            "protocol": {"environment": "donkey-test-v0", "fixed_throttle": 0.2, "scenarios": scenarios,
                         "rules": {"cte_threshold": 2.0, "consecutive_frames": 3}},
            "models": [{"id": "base", "path": "models/base", "offline_angle_mae": 0.02}, {"id": "cand", "path": "m/c"}],
            "device": "cpu", "session": {"start_position_control": "simulator default reset"}}), encoding="utf-8")
        (directory / "cand-offline.json").write_text(json.dumps({"results": [{"angle_mae": 0.03}]}), encoding="utf-8")
        stat = lambda mean, worst: {"mean": mean, "worst": worst, "n": 2}
        (directory / "result.json").write_text(json.dumps({
            "status": "succeeded", "baseline": "base", "fixed_throttle": 0.2,
            "models": {
                "base": {"episodes": 2, "collisions": 0, "time_limits": 2,
                         "metrics": {"mean_abs_cte": stat(0.5, 0.6), "survival_s": stat(30.0, 30.0)},
                         "paired_difference_from_baseline": {"mean_abs_cte": {"mean": 0.0, "std": 0.0, "n": 2}}},
                "cand": {"episodes": 2, "collisions": 1, "time_limits": 1,
                         "metrics": {"mean_abs_cte": stat(1.5, float("nan")), "survival_s": stat(20.0, 10.0)},
                         "paired_difference_from_baseline": {"mean_abs_cte": {"mean": 1.0, "std": 0.5, "n": 2}}}}},
        ).replace("NaN", "null"), encoding="utf-8")
        episodes = []
        for model, cte, survival in (("base", 0.5, 30.0), ("cand", 1.5, 20.0)):
            for scenario in scenarios:
                episodes.append({"model": model, "scenario": scenario["id"], "seed": scenario["seed"], "frames": 6,
                                 "mean_abs_cte": cte, "max_abs_cte": cte * 2, "excursions": int(cte > 1),
                                 "survival_s": survival, "termination": "time_limit", "private_note": "not served"})
        _write_jsonl(directory / "episodes.jsonl", episodes)
        if frames:
            rows = []
            for model, scale in (("base", 0.5), ("cand", 1.0)):
                for i in range(40):
                    cte = scale * (3.0 if 20 <= i < 30 else 1.0)  # cand: frames 20..29 are beyond the threshold
                    rows.append({"model": model, "scenario": "reset-01", "elapsed_s": i * 0.05, "cte": cte,
                                 "steering": 0.1, "speed": 2.0, "inference_ms": 5.0, "pos": [float(i), 0.5, 2.0 * i],
                                 "hit": "wall" if model == "cand" and i == 39 else "none"})
            _write_jsonl(directory / "frames.jsonl", rows)
            with (directory / "frames.jsonl").open("a", encoding="utf-8") as f:
                f.write('not json\n{"model": "cand", "scenario": "reset-01", "elapsed_s": 9')  # damaged + unfinished
        return directory

    def test_closed_loop_evaluations_are_listed_compared_and_traced(self):
        self.assertEqual(self.client.get("/api/closed-loop").json(), [])
        self.make_closed_loop("eval-001")
        self.make_closed_loop("eval-002", frames=False)
        (self.root / "docs/experiments/semester_2026_fall/closed_loop/no-result").mkdir()
        listed = self.client.get("/api/closed-loop").json()
        self.assertEqual([e["id"] for e in listed], ["eval-002", "eval-001"])
        first = listed[1]
        self.assertEqual((first["baseline"], first["scenarios"], first["duration_s"], first["has_frames"]),
                         ("base", 2, 30.0, True))
        self.assertFalse(listed[0]["has_frames"])
        models = {m["id"]: m for m in first["models"]}
        self.assertEqual((models["base"]["baseline"], models["base"]["offline_angle_mae"], models["base"]["paired"]),
                         (True, 0.02, None))
        self.assertEqual(models["cand"]["offline_angle_mae"], 0.03, "read from cand-offline.json")
        self.assertEqual(models["cand"]["metrics"]["mean_abs_cte"], {"mean": 1.5, "worst": None})
        self.assertEqual(models["cand"]["paired"]["mean_abs_cte"], {"mean": 1.0, "std": 0.5, "n": 2})
        self.assertIsNone(models["cand"]["metrics"]["excursions"]["mean"], "a metric the result lacks is empty")

        detail = self.client.get("/api/closed-loop/eval-001").json()
        self.assertEqual(len(detail["episodes"]), 4)
        self.assertNotIn("private_note", detail["episodes"][0])
        cand = next(e for e in detail["episodes"] if e["model"] == "cand" and e["scenario"] == "reset-02")
        self.assertEqual(cand["difference"]["mean_abs_cte"], 1.0)
        self.assertEqual(cand["difference"]["survival_s"], -10.0)
        self.assertIsNone(next(e for e in detail["episodes"] if e["model"] == "base")["difference"])
        self.assertIn("survival_s", detail["higher_is_better"])

        trace = self.client.get("/api/closed-loop/eval-001/trace", params={"model": "cand", "scenario": "reset-01"}).json()
        self.assertEqual((trace["frames"], trace["points"], trace["outside_frames"], trace["collision_frames"]),
                         (40, 40, 10, 1), "the damaged and the unfinished line are not frames")
        self.assertEqual(trace["series"]["x"][:2], [0.0, 1.0])
        self.assertEqual(trace["series"]["z"][1], 2.0)
        self.assertEqual(sum(trace["series"]["outside"]), 10)
        thin = self.client.get("/api/closed-loop/eval-001/trace",
                               params={"model": "cand", "scenario": "reset-01", "max_points": 4}).json()
        self.assertLess(thin["points"], 12)
        kept = thin["series"]["elapsed_s"]
        for wanted in (20 * 0.05, 29 * 0.05, 39 * 0.05):  # both ends of the excursion and the collision frame
            self.assertTrue(any(abs(t - wanted) < 1e-9 for t in kept), (wanted, kept))
        base = self.client.get("/api/closed-loop/eval-001/trace", params={"model": "base", "scenario": "reset-01"}).json()
        self.assertEqual((base["outside_frames"], base["collision_frames"]), (0, 0))

        get = self.client.get
        self.assertEqual(get("/api/closed-loop/eval-001/trace", params={"model": "cand", "scenario": "nope"}).status_code, 404)
        self.assertEqual(get("/api/closed-loop/eval-002/trace", params={"model": "cand", "scenario": "reset-01"}).status_code, 404)
        self.assertEqual(get("/api/closed-loop/eval-001/trace", params={"model": "cand"}).status_code, 422)
        self.assertEqual(get("/api/closed-loop/eval-001/trace",
                             params={"model": "cand", "scenario": "reset-01", "max_points": 999999}).status_code, 422)
        (self.root / "docs" / "secret").mkdir()
        (self.root / "docs" / "secret" / "result.json").write_text("{}", encoding="utf-8")
        for bad in ("missing", "no-result", "%2E%2E", "..%2F..%2Fsecret", "..%5C..%5Csecret", ".hidden", "a%20b",
                    "x" * 101):
            self.assertEqual(get(f"/api/closed-loop/{bad}").status_code, 404, bad)
            self.assertEqual(get(f"/api/closed-loop/{bad}/trace", params={"model": "cand", "scenario": "reset-01"})
                             .status_code, 404, bad)

    def test_a_udacity_evaluation_shows_target_speed_failures_and_no_top_view(self):
        directory = self.make_closed_loop("udacity-001", frames=False)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        manifest["protocol"]["scenarios"][0]["speed_mph"] = 9.
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        result.update(fixed_throttle=None, target_speed_mph=12.0, cte_source="camera_estimate", distance_unit="m")
        result["models"]["cand"]["failures"] = 1
        (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
        rows = [{"model": "cand", "scenario": "reset-01", "elapsed_s": i * 0.05, "cte": 0.1, "steering": 0.0,
                 "speed": 12.0, "inference_ms": 4.0, "odometer": i * 0.27} for i in range(10)]
        _write_jsonl(directory / "frames.jsonl", rows)  # the link carries no position
        detail = self.client.get("/api/closed-loop/udacity-001").json()
        self.assertEqual((detail["fixed_throttle"], detail["target_speed_mph"], detail["cte_source"], detail["distance_unit"]),
                         (None, 12.0, "camera_estimate", "m"))
        self.assertEqual({m["id"]: m["failures"] for m in detail["models"]}, {"base": None, "cand": 1})
        self.assertEqual([row["target_speed_mph"] for row in detail["episodes"]], [9., 12., 9., 12.])
        trace = self.client.get("/api/closed-loop/udacity-001/trace", params={"model": "cand", "scenario": "reset-01"}).json()
        self.assertFalse(trace["has_positions"])
        self.assertEqual(set(trace["series"]["x"]), {None})
        donkey = self.make_closed_loop("eval-001")
        self.assertTrue(donkey.is_dir())
        self.assertTrue(self.client.get("/api/closed-loop/eval-001/trace",
                                        params={"model": "cand", "scenario": "reset-01"}).json()["has_positions"])

    def test_requests_from_other_sites_and_rebound_hosts_are_refused(self):
        # A page of another site (or a DNS name rebound to 127.0.0.1) must not reach the API that starts jobs.
        get = self.client.get
        self.assertEqual(get("/api/health").status_code, 200)
        self.assertEqual(get("/api/health", headers={"host": "127.0.0.1:8790"}).status_code, 200)
        self.assertEqual(get("/api/health", headers={"host": "localhost:8790"}).status_code, 200)
        self.assertEqual(get("/api/health", headers={"host": "[::1]:8790"}).status_code, 200)
        for host in ("evil.example", "evil.example:8790", "127.0.0.1.evil.example", "localhost@evil.example", ""):
            response = get("/api/health", headers={"host": host})
            self.assertEqual(response.status_code, 400, host)
            self.assertEqual(get("/", headers={"host": host}).status_code, 400, host)
        self.assertEqual(get("/api/health", headers={"origin": "http://testserver"}).status_code, 200)
        for headers in ({"origin": "http://evil.example"}, {"origin": "null"}, {"sec-fetch-site": "cross-site"},
                        {"origin": "https://testserver"}, {"origin": "http://testserver:81"}):
            self.assertEqual(get("/api/health", headers=headers).status_code, 403, headers)
            with mock.patch.object(self.app.state.jobs, "_launch") as launch:
                posted = self.client.post("/api/jobs", headers=headers, json={"type": "single", "args": {"epochs": 1}})
            self.assertEqual(posted.status_code, 403, headers)
            launch.assert_not_called()
        self.assertEqual(self.app.state.jobs.list(), [])
        # Same-site requests of the page itself and non-browser clients (no Origin) pass.
        self.assertEqual(get("/api/health", headers={"sec-fetch-site": "same-origin"}).status_code, 200)
        self.assertEqual(get("/", headers={"sec-fetch-site": "cross-site"}).status_code, 200, "the page may be linked")

        allowed = create_app(repo_root=self.root, console_root=self.root / "console-lan", registry=self.registry,
                             sampler=_FakeSampler(), start_background=False, web_dist=self.root / "no-dist",
                             allowed_hosts=("console.lan",), enable_dsh=False)
        try:
            with TestClient(allowed) as client:
                self.assertEqual(client.get("/api/health", headers={"host": "console.lan:8790"}).status_code, 200)
                self.assertEqual(client.get("/api/health", headers={"host": "other.lan:8790"}).status_code, 400)
        finally:
            allowed.state.jobs.shutdown()

    def test_malformed_submissions_are_refused_before_anything_is_queued(self):
        post = lambda body: self.client.post("/api/jobs", json=body)
        smoke = next(p for p in self.client.get("/api/presets").json()["distributed"] if p["id"] == "synthetic-smoke")
        cases = [
            {"type": "single", "args": [1, 2]},
            {"type": "single", "args": {"epochs": 1.5}},
            {"type": "single", "args": {"epochs": True}},
            {"type": "single", "args": {"augment": "yes"}},
            {"type": "single", "args": {"lr": "nan"}},
            {"type": "single", "args": {"lr": 1e400}},
            {"type": "single", "args": {"epochs": 1}, "label": {"x": 1}},
            {"type": "single", "args": {"epochs": 1}, "label": "x" * 201},
            {"type": "distributed", "config": [smoke["config"]]},
            {"type": "distributed", "config": dict(smoke["config"], task=["synthetic"])},
            {"type": "distributed", "config": dict(smoke["config"], learning_rate="inf")},
            {"type": "distributed", "config": dict(smoke["config"], learning_rate=None)},
            {"type": "distributed", "config": dict(smoke["config"], train_workers=1e400)},
            {"type": "distributed", "config": dict(smoke["config"], mode="ps", transport="grpc_proto", train_workers=2,
                                                   shard_sizes=[16.0, 16.0])},
        ]
        with mock.patch.object(self.app.state.jobs, "_launch") as launch:
            for body in cases:
                # 1e400 is not JSON; send it the way a hand-written client could.
                raw = json.dumps(body).replace("Infinity", "1e400")
                response = self.client.post("/api/jobs", content=raw, headers={"content-type": "application/json"})
                self.assertEqual(response.status_code, 422, (body, response.text))
            launch.assert_not_called()
            ok = post({"type": "single", "args": {"epochs": 2.0, "batch": 2, "augment": True}, "label": "整数值的浮点数"})
            self.assertEqual(ok.status_code, 200, ok.text)
            self.assertEqual(ok.json()["config"]["args"]["epochs"], 2)
        self.assertEqual(len(self.app.state.jobs.list()), 1)

    def test_conv_backend_options_and_validation(self):
        presets = self.client.get("/api/presets").json()
        options = {o["value"]: o for o in presets["conv_backends"]}
        self.assertEqual([v for v, o in options.items() if o["default"]], ["cuda_native_register", "numpy"])
        self.assertEqual(options["cuda_native_cublas"]["device"], "cuda")
        self.assertTrue(all(p["args"]["conv_backend"] == "auto" for p in presets["single"]))

        smoke = next(p for p in presets["distributed"] if p["id"] == "synthetic-smoke")
        base = dict(smoke["config"], mode="single", transport="none", train_workers=1)
        for config, message in (
            (dict(base, conv_backend="cuda_native_magic"), "conv_backend 只能是"),
            (dict(base, device="cpu", conv_backend="cuda_native_register"), "不能用于 device=cpu"),
        ):
            response = self.client.post("/api/jobs", json={"type": "distributed", "config": config})
            self.assertEqual(response.status_code, 422)
            self.assertIn(message, response.json()["detail"])
        for args, message in (
            ({"device": "cpu", "conv_backend": "cuda_native_cublas"}, "不能用于 device=cpu"),
            ({"dtype": "float64", "conv_backend": "cuda_native_register"}, "只支持 float32"),
        ):
            response = self.client.post("/api/jobs", json={"type": "single", "args": {"epochs": 1, "batch": 2, **args}})
            self.assertEqual(response.status_code, 422)
            self.assertIn(message, response.json()["detail"])

        with mock.patch.object(self.app.state.jobs, "_launch"):
            single = self.client.post("/api/jobs", json={"type": "single", "args": {
                "epochs": 1, "batch": 2, "device": "cuda", "conv_backend": "cuda_native_cublas"}}).json()
            dist = self.client.post("/api/jobs", json={"type": "distributed", "config": dict(
                base, device="cuda", conv_backend="cuda_native_register")}).json()
        command = single["command"]
        self.assertEqual(command[command.index("--conv-backend") + 1], "cuda_native_cublas")
        self.assertEqual(dist["status"], "pending")

    def test_formal_training_parallel_args(self):
        presets = self.client.get("/api/presets").json()
        self.assertIn("current", presets["native_library"])
        ring = next(p for p in presets["single"] if p["args"]["distributed"] == "ring")
        self.assertEqual(ring["args"]["workers"], 2)
        for args, message in (
            ({"distributed": "mesh"}, "distributed 只能是"),
            ({"distributed": "ring", "transport": "socket_json"}, "ring 只支持"),
            ({"distributed": "ps", "workers": 4, "batch": 2}, "必须不小于 Worker 数"),
            ({"distributed": "ps", "dtype": "float64"}, "只支持 float32"),
            ({"distributed": "ps", "workers": 99}, "Worker 数必须在"),
            ({"distributed": "ps", "transport": "socket_json"}, "宽度不超过 32"),
            ({"arch": "resnet34", "base_width": 32, "distributed": "ps", "transport": "socket_json"}, "宽度不超过 24"),
        ):
            response = self.client.post("/api/jobs", json={"type": "single", "args": {"epochs": 1, "batch": 8, **args}})
            self.assertEqual(response.status_code, 422, args)
            self.assertIn(message, response.json()["detail"])
        with mock.patch.object(self.app.state.jobs, "_launch"):
            job = self.client.post("/api/jobs", json={"type": "single", "args": {
                "epochs": 1, "batch": 8, "distributed": "ring", "workers": 2, "export_onnx": True}}).json()
            plain = self.client.post("/api/jobs", json={"type": "single", "args": {
                "epochs": 1, "batch": 8, "workers": 4, "transport": "grpc_proto"}}).json()
            r34_socket = self.client.post("/api/jobs", json={"type": "single", "args": {
                "arch": "resnet34", "base_width": 24, "epochs": 1, "batch": 8,
                "distributed": "ps", "workers": 2, "transport": "socket_json"}})
            r34_grpc = self.client.post("/api/jobs", json={"type": "single", "args": {
                "arch": "resnet34", "base_width": 32, "epochs": 1, "batch": 8,
                "distributed": "ps", "workers": 2, "transport": "grpc_proto"}})
            self.assertEqual(r34_socket.status_code, 200, r34_socket.text)
            self.assertEqual(r34_grpc.status_code, 200, r34_grpc.text)
        command = job["command"]
        self.assertEqual([command[command.index(flag) + 1] for flag in ("--distributed", "--workers", "--transport")],
                         ["ring", "2", "grpc_proto"])
        self.assertIn("--export-onnx", command)
        self.assertIn("Ring AllReduce · gRPC · 2w", job["label"])
        # Without a parallel mode the worker settings are dropped instead of reaching apps.train.
        self.assertEqual(plain["command"][plain["command"].index("--distributed") + 1], "none")
        self.assertNotIn("--workers", plain["command"])
        self.assertIn("单进程", plain["label"])


if __name__ == "__main__":
    unittest.main()


class DatasetChoiceTest(unittest.TestCase):
    """A job may name the default DonkeyCar set or a collected one, and nothing else."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "mycar" / "data").mkdir(parents=True)
        collected = self.root / "runs" / "datasets" / "udacity-lake-001"
        (collected / "images").mkdir(parents=True)
        import cv2
        import numpy as np

        for index in range(3):
            cv2.imwrite(str(collected / "images" / f"{index}_0.1000.jpg"), np.full((8, 16, 3), 90, np.uint8))
        (collected / "dataset.json").write_text(json.dumps({
            "id": "udacity-lake-001", "environment": "udacity-lake", "simulator": "udacity-term1", "frames": 3,
            "image": {"width": 320, "height": 160}, "image_size": "80x160", "created_unix_s": 10.,
            "source": "自动采集：前视图像巡线控制器", "summary": {"accepted": True}}), encoding="utf-8")
        self.app = create_app(repo_root=self.root, console_root=self.root / "console", registry=make_registry(self.root),
                              sampler=_FakeSampler(), start_background=False, web_dist=self.root / "no-dist")
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.app.state.jobs.shutdown()
        self._tmp.cleanup()

    def test_validation_keeps_the_default_implicit_and_refuses_unknown_places(self):
        from apps.console.presets import ValidationError, single_cli_args, single_label, validate_single

        self.assertNotIn("data", validate_single({"epochs": 1}, self.root))
        self.assertNotIn("data", validate_single({"epochs": 1, "data": "mycar/data"}, self.root))
        for spelling in ("udacity-lake-001", "runs/datasets/udacity-lake-001", "runs\\datasets\\udacity-lake-001\\"):
            args = validate_single({"epochs": 1, "data": spelling, "image_size": "80x160"}, self.root)
            self.assertEqual(args["data"], "runs/datasets/udacity-lake-001")
        self.assertIn("udacity-lake-001", single_label(args))
        cli = single_cli_args(args)
        self.assertEqual(cli[cli.index("--data") + 1], "runs/datasets/udacity-lake-001")
        for bad in ("../outside", "C:/Windows", "/etc/passwd", "runs/datasets/../../mycar", "docs", 7):
            with self.assertRaises(ValidationError, msg=bad):
                validate_single({"data": bad}, self.root)
        with self.assertRaisesRegex(ValidationError, "不存在"):
            validate_single({"data": "never-collected"}, self.root)

    def test_an_active_collection_is_visible_but_cannot_start_training(self):
        meta = self.root / "runs/datasets/udacity-lake-001/dataset.json"
        value = json.loads(meta.read_text(encoding="utf-8"))
        value["status"] = "collecting"
        meta.write_text(json.dumps(value), encoding="utf-8")
        listed = self.client.get("/api/datasets").json()
        self.assertEqual((listed[1]["status"], listed[1]["ready"]), ("collecting", False))
        refused = self.client.post("/api/jobs", json={"type": "single", "args": {"data": "udacity-lake-001"}})
        self.assertEqual(refused.status_code, 422)
        self.assertEqual(self.client.get("/api/jobs").json(), [])

    def test_interrupted_and_rejected_sets_need_recovery_before_automatic_training(self):
        meta = self.root / "runs/datasets/udacity-lake-001/dataset.json"
        original = json.loads(meta.read_text(encoding="utf-8"))
        for status, accepted in (("interrupted", True), ("complete", False)):
            with self.subTest(status=status, accepted=accepted):
                meta.write_text(json.dumps({**original, "status": status, "summary": {"accepted": accepted}}), encoding="utf-8")
                self.assertFalse(self.client.get("/api/datasets").json()[1]["ready"])
                refused = self.client.post("/api/jobs", json={"type": "single", "args": {"data": "udacity-lake-001"}})
                self.assertEqual(refused.status_code, 422)
                self.assertEqual(self.client.get("/api/jobs").json(), [])

    def test_the_api_lists_the_sets_and_a_job_on_a_collected_set_passes_it_on(self):
        listed = self.client.get("/api/datasets").json()
        self.assertEqual([row["data"] for row in listed], ["mycar/data", "runs/datasets/udacity-lake-001"])
        self.assertEqual((listed[0]["default"], listed[0]["ready"]), (True, False))  # the default folder holds no frames here
        self.assertEqual((listed[1]["frames"], listed[1]["image_size"], listed[1]["environment"], listed[1]["ready"]),
                         (3, "80x160", "udacity-lake", True))
        self.assertEqual(self.client.get("/api/presets").json()["datasets"], listed)
        refused = self.client.post("/api/jobs", json={"type": "single", "args": {"data": "../../etc"}})
        self.assertEqual(refused.status_code, 422)
        self.assertEqual(self.client.get("/api/jobs").json(), [])


class ConsoleHardeningTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_kill_tree_ignores_pid_zero_and_tolerates_access_denied(self):
        from apps.console.jobs import kill_tree

        with mock.patch("apps.console.jobs.psutil.Process") as proc_cls:
            self.assertEqual(kill_tree(0), [])
        proc_cls.assert_not_called()

        with mock.patch("apps.console.jobs.psutil.Process", side_effect=psutil.AccessDenied(pid=1234)):
            self.assertEqual(kill_tree(1234), [])

        denied = mock.MagicMock()
        denied.pid = 4321
        denied.terminate.side_effect = psutil.AccessDenied(pid=4321)
        denied.kill.side_effect = psutil.AccessDenied(pid=4321)
        with mock.patch("apps.console.jobs.psutil.wait_procs", return_value=([], [])):
            self.assertEqual(kill_tree(0, descendants=[denied]), [])

    def test_recover_restarts_resource_recorder_and_finalize_recognizes_result_and_evaluation(self):
        import subprocess

        started = []
        stopped = []

        class _FakeRecorder:
            def __init__(self, job_id):
                self.job_id = job_id

            def stop(self):
                stopped.append(self.job_id)

        def factory(job, pid):
            started.append((job["id"], pid))
            return _FakeRecorder(job["id"])

        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        try:
            jobs_root = self.root / "jobs"
            jobs_root.mkdir(parents=True, exist_ok=True)
            live_job = {
                "id": "live-adopted", "type": "single", "label": "live", "task": "t",
                "status": "running", "created_unix_s": time.time(), "pid": proc.pid,
                "pid_create_time": psutil.Process(proc.pid).create_time(),
                "stop_requested": False, "record_resources": True,
                "run_dir": str(jobs_root / "live-adopted"), "log_path": str(jobs_root / "live.log"), "command": [],
            }
            (jobs_root / "live-adopted.job.json").write_text(json.dumps(live_job), encoding="utf-8")
            manager = JobManager(jobs_root, self.root, poll_interval_s=0.05, resource_recorder_factory=factory)
            try:
                self.assertEqual(started, [("live-adopted", proc.pid)])
                self.assertIn("live-adopted", manager._recorders)
            finally:
                manager.shutdown()
            self.assertEqual(stopped, ["live-adopted"])
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)

        mgr = JobManager(self.root / "jobs2", self.root, poll_interval_s=0.5)
        try:
            for name, files, code, expected in (
                ("res-ok", {"result.json": {"status": "succeeded"}}, None, "succeeded"),
                ("eval-ok", {"evaluation.json": {"results": [{"key": "m"}], "failures": []}}, None, "succeeded"),
                ("eval-fail", {"evaluation.json": {"results": [], "failures": [{"key": "m"}]}}, 0, "failed"),
            ):
                run_dir = self.root / "jobs2" / name
                run_dir.mkdir(parents=True, exist_ok=True)
                for fname, payload in files.items():
                    (run_dir / fname).write_text(json.dumps(payload), encoding="utf-8")
                job = {"id": name, "run_dir": str(run_dir), "pid": None, "stop_requested": False}
                mgr._finalize(job, code)
                self.assertEqual(job["status"], expected, (name, job))
        finally:
            mgr.shutdown()

    def test_single_run_summary_val_angle_mae_negative_seed_and_local_hosts(self):
        from apps.console.presets import ValidationError, validate_single
        from apps.console.server import _local_interface_hosts

        console_root = self.root / "console"
        run_dir = console_root / "job-single"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({
            "task": "donkey_resnet18_single", "mode": "single", "args": {"epochs": 1, "batch": 4},
        }), encoding="utf-8")
        (run_dir / "results.json").write_text(json.dumps({
            "status": "passed", "last_mean_loss": 0.05, "best_loss": 0.04,
            "val_loss": 0.045, "val_angle_mae": 0.09, "steps": 10,
            "samples_per_s": 50.0, "epoch_train_wall_s": 0.2,
        }), encoding="utf-8")
        registry = make_registry(self.root, jobs=[{
            "id": "job-single", "type": "single", "label": "single", "task": "donkey_resnet18_single",
            "status": "succeeded", "run_dir": str(run_dir), "created_unix_s": 1.0,
        }])
        ref = registry.resolve("console/job-single")
        summary = registry.summary(ref)
        self.assertEqual(summary["final"]["val_angle_mae"], 0.09)
        self.assertEqual(summary["final"]["val"], {"loss": 0.045, "angle_mae": 0.09})

        with self.assertRaisesRegex(ValidationError, "seed"):
            validate_single({"epochs": 1, "seed": -1}, self.root)

        hosts = _local_interface_hosts()
        self.assertIsInstance(hosts, tuple)
        self.assertTrue(all(isinstance(h, str) and "%" not in h for h in hosts))
