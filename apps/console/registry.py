"""Discover runs from archives, console jobs and TensorBoard logs; serve cached, incremental data."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .readers import (
    JsonlTail,
    columns,
    downsample_indices,
    read_csv_rows,
    read_json,
    read_tensorboard,
)

STEP_FIELDS = (
    "rank", "epoch", "step", "start_s", "end_s", "step_wall_s", "data_s", "gradient_sync_s", "sync_update_s",
    "update_confirm_s", "digest_s", "n_samples", "loss", "input_h2d_wall_s", "forward_wall_s", "forward_gpu_s",
    "backward_wall_s", "backward_gpu_s", "optimizer_wall_s", "optimizer_gpu_s", "gradient_d2h_s",
    "gradient_to_device_wall_s", "metrics_wall_s", "request_bytes", "response_bytes", "outgoing_gradient_bytes",
    "rpc_observed_s", "encode_s", "decode_s",
)
EPOCH_FIELDS = (
    "rank", "epoch", "steps", "start_s", "end_s", "epoch_train_wall_s", "samples", "step_wall_s", "data_s",
    "gradient_sync_s", "forward_wall_s", "backward_wall_s", "optimizer_wall_s", "forward_gpu_s", "backward_gpu_s",
    "optimizer_gpu_s", "request_bytes", "response_bytes", "outgoing_gradient_bytes",
)
RESOURCE_FIELDS = (
    "monotonic_s", "unix_s", "cpu_utilization_pct", "ram_used_bytes", "gpu_utilization_pct", "gpu_memory_mib",
    "gpu_temperature_c", "gpu_power_w", "process_rss_bytes", "sampling_error",
)
ACTIVE_STATUSES = ("pending", "running", "stopping")
FINAL_SPLITS = ("val", "test", "train")


@dataclass
class Source:
    key: str
    label: str
    kind: str  # console | archive | tensorboard
    root: Path


@dataclass
class RunRef:
    run_id: str
    source: Source
    label: str
    path: Path
    job: dict | None = None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _small_split(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    return {k: v for k, v in value.items() if isinstance(v, (int, float)) and k != "evaluation_s"}


class RunData:
    """Cached, incrementally refreshed data of one run directory."""

    def __init__(self, ref: RunRef, kind: str):
        self.ref = ref
        self.path = ref.path
        self.kind = kind
        self.lock = threading.Lock()
        self.rank_steps: dict[int, JsonlTail] = {}
        self.rank_epochs: dict[int, JsonlTail] = {}
        self.resources = JsonlTail(self.path / "resources.jsonl", RESOURCE_FIELDS)
        self.metrics = JsonlTail(self.path / "metrics.jsonl") if kind == "single" else None
        self.rank_pids: dict[int, int] = {}
        self.tb: dict | None = None

    def _discover_ranks(self):
        for rank_dir in self.path.glob("rank-*"):
            try:
                rank = int(rank_dir.name.split("-", 1)[1])
            except ValueError:
                continue
            if rank not in self.rank_steps:
                self.rank_steps[rank] = JsonlTail(rank_dir / "steps.jsonl", STEP_FIELDS)
                self.rank_epochs[rank] = JsonlTail(rank_dir / "epochs.jsonl", EPOCH_FIELDS, nested=("val",))
            if rank not in self.rank_pids:
                prepared = read_json(rank_dir / "prepare.json")
                if prepared and prepared.get("pid"):
                    self.rank_pids[rank] = int(prepared["pid"])

    def poll(self):
        with self.lock:
            if self.kind == "tensorboard":
                if self.tb is None:
                    self.tb = read_tensorboard(self.path)
                return
            if self.kind == "distributed":
                self._discover_ranks()
                for tail in (*self.rank_steps.values(), *self.rank_epochs.values()):
                    tail.poll()
            if self.metrics is not None:
                self.metrics.poll()
            self.resources.poll()

    def ranks(self) -> list[int]:
        return sorted(self.rank_steps)

    def origin(self) -> float | None:
        """perf_counter origin shared by steps and resources (system-wide clock on Windows)."""
        candidates = []
        if self.resources.rows:
            candidates.append(self.resources.rows[0].get("monotonic_s"))
        for tail in self.rank_steps.values():
            if tail.rows:
                candidates.append(tail.rows[0].get("start_s"))
        if self.metrics is not None and self.metrics.rows:
            candidates.append(self.metrics.rows[0].get("monotonic_s"))
        candidates = [c for c in candidates if c is not None]
        return min(candidates) if candidates else None

    def cursor(self) -> dict:
        cur = {"resources": len(self.resources.rows)}
        for rank, tail in self.rank_steps.items():
            cur[f"steps:{rank}"] = len(tail.rows)
        for rank, tail in self.rank_epochs.items():
            cur[f"epochs:{rank}"] = len(tail.rows)
        if self.metrics is not None:
            cur["metrics"] = len(self.metrics.rows)
        return cur

    def rank_status(self, active: bool = False) -> list[dict]:
        now_pc = time.perf_counter()
        out = []
        for rank in self.ranks():
            rows = self.rank_steps[rank].rows
            last = rows[-1] if rows else None
            epochs = self.rank_epochs[rank].rows
            item = {"rank": rank, "pid": self.rank_pids.get(rank), "steps_done": len(rows),
                    "epochs_done": len(epochs)}
            if last:
                wall = last.get("step_wall_s") or 0.0
                sync = last.get("gradient_sync_s") or 0.0
                item.update(epoch=last.get("epoch"), step=last.get("step"), loss=last.get("loss"),
                            step_wall_s=wall, gradient_sync_s=sync,
                            sync_fraction=(sync / wall) if wall else None,
                            seconds_since_last=max(0.0, now_pc - last["end_s"]) if active and last.get("end_s") else None)
            out.append(item)
        return out


class RunRegistry:
    def __init__(self, repo_root: Path, *, console_root: Path, archive_roots: list[Path], tensorboard_root: Path | None,
                 jobs_provider: Callable[[], list[dict]] | None = None, cache_size: int = 8):
        self.repo_root = Path(repo_root)
        self.console = Source("console", "控制台任务", "console", Path(console_root))
        self.archives = [Source(f"archive-{p.name}", f"归档 {p.name}", "archive", Path(p)) for p in archive_roots]
        self.tensorboard = (Source("tensorboard", "TensorBoard 单进程", "tensorboard", Path(tensorboard_root))
                            if tensorboard_root else None)
        self.jobs_provider = jobs_provider or (list)
        self._summary_cache: dict[str, tuple[tuple, dict]] = {}
        self._data: OrderedDict[str, RunData] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    @classmethod
    def default(cls, repo_root: Path, jobs_provider=None) -> RunRegistry:
        repo_root = Path(repo_root)
        dist = repo_root / "docs" / "experiments" / "semester_2026_fall" / "distributed_gpu"
        archives = sorted((p for p in dist.iterdir() if (p / "run_index.json").exists()), reverse=True) if dist.exists() else []
        tb = repo_root / "mycar" / "logs" / "tensorboard"
        return cls(repo_root, console_root=repo_root / "runs" / "console", archive_roots=archives,
                   tensorboard_root=tb if tb.exists() else None, jobs_provider=jobs_provider)

    # ---- sources and refs -------------------------------------------------
    def sources(self) -> list[dict]:
        items = [self.console, *self.archives] + ([self.tensorboard] if self.tensorboard else [])
        return [{"key": s.key, "label": s.label, "kind": s.kind, "root": self._rel(s.root)} for s in items]

    def _rel(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.repo_root.resolve())).replace("\\", "/")
        except ValueError:
            return str(path)

    def _refs(self, source_key: str | None = None) -> list[RunRef]:
        refs: list[RunRef] = []
        if source_key in (None, "console"):
            for job in self.jobs_provider():
                refs.append(RunRef(f"console/{job['id']}", self.console, job.get("label") or job["id"],
                                   Path(job["run_dir"]), job))
        for source in self.archives:
            if source_key not in (None, source.key):
                continue
            index = read_json(source.root / "run_index.json", {}) or {}
            for label, rel in index.items():
                path = source.root / rel
                if path.is_dir():
                    refs.append(RunRef(f"{source.key}/{rel}".replace("\\", "/"), source, label, path))
        if self.tensorboard and source_key in (None, "tensorboard"):
            for path in sorted(self.tensorboard.root.iterdir(), reverse=True):
                if path.is_dir() and any(path.glob("events.out.tfevents*")):
                    refs.append(RunRef(f"tensorboard/{path.name}", self.tensorboard, path.name, path))
        return refs

    def resolve(self, run_id: str) -> RunRef | None:
        run_id = run_id.strip("/")
        head, _, rest = run_id.partition("/")
        if head == "console":
            job = next((j for j in self.jobs_provider() if j["id"] == rest), None)
            return RunRef(run_id, self.console, job.get("label") or rest, Path(job["run_dir"]), job) if job else None
        if head == "tensorboard" and self.tensorboard:
            path = (self.tensorboard.root / rest).resolve()
            if path.parent == self.tensorboard.root.resolve() and path.is_dir():
                return RunRef(run_id, self.tensorboard, rest, path)
            return None
        source = next((s for s in self.archives if s.key == head), None)
        if source is None:
            return None
        path = (source.root / rest).resolve()
        root = source.root.resolve()
        if root not in path.parents or not (path / "config.json").exists():
            return None
        index = read_json(source.root / "run_index.json", {}) or {}
        label = next((k for k, v in index.items() if v.replace("\\", "/") == rest), rest)
        return RunRef(run_id, source, label, path)

    # ---- summaries -----------------------------------------------------------
    @staticmethod
    def _kind(ref: RunRef, config: dict) -> str:
        if ref.source.kind == "tensorboard":
            return "tensorboard"
        if (ref.job or {}).get("type") == "single" or config.get("kind") == "single":
            return "single"
        return "distributed"

    def summary(self, ref: RunRef) -> dict:
        path = ref.path
        stamp = (tuple(_mtime(path / n) for n in ("config.json", "results.json", "failure.json")),
                 tuple(sorted((ref.job or {}).items(), key=lambda kv: kv[0])) if ref.job else None)
        stamp = (stamp[0], repr(stamp[1]))
        cached = self._summary_cache.get(ref.run_id)
        if cached and cached[0] == stamp:
            return cached[1]
        summary = self._build_summary(ref)
        self._summary_cache[ref.run_id] = (stamp, summary)
        return summary

    def _build_summary(self, ref: RunRef) -> dict:
        path, job = ref.path, ref.job or {}
        if ref.source.kind == "tensorboard":
            name = path.name
            stamp = "_".join(name.rsplit("_", 2)[-2:])
            try:
                started = time.mktime(time.strptime(stamp, "%Y%m%d_%H%M%S"))
            except ValueError:
                started = None
            finished = max((_mtime(p) or 0 for p in path.glob("events.out.tfevents*")), default=None)
            return {"id": ref.run_id, "source": ref.source.key, "label": ref.label, "kind": "tensorboard",
                    "task": name.rsplit("_", 2)[0] if "_" in name else name, "mode": "single", "transport": "none",
                    "workers": 1, "status": "finished", "started_unix_s": started, "finished_unix_s": finished,
                    "elapsed_s": (finished - started) if started and finished and finished > started else None, "device": None, "final": {}, "path": self._rel(path)}
        config = read_json(path / "config.json", None) or job.get("config") or {}
        results = read_json(path / "results.json")
        kind = self._kind(ref, config)
        status = "incomplete"
        if results:
            status = results.get("status") or "finished"
        elif (path / "failure.json").exists():
            status = "failed"
        if job.get("status") in (*ACTIVE_STATUSES, "cancelled") or (job.get("status") == "failed" and not results):
            status = job["status"]
        final = {}
        elapsed = (results or {}).get("elapsed_s")
        if kind == "distributed" and results:
            fin = results.get("final") or {}
            for split in FINAL_SPLITS:
                small = _small_split(fin.get(split))
                if small:
                    final[split] = small
            epochs = results.get("epochs") or []
            if epochs:
                final["samples_per_s"] = epochs[-1].get("samples_per_s")
                final["epoch_train_wall_s"] = epochs[-1].get("epoch_train_wall_s")
        elif kind == "single" and results:
            final = {k: results.get(k) for k in ("last_mean_loss", "best_loss", "steps")}
        args = config.get("args") or {}
        started = job.get("started_unix_s") or config.get("started_unix_s") or _mtime(path / "config.json")
        finished = job.get("finished_unix_s") or _mtime(path / "results.json") or _mtime(path / "failure.json")
        if elapsed is None and started and finished and status not in ACTIVE_STATUSES:
            elapsed = finished - started
        return {
            "id": ref.run_id, "source": ref.source.key, "label": ref.label, "kind": kind,
            "task": config.get("task") or job.get("task"),
            "mode": config.get("mode", "single" if kind == "single" else None),
            "transport": config.get("transport", "none" if kind == "single" else None),
            "workers": config.get("train_workers", 1),
            "status": status, "started_unix_s": started, "finished_unix_s": finished if status not in ACTIVE_STATUSES else None,
            "elapsed_s": elapsed, "device": config.get("device") or args.get("device"),
            "optimizer": config.get("optimizer"), "global_batch": config.get("global_batch") or args.get("batch"),
            "epochs": config.get("epochs") or args.get("epochs"), "learning_rate": config.get("learning_rate") or args.get("lr"),
            "final": final, "error": (results or {}).get("error") or job.get("error"),
            "job_id": job.get("id"), "path": self._rel(path),
        }

    def list_runs(self, source: str | None = None) -> list[dict]:
        runs = []
        for ref in self._refs(source):
            try:
                runs.append(self.summary(ref))
            except Exception as exc:  # a broken archive entry must not hide the others
                runs.append({"id": ref.run_id, "source": ref.source.key, "label": ref.label, "status": "error",
                             "error": f"{type(exc).__name__}: {exc}", "path": self._rel(ref.path)})
        return runs

    # ---- cached run data ---------------------------------------------------------
    def data(self, ref: RunRef) -> RunData:
        with self._lock:
            item = self._data.get(ref.run_id)
            if item is None or item.path != ref.path:
                config = read_json(ref.path / "config.json", None) or (ref.job or {}).get("config") or {}
                item = RunData(ref, self._kind(ref, config))
                self._data[ref.run_id] = item
            else:
                item.ref = ref
            self._data.move_to_end(ref.run_id)
            while len(self._data) > self._cache_size:
                self._data.popitem(last=False)
        item.poll()
        return item

    def detail(self, ref: RunRef) -> dict:
        data = self.data(ref)
        summary = self.summary(ref)
        out = {"summary": summary, "kind": data.kind, "ranks": data.ranks(), "cursor": data.cursor()}
        if data.kind == "tensorboard":
            tb = data.tb or {}
            out.update(config=tb.get("config") or {}, epochs=tb.get("epochs", []), manifest=None,
                       results=None, scalar_tags=tb.get("scalar_tags", []))
            return out
        config = read_json(ref.path / "config.json", None) or (ref.job or {}).get("config") or {}
        manifest = read_json(ref.path / "manifest.json") or {}
        results = read_json(ref.path / "results.json")
        out["config"] = config
        out["manifest"] = {k: manifest.get(k) for k in ("platform", "python", "packages", "root_head", "myflows_head",
                                                        "measurement", "bn_statistics_sha256")} if manifest else None
        if results:
            final = results.get("final") or {}
            out["results"] = {k: results.get(k) for k in ("status", "error", "elapsed_s", "cleanup_s",
                                                          "rank_state_consistent", "exitcodes", "data_meta")}
            out["results"]["final"] = {s: _small_split(final.get(s)) for s in FINAL_SPLITS if final.get(s)}
            out["results"]["final"].update({k: final.get(k) for k in ("device", "backends", "version",
                                                                      "allocator_reserved_bytes_at_finish")})
        else:
            out["results"] = None
        if data.kind == "distributed":
            out["epochs"] = (results or {}).get("epochs") or read_csv_rows(ref.path / "epochs.csv") or self._live_epochs(data)
            out["rank_status"] = data.rank_status(active=summary.get("status") in ACTIVE_STATUSES)
            out["rank_pids"] = data.rank_pids
        else:
            out["epochs"] = [r for r in (data.metrics.rows if data.metrics else []) if r.get("kind") == "epoch"]
        out["job"] = ref.job
        return out

    @staticmethod
    def _live_epochs(data: RunData) -> list[dict]:
        by_epoch: dict[int, list[dict]] = {}
        for tail in data.rank_epochs.values():
            for row in tail.rows:
                by_epoch.setdefault(row["epoch"], []).append(row)
        rows = []
        for epoch in sorted(by_epoch):
            items = by_epoch[epoch]
            total = max(i["end_s"] for i in items) - min(i["start_s"] for i in items)
            row = {"epoch": epoch, "epoch_train_wall_s": total, "samples": sum(i.get("samples", 0) for i in items),
                   "gradient_sync_s": max(i.get("gradient_sync_s") or 0 for i in items), "ranks_reported": len(items)}
            row["samples_per_s"] = row["samples"] / total if total else None
            for item in items:
                row.update({k: v for k, v in item.items() if k.startswith("val_") and k != "val_split"})
            rows.append(row)
        return rows

    # ---- series ------------------------------------------------------------------
    def steps(self, ref: RunRef, *, ranks: list[int] | None = None, fields: list[str] | None = None,
              max_points: int | None = 3000, cursor: dict | None = None) -> dict:
        data = self.data(ref)
        origin = data.origin()
        cursor = cursor or {}
        if data.kind == "tensorboard":
            rows = (data.tb or {}).get("steps", [])
            start = int(cursor.get("steps:0", 0))
            new = rows[start:]
            idx = downsample_indices(len(new), max_points)
            t0 = rows[0]["unix_s"] if rows else None
            series = columns(new, fields or ("step", "loss", "running_loss", "step_time_ms", "data_load_ms",
                                             "train_step_ms", "samples_per_sec"), idx)
            series["t"] = [new[i]["unix_s"] - t0 for i in idx] if t0 is not None else []
            series["x"] = [new[i]["step"] for i in idx]
            return {"origin": None, "ranks": {"0": series}, "cursor": {"steps:0": len(rows)}}
        if data.kind == "single":
            rows = [r for r in data.metrics.rows if r.get("kind") == "step"]
            start = int(cursor.get("steps:0", 0))
            new = rows[start:]
            idx = downsample_indices(len(new), max_points)
            series = columns(new, fields or ("epoch", "step", "loss", "running_loss", "step_time_ms", "data_load_ms",
                                             "train_step_ms", "samples_per_sec"), idx)
            series["t"] = [new[i]["monotonic_s"] - origin for i in idx] if origin is not None else []
            series["x"] = [new[i]["step"] for i in idx]
            return {"origin": origin, "ranks": {"0": series}, "cursor": {"steps:0": len(rows)}}
        out, new_cursor = {}, {}
        wanted = fields or ("epoch", "step", "loss", "step_wall_s", "data_s", "input_h2d_wall_s", "forward_wall_s", "backward_wall_s",
                            "optimizer_wall_s", "gradient_sync_s", "forward_gpu_s", "backward_gpu_s", "optimizer_gpu_s",
                            "request_bytes", "response_bytes", "n_samples")
        for rank in (ranks if ranks is not None else data.ranks()):
            tail = data.rank_steps.get(rank)
            if tail is None:
                continue
            rows = tail.rows
            start = int(cursor.get(f"steps:{rank}", 0))
            new = rows[start:]
            idx = downsample_indices(len(new), max_points)
            series = columns(new, wanted, idx)
            series["t"] = [new[i]["end_s"] - origin for i in idx] if origin is not None else []
            series["x"] = [start + i for i in idx]
            out[str(rank)] = series
            new_cursor[f"steps:{rank}"] = len(rows)
        return {"origin": origin, "ranks": out, "cursor": new_cursor}

    def rank_epochs(self, ref: RunRef, cursor: dict | None = None) -> dict:
        data = self.data(ref)
        cursor = cursor or {}
        origin = data.origin()
        out, new_cursor = {}, {}
        for rank, tail in data.rank_epochs.items():
            start = int(cursor.get(f"epochs:{rank}", 0))
            rows = [dict(r, t_start=(r["start_s"] - origin) if origin is not None else None,
                         t_end=(r["end_s"] - origin) if origin is not None else None) for r in tail.rows[start:]]
            out[str(rank)] = rows
            new_cursor[f"epochs:{rank}"] = len(tail.rows)
        if data.kind == "single" and data.metrics is not None:
            rows = [r for r in data.metrics.rows if r.get("kind") == "epoch"]
            start = int(cursor.get("epochs:0", 0))
            out["0"] = [dict(r, t_end=(r["monotonic_s"] - origin) if origin is not None else None) for r in rows[start:]]
            new_cursor["epochs:0"] = len(rows)
        return {"ranks": out, "cursor": new_cursor}

    def resources(self, ref: RunRef, *, max_points: int | None = 3000, cursor: dict | None = None) -> dict:
        data = self.data(ref)
        origin = data.origin()
        start = int((cursor or {}).get("resources", 0))
        rows = data.resources.rows[start:]
        idx = downsample_indices(len(rows), max_points)
        series = columns(rows, ("cpu_utilization_pct", "ram_used_bytes", "gpu_utilization_pct", "gpu_memory_mib",
                                "gpu_temperature_c", "gpu_power_w"), idx)
        series["t"] = [rows[i]["monotonic_s"] - origin for i in idx] if origin is not None else []
        pid_to_rank = {str(pid): rank for rank, pid in data.rank_pids.items()}
        rank_rss = {str(rank): [] for rank in data.rank_pids}
        for i in idx:
            rss = rows[i].get("process_rss_bytes") or {}
            for rank in rank_rss:
                rank_rss[rank].append(None)
            for pid, value in rss.items():
                rank = pid_to_rank.get(pid)
                if rank is not None:
                    rank_rss[str(rank)][-1] = value / (1024 * 1024)
        series["rank_rss_mib"] = rank_rss
        errors = sum(1 for r in rows if r.get("sampling_error"))
        return {"origin": origin, "series": series, "sampling_errors": errors,
                "cursor": {"resources": len(data.resources.rows)}}

    def log_path(self, ref: RunRef) -> Path | None:
        candidates = []
        if ref.job and ref.job.get("log_path"):
            candidates.append(Path(ref.job["log_path"]))
        candidates += [ref.path.with_name(ref.path.name + ".log"), ref.path / "stdout.log"]
        return next((p for p in candidates if p.exists()), None)
