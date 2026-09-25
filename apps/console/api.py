"""FastAPI routes for the training console (REST + Server-Sent Events)."""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

import psutil
from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .jobs import JobManager
from .presets import (
    MAX_WORKERS,
    MODE_LABELS,
    TASK_LABELS,
    ValidationError,
    distributed_presets,
    modes,
    single_cli_args,
    single_presets,
    validate_distributed,
    validate_single,
)
from .readers import read_json, tail_text
from .registry import ACTIVE_STATUSES, RunRef, RunRegistry


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(_clean(content), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(_clean(data), ensure_ascii=False, separators=(',', ':'))}\n\n"


def _parse_cursor(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        raise HTTPException(400, "cursor 必须是 JSON 对象")
    if not isinstance(value, dict):
        raise HTTPException(400, "cursor 必须是 JSON 对象")
    return value


def _csv(raw: str | None, cast=str) -> list | None:
    if not raw:
        return None
    try:
        return [cast(x) for x in raw.split(",") if x.strip() != ""]
    except ValueError:
        raise HTTPException(400, f"参数格式错误：{raw}")


def build_router(*, registry: RunRegistry, jobs: JobManager, live, sampler, repo_root: Path,
                 archive_root: Path | None, python: str | None = None) -> APIRouter:
    router = APIRouter(prefix="/api", default_response_class=SafeJSONResponse)
    python = python or sys.executable

    def ref_or_404(run_id: str) -> RunRef:
        ref = registry.resolve(run_id)
        if ref is None:
            raise HTTPException(404, f"运行不存在：{run_id}")
        return ref

    def pid_owners() -> dict[int, dict]:
        owners = {}
        for job in jobs.list():
            if job["status"] not in ("running", "stopping") or not job.get("pid"):
                continue
            base = {"job_id": job["id"], "label": job["label"]}
            try:
                root = psutil.Process(job["pid"])
                for proc in [root, *root.children(recursive=True)]:
                    owners[proc.pid] = dict(base)
            except psutil.NoSuchProcess:
                continue
            for rank_dir in Path(job["run_dir"]).glob("rank-*"):
                prepared = read_json(rank_dir / "prepare.json") or {}
                if prepared.get("pid") in owners:
                    owners[prepared["pid"]]["rank"] = int(rank_dir.name.split("-", 1)[1])
        return owners

    def gpu_payload(history_since: float | None = None, include_history: bool = True) -> dict:
        latest = live.latest()
        owners = pid_owners()
        if latest:
            latest = dict(latest)
            gpus = []
            for gpu in latest.get("gpus", []):
                gpu = dict(gpu)
                if gpu.get("processes"):
                    gpu["processes"] = [dict(p, owner=owners.get(p["pid"])) for p in gpu["processes"]]
                gpus.append(gpu)
            latest["gpus"] = gpus
        payload = {"backend": sampler.backend_name, "driver_version": sampler.driver_version,
                   "backend_errors": sampler.errors, "interval_s": live.interval_s, "latest": latest}
        if include_history:
            payload["history"] = [{k: r.get(k) for k in ("unix_s", "cpu_utilization_pct", "ram_used_bytes",
                                                         "ram_total_bytes", "gpus")} for r in live.history(history_since)]
        return payload

    # ---- health / gpu --------------------------------------------------------------
    @router.get("/health")
    def health():
        return {"ok": True, "gpu_backend": sampler.backend_name, "jobs": len(jobs.list())}

    @router.get("/gpu")
    def gpu():
        return gpu_payload()

    @router.get("/gpu/stream")
    async def gpu_stream(request: Request):
        async def gen():
            last = None
            while not await request.is_disconnected():
                latest = live.latest()
                if latest and latest.get("unix_s") != last:
                    last = latest.get("unix_s")
                    payload = await asyncio.to_thread(gpu_payload, None, False)
                    yield _sse("gpu", payload)
                await asyncio.sleep(max(0.2, live.interval_s / 2))
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # ---- runs ------------------------------------------------------------------------
    @router.get("/sources")
    def sources():
        return registry.sources()

    @router.get("/runs")
    def runs(source: str | None = None, kind: str | None = None, task: str | None = None, mode: str | None = None,
             status: str | None = None, q: str | None = None):
        items = registry.list_runs(source)
        if kind:
            items = [r for r in items if r.get("kind") == kind]
        if task:
            items = [r for r in items if r.get("task") == task]
        if mode:
            items = [r for r in items if r.get("mode") == mode]
        if status:
            items = [r for r in items if r.get("status") == status]
        if q:
            needle = q.lower()
            items = [r for r in items if needle in (r.get("label") or "").lower() or needle in r["id"].lower()]
        return items

    @router.get("/runs/{run_id:path}/steps")
    def run_steps(run_id: str, ranks: str | None = None, fields: str | None = None,
                  max_points: int = Query(3000, ge=0, le=200000), cursor: str | None = None):
        return registry.steps(ref_or_404(run_id), ranks=_csv(ranks, int), fields=_csv(fields),
                              max_points=max_points or None, cursor=_parse_cursor(cursor))

    @router.get("/runs/{run_id:path}/epochs")
    def run_epochs(run_id: str, cursor: str | None = None):
        return registry.rank_epochs(ref_or_404(run_id), cursor=_parse_cursor(cursor))

    @router.get("/runs/{run_id:path}/resources")
    def run_resources(run_id: str, max_points: int = Query(3000, ge=0, le=200000), cursor: str | None = None):
        return registry.resources(ref_or_404(run_id), max_points=max_points or None, cursor=_parse_cursor(cursor))

    @router.get("/runs/{run_id:path}/log")
    def run_log(run_id: str, tail_kb: int = Query(64, ge=1, le=4096)):
        ref = ref_or_404(run_id)
        path = registry.log_path(ref)
        return {"path": str(path) if path else None, "text": tail_text(path, tail_kb * 1024) if path else ""}

    @router.get("/runs/{run_id:path}/stream")
    async def run_stream(run_id: str, request: Request, cursor: str | None = None, fields: str | None = None,
                         interval_s: float = Query(1.0, ge=0.2, le=30)):
        ref = ref_or_404(run_id)
        state = {"cursor": _parse_cursor(cursor)}
        field_list = _csv(fields)

        def collect():
            fresh = registry.resolve(run_id) or ref
            steps = registry.steps(fresh, fields=field_list, max_points=None, cursor=state["cursor"])
            epochs = registry.rank_epochs(fresh, cursor=state["cursor"])
            resources = registry.resources(fresh, max_points=None, cursor=state["cursor"])
            summary = registry.summary(fresh)
            data = registry.data(fresh)
            state["cursor"].update(steps["cursor"])
            state["cursor"].update(epochs["cursor"])
            state["cursor"].update(resources["cursor"])
            return {
                "steps": steps if any(len(s.get("x", [])) for s in steps["ranks"].values()) else None,
                "epochs": epochs if any(epochs["ranks"].values()) else None,
                "resources": resources if resources["series"]["t"] else None,
                "status": {"summary": summary, "ranks": data.ranks(),
                           "rank_status": data.rank_status(active=summary.get("status") in ACTIVE_STATUSES)
                           if data.kind == "distributed" else []},
                "cursor": dict(state["cursor"]),
            }

        async def gen():
            last_status = None
            while not await request.is_disconnected():
                update = await asyncio.to_thread(collect)
                for key in ("steps", "epochs", "resources"):
                    if update[key] is not None:
                        yield _sse(key, update[key])
                status = json.dumps(_clean(update["status"]), sort_keys=True, default=str)
                if status != last_status:
                    last_status = status
                    yield _sse("status", dict(update["status"], cursor=update["cursor"]))
                if update["status"]["summary"].get("status") not in ACTIVE_STATUSES:
                    yield _sse("end", {"status": update["status"]["summary"].get("status")})
                    break
                await asyncio.sleep(interval_s)
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @router.get("/runs/{run_id:path}")
    def run_detail(run_id: str):
        return registry.detail(ref_or_404(run_id))

    # ---- presets / jobs -----------------------------------------------------------------
    @router.get("/presets")
    def presets():
        return {
            "modes": modes(), "tasks": [{"task": t, "label": l} for t, l in TASK_LABELS.items()],
            "max_workers": MAX_WORKERS, "worker_choices": [1, 2, 4],
            "distributed": distributed_presets(repo_root, archive_root), "single": single_presets(repo_root),
        }

    @router.get("/jobs")
    def list_jobs():
        return jobs.list()

    @router.post("/jobs")
    def submit_job(body: dict = Body(...)):
        job_type = body.get("type")
        try:
            if job_type == "distributed":
                config = validate_distributed(body.get("config") or {})
                label = body.get("label") or (
                    f"{config['task']} · {MODE_LABELS.get((config['mode'], config['transport']), config['mode'])}"
                    f" · {config['train_workers']}w")
                command = [python, "-m", "benchmark.distributed_experiment", "--config", "{config_path}",
                           "--out", "{run_dir}"]
                return jobs.submit(job_type="distributed", label=label, task=config["task"], command=command,
                                   config=config)
            if job_type == "single":
                args = validate_single(body.get("args") or {}, repo_root)
                label = body.get("label") or f"单进程 ResNet18 · {args.get('max_samples', '全部')} 样本"
                command = [python, "-m", "apps.train.train_myflows_donkey", *single_cli_args(args),
                           "--run-dir", "{run_dir}", "--log-dir", "{run_dir}", "--log-file", "{run_dir}/train.log",
                           "--logdir", "{run_dir}/tensorboard", "--out", "{run_dir}/checkpoints/checkpoint",
                           "--best-out", "{run_dir}/checkpoints/best", "--stop-file", "{run_dir}/STOP_TRAINING"]
                return jobs.submit(job_type="single", label=label, task="donkey_resnet18_single", command=command,
                                   config={"kind": "single", "task": "donkey_resnet18_single", "args": args},
                                   extra={"record_resources": True, "stop_file": "{run_dir}/STOP_TRAINING"})
        except ValidationError as exc:
            raise HTTPException(422, str(exc))
        raise HTTPException(422, "type 必须是 distributed 或 single")

    @router.post("/jobs/{job_id}/stop")
    def stop_job(job_id: str):
        try:
            return jobs.stop(job_id)
        except KeyError:
            raise HTTPException(404, f"任务不存在：{job_id}")

    return router
