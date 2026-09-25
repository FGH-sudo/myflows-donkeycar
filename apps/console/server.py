"""Training console server.

Usage (repository root)::

    python -m apps.console.server                 # http://127.0.0.1:8790
    python -m apps.console.server --port 9000

Build the web UI first: ``cd apps/console/web && npm ci && npm run build``.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from MyFlows.monitoring import GpuSampler, ResourceRecorder

from .api import build_router
from .jobs import JobManager
from .registry import RunRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"
DEFAULT_PORT = 8790
LIVE_INTERVAL_S = 1.0
JOB_SAMPLE_INTERVAL_S = 0.5

_NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>MyFlows 训练控制台</title>
<body style="font-family:sans-serif;padding:40px">
<h2>前端尚未构建</h2>
<p>在 <code>apps/console/web</code> 下执行 <code>npm ci &amp;&amp; npm run build</code> 后刷新；
开发模式可执行 <code>npm run dev</code> 并访问 Vite 提示的地址。</p>
<p>API 已可用：<a href="/api/health">/api/health</a>、<a href="/docs">/docs</a></p></body>"""


def create_app(*, repo_root: Path = REPO_ROOT, console_root: Path | None = None, registry: RunRegistry | None = None,
               sampler: GpuSampler | None = None, python: str | None = None, max_running: int = 1,
               start_background: bool = True, web_dist: Path = WEB_DIST) -> FastAPI:
    repo_root = Path(repo_root)
    console_root = Path(console_root) if console_root else repo_root / "runs" / "console"
    sampler = sampler or GpuSampler()

    def recorder_for_job(job: dict, pid: int):
        return ResourceRecorder(sampler, interval_s=JOB_SAMPLE_INTERVAL_S,
                                jsonl_path=Path(job["run_dir"]) / "resources.jsonl", root_pid=pid).start()

    jobs = JobManager(console_root, repo_root, python=python, max_running=max_running,
                      resource_recorder_factory=recorder_for_job)
    if registry is None:
        registry = RunRegistry.default(repo_root, jobs_provider=jobs.list)
    else:
        registry.jobs_provider = jobs.list
    live = ResourceRecorder(sampler, interval_s=LIVE_INTERVAL_S, history=300)
    archive_root = registry.archives[0].root if registry.archives else None

    @asynccontextmanager
    async def lifespan(_app):
        if start_background:
            live.start()
            jobs.start()
        yield
        jobs.shutdown()
        live.stop()
        sampler.close()

    app = FastAPI(title="MyFlows 训练控制台", lifespan=lifespan)
    app.state.jobs, app.state.registry, app.state.live, app.state.sampler = jobs, registry, live, sampler
    app.include_router(build_router(registry=registry, jobs=jobs, live=live, sampler=sampler, repo_root=repo_root,
                                    archive_root=archive_root, python=python))

    index = Path(web_dist) / "index.html"
    if (Path(web_dist) / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=Path(web_dist) / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            return HTMLResponse("Not Found", status_code=404)
        candidate = (Path(web_dist) / full_path).resolve()
        if full_path and candidate.is_file() and Path(web_dist).resolve() in candidate.parents:
            return FileResponse(candidate)
        if index.exists():
            return FileResponse(index)
        return HTMLResponse(_NOT_BUILT)

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="MyFlows 训练控制台")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--max-running", type=int, default=1, help="同时运行的任务数；单 GPU 建议保持 1")
    ap.add_argument("--console-root", default=None, help="控制台任务目录，默认 runs/console")
    args = ap.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[warning] 控制台可以在本机启动训练进程，监听 {args.host} 会把该能力暴露给网络中的其他机器。",
              file=sys.stderr)
    app = create_app(console_root=Path(args.console_root).resolve() if args.console_root else None,
                     max_running=args.max_running)
    print(f"MyFlows 训练控制台: http://{args.host}:{args.port}  (GPU 采样: {app.state.sampler.backend_name})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
