"""Drive the Udacity simulator with a model of this project: the counterpart of the simulator's own ``drive.py``.

    python -m apps.sim.udacity_drive --model runs/.../checkpoints/best.onnx
    python -m apps.sim.udacity_drive --endpoint http://127.0.0.1:8000       # through apps.serve.serve_fastapi
    python -m apps.sim.udacity_drive --endpoint grpc://127.0.0.1:50051      # through apps.serve.serve_grpc

Without ``--sim`` the simulator is started and put into autonomous mode (Windows); ``--sim remote`` waits for a
simulator that was started by hand, ``--sim fake`` uses the protocol stand-in. The model steers, a speed
controller holds ``--speed``; the run ends after ``--seconds`` or when the car is stuck or the road is lost.
One line of JSON with the episode summary is printed at the end; ``--out`` also keeps the per-frame log.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from apps.collect.policy import finite_float
from apps.common.gpu_lock import GpuLock
from apps.eval import environments
from apps.eval.closed_loop import episode, model_predictor
from apps.eval.simulator import ROOT, guarded_worker

PROTOCOL = Path(__file__).resolve().parents[1] / "eval" / "scenarios" / "udacity_lake.json"


def main(argv=None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="ONNX file or checkpoint base name, relative to the repository")
    source.add_argument("--endpoint", help="a running inference service: http://host:port or grpc://host:port")
    parser.add_argument("--config", help="training config.json of a checkpoint (input size, width, backend)")
    parser.add_argument("--speed", type=finite_float, default=12., help="target speed in mph")
    parser.add_argument("--seconds", type=finite_float, default=120.)
    parser.add_argument("--track", choices=environments.UDACITY_TRACKS, default="lake")
    parser.add_argument("--sim", help="simulator executable, remote, or fake")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--out", type=Path, help="directory for frames.jsonl and summary.json (must not exist)")
    parser.add_argument("--camera-every", type=int, default=0, metavar="N", help="save every Nth camera frame under --out (0 disables)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--guard-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(raw_argv)
    if args.speed <= 0 or args.seconds <= 0:
        parser.error("--speed and --seconds must be positive")
    if args.camera_every < 0 or args.camera_every and args.out is None:
        parser.error("--camera-every must be non-negative and requires --out when enabled")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol.update(environment=environments.UDACITY_FAKE if args.sim == "fake" else f"udacity-{args.track}",
                    target_speed_mph=args.speed)
    environment = environments.for_protocol(protocol)
    item = {"id": "pilot", "endpoint": args.endpoint} if args.endpoint else {"id": "pilot", "path": args.model}
    if args.config:
        item["config"] = args.config
    out = (ROOT / args.out).resolve() if args.out else None
    if not args.worker and args.sim not in ("fake", "remote"):
        guard_ctx = None
        if out is not None:
            out.mkdir(parents=True, exist_ok=False)
            guard_dir = out
            worker_argv = raw_argv
        else:
            guard_ctx = tempfile.TemporaryDirectory(prefix="udacity-drive-")
            guard_dir = Path(guard_ctx.name)
            worker_argv = [*raw_argv, "--guard-dir", str(guard_dir)]
        try:
            code = guarded_worker("apps.sim.udacity_drive", worker_argv, 150 + args.seconds, guard_dir,
                                  accepted_codes=(0, 1, 2, 130))
            log_file = guard_dir / "session.log"
            if log_file.is_file():
                lines = [ln for ln in log_file.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
                if lines:
                    print(lines[-1])
            return code
        finally:
            if guard_ctx is not None:
                guard_ctx.cleanup()
    if out is not None:
        out.mkdir(parents=True, exist_ok=args.worker)
    session_dir = out or args.guard_dir
    scenario = {"id": "drive", "seed": 0, "start": "simulator_default", "duration_s": args.seconds, "speed_mph": args.speed}
    protocol["scenarios"] = [scenario]
    log = (out / "frames.jsonl").open("w", encoding="utf-8", buffering=1) if out is not None else open(os.devnull, "w")
    saved = seen = 0

    def camera(picture, row):
        nonlocal saved, seen
        if seen % args.camera_every == 0:
            import cv2

            directory = out / "images"
            directory.mkdir(exist_ok=True)
            name = f"{seen:06d}.png"
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
            if not ok:
                raise OSError("camera PNG encoding failed")
            (directory / name).write_bytes(encoded.tobytes())
            row["camera"] = f"images/{name}"
            saved += 1
        seen += 1

    described = None
    try:
        with GpuLock(label="Udacity driving", run_dir=out):
            predict = model_predictor(item, args.device)
            with environment.session(session_dir, executable=None if args.sim == "fake" else args.sim) as (stream, described):
                row = episode(stream, predict, scenario, protocol["rules"], log, model_id="pilot", clock=environment.clock,
                              monitor=environment.monitor(), on_frame=camera if args.camera_every else None)
        checks = {"time_limit": row["termination"] == "time_limit", "no_excursions": row["excursions"] == 0,
                  "cte_coverage": row["cte_frames"] > 0 and (row["cte_coverage"] or 0.) >= .9}
        passed = all(checks.values())
        summary = {**row, "status": "completed", "passed": passed, "checks": checks,
                   "acceptance": {"max_excursions": 0, "min_cte_coverage": .9}}
        code = 0 if passed else 2
    except KeyboardInterrupt:
        summary = {"status": "interrupted", "termination": "interrupted", "passed": False}
        code = 130
    except Exception as error:
        summary = {"status": "failed", "termination": "error", "passed": False,
                   "error": f"{type(error).__name__}: {error}"}
        code = 1
    finally:
        log.close()
    summary.update(environment=environment.name, **environment.describe(), source=item, session=described,
                   camera_frames=saved, protocol=protocol, scenario_config=scenario)
    if out is not None:
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "passed", "error", "termination", "survival_s", "distance_units", "mean_abs_cte",
                                                     "max_abs_cte", "excursions", "frames", "inference_p50_ms",
                                                     "inference_p95_ms", "environment", "cte_source") if key in summary}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
