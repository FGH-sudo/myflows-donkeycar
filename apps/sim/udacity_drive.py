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
import io
import json
from pathlib import Path

from apps.eval import environments
from apps.eval.closed_loop import episode, model_predictor
from apps.eval.simulator import ROOT

PROTOCOL = Path(__file__).resolve().parents[1] / "eval" / "scenarios" / "udacity_lake.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="ONNX file or checkpoint base name, relative to the repository")
    source.add_argument("--endpoint", help="a running inference service: http://host:port or grpc://host:port")
    parser.add_argument("--config", help="training config.json of a checkpoint (input size, width, backend)")
    parser.add_argument("--speed", type=float, default=12., help="target speed in mph")
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--track", choices=environments.UDACITY_TRACKS, default="lake")
    parser.add_argument("--sim", help="simulator executable, remote, or fake")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--out", type=Path, help="directory for frames.jsonl and summary.json (must not exist)")
    args = parser.parse_args(argv)
    if args.speed <= 0 or args.seconds <= 0:
        parser.error("--speed and --seconds must be positive")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol.update(environment=environments.UDACITY_FAKE if args.sim == "fake" else f"udacity-{args.track}",
                    target_speed_mph=args.speed)
    environment = environments.for_protocol(protocol)
    item = {"id": "pilot", "endpoint": args.endpoint} if args.endpoint else {"id": "pilot", "path": args.model}
    if args.config:
        item["config"] = args.config
    out = (ROOT / args.out).resolve() if args.out else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
    predict = model_predictor(item, args.device)
    scenario = {"id": "drive", "seed": 0, "start": "simulator_default", "duration_s": args.seconds, "speed_mph": args.speed}
    log = (out / "frames.jsonl").open("w", encoding="utf-8", buffering=1) if out is not None else io.StringIO()
    try:
        with environment.session(out, executable=None if args.sim == "fake" else args.sim) as (stream, described):
            row = episode(stream, predict, scenario, protocol["rules"], log, model_id="pilot", clock=environment.clock,
                          monitor=environment.monitor())
    finally:
        log.close()
    summary = {**row, "environment": environment.name, **environment.describe(), "source": item, "session": described}
    if out is not None:
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("termination", "survival_s", "distance_units", "mean_abs_cte",
                                                     "max_abs_cte", "excursions", "frames", "inference_p50_ms",
                                                     "inference_p95_ms", "environment", "cte_source")}, ensure_ascii=False))
    return 0 if row["termination"] == "time_limit" else 2


if __name__ == "__main__":
    raise SystemExit(main())
