"""Stand-in for apps.sim.udacity_drive for AutoPilot closed-loop tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config")
    ap.add_argument("--speed", type=float, default=12.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--track", default="lake")
    ap.add_argument("--sim", default="fake")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, required=True)
    args, _ = ap.parse_known_args(argv)

    if args.out.exists():
        raise FileExistsError(f"closed-loop out directory already exists: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)

    is_best = "best" in str(args.out.parent.name)
    mean_cte = 0.14 if is_best else 0.31
    summary = {
        "status": "completed",
        "passed": True,
        "termination": "time_limit",
        "survival_s": float(args.seconds),
        "distance_units": float(args.seconds) * float(args.speed) * 0.44704,
        "mean_abs_cte": mean_cte,
        "max_abs_cte": mean_cte * 2.1,
        "excursions": 0,
        "cte_frames": 300,
        "cte_coverage": 1.0,
        "frames": 300,
        "inference_p50_ms": 1.25,
        "inference_p95_ms": 2.10,
        "environment": f"udacity-{args.sim or args.track}",
        "model": args.model,
        "config": args.config,
        "device": args.device,
    }
    (args.out / "frames.jsonl").write_text("", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
