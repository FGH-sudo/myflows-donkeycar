"""Run the Stage-1 compare matrix for a single task (synthetic by default)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from MyFlows.distributed.launcher import run_training


MATRIX = (
    ("single-1", "single", "none", 1),
    ("ps-json-2", "ps", "socket_json", 2),
    ("ps-json-4", "ps", "socket_json", 4),
    ("ps-grpc-2", "ps", "grpc_proto", 2),
    ("ps-grpc-4", "ps", "grpc_proto", 4),
    ("ring-grpc-2", "ring", "grpc_proto", 2),
    ("ring-grpc-4", "ring", "grpc_proto", 4),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="synthetic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--global-batch", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for config_id, mode, transport, workers in MATRIX:
        result = run_training(
            mode=mode, transport=transport, task=args.task, train_workers=workers,
            device=args.device, steps=args.steps, epochs=args.epochs,
            global_batch=args.global_batch, timeout=args.timeout, run_id=config_id,
            collect_history=args.task == "synthetic", optimizer="mbgd",
        )
        metrics = result.get("metrics") or []
        final_metrics = metrics[-1] if metrics else {}
        losses = result.get("losses") or []
        row = {
            "config_id": config_id,
            "task": args.task,
            "mode": mode,
            "transport": transport,
            "N": workers,
            "status": result.get("status"),
            "error": result.get("error"),
            "elapsed_s": result.get("elapsed_s"),
            "cleanup_s": result.get("cleanup_s"),
            "final_loss": losses[-1] if losses else None,
            "final_accuracy": final_metrics.get("accuracy"),
            "final_angle_mae": final_metrics.get("angle_mae"),
            "completed_steps": len(losses),
            "steps_semantics": (result.get("config") or {}).get("steps_semantics"),
        }
        rows.append(row)
        (out / f"{config_id}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(row)
        if result.get("status") != "passed":
            raise RuntimeError(f"{config_id} failed: {result.get('error')}")
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
