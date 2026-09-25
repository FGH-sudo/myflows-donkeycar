"""Unified distributed training entry: single / PS / Ring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.stage1_common import RunArtifacts
from MyFlows.distributed.launcher import run_training


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "ps", "ring"), default="ps")
    parser.add_argument("--transport", choices=("none", "socket_json", "grpc_proto"), default="socket_json")
    parser.add_argument("--task", default="synthetic")
    parser.add_argument("--train-workers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--global-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1,
                        help="real-task epoch count; ignored as a loop expander for synthetic")
    parser.add_argument("--steps", type=int, default=20,
                        help="synthetic: total updates; mnist_mlp/donkey_cnn: max batches per epoch")
    parser.add_argument("--optimizer", choices=("mbgd", "adam"), default="mbgd")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--ps-wait-strategy", choices=("poll", "notify"), default="poll",
                        help="PS server wait policy; notify is opt-in because shared-GPU gains depend on workload")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-dir")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    transport = "none" if args.mode == "single" else ("grpc_proto" if args.mode == "ring" else args.transport)
    with RunArtifacts(args.out_dir, vars(args)) as run:
        result = run_training(
            mode=args.mode,
            transport=transport,
            task=args.task,
            train_workers=args.train_workers,
            device=args.device,
            global_batch=args.global_batch,
            seed=args.seed,
            epochs=args.epochs,
            steps=args.steps,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            timeout=args.timeout,
            ps_wait_strategy=args.ps_wait_strategy,
            run_id=run.path.name,
            data_dir=args.data_dir,
            collect_history=args.task == "synthetic",
        )
        events = result.get("events") or []
        (run.path / "events.jsonl").write_text(
            "".join(json.dumps(_jsonable(e)) + "\n" for e in events), encoding="utf-8")
        losses = result.get("losses") or []
        metrics = result.get("metrics") or []
        final_metrics = metrics[-1] if metrics else {}
        run.results["distributed"] = {
            "status": result.get("status"),
            "error": result.get("error"),
            "elapsed_s": result.get("elapsed_s"),
            "cleanup_s": result.get("cleanup_s"),
            "exitcodes": result.get("exitcodes"),
            "losses": losses,
            "metrics": _jsonable(metrics),
            "final_loss": losses[-1] if losses else None,
            "final_metrics": _jsonable(final_metrics),
            "completed_steps": len(losses),
            "steps_semantics": (result.get("config") or {}).get("steps_semantics"),
            "communication": result.get("communication"),
            "worker_metrics": _jsonable(result.get("worker_metrics") or {}),
            "metrics_scope": result.get("metrics_scope") or "train_batch",
            "metrics_valid": result.get("metrics_valid"),
            "metrics_issues": result.get("metrics_issues") or [],
        }
        run.manifest["mode"] = args.mode
        run.manifest["transport"] = transport
        run.manifest["optimizer"] = args.optimizer
        run.manifest["steps_semantics"] = (result.get("config") or {}).get("steps_semantics")
        run.manifest["planned_steps"] = (result.get("config") or {}).get("planned_steps")
        meta = result.get("data_meta") or {}
        run.manifest["data_split"] = {
            "source": meta.get("source"),
            "n_train": meta.get("n_train"),
            "n_val": meta.get("n_val"),
            "n_test": meta.get("n_test"),
            "seed": meta.get("seed", args.seed),
            "split_fingerprint": meta.get("split_fingerprint"),
        }
        if result.get("status") != "passed":
            raise RuntimeError(result.get("error") or "distributed run failed")
        print(f"mode={args.mode} transport={transport} workers={args.train_workers} "
              f"status={result['status']} elapsed={result.get('elapsed_s'):.3f}")


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items() if k != "history"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return None
    return value


if __name__ == "__main__":
    main()
