"""Run the synchronous CPU PS and verify every update against one process."""

import argparse
import json
import time

import numpy as np

from benchmark.stage1_common import RunArtifacts, array_hash
from MyFlows.distributed.launcher import run_ps
from MyFlows.distributed.model import make_batches, single_process
from MyFlows.distributed.protocol import payload_hash


def verify(result, reference):
    if result["status"] != "passed":
        raise RuntimeError(result["error"])
    if len(result["history"]) != len(reference["history"]):
        raise AssertionError("history length differs")
    max_abs = 0.0
    for actual, expected in zip(result["history"], reference["history"]):
        if set(actual) != set(expected):
            raise AssertionError("parameter mapping differs")
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-8, rtol=1e-6)
            max_abs = max(max_abs, float(np.max(np.abs(actual[key] - expected[key]))))
    np.testing.assert_allclose(result["losses"], reference["losses"], atol=1e-8, rtol=1e-6)
    updates = [e for e in result["events"] if e["role"] == "server" and e["phase"] == "updated"]
    if len(updates) != result["config"]["steps"]:
        raise AssertionError("missing server update events")
    for step, event in enumerate(updates):
        if (event["step_id"], event["parameter_version"], event["n_samples"]) != (step, step + 1, result["config"]["global_batch"]):
            raise AssertionError("wrong step/version/sample count")
    for worker in range(result["config"]["workers"]):
        acknowledged = [e for e in result["events"] if e["worker_id"] == worker and e["phase"] == "gradient_acknowledged"]
        parameters = [e for e in result["events"] if e["worker_id"] == worker and e["phase"] == "parameters_received"]
        if len(acknowledged) != result["config"]["steps"] or len(parameters) != result["config"]["steps"] + 1:
            raise AssertionError("missing worker acknowledgement/parameter events")
    if result["alive_pids"] or result["cleanup_s"] > 5 or any(code != 0 for code in result["exitcodes"].values()):
        raise AssertionError("unclean successful process exit")
    return max_abs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--global-batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-sizes", help="e.g. 20,12")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--fault", choices=("crash", "timeout", "duplicate", "schema"))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    shard_sizes = [int(s) for s in args.shard_sizes.split(",")] if args.shard_sizes else None
    with RunArtifacts(args.out_dir, vars(args)) as run:
        result = run_ps(workers=args.workers, global_batch=args.global_batch, steps=args.steps,
                        seed=args.seed, shard_sizes=shard_sizes, timeout=args.timeout,
                        fault=args.fault, run_id=run.path.name)
        events = result.pop("events")
        (run.path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        run.timings.extend(events)
        run.results["ps"] = {k: v for k, v in result.items() if k != "history"}
        run.manifest["timing_scope"] = "Single-host perf_counter; Queue submit, receive/ack wait, compute and server phases. Bytes are array payload only; protocol overhead unmeasured."
        if result["status"] != "passed":
            raise RuntimeError(result["error"])
        reference_started = time.perf_counter()
        reference = single_process(args.seed, args.global_batch, args.steps)
        run.results["single_process_wall_s"] = time.perf_counter() - reference_started
        run.notes.append("Single-process wall time includes model/data construction and 20-step history (or configured steps); PS elapsed time also includes spawn, IPC and cleanup. These are one-run demonstration timings, not a scaling benchmark.")
        max_abs = verify({**result, "events": events}, reference)
        arrays = {f"version_{step}_{name}": a for step, state in enumerate(result["history"]) for name, a in state.items()}
        np.savez(run.path / "parameters.npz", **arrays)
        np.savez(run.path / "reference.npz", **{f"version_{step}_{name}": a for step, state in enumerate(reference["history"]) for name, a in state.items()})
        x, y = make_batches(args.seed, args.global_batch, args.steps)
        np.savez(run.path / "fixture.npz", x=x, y=y)
        run.manifest["fixture_sha256"] = array_hash(x, y)
        run.manifest["initial_parameters_sha256"] = payload_hash(reference["history"][0])
        run.results["equivalence_max_abs"] = max_abs
        run.notes.append(f"Verified every update against single-process FP64 MBGD, including step 1 and step {args.steps}. Maximum parameter absolute difference: {max_abs:.6g}.")
        run.notes.append("Workers never update weights; server weights local mean gradients by actual shard sample counts. Launcher cleanup and individual exit codes are in results.json.")
        print(f"workers={args.workers} steps={args.steps} max_abs={max_abs:.6g} cleanup_s={result['cleanup_s']:.4f}")


if __name__ == "__main__":
    main()
