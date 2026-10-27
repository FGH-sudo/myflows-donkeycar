"""A stand-in for apps.train.train_myflows_donkey with the same command line and the same output files.

It takes the GPU lock, writes config.json / metrics.jsonl / resources.jsonl / result.json like the real entry,
honours the stop file and exits with the same codes. Scenarios (``FAKE_TRAIN_SCENARIO``, a JSON file) pick a
behaviour per configuration::

    {"step_s": 0.01, "objective": {...}, "rules": [{"when": {"lr": 0.001, "attempt": 1}, "behavior": "crash"}]}

Behaviours: ``nan`` (exit 3), ``nan_online`` (NaN row, keeps running), ``diverge`` (loss explodes, keeps
running), ``explode`` (loss jumps 10^4x on the second step and only decays slowly, like Adam with a far too
large lr), ``oom`` (exit 4), ``hang`` (stops writing and ignores the stop file), ``crash`` (exit 1 without
result.json), ``slow`` (long steps), ``data_bound`` (slow data loading while num_workers == 0).
``--state-every`` writes real (empty-model) TrainingStates, the first one at step 0; ``--resume-state``
continues from one. With ``--pretrained`` the result carries a ``transfer`` record like apps.train (load report,
frozen stages, frozen-tensor check); ``"objective": {"freeze": {"2": -0.01}}`` adds a term per frozen level and
the ``frozen_leak`` behaviour reports frozen tensors that changed.

Data-parallel runs (``--distributed ps|ring``) also write ``rank-*/steps.jsonl`` like apps.train. The step's
compute splits across the workers; ``"parallel": {"sync_ms": {"ps": 40, "ring": 15}, "straggler": true}``
sets the gradient synchronisation cost per step (default 30 ms × workers) and makes rank 1 twice as slow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from apps.common.gpu_lock import GpuLock
from apps.train.common import outcome


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    for name, kind, default in (("lr", float, 1e-3), ("batch", int, 1), ("num_workers", int, 0), ("seed", int, None),
                                ("epochs", int, 1), ("max_samples", int, 0), ("split_file", str, None),
                                ("conv_backend", str, "auto"), ("device", str, "auto"), ("run_dir", str, None),
                                ("log_file", str, None), ("stop_file", str, None), ("out", str, None),
                                ("best_out", str, None), ("dtype", str, "float32"), ("distributed", str, "none"),
                                ("workers", int, 1), ("transport", str, "none"),
                                ("state_every", int, 0), ("keep_states", int, 3), ("resume_state", str, None),
                                ("arch", str, "resnet18"), ("pretrained", str, None), ("freeze_stages", int, 0)):
        ap.add_argument("--" + name.replace("_", "-"), type=kind, default=default)
    return ap


def _scenario() -> dict:
    path = os.environ.get("FAKE_TRAIN_SCENARIO")
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _matches(when: dict, args, attempt: int) -> bool:
    for key, value in when.items():
        if key == "attempt":
            if attempt != value:
                return False
        elif getattr(args, key, None) != value:
            return False
    return True


def objective(args, spec: dict) -> float:
    """Deterministic val_angle_mae: a bowl in log10(lr) plus a batch term and a small seed-dependent noise."""
    lr_opt = spec.get("lr_opt", -3.0)
    value = spec.get("base", 0.2) + spec.get("lr_coef", 0.05) * abs(math.log10(args.lr) - lr_opt)
    value += float((spec.get("batch") or {}).get(str(args.batch), 0.0))
    value += spec.get("seed_noise", 0.002) * (((args.seed or 0) * 7919) % 5 - 2) / 2.0
    value += float((spec.get("freeze") or {}).get(str(args.freeze_stages), 0.0))
    return value


STAGES = ("stem", "layer1", "layer2", "layer3", "layer4")


def transfer_record(args, behavior) -> dict | None:
    """The ``transfer`` block of apps.train's result.json for a pretrained run."""
    if not args.pretrained:
        return None
    frozen = list(STAGES[:args.freeze_stages])
    before = {"tensors": 4 * len(frozen), "sha256": f"frozen-{args.freeze_stages}"}
    after = dict(before, sha256="changed") if behavior == "frozen_leak" else dict(before)
    return {"arch": args.arch, "pretrained": args.pretrained, "freeze_stages": args.freeze_stages,
            "input_norm": "imagenet",
            "load_report": {"arch": args.arch, "expected": 102, "loaded": 102, "success_rate": 1.0, "missing": [],
                            "unexpected": [], "shape_mismatch": [], "npz_sha256": "ed6abcb7256afa3d",
                            "replaced_head": [{"source": "fc.weight"}, {"source": "fc.bias"}],
                            "source": {"source_sha256": "f37072fd47e89c5e"}},
            "freeze": {"stages": list(STAGES), "freeze_stages": args.freeze_stages, "frozen": frozen,
                       "frozen_values": 1_000_000 * len(frozen), "trainable_values": 11_000_000 - 1_000_000 * len(frozen)},
            "frozen_before": before, "frozen_after": after, "frozen_unchanged": after == before}


def main() -> None:
    args, _unknown = _parser().parse_known_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    scenario = _scenario()
    attempt = int(run_dir.name.split("-")[-1]) if run_dir.name.startswith("attempt-") else 1
    behavior = next((rule["behavior"] for rule in scenario.get("rules", [])
                     if _matches(rule.get("when", {}), args, attempt)), None)
    log = open(args.log_file, "a", encoding="utf-8") if args.log_file else sys.stdout

    def say(message):
        print(message, file=log, flush=True)

    started = time.time()
    lock = GpuLock(label="fake_train", run_dir=run_dir, say=say).acquire()
    if scenario.get("spawn_child"):
        import subprocess

        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        (run_dir / "child.pid").write_text(str(child.pid), encoding="utf-8")
    from apps.autopilot.data import describe_split

    split = describe_split(Path.cwd(), args.split_file, max_samples=args.max_samples) if args.split_file else {}
    train = int(split.get("train_count") or 20)
    steps_per_epoch = math.ceil(train / args.batch)
    (run_dir / "config.json").write_text(json.dumps({"kind": "single", "task": "donkey_resnet18_single",
                                                     "args": vars(args), "pid": os.getpid()}), encoding="utf-8")
    metrics = (run_dir / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
    resources = (run_dir / "resources.jsonl").open("a", encoding="utf-8", buffering=1)
    stop_file = Path(args.stop_file) if args.stop_file else run_dir / "STOP"
    step_s = float(scenario.get("step_s", 0.01)) * (40.0 if behavior == "slow" else 1.0)
    data_bound = behavior == "data_bound" or (scenario.get("data_bound") and args.num_workers == 0)
    data_ms, step_ms, util = (30.0, 20.0, 30.0) if data_bound else (2.0, 20.0, 85.0)
    mem_mib = float(scenario.get("mem_mib_per_sample", 300.0)) * args.batch  # CuPy pool peak; 8192 MiB GPU
    target = objective(args, scenario.get("objective") or {})
    loss = 1.0
    global_step = 0
    start = 0
    if args.resume_state:  # apps.train --resume-state: continue from the TrainingState's step and loss
        from MyFlows.utils.checkpoint import read_checkpoint_metadata

        state = (read_checkpoint_metadata(Path(args.resume_state) / "checkpoint") or {}).get("training") or {}
        start = global_step = int(state["global_step"])
        loss = float(state["loss"])

    def finish(status, failure=None, code=0):
        result = {"status": status, "exit_code": code, "failure": failure, "task": "donkey_resnet18_single",
                  "metrics": {"val_angle_mae": target, "val_loss": target * 2, "best_epoch": args.epochs},
                  "epochs": [], "throughput": {"samples_per_s": args.batch / ((data_ms + step_ms) / 1000.0)},
                  "memory": {"source": "cupy_memory_pool", "peak_used_mb": mem_mib * 5 / 6,
                             "peak_pool_mb": mem_mib, "device_total_mb": 8192.0},
                  "samples": {"train": train, "val": split.get("val_count"),
                              "trained": global_step * args.batch,
                              "expected_trained": args.epochs * steps_per_epoch * args.batch},
                  "data": {"val_hash": split.get("val_hash"), "split_file": args.split_file},
                  "backend": {"requested": args.conv_backend, "actual": args.conv_backend if args.conv_backend
                              != "auto" else "cupy", "device": "cuda"},
                  "timing": {"lock_wait_s": lock.wait_s, "lock_acquired_unix_s": lock.acquired_unix_s,
                             "started_unix_s": started, "run_s": time.time() - lock.acquired_unix_s},
                  "transfer": transfer_record(args, behavior)}
        if failure is not None:
            result["metrics"] = {}
        outcome.write_result(run_dir, result)
        (run_dir / "results.json").write_text(json.dumps({"status": "passed" if failure is None else "failed"}),
                                              encoding="utf-8")
        lock.release()
        sys.exit(code)

    parallel = scenario.get("parallel") or {}
    ranks = []
    if args.distributed != "none":
        for r in range(args.workers):
            (run_dir / f"rank-{r}").mkdir(exist_ok=True)
            ranks.append((run_dir / f"rank-{r}" / "steps.jsonl").open("a", encoding="utf-8", buffering=1))
        sync_ms = float((parallel.get("sync_ms") or {}).get(args.distributed, 30.0 * args.workers))

    def save_state(epoch, next_step):
        from MyFlows.utils.checkpoint import save_training_state

        save_training_state(run_dir / "states", [], None, global_step, {
            "epoch": epoch, "epoch_complete": False, "next_step": next_step, "global_step": global_step,
            "loss": loss, "running_loss": loss if global_step else None, "last_mean_loss": None,
            "first_loss": 1.0, "lr": args.lr, "end_epoch": args.epochs}, keep=args.keep_states)

    if args.state_every and not args.resume_state:
        save_state(0, 0)
    for epoch in range(args.epochs):
        for step in range(steps_per_epoch):
            if epoch * steps_per_epoch + step < start:
                continue
            if behavior == "hang" and global_step >= 3:
                while True:  # ignores the stop file; only killing the process ends it
                    time.sleep(1.0)
            if behavior == "oom" and global_step == 2:
                say("cupy.cuda.memory.OutOfMemoryError: Out of memory allocating 1,073,741,824 bytes")
                finish("failed", {"kind": "oom", "message": "OutOfMemoryError: Out of memory"}, outcome.EXIT_OOM)
            if behavior == "crash" and global_step == 2:
                say("Traceback: simulated worker crash")
                os._exit(1)
            if behavior == "diverge":
                loss = 1.0 * (1.25 ** global_step)
            elif behavior == "explode":
                loss = 1.0 if global_step == 0 else 1e4 * (0.97 ** global_step)
            else:
                loss = max(target, loss * 0.9)
            if behavior in ("nan", "nan_online") and global_step == 4:
                loss = float("nan")
            time.sleep(step_s)
            global_step += 1
            now = time.time()
            if ranks:
                local = args.batch // args.workers
                compute = [step_ms * local / args.batch * (2.0 if parallel.get("straggler") and r == 1 else 1.0)
                           for r in range(args.workers)]
                slowest = max(compute)
                for r, out in enumerate(ranks):
                    wait = slowest - compute[r] + sync_ms  # fast ranks wait for the slowest one
                    wall = data_ms + compute[r] + wait
                    out.write(json.dumps({"rank": r, "epoch": epoch, "step": global_step, "step_wall_s": wall / 1000,
                                          "data_s": data_ms / 1000, "gradient_sync_s": wait / 1000,
                                          "update_confirm_s": 0.0, "n_samples": local, "loss": loss}) + "\n")
            metrics.write(json.dumps({"kind": "step", "epoch": epoch + 1, "step": global_step, "loss": loss,
                                      "unix_s": now, "monotonic_s": time.perf_counter(),
                                      "data_load_ms": data_ms, "step_time_ms": step_ms, "batch_size": args.batch,
                                      "samples_per_sec": args.batch / (step_ms / 1000.0)}) + "\n")
            resources.write(json.dumps({"unix_s": now, "monotonic_s": time.perf_counter(),
                                        "gpu_utilization_pct": util, "gpu_memory_mib": mem_mib,
                                        "gpu_temperature_c": 60.0, "gpu_power_w": 80.0,
                                        "gpus": [{"memory_used_mib": mem_mib,
                                                  "memory_total_mib": 8192.0}]}) + "\n")
            if behavior == "nan" and not math.isfinite(loss):
                finish("failed", {"kind": "nan", "message": f"非有限 loss: {loss}", "step": global_step},
                       outcome.EXIT_NONFINITE)
            if args.state_every and global_step % args.state_every == 0 and math.isfinite(loss):
                save_state(epoch, step + 1)
            if stop_file.exists():
                stop_file.unlink()
                finish("stopped")
        metrics.write(json.dumps({"kind": "epoch", "epoch": epoch + 1, "loss": loss, "val_loss": target * 2,
                                  "val_angle_mae": target, "unix_s": time.time()}) + "\n")
    finish("succeeded")


if __name__ == "__main__":
    main()
