"""Train CIFAR-10 on a locked train/validation split. Score the official test set explicitly once."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from apps.common import cifar10
from apps.common.environment import environment
from apps.common.gpu_lock import GpuLock
from apps.train.common.outcome import write_result
from MyFlows.monitoring.record import MonitorProcess, monitoring_enabled
from MyFlows.monitoring.schema import record

ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def build(config):
    import MyFlows as ms
    from apps.train.classification_model import SmallClassifier
    from apps.train.train_myflows_donkey import _resolve_conv_backend
    from MyFlows.core.device import set_device
    from MyFlows.utils.initializers import make_initializer

    device = config["device"]
    set_device(device)
    ms.set_default_dtype(np.float32)
    backend = _resolve_conv_backend("auto", device, np.float32)
    batch, width, seed = (config[k] for k in ("batch", "width", "seed"))
    if config["arch"] == "small":
        model = SmallClassifier(width=width, seed=seed, backend=backend)
    else:
        model = ms.ResNet18(output_dim=10, stem="cifar", base_width=width, conv_backend=backend,
                            initializer=make_initializer("kaiming_normal", seed=seed))
    x = ms.Variable(np.zeros((batch, 3, 32, 32), np.float32), name="image", requires_grad=False)
    y = ms.Variable(np.zeros(batch, np.int64), name="label", requires_grad=False)
    logits = model(x)
    loss = ms.CrossEntropy(logits, y)
    graph = ms.Graph(loss)
    return model, (x, y, logits, loss, graph), backend


def evaluate(model, nodes, images, labels, indices, batch):
    from MyFlows.core.device import asnumpy

    x, y, logits, _, graph = nodes
    model.eval()
    total_loss = 0.
    correct = count = 0
    try:
        for first in range(0, len(indices), batch):
            actual = indices[first:first + batch]
            ids = np.pad(actual, (0, batch - len(actual)), mode="edge")
            x.value, y.value = images[ids].astype(np.float32) / 255., labels[ids]
            graph.forward()
            z = np.asarray(asnumpy(logits.value)[:len(actual)], np.float64)
            z -= z.max(axis=1, keepdims=True)
            total_loss += (np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(actual)), labels[actual]]).sum()
            correct += int((z.argmax(axis=1) == labels[actual]).sum())
            count += len(actual)
    finally:
        model.train()
    return {"samples": count, "loss": float(total_loss / count), "accuracy": correct / count}


def train(args, directory):
    import MyFlows as ms
    from MyFlows.core.device import asnumpy
    from MyFlows.utils.training_dashboard import TrainingDashboard
    from apps.train.train_myflows_donkey import _DeviceSpan, _StepPhases

    images, labels = cifar10.load(ROOT / args.data, "train")
    training, validation = cifar10.stratified_split(labels)
    if args.train_limit:
        training = cifar10.balanced_subset(training, labels, args.train_limit, seed=42)
    if args.val_limit:
        validation = cifar10.balanced_subset(validation, labels, args.val_limit, seed=43)
    if len(training) < args.batch:
        raise ValueError("training split must contain at least one complete batch")
    rng = np.random.default_rng(args.seed)
    batches = np.stack([rng.permutation(training)[:len(training)//args.batch*args.batch].reshape(-1, args.batch) for _ in range(args.epochs)])
    np.savez(directory / "split-and-order.npz", train=training, val=validation, batches=batches)
    config = record({"kind": "single", "task": "cifar10_classification", "args": vars(args) | {"run_dir": str(args.run_dir)},
                     "samples": len(training), "val_samples": len(validation), "split_seed": 42, "tail": "drop_last",
                     "order_sha256": _sha(directory / "split-and-order.npz"),
                     "monitoring": {"enabled": monitoring_enabled(args.monitor, directory), "requested": args.monitor, "interval_s": .5}}, clocks=True)
    _write(directory / "config.json", config)
    model, nodes, backend = build(config["args"])
    x, y, logits, loss, graph = nodes
    opt = ms.Adam(graph, learning_rate=args.lr)
    if args.arch == "small":
        np.savez(directory / "initial.npz", **model.torch_state())
    phases = _StepPhases(args.device, args.phase_timing_every)
    train_span = _DeviceSpan(args.device)
    epochs, best, step = [], -1., 0
    with ExitStack() as stack:
        if monitoring_enabled(args.monitor, directory):
            monitor = MonitorProcess(directory, .5).start()
            stack.callback(monitor.stop)
        dashboard = TrainingDashboard(directory / "tensorboard", enabled=args.tensorboard, jsonl_path=directory / "metrics.jsonl",
                                      resources_path=directory / "resources.jsonl")
        stack.callback(dashboard.close)
        for epoch, ids_epoch in enumerate(batches, 1):
            start = time.perf_counter()
            losses, correct = [], 0
            model.train()
            for ids in ids_epoch:
                begin = time.perf_counter()
                x.value, y.value = images[ids].astype(np.float32) / 255., labels[ids]
                if args.device == "cuda":
                    from MyFlows.core.device import xp
                    xp.cuda.get_current_stream().synchronize()
                loaded = time.perf_counter()
                phases.begin(step)
                train_span.start()
                opt.one_step(phases)
                train_span.stop()
                with phases("optimizer"):
                    opt.update()
                value = float(asnumpy(loss.value))
                good = int((asnumpy(logits.value).argmax(axis=1) == labels[ids]).sum())
                end = time.perf_counter()
                extra = phases.row()
                step += 1
                correct += good
                losses.append(value)
                dashboard.log_train_step(step, epoch=epoch, loss=value, accuracy=good / args.batch,
                                         data_load_ms=(loaded-begin)*1000, train_step_ms=train_span.elapsed_ms(), step_time_ms=(end-loaded)*1000,
                                         batch_size=args.batch, labels=labels[ids], task="classification", num_classes=10, extra=extra)
            val = evaluate(model, nodes, images, labels, validation, args.batch)
            train_loss, accuracy = float(np.mean(losses)), correct / (len(ids_epoch) * args.batch)
            elapsed = time.perf_counter()-start
            dashboard.log_epoch(epoch, loss=train_loss, accuracy=accuracy, learning_rate=args.lr, epoch_time_s=elapsed,
                                validation_loss=val["loss"], validation_accuracy=val["accuracy"])
            epochs.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": accuracy, "val_loss": val["loss"],
                           "val_accuracy": val["accuracy"], "epoch_s": elapsed})
            if val["accuracy"] > best:
                best = val["accuracy"]
                ms.save_checkpoint([model], None, epoch, filepath=str(directory / "best"), score=best, score_name="val_accuracy")
            print(f"epoch {epoch} train {accuracy:.3f}, val {val['accuracy']:.3f}", flush=True)
    result = {"status": "succeeded", "exit_code": 0, "task": "cifar10_classification", "environment": environment(include_cuda=args.device=="cuda"),
              "epochs": epochs, "steps": step, "metrics": {"best_val_accuracy": best, "final_val_accuracy": epochs[-1]["val_accuracy"]},
              "memory": phases.summary(), "backend": backend, "test": None, "checkpoint_sha256": _sha(directory / "best.npz"),
              "config_sha256": _sha(directory / "config.json"), "checkpoint_metadata_sha256": _sha(directory / "best.json")}
    write_result(directory, result)


def score_test(directory):
    import MyFlows as ms

    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))["args"]
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if result["status"] != "succeeded" or result["checkpoint_sha256"] != _sha(directory / "best.npz"):
        raise ValueError("expected the completed validation-selected checkpoint")
    if result.get("config_sha256") != _sha(directory / "config.json") or result.get("checkpoint_metadata_sha256") != _sha(directory / "best.json"):
        raise ValueError("training configuration or checkpoint metadata changed after validation")
    target = directory / "official-test.json"
    if target.exists():
        scored = json.loads(target.read_text(encoding="utf-8"))
        if scored["checkpoint_sha256"] != result["checkpoint_sha256"]:
            raise ValueError("test score belongs to another checkpoint")
        if scored["config_sha256"] != result["config_sha256"]:
            raise ValueError("test score belongs to another training configuration")
        if result.get("test") != scored["test"]:
            result["test"] = scored["test"]
            write_result(directory, result)
        print(json.dumps(scored["test"]))
        return scored
    locked = {"schema_version": 1, "config_sha256": _sha(directory / "config.json"), "checkpoint_sha256": result["checkpoint_sha256"],
              "selection": "highest validation accuracy, earliest epoch on ties"}
    _write(directory / "locked.json", locked)
    model, nodes, _ = build(config)
    ms.load_checkpoint([model], None, str(directory / "best"))
    images, labels = cifar10.load(ROOT / config["data"], "test")
    test = evaluate(model, nodes, images, labels, np.arange(len(labels)), config["batch"])
    scored = {**locked, "test": test, "majority_baseline": float(np.bincount(labels).max() / len(labels))}
    _write(target, scored)
    result["test"] = test
    write_result(directory, result)
    print(json.dumps(scored))
    return scored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", default="runs/datasets/cifar10")
    parser.add_argument("--arch", choices=("small", "resnet18"), default="small")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--phase-timing-every", type=int, default=20)
    parser.add_argument("--monitor", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--score-test", action="store_true", help="lock a validation-selected checkpoint and score the official test split")
    args = parser.parse_args(argv)
    directory = (ROOT / args.run_dir).resolve()
    if min(args.epochs, args.batch, args.width) < 1 or not 0 < args.lr < float("inf"):
        parser.error("invalid training budget or learning rate")
    if args.score_test:
        with GpuLock(label="CIFAR-10 official test", run_dir=directory):
            score_test(directory)
        return
    directory.mkdir(parents=True, exist_ok=False)
    with GpuLock(label="CIFAR-10 training", run_dir=directory, enabled=args.device=="cuda"):
        try:
            train(args, directory)
        except Exception as exc:
            write_result(directory, {"status": "failed", "exit_code": 1, "failure": {"kind": type(exc).__name__, "message": str(exc)},
                                     "environment": environment(include_cuda=args.device=="cuda")})
            raise


if __name__ == "__main__":
    main()
