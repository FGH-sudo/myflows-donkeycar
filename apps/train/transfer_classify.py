"""E2 transfer validation on an image classification task: ImageNet-pretrained ResNet18 / MobileNetV2 / AlexNet,
head replaced, stages frozen, trained on DonkeyCar camera frames classified by steering direction.

For the DonkeyCar task each frame's steering angle falls in one of ``--classes`` bins whose edges are the
quantiles of the training angles (balanced classes: left / straight / right for 3). The rest follows §4.6:

- the source weights' input: frames resized to 224x224 and normalised with the ImageNet mean/std (in-graph);
- a fixed split: contiguous time blocks (neighbouring frames never on both sides) into train/val/test; the
  test split is scored once, at the end;
- the run records the pretrained load report (success rate, missing, checksums), the freeze summary and a
  byte-level fingerprint of the frozen tensors before and after training, the validation loss before
  training (pretrained backbone, fresh head) and after every epoch, and saves the new checkpoint.

    python -m apps.train.transfer_classify --arch mobilenet_v2 --freeze-stages 3 --max-samples 2000 --epochs 2
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from apps.common.donkey_data import load_donkey_index
from apps.common.gpu_lock import GpuLock
from apps.common.image_preprocess import imread_nchw
from apps.common.splits import build_time_block_split, frame_numbers

ROOT = Path(__file__).resolve().parents[2]


def _load(sample):
    path, label, w, h = sample
    if isinstance(path, np.ndarray):
        import cv2
        image = cv2.resize(path.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LINEAR)
        return image.transpose(2, 0, 1).astype(np.float32) / 255., np.int64(label)
    return imread_nchw(Path(path), (int(w), int(h)), np.float32)[0], np.int64(label)


def class_edges(angles, classes: int) -> list[float]:
    """Bin edges: the 1/k … (k-1)/k quantiles of the training angles."""
    return [float(np.quantile(np.asarray(angles, dtype=np.float64), q / classes)) for q in range(1, classes)]


def classify(angle: float, edges: list[float]) -> int:
    return int(np.searchsorted(np.asarray(edges), angle, side="right"))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", default="resnet18", choices=("small", "resnet18", "mobilenet_v2", "alexnet"))
    ap.add_argument("--dataset", default="donkey", choices=("donkey", "cifar10"))
    ap.add_argument("--cifar-root", default="runs/datasets/cifar10")
    ap.add_argument("--dropout", type=float, default=0.)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--feature-cache", action="store_true")
    ap.add_argument("--require-loss-decrease", action=argparse.BooleanOptionalAction, default=True,
                    help="use the historical loss-decrease check as an exit gate")
    ap.add_argument("--freeze-stages", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true", help="随机初始化（对照）")
    ap.add_argument("--weights", help="转换后的 ImageNet NPZ，默认使用仓库 runs/pretrained/converted")
    ap.add_argument("--data", default="mycar/data")
    ap.add_argument("--max-samples", type=int, default=2000, help="按帧序取前 N 条（0 = 全部）")
    ap.add_argument("--val-size", type=int, default=300)
    ap.add_argument("--test-size", type=int, default=300)
    ap.add_argument("--block-size", type=int, default=50)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr-mult", type=float, default=1.)
    ap.add_argument("--stage-lr-mults", help="stem:0.01,layer1:0.1")
    ap.add_argument("--unfreeze-schedule", help="0:4,2:3,4:2")
    ap.add_argument("--freeze-mode", choices=("both", "params", "bn"), default="both")
    ap.add_argument("--bn-unfreeze", choices=("follow", "keep-frozen"), default="follow")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--conv-backend", default="auto")
    ap.add_argument("--out-dir", default=None, help="默认 runs/transfer/<arch>-f<N>-<时间>")
    ap.add_argument("--gpu-lock-timeout", type=float, default=0.0)
    return ap


def evaluate(graph, x_var, y_var, logits, model, samples, batch, xp_module) -> dict:
    from MyFlows.core.device import asnumpy
    from MyFlows.data.pipeline import MultiprocessDataLoader

    model.eval()
    total_loss = correct = count = 0.0
    for x_items, y_items in MultiprocessDataLoader(samples, batch, num_workers=0, shuffle=False, load_fn=_load):
        real = len(x_items)
        x = np.asarray(x_items, dtype=np.float32)
        y = np.asarray(y_items, dtype=np.int64)
        if real < batch:
            x = np.concatenate([x, np.repeat(x[-1:], batch - real, axis=0)])
            y = np.concatenate([y, np.repeat(y[-1:], batch - real)])
        x_var.value, y_var.value = xp_module.asarray(x), xp_module.asarray(y)
        graph.forward()
        z = np.asarray(asnumpy(logits.value), dtype=np.float64)[:real]
        z = z - z.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        total_loss += float(-log_p[np.arange(real), y[:real]].sum())
        correct += float((log_p.argmax(axis=1) == y[:real]).sum())
        count += real
    model.train()
    return {"loss": total_loss / count, "accuracy": correct / count, "samples": int(count)}


def _datasets(args):
    size = int(args.image_size)
    if args.dataset == "cifar10":
        from apps.common import cifar10

        images, labels = cifar10.load(ROOT / args.cifar_root, "train")
        training, validation = cifar10.stratified_split(labels, seed=args.split_seed)
        retained_test = cifar10.balanced_subset(training, labels, args.test_size, seed=args.split_seed+1)
        training = np.setdiff1d(training, retained_test)
        ntrain = args.max_samples - args.val_size - args.test_size if args.max_samples else len(training)
        split = {"train": cifar10.balanced_subset(training, labels, ntrain, seed=args.split_seed),
                 "val": cifar10.balanced_subset(validation, labels, args.val_size, seed=args.split_seed), "test": retained_test}
        parts = {name: [(images[i], int(labels[i]), size, size) for i in ids] for name, ids in split.items()}
        manifest = {name: [[int(i), int(labels[i])] for i in ids] for name, ids in split.items()}
        return parts, split, [], manifest
    data_dir = (ROOT / args.data).resolve()
    rows = load_donkey_index(data_dir, fixed_throttle=0.5, angle_scale=1.0)
    rows.sort(key=lambda r: (frame_numbers([str(r[0])])[0], str(r[0])))
    if args.max_samples:
        rows = rows[:args.max_samples]
    split = build_time_block_split(frame_numbers([str(r[0]) for r in rows]), val_size=args.val_size,
                                   test_size=args.test_size, block_size=args.block_size, seed=args.split_seed)
    edges = class_edges([rows[i][1] for i in split["train"]], args.classes)
    size = int(args.image_size)

    def samples(name):
        return [(str(data_dir / rows[i][0]), classify(rows[i][1], edges), size, size) for i in split[name]]

    parts = {name: samples(name) for name in ("train", "val", "test")}
    return parts, split, edges, {name: [list(s) for s in values] for name, values in parts.items()}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import MyFlows as ms
    from MyFlows.core.device import asnumpy, set_device, xp
    from MyFlows.data.pipeline import MultiprocessDataLoader
    from MyFlows.utils import pretrained
    from MyFlows.transfer import TransferLearningManager
    from MyFlows.transfer.manager import parse_lr_multipliers
    from apps.train.train_myflows_donkey import _resolve_conv_backend, _DeviceSpan, _StepPhases
    from MyFlows.monitoring.schema import record
    from MyFlows.utils.training_dashboard import TrainingDashboard
    from MyFlows.utils.transforms import augment_chw_batch, default_train_transforms
    from apps.common.environment import environment
    from apps.train.common.outcome import write_result

    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{args.arch}-f{args.freeze_stages}" + ("-scratch" if args.no_pretrained else "")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "runs" / "transfer" / f"{tag}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "cifar10":
        args.classes = 10
    if args.arch == "small":
        args.no_pretrained = True
    if args.feature_cache and args.augment:
        raise ValueError("feature cache requires deterministic input; disable random augmentation")
    size = int(args.image_size)
    parts, split, edges, sample_manifest = _datasets(args)
    train_s, val_s, test_s = (parts[name] for name in ("train", "val", "test"))
    if min(args.batch, args.epochs, len(val_s), len(test_s)) < 1 or len(train_s) < args.batch:
        raise ValueError("need positive epochs, validation/test samples and a complete training batch")
    counts = {name: np.bincount([s[1] for s in part], minlength=args.classes).tolist()
              for name, part in (("train", train_s), ("val", val_s), ("test", test_s))}
    print(f"[data] train={len(train_s)} val={len(val_s)} test={len(test_s)} purged={len(split.get('purged', []))} "
          f"class edges={[round(e, 4) for e in edges]} counts={counts}", flush=True)

    lock = GpuLock(label=f"transfer_classify {tag}", run_dir=out_dir) if args.device == "cuda" else None
    if lock is not None:
        lock.acquire(timeout_s=args.gpu_lock_timeout or None)
    outputs = ExitStack()
    try:
        set_device(args.device)
        ms.set_default_dtype(np.float32)
        conv_backend = _resolve_conv_backend(args.conv_backend, args.device, np.float32)
        initializer = ms.make_initializer("kaiming_normal", seed=args.seed)
        strategy = dict(freeze_stages=args.freeze_stages, backbone_lr_mult=args.backbone_lr_mult,
                        stage_lr_mults=parse_lr_multipliers(args.stage_lr_mults), unfreeze_schedule=args.unfreeze_schedule,
                        freeze_mode=args.freeze_mode, bn_unfreeze=args.bn_unfreeze)
        if args.arch == "small":
            from apps.train.classification_model import SmallClassifier
            manager = TransferLearningManager(SmallClassifier(args.classes, dropout=args.dropout, seed=args.seed, backend=conv_backend),
                                              **strategy)
        else:
            manager = TransferLearningManager.build(args.arch, args.classes, dropout=args.dropout, initializer=initializer,
                                                   dropout_seed=args.seed, conv_backend=conv_backend, input_norm="imagenet",
                                                   name=f"{args.arch}_transfer", pretrained=None if args.no_pretrained else
                                                   (Path(args.weights) if args.weights else ROOT / "runs/pretrained/converted" / f"{args.arch}.npz"), **strategy)
        model, report, freeze = manager.model, manager.load_report, manager.freeze
        model.train()
        x_var = ms.Variable(np.zeros((args.batch, 3, size, size), np.float32), name="X", requires_grad=False)
        y_var = ms.Variable(np.zeros((args.batch,), np.int64), name="y", requires_grad=False)
        logits = model(x_var)
        loss = ms.CrossEntropy(logits, y_var)
        graph = ms.Graph(loss)
        opt = ms.Adam(graph, learning_rate=args.lr, param_groups=manager.parameter_groups())
        manager.begin_tracking()
        manager.epoch(0)
        frozen_before = pretrained.frozen_fingerprint(model)
        print(f"[model] {args.arch} conv={conv_backend} pretrained={report['loaded'] if report else 0}/"
              f"{report['expected'] if report else 0} frozen={freeze['frozen']} trainable={freeze['trainable_values']:,}",
              flush=True)

        initial = evaluate(graph, x_var, y_var, logits, model, val_s, args.batch, xp)
        print(f"[val] before training: loss={initial['loss']:.4f} acc={initial['accuracy']:.3f}", flush=True)
        eval_nodes = graph, x_var, y_var, logits
        cache = None
        if args.feature_cache:
            arrays, cache = manager.cache_features(out_dir / "feature-cache", parts, sample_manifest=sample_manifest,
                                                   input_node=x_var, batch=args.batch, load_input=lambda s: _load(s)[0],
                                                   preprocessing={"size": size, "norm": getattr(model, "input_norm", None),
                                                                  "backend": conv_backend, "dataset": args.dataset})
            x_var = ms.Variable(np.zeros((args.batch, arrays["train"].shape[1]), np.float32), requires_grad=False)
            logits = model.forward_head(x_var)
            loss = ms.CrossEntropy(logits, y_var)
            graph = ms.Graph(loss)
            opt.graph = graph
            training_samples = [(arrays["train"][i], s[1]) for i, s in enumerate(train_s)]
        else:
            training_samples = train_s
        transform = default_train_transforms(seed=args.seed) if args.augment else None
        dashboard = TrainingDashboard(out_dir / "tensorboard", enabled=False, jsonl_path=out_dir / "metrics.jsonl")
        outputs.callback(dashboard.close)
        (out_dir / "config.json").write_text(json.dumps(record({"kind": "single", "task": "transfer_classify",
                           "args": vars(args), "samples": len(train_s), "val_samples": len(val_s),
                           "monitoring": {"requested": "off", "enabled": False}}, clocks=True), indent=2), encoding="utf-8")
        train_span, phases = _DeviceSpan(args.device), _StepPhases(args.device, 20)
        history, step_losses, step_ms = [], [], []
        global_step = 0
        started = time.time()
        for epoch in range(args.epochs):
            freeze_entry = manager.epoch(epoch)
            epoch_start = time.perf_counter()
            correct = seen = 0
            epoch_start_step = len(step_losses)
            loader = MultiprocessDataLoader(training_samples, args.batch, num_workers=0 if cache else args.num_workers,
                                            shuffle=True, seed=args.seed * 1000 + epoch,
                                            load_fn=(lambda sample: sample) if cache else _load)
            load_start = time.perf_counter()
            for x_items, y_items in loader:
                if len(x_items) < args.batch:  # fixed graph batch: the last partial batch is dropped
                    continue
                t0 = time.perf_counter()
                xb, yb = np.asarray(x_items, dtype=np.float32), np.asarray(y_items, dtype=np.int64)
                if transform:
                    augment_chw_batch(xb, yb, transform)
                x_var.value, y_var.value = xp.asarray(xb), xp.asarray(yb)
                phases.begin(global_step)
                train_span.start()
                opt.one_step(phases)
                train_span.stop()
                with phases("optimizer"):
                    opt.update()
                value = float(loss.value)
                good = int((asnumpy(logits.value).argmax(axis=1) == yb).sum())
                correct += good
                seen += len(yb)
                step_ms.append((time.perf_counter() - t0) * 1000.0)
                step_losses.append(value)
                global_step += 1
                if not math.isfinite(value):
                    raise SystemExit(f"loss became {value} at step {global_step}")
                dashboard.log_train_step(global_step, epoch=epoch+1, loss=value, accuracy=good/len(yb),
                                         data_load_ms=(t0-load_start)*1000, train_step_ms=train_span.elapsed_ms(),
                                         step_time_ms=step_ms[-1], batch_size=len(yb), labels=yb,
                                         task="classification", num_classes=args.classes, extra=phases.row())
                if global_step % 20 == 0:
                    print(f"[train] epoch {epoch + 1} step {global_step} loss={value:.4f} "
                          f"({np.median(step_ms[-20:]):.0f} ms/step)", flush=True)
                load_start = time.perf_counter()
            train_epoch_s = time.perf_counter() - epoch_start
            val = evaluate(*eval_nodes, model, val_s, args.batch, xp)
            train_loss, train_accuracy = float(np.mean(step_losses[epoch_start_step:])), correct / seen
            epoch_wall_s = time.perf_counter() - epoch_start
            dashboard.log_epoch(epoch+1, loss=train_loss, accuracy=train_accuracy, learning_rate=args.lr, epoch_time_s=epoch_wall_s,
                                validation_loss=val["loss"], validation_accuracy=val["accuracy"],
                                extra={"freeze_stages": freeze_entry["freeze_stages"]})
            history.append({"epoch": epoch + 1, "step": global_step, "freeze_stages": freeze_entry["freeze_stages"],
                            "group_learning_rates": opt.group_learning_rates(), **{f"val_{k}": v for k, v in val.items()},
                            "train_loss_epoch_mean": train_loss, "train_accuracy": train_accuracy, "train_samples": seen,
                            "train_epoch_s": train_epoch_s, "epoch_wall_s": epoch_wall_s})
            print(f"[val] epoch {epoch + 1}: loss={val['loss']:.4f} acc={val['accuracy']:.3f}", flush=True)
        train_s_elapsed = time.time() - started
        test = evaluate(*eval_nodes, model, test_s, args.batch, xp)
        print(f"[test] loss={test['loss']:.4f} acc={test['accuracy']:.3f} (scored once)", flush=True)
        frozen_after = pretrained.frozen_fingerprint(model)
        frozen_unchanged = manager.verify_frozen()
        checkpoint = out_dir / "checkpoint"
        ms.save_checkpoint([model], opt, args.epochs, filepath=str(checkpoint), loss=history[-1]["val_loss"],
                           training={"task": "transfer_classify", "arch": args.arch, "classes": args.classes,
                                     "freeze_stages": args.freeze_stages, "class_edges": edges})
        peak_mb = xp.get_default_memory_pool().total_bytes() / 2**20 if args.device == "cuda" else None
    finally:
        outputs.close()
        if lock is not None:
            lock.release()

    head = max(10, len(step_losses) // 10)
    result = {
        "task": "transfer_classify", "dataset": "DonkeyCar steering bins" if args.dataset == "donkey" else "CIFAR-10 training archive subsets",
        "args": vars(args), "arch": args.arch, "pretrained": not args.no_pretrained,
        "load_report": pretrained.summarize_load_report(report) if report else None,
        "freeze": freeze, "frozen_before": frozen_before, "frozen_after": frozen_after,
        "frozen_unchanged": frozen_unchanged, "transfer": {**manager.result(), "cache": cache},
        "classes": args.classes, "class_edges": edges, "class_counts": counts,
        "split": {"kind": "time_block" if args.dataset == "donkey" else "balanced", "train": len(train_s), "val": len(val_s), "test": len(test_s),
                  "purged": len(split.get("purged", [])), "block_size": args.block_size, "seed": args.split_seed},
        "input": {"size": [size, size], "normalization": "imagenet mean/std" if args.arch != "small" else "RGB /255"},
        "val_before": initial, "epochs": history, "test": test,
        "train_loss": {"first_steps_mean": float(np.mean(step_losses[:head])),
                       "last_steps_mean": float(np.mean(step_losses[-head:])), "steps": global_step},
        "loss_decreased": history[-1]["val_loss"] < initial["loss"]
                          and float(np.mean(step_losses[-head:])) < float(np.mean(step_losses[:head])),
        "timing": {"train_s": train_s_elapsed, "step_ms_median": float(np.median(step_ms)) if step_ms else None,
                   "lock_wait_s": lock.wait_s if lock is not None else None},
        "memory": {"peak_pool_mb": peak_mb}, "checkpoint": str(checkpoint), "conv_backend": conv_backend,
    }
    passed = bool(result["frozen_unchanged"] and (result["loss_decreased"] or not args.require_loss_decrease))
    result.update(status="succeeded" if passed else "failed", exit_code=0 if passed else 1,
                  environment=environment(include_cuda=args.device == "cuda"))
    write_result(out_dir, result)
    print(f"[result] val loss {initial['loss']:.4f} -> {history[-1]['val_loss']:.4f}, frozen unchanged="
          f"{result['frozen_unchanged']}, loss decreased={result['loss_decreased']} -> {out_dir / 'result.json'}",
          flush=True)
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
