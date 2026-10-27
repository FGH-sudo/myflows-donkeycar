"""E2 transfer validation on an image classification task: ImageNet-pretrained ResNet18 / MobileNetV2 / AlexNet,
head replaced, stages frozen, trained on DonkeyCar camera frames classified by steering direction.

semester_spec §4.6 suggests CIFAR-10; it is not on this machine, so the classification task is built from the
DonkeyCar tub instead: each frame's steering angle falls in one of ``--classes`` bins whose edges are the
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
    return imread_nchw(Path(path), (int(w), int(h)), np.float32)[0], np.int64(label)


def class_edges(angles, classes: int) -> list[float]:
    """Bin edges: the 1/k … (k-1)/k quantiles of the training angles."""
    return [float(np.quantile(np.asarray(angles, dtype=np.float64), q / classes)) for q in range(1, classes)]


def classify(angle: float, edges: list[float]) -> int:
    return int(np.searchsorted(np.asarray(edges), angle, side="right"))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", default="resnet18", choices=("resnet18", "mobilenet_v2", "alexnet"))
    ap.add_argument("--freeze-stages", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true", help="随机初始化（对照）")
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


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import MyFlows as ms
    from MyFlows.core.device import set_device, xp
    from MyFlows.data.pipeline import MultiprocessDataLoader
    from MyFlows.utils import pretrained
    from apps.train.train_myflows_donkey import _resolve_conv_backend

    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{args.arch}-f{args.freeze_stages}" + ("-scratch" if args.no_pretrained else "")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "runs" / "transfer" / f"{tag}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data: frame-ordered rows, a time-block split, balanced steering classes from the training angles
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

    train_s, val_s, test_s = samples("train"), samples("val"), samples("test")
    counts = {name: np.bincount([s[1] for s in part], minlength=args.classes).tolist()
              for name, part in (("train", train_s), ("val", val_s), ("test", test_s))}
    print(f"[data] train={len(train_s)} val={len(val_s)} test={len(test_s)} purged={len(split.get('purged', []))} "
          f"class edges={[round(e, 4) for e in edges]} counts={counts}", flush=True)

    lock = GpuLock(label=f"transfer_classify {tag}", run_dir=out_dir) if args.device == "cuda" else None
    if lock is not None:
        lock.acquire(timeout_s=args.gpu_lock_timeout or None)
    try:
        set_device(args.device)
        ms.set_default_dtype(np.float32)
        conv_backend = _resolve_conv_backend(args.conv_backend, args.device, np.float32)
        initializer = ms.make_initializer("kaiming_normal", seed=args.seed)
        model = pretrained.build_model(args.arch, args.classes, dropout=0.0, initializer=initializer,
                                       conv_backend=conv_backend, input_norm="imagenet", name=f"{args.arch}_transfer")
        report = None
        if not args.no_pretrained:
            report = pretrained.load_torchvision_weights(
                model, ROOT / "runs" / "pretrained" / "converted" / f"{args.arch}.npz", replace_head=True)
        freeze = pretrained.freeze_stages(model, args.freeze_stages)
        model.train()
        x_var = ms.Variable(np.zeros((args.batch, 3, size, size), np.float32), name="X", requires_grad=False)
        y_var = ms.Variable(np.zeros((args.batch,), np.int64), name="y", requires_grad=False)
        logits = model(x_var)
        loss = ms.CrossEntropy(logits, y_var)
        graph = ms.Graph(loss)
        opt = ms.Adam(graph, learning_rate=args.lr)
        frozen_before = pretrained.frozen_fingerprint(model)
        print(f"[model] {args.arch} conv={conv_backend} pretrained={report['loaded'] if report else 0}/"
              f"{report['expected'] if report else 0} frozen={freeze['frozen']} trainable={freeze['trainable_values']:,}",
              flush=True)

        initial = evaluate(graph, x_var, y_var, logits, model, val_s, args.batch, xp)
        print(f"[val] before training: loss={initial['loss']:.4f} acc={initial['accuracy']:.3f}", flush=True)
        history, step_losses, step_ms = [], [], []
        global_step = 0
        started = time.time()
        for epoch in range(args.epochs):
            loader = MultiprocessDataLoader(train_s, args.batch, num_workers=args.num_workers, shuffle=True,
                                            seed=args.seed * 1000 + epoch, load_fn=_load)
            for x_items, y_items in loader:
                if len(x_items) < args.batch:  # fixed graph batch: the last partial batch is dropped
                    continue
                t0 = time.perf_counter()
                x_var.value = xp.asarray(np.asarray(x_items, dtype=np.float32))
                y_var.value = xp.asarray(np.asarray(y_items, dtype=np.int64))
                opt.one_step()
                opt.update()
                value = float(loss.value)
                step_ms.append((time.perf_counter() - t0) * 1000.0)
                step_losses.append(value)
                global_step += 1
                if not math.isfinite(value):
                    raise SystemExit(f"loss became {value} at step {global_step}")
                if global_step % 20 == 0:
                    print(f"[train] epoch {epoch + 1} step {global_step} loss={value:.4f} "
                          f"({np.median(step_ms[-20:]):.0f} ms/step)", flush=True)
            val = evaluate(graph, x_var, y_var, logits, model, val_s, args.batch, xp)
            history.append({"epoch": epoch + 1, "step": global_step, **{f"val_{k}": v for k, v in val.items()},
                            "train_loss_epoch_mean": float(np.mean(step_losses[-len(train_s) // args.batch:]))})
            print(f"[val] epoch {epoch + 1}: loss={val['loss']:.4f} acc={val['accuracy']:.3f}", flush=True)
        train_s_elapsed = time.time() - started
        test = evaluate(graph, x_var, y_var, logits, model, test_s, args.batch, xp)
        print(f"[test] loss={test['loss']:.4f} acc={test['accuracy']:.3f} (scored once)", flush=True)
        frozen_after = pretrained.frozen_fingerprint(model)
        checkpoint = out_dir / "checkpoint"
        ms.save_checkpoint([model], opt, args.epochs, filepath=str(checkpoint), loss=history[-1]["val_loss"],
                           training={"task": "transfer_classify", "arch": args.arch, "classes": args.classes,
                                     "freeze_stages": args.freeze_stages, "class_edges": edges})
        peak_mb = xp.get_default_memory_pool().total_bytes() / 2**20 if args.device == "cuda" else None
    finally:
        if lock is not None:
            lock.release()

    head = max(10, len(step_losses) // 10)
    result = {
        "task": "transfer_classify", "dataset": "DonkeyCar steering bins (CIFAR-10 not available locally)",
        "args": vars(args), "arch": args.arch, "pretrained": not args.no_pretrained,
        "load_report": pretrained.summarize_load_report(report) if report else None,
        "freeze": freeze, "frozen_before": frozen_before, "frozen_after": frozen_after,
        "frozen_unchanged": frozen_before == frozen_after,
        "classes": args.classes, "class_edges": edges, "class_counts": counts,
        "split": {"kind": "time_block", "train": len(train_s), "val": len(val_s), "test": len(test_s),
                  "purged": len(split.get("purged", [])), "block_size": args.block_size, "seed": args.split_seed},
        "input": {"size": [size, size], "normalization": "imagenet mean/std"},
        "val_before": initial, "epochs": history, "test": test,
        "train_loss": {"first_steps_mean": float(np.mean(step_losses[:head])),
                       "last_steps_mean": float(np.mean(step_losses[-head:])), "steps": global_step},
        "loss_decreased": history[-1]["val_loss"] < initial["loss"]
                          and float(np.mean(step_losses[-head:])) < float(np.mean(step_losses[:head])),
        "timing": {"train_s": train_s_elapsed, "step_ms_median": float(np.median(step_ms)) if step_ms else None,
                   "lock_wait_s": lock.wait_s if lock is not None else None},
        "memory": {"peak_pool_mb": peak_mb}, "checkpoint": str(checkpoint), "conv_backend": conv_backend,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[result] val loss {initial['loss']:.4f} -> {history[-1]['val_loss']:.4f}, frozen unchanged="
          f"{result['frozen_unchanged']}, loss decreased={result['loss_decreased']} -> {out_dir / 'result.json'}",
          flush=True)
    return 0 if result["frozen_unchanged"] and result["loss_decreased"] else 1


if __name__ == "__main__":
    sys.exit(main())
