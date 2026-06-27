#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MyFlows VGG-11 在 Donkey 道路图像上做转向角离散分类。

标签：将文件名解析的 angle 离散为 num_classes 类（默认 5 档）。
用法（项目根目录 d:\\DL\\testmyflow）:
  python -m apps.train.train_vgg_donkey_classify --data mycar/data --epochs 10 --batch 2 \\
      --max-samples 0 --augment --device auto --export-onnx
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import MyFlows as ms
from apps.common.donkey_data import load_donkey_index
from apps.common.image_preprocess import imread_nchw, pad_fixed_batch
from MyFlows.data.pipeline import MultiprocessDataLoader
from tools.device_runtime import (
    myflows_asnumpy,
    myflows_scalar_float,
    print_myflows_device,
    resolve_myflows_device,
)
from MyFlows.layers.vgg import VGG11, angle_to_class
from MyFlows.utils.training_dashboard import TrainingDashboard
from MyFlows.utils.training_observer import global_gradient_norm
from MyFlows.utils.transforms import augment_chw_batch, default_train_transforms
from MyFlows.utils.viz import TrainingHistory
from apps.train.common.checkpoints import checkpoint_stem
from apps.train.common.logging import TeeLogger


def _accuracy_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    labels = y_true.reshape(-1).astype(int)
    return float(np.mean(preds == labels))


def _load_classification_sample(sample: tuple[str, int, int, int, str]) -> tuple[np.ndarray, np.ndarray]:
    path_str, label, image_w, image_h, dtype_name = sample
    dtype = np.float32 if dtype_name == "float32" else np.float64
    x = imread_nchw(Path(path_str), (int(image_w), int(image_h)), dtype)[0]
    y = np.array([int(label)], dtype=dtype)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(description="MyFlows VGG-11 Donkey 角度分类")
    ap.add_argument("--data", type=str, default="mycar/data")
    ap.add_argument("--catalog", type=str, default="catalog_generated.catalog")
    ap.add_argument("--num-classes", type=int, default=5, choices=(3, 5))
    ap.add_argument("--fixed-throttle", type=float, default=0.5)
    ap.add_argument("--angle-scale", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"))
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="MyFlows 计算设备：auto 有 GPU 则用 CUDA(CuPy)",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="mycar/logs/vgg11_classify_checkpoint")
    ap.add_argument("--best-out", type=str, default="mycar/models/vgg11_classify_best")
    ap.add_argument("--log-dir", type=str, default="mycar/logs")
    ap.add_argument("--log-file", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument("--graph-opt", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--mixup", action="store_true")
    ap.add_argument("--cutmix", action="store_true")
    ap.add_argument("--logdir", type=str, default=None)
    ap.add_argument("--tb-log-interval", type=int, default=10)
    ap.add_argument("--no-tensorboard", action="store_true")
    ap.add_argument("--tb-image-interval", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--tb-grad-interval", type=int, default=100)
    ap.add_argument("--tb-param-interval", type=int, default=200)
    ap.add_argument("--tb-activation-interval", type=int, default=200)
    ap.add_argument("--tb-max-hist-params", type=int, default=24)
    ap.add_argument("--tb-feature-channels", type=int, default=16)
    args = ap.parse_args()

    data_dir = (ROOT / args.data).resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"数据目录不存在: {data_dir}")

    log_dir = (ROOT / args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = (ROOT / args.log_file).resolve() if args.log_file else (
        log_dir / f"vgg_train_{run_stamp}.log"
    )
    tb_logdir = (ROOT / args.logdir).resolve() if args.logdir else (
        log_dir / "tensorboard" / f"vgg11_{run_stamp}"
    )
    tee = TeeLogger(log_file)
    sys.stdout = tee
    sys.stderr = tee

    raw_index = load_donkey_index(
        data_dir, args.fixed_throttle, args.angle_scale, args.catalog
    )
    if not raw_index:
        raise SystemExit("未找到训练样本")
    if args.max_samples and args.max_samples > 0:
        raw_index = raw_index[: args.max_samples]

    index: list[tuple[Path, int]] = []
    for rel, angle, _ in raw_index:
        index.append((rel, angle_to_class(angle, args.num_classes)))

    first_p = data_dir / index[0][0]
    img0 = cv2.imread(str(first_p), cv2.IMREAD_COLOR)
    if img0 is None:
        raise SystemExit(f"首帧读取失败: {first_p}")
    h0, w0 = img0.shape[:2]
    print(f"输入 H×W = {h0}×{w0}, num_classes={args.num_classes}")

    dtype = np.float32 if args.dtype == "float32" else np.float64
    B = int(args.batch)
    h, w = h0, w0
    K = int(args.num_classes)

    device = resolve_myflows_device(args.device)
    print_myflows_device(device, args.device)

    x_var = ms.Variable(np.zeros((B, 3, h, w), dtype=dtype), name="X")
    y_var = ms.Variable(np.zeros((B, 1), dtype=dtype), name="y")

    model = VGG11(
        in_channels=3,
        num_classes=K,
        image_h=h,
        image_w=w,
        name="vgg11_donkey",
    )
    logits = model(x_var)
    loss_node = ms.CrossEntropy(logits, y_var, name="loss")
    graph = ms.Graph(loss_node, optimize=bool(args.graph_opt))
    opt = ms.Adam(graph, learning_rate=args.lr)

    out_base = checkpoint_stem((ROOT / args.out).resolve())
    best_base = checkpoint_stem((ROOT / args.best_out).resolve())
    out_base.parent.mkdir(parents=True, exist_ok=True)
    best_base.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_loss = None
    if args.resume:
        loaded_epoch, saved_score = ms.load_checkpoint([model], opt, str(out_base))
        start_epoch = max(0, int(loaded_epoch) + 1)
        best_loss = -saved_score if saved_score else None

    n = len(index)
    steps = (n + B - 1) // B
    dashboard = TrainingDashboard(
        tb_logdir,
        enabled=not args.no_tensorboard,
        grad_interval=args.tb_grad_interval,
        param_interval=args.tb_param_interval,
        activation_interval=args.tb_activation_interval,
        image_interval=args.tb_image_interval,
        log_interval=args.tb_log_interval,
        max_hist_params=args.tb_max_hist_params,
        feature_channels=args.tb_feature_channels,
    )
    if dashboard.active:
        dashboard.log_run_config(args, {"samples": n, "input_h": h, "input_w": w, "steps_per_epoch": steps})
        all_labels = np.asarray([cls for _, cls in index], dtype=np.int64)
        dashboard.log_dataset_summary(all_labels, task="classification", num_classes=K)
    train_transform = default_train_transforms(seed=42) if args.augment else None
    history = TrainingHistory()
    global_step = 0
    dataset = [
        (str(data_dir / rel), cls, w, h, args.dtype)
        for rel, cls in index
    ]

    for ep in range(start_epoch, start_epoch + args.epochs):
        total_loss = 0.0
        total_acc = 0.0
        count = 0
        epoch_start = time.perf_counter()
        loader = MultiprocessDataLoader(
            dataset,
            B,
            num_workers=int(args.num_workers),
            shuffle=True,
            seed=ep,
            load_fn=_load_classification_sample,
        )
        load_start = time.perf_counter()
        for s, (x_items, y_items) in enumerate(loader):
            after_load = time.perf_counter()
            x_batch = np.asarray(x_items, dtype=dtype)
            y_batch = np.asarray(y_items, dtype=dtype).reshape(-1, 1)
            x_batch, y_batch = pad_fixed_batch(x_batch, y_batch, B)

            if train_transform or args.mixup or args.cutmix:
                x_orig = x_batch.copy()
                y_float = y_batch.astype(np.float64)
                augment_chw_batch(
                    x_batch,
                    y_float,
                    train_transform,
                    mixup=bool(args.mixup),
                    cutmix=bool(args.cutmix),
                )
                y_batch = np.round(y_float).astype(dtype).reshape(B, 1)
                dashboard.log_augmentation(global_step, x_orig, x_batch, batch_size=B)

            x_var.value = x_batch
            y_var.value = y_batch

            train_start = time.perf_counter()
            opt.one_step()
            train_elapsed_ms = (time.perf_counter() - train_start) * 1000.0
            grad_norm = global_gradient_norm(model)
            if dashboard.active and dashboard.grad_interval > 0 and global_step % dashboard.grad_interval == 0:
                dashboard.tb.log_scalar("gradients/global_norm", grad_norm, global_step)
            dashboard.log_gradients(global_step, model)
            dashboard.log_parameters(global_step, model, args.lr)
            opt.update()
            step_elapsed_ms = (time.perf_counter() - after_load) * 1000.0
            data_load_ms = (after_load - load_start) * 1000.0
            step_loss = myflows_scalar_float(loss_node.value)
            step_acc = _accuracy_from_logits(myflows_asnumpy(logits.value), y_batch)
            total_loss += step_loss
            total_acc += step_acc
            count += 1
            global_step += 1
            history.record_step(global_step, step_loss)

            dashboard.log_train_step(
                global_step,
                loss=step_loss,
                accuracy=step_acc,
                data_load_ms=data_load_ms,
                train_step_ms=train_elapsed_ms,
                step_time_ms=step_elapsed_ms,
                batch_size=B,
                labels=y_batch,
                task="classification",
                num_classes=K,
            )
            dashboard.log_activations(global_step, getattr(model, "_last_feature_nodes", {}))

            if s % 10 == 0 or s == steps - 1:
                print(
                    f"epoch {ep+1} step {s+1}/{steps} loss={step_loss:.4f} acc={step_acc:.4f}"
                )
            load_start = time.perf_counter()

        if count:
            mean_loss = total_loss / count
            mean_acc = total_acc / count
            history.record_epoch(ep + 1, mean_loss)
            dashboard.log_epoch(ep + 1, loss=mean_loss, accuracy=mean_acc, learning_rate=args.lr, epoch_time_s=time.perf_counter() - epoch_start)
            print(f"epoch {ep+1} mean loss={mean_loss:.4f} acc={mean_acc:.4f}")

            ms.save_checkpoint([model], opt, ep, -mean_loss, str(out_base))
            dashboard.log_checkpoint("latest_epoch", f"epoch={ep+1} path={out_base}.json mean_loss={mean_loss:.6f}", ep + 1)
            if best_loss is None or mean_loss < best_loss:
                best_loss = mean_loss
                ms.save_checkpoint([model], None, ep, -best_loss, str(best_base))
                print(f"已保存最佳模型 loss={best_loss:.4f}")
                dashboard.log_checkpoint("best", f"epoch={ep+1} path={best_base}.json best_loss={best_loss:.6f}", ep + 1)

    dashboard.close()
    if args.export_onnx:
        onnx_path = best_base.with_suffix(".onnx")
        infer_graph = ms.Graph(logits, optimize=bool(args.graph_opt))
        infer_graph.forward()
        ms.export_onnx(infer_graph, str(onnx_path), input_nodes=[x_var], output_names=["logits"])
        print(f"已导出 ONNX: {onnx_path}")

    tee.close()


if __name__ == "__main__":
    main()
