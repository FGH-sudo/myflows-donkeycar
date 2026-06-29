#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MyFlows VGG-11 在 Donkey 道路图像上做转向角和油门回归。

标签：从 DonkeyCar catalog 或文件名解析出 [angle, throttle]。
用法（项目根目录 d:\\DL\\testmyflow）:
  python -m apps.train.train_vgg_donkey_regression --data mycar/data --epochs 10 --batch 2 \\
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
from apps.common.splits import resolve_splits, select_split
from apps.train.common.training_control import BestScore, EarlyStopping
from apps.train.common.validation import run_loss_validation
from MyFlows.data.pipeline import MultiprocessDataLoader
from MyFlows.train.regularization import RegularizationConfig, apply_regularization
from tools.device_runtime import (
    myflows_scalar_float,
    print_myflows_device,
    resolve_myflows_device,
)
from MyFlows.layers.vgg import VGG11
from MyFlows.utils.model_inspector import format_inspection_report, format_model_summary, inspect_graph, model_summary
from MyFlows.utils.training_dashboard import TrainingDashboard
from MyFlows.utils.training_observer import global_gradient_norm
from MyFlows.utils.transforms import augment_chw_batch, default_train_transforms
from MyFlows.utils.viz import TrainingHistory
from apps.train.common.checkpoints import checkpoint_score_to_loss, checkpoint_stem
from apps.train.common.logging import TeeLogger
from tools.export_resnet_onnx import export_vgg11_onnx


def _load_regression_sample(sample: tuple[str, float, float, int, int, str]) -> tuple[np.ndarray, np.ndarray]:
    path_str, angle, throttle, image_w, image_h, dtype_name = sample
    dtype = np.float32 if dtype_name == "float32" else np.float64
    x = imread_nchw(Path(path_str), (int(image_w), int(image_h)), dtype)[0]
    y = np.array([float(angle), float(throttle)], dtype=dtype)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(description="MyFlows VGG-11 Donkey 控制回归")
    ap.add_argument("--data", type=str, default="mycar/data")
    ap.add_argument("--catalog", type=str, default="catalog_generated.catalog")
    ap.add_argument("--fixed-throttle", type=float, default=0.5)
    ap.add_argument("--force-fixed-throttle", action="store_true", help="忽略 catalog 中的 user/throttle，强制使用 --fixed-throttle")
    ap.add_argument("--angle-scale", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.0)
    ap.add_argument("--test-ratio", type=float, default=0.0)
    ap.add_argument("--val-size", type=int, default=0)
    ap.add_argument("--test-size", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--split-out", type=str, default=None)
    ap.add_argument("--split-file", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"))
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="MyFlows 计算设备：auto 有 GPU 则用 CUDA(CuPy)",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--l1-coeff", type=float, default=0.0)
    ap.add_argument("--regularize-bias-bn", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--early-stopping", action="store_true")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--initializer", type=str, default="kaiming_normal")
    ap.add_argument("--out", type=str, default="mycar/logs/vgg11_regression_checkpoint")
    ap.add_argument("--best-out", type=str, default="mycar/models/vgg11_regression_best")
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
    ap.add_argument("--check-shape", action="store_true")
    ap.add_argument("--check-content", action="store_true")
    ap.add_argument("--summary-once", action="store_true")
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
        data_dir,
        args.fixed_throttle,
        args.angle_scale,
        args.catalog,
        force_fixed_throttle=bool(args.force_fixed_throttle),
    )
    if not raw_index:
        raise SystemExit("未找到训练样本")
    if args.max_samples and args.max_samples > 0:
        raw_index = raw_index[: args.max_samples]

    index: list[tuple[Path, float, float]] = list(raw_index)

    splits, split_path = resolve_splits(
        index,
        split_file=(ROOT / args.split_file).resolve() if args.split_file else None,
        split_out=(ROOT / args.split_out).resolve() if args.split_out else None,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.split_seed,
    )
    val_index = []
    if splits is not None:
        full_count = len(index)
        train_index = select_split(index, splits, "train")
        val_index = select_split(index, splits, "val")
        test_index = select_split(index, splits, "test")
        if not train_index:
            raise SystemExit("split 后训练集为空，请减小 val/test 规模")
        index = train_index
        split_msg = f"[split] source={full_count} train={len(index)} val={len(val_index)} test={len(test_index)}"
        if split_path is not None:
            split_msg += f" file={split_path}"
        print(split_msg)

    first_p = data_dir / index[0][0]
    img0 = cv2.imread(str(first_p), cv2.IMREAD_COLOR)
    if img0 is None:
        raise SystemExit(f"首帧读取失败: {first_p}")
    h0, w0 = img0.shape[:2]
    print(f"输入 H×W = {h0}×{w0}, output_dim=2")

    dtype = np.float32 if args.dtype == "float32" else np.float64
    B = int(args.batch)
    h, w = h0, w0

    device = resolve_myflows_device(args.device)
    print_myflows_device(device, args.device)

    x_var = ms.Variable(np.zeros((B, 3, h, w), dtype=dtype), name="X")
    y_var = ms.Variable(np.zeros((B, 2), dtype=dtype), name="y")

    model = VGG11(
        in_channels=3,
        output_dim=2,
        image_h=h,
        image_w=w,
        name="vgg11_donkey",
        dropout=args.dropout,
        initializer=args.initializer,
    )
    pred = model(x_var)
    loss_node = ms.MSELoss(pred, y_var, name="loss")
    graph = ms.Graph(loss_node, optimize=bool(args.graph_opt))
    opt = ms.Adam(graph, learning_rate=args.lr)
    reg_config = RegularizationConfig(
        l1=float(args.l1_coeff),
        l2=float(args.weight_decay),
        regularize_bias_bn=bool(args.regularize_bias_bn),
    )

    out_base = checkpoint_stem((ROOT / args.out).resolve())
    best_base = checkpoint_stem((ROOT / args.best_out).resolve())
    out_base.parent.mkdir(parents=True, exist_ok=True)
    best_base.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_loss = None
    if args.resume:
        loaded_epoch, saved_score = ms.load_checkpoint([model], opt, str(out_base))
        start_epoch = max(0, int(loaded_epoch) + 1)
        best_loss = checkpoint_score_to_loss(saved_score)
    best_tracker = BestScore()
    best_tracker.best = best_loss
    early_stopper = EarlyStopping(args.patience, args.min_delta) if args.early_stopping else None
    if args.summary_once:
        print("[model-summary]")
        print(format_model_summary(model_summary(model)))

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
        all_labels = np.asarray([[angle, throttle] for _, angle, throttle in index], dtype=dtype)
        dashboard.log_dataset_summary(all_labels, task="regression")
    train_transform = default_train_transforms(seed=42) if args.augment else None
    history = TrainingHistory()
    global_step = 0
    dataset = [
        (str(data_dir / rel), angle, throttle, w, h, args.dtype)
        for rel, angle, throttle in index
    ]
    val_dataset = [
        (str(data_dir / rel), angle, throttle, w, h, args.dtype)
        for rel, angle, throttle in val_index
    ]
    inspect_done = False

    def _validation_step(x_items, y_items):
        real_count = len(x_items)
        x_batch = np.asarray(x_items, dtype=dtype)
        y_batch = np.asarray(y_items, dtype=dtype).reshape(-1, 2)
        x_batch, y_batch = pad_fixed_batch(x_batch, y_batch, B)
        x_var.value = x_batch
        y_var.value = y_batch
        model.eval()
        graph.forward()
        return myflows_scalar_float(loss_node.value), real_count

    for ep in range(start_epoch, start_epoch + args.epochs):
        total_loss = 0.0
        count = 0
        epoch_start = time.perf_counter()
        loader = MultiprocessDataLoader(
            dataset,
            B,
            num_workers=int(args.num_workers),
            shuffle=True,
            seed=ep,
            load_fn=_load_regression_sample,
        )
        load_start = time.perf_counter()
        for s, (x_items, y_items) in enumerate(loader):
            after_load = time.perf_counter()
            x_batch = np.asarray(x_items, dtype=dtype)
            y_batch = np.asarray(y_items, dtype=dtype).reshape(-1, 2)
            x_batch, y_batch = pad_fixed_batch(x_batch, y_batch, B)

            if train_transform or args.mixup or args.cutmix:
                x_orig = x_batch.copy()
                augment_chw_batch(
                    x_batch,
                    y_batch,
                    train_transform,
                    mixup=bool(args.mixup),
                    cutmix=bool(args.cutmix),
                )
                dashboard.log_augmentation(global_step, x_orig, x_batch, batch_size=B)

            x_var.value = x_batch
            y_var.value = y_batch

            model.train(True)
            train_start = time.perf_counter()
            opt.one_step()
            train_elapsed_ms = (time.perf_counter() - train_start) * 1000.0
            if reg_config.enabled:
                apply_regularization(opt, model.params, reg_config)
            if (args.check_shape or args.check_content) and not inspect_done:
                report = inspect_graph(graph, check_shape=args.check_shape, check_content=args.check_content)
                print("[graph-inspection]")
                print(format_inspection_report(report))
                if not report["ok"]:
                    raise SystemExit("graph inspection failed")
                inspect_done = True
            grad_norm = global_gradient_norm(model)
            if dashboard.active and dashboard.grad_interval > 0 and global_step % dashboard.grad_interval == 0:
                dashboard.tb.log_scalar("gradients/global_norm", grad_norm, global_step)
            dashboard.log_gradients(global_step, model)
            dashboard.log_parameters(global_step, model, args.lr)
            opt.update()
            step_elapsed_ms = (time.perf_counter() - after_load) * 1000.0
            data_load_ms = (after_load - load_start) * 1000.0
            step_loss = myflows_scalar_float(loss_node.value)
            total_loss += step_loss
            count += 1
            global_step += 1
            history.record_step(global_step, step_loss)

            dashboard.log_train_step(
                global_step,
                loss=step_loss,
                data_load_ms=data_load_ms,
                train_step_ms=train_elapsed_ms,
                step_time_ms=step_elapsed_ms,
                batch_size=B,
                labels=y_batch,
                task="regression",
            )
            dashboard.log_activations(global_step, getattr(model, "_last_feature_nodes", {}), preferred_last="stage5")

            if s % 10 == 0 or s == steps - 1:
                print(
                    f"epoch {ep+1} step {s+1}/{steps} loss={step_loss:.4f}"
                )
            load_start = time.perf_counter()

        if count:
            mean_loss = total_loss / count
            history.record_epoch(ep + 1, mean_loss)
            val_result = run_loss_validation(
                val_dataset,
                B,
                load_fn=_load_regression_sample,
                step_fn=_validation_step,
                num_workers=0,
            )
            validation_loss = val_result.mean_loss if val_result is not None else None
            dashboard.log_epoch(
                ep + 1,
                loss=mean_loss,
                validation_loss=validation_loss,
                learning_rate=args.lr,
                epoch_time_s=time.perf_counter() - epoch_start,
            )
            print(f"epoch {ep+1} mean loss={mean_loss:.4f}")
            if validation_loss is not None:
                print(f"epoch {ep+1} val loss={validation_loss:.4f} ({val_result.samples} samples)")

            ms.save_checkpoint([model], opt, ep, filepath=str(out_base), loss=mean_loss)
            dashboard.log_checkpoint("latest_epoch", f"epoch={ep+1} path={out_base}.json mean_loss={mean_loss:.6f}", ep + 1)
            score_for_best = float(validation_loss if validation_loss is not None else mean_loss)
            if best_tracker.update(score_for_best):
                best_loss = best_tracker.best
                ms.save_checkpoint([model], None, ep, filepath=str(best_base), loss=best_loss)
                print(f"已保存最佳模型 loss={best_loss:.4f}")
                dashboard.log_checkpoint("best", f"epoch={ep+1} path={best_base}.json best_loss={best_loss:.6f}", ep + 1)
            if early_stopper is not None:
                stop_state = early_stopper.update(score_for_best)
                if stop_state.should_stop:
                    print(
                        f"early stopping: best={stop_state.best:.6f} "
                        f"bad_epochs={stop_state.bad_epochs}/{early_stopper.patience}"
                    )
                    break

    dashboard.close()
    if args.export_onnx:
        onnx_path = best_base.with_suffix(".onnx")
        export_vgg11_onnx(
            best_base,
            output=onnx_path,
            batch_size=1,
            image_w=w,
            image_h=h,
            device=args.device,
            graph_opt=bool(args.graph_opt),
        )
        print(f"已导出 ONNX: {onnx_path} (batch=1)")

    tee.close()


if __name__ == "__main__":
    main()
