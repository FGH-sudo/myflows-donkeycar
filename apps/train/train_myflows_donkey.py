#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 MyFlows 的 ResNet-18 在 DonkeyCar 图像数据上做转向回归（OpenCV 读取）：
- 数据目录：`--data`（默认 `mycar/data`），其中必须包含 `images/*.jpg`
- 优先读取 `--catalog` 指定的 catalog；不存在时回退到从文件名解析 angle
- catalog 缺少 throttle 或文件名模式下，`throttle` 固定为 `--fixed-throttle`（默认 0.5）

用法（在项目根目录 d:\\DL\\testmyflow 下）:
  # 试跑
  python -m apps.train.train_myflows_donkey --max-samples 200 --epochs 1 --device auto

  # 正式训练（全量可解析样本；GPU=CUDA+CuPy；不覆盖旧模型）
  python -m apps.train.train_myflows_donkey --unique-run --max-samples 0 --epochs 20 --batch 2 \\
      --augment --graph-opt --checkpoint-every 500 --export-onnx --device auto

  # 续训（指向当次 --out 断点，不要加 --unique-run）
  python -m apps.train.train_myflows_donkey --resume --epochs 10 --batch 2 --device auto

说明:
  --max-samples 0 表示不限制条数（用全部可用样本），不是 0 张图。
  --unique-run / --run-id 为 checkpoint 与 best 文件名追加后缀，避免覆盖。
  GPU 训练需 cupy-cuda12x（见 MyFlows/requirements-gpu.txt）；导出 ONNX 在训练正常结束后进行。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import MyFlows as ms
from apps.common.donkey_data import (
    image_rel_path as _image_rel_path,
    load_donkey_index as _load_donkey_index,
    load_generated_road_images_index as _load_generated_road_images_index,
    parse_angle_from_filename as _parse_angle_from_filename,
)
from apps.common.image_preprocess import imread_nchw as _imread_nchw
from apps.common.image_preprocess import pad_fixed_batch as _pad_fixed_batch
from apps.common.splits import resolve_splits, select_split
from apps.train.common.checkpoints import (
    DEFAULT_RESNET_BEST_OUT,
    DEFAULT_RESNET_CHECKPOINT_OUT,
    checkpoint_score_to_loss,
    checkpoint_stem as _checkpoint_stem,
    with_run_id as _with_run_id,
)
from apps.train.common.training_control import BestScore, EarlyStopping
from apps.train.common.validation import run_loss_validation
from apps.train.common.logging import TeeLogger
from MyFlows.data.pipeline import MultiprocessDataLoader
from MyFlows.train.regularization import RegularizationConfig, apply_regularization
from MyFlows.utils.model_inspector import format_inspection_report, format_model_summary, inspect_graph, model_summary
from tools.export_resnet_onnx import export_resnet18_onnx
from tools.device_runtime import myflows_scalar_float, print_myflows_device, resolve_myflows_device
from MyFlows.utils.training_observer import (
    global_gradient_norm,
    label_stats_regression,
)
from MyFlows.utils.training_dashboard import TrainingDashboard
from MyFlows.utils.transforms import augment_chw_batch, chw_to_hwc, default_train_transforms
from MyFlows.utils.viz import TrainingHistory, plot_training_curves


def _load_training_sample(sample: tuple[str, float, float, int, int, str]) -> tuple[np.ndarray, np.ndarray]:
    path_str, angle, throttle, image_w, image_h, dtype_name = sample
    dtype = np.float32 if dtype_name == "float32" else np.float64
    x = _imread_nchw(Path(path_str), (int(image_w), int(image_h)), dtype)[0]
    y = np.array([float(angle), float(throttle)], dtype=dtype)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(description="MyFlows ResNet-18 训练 Donkey tub")
    ap.add_argument("--data", type=str, default="mycar/data", help="Donkey tub 根目录(含 images 与 catalog)")
    ap.add_argument("--catalog", type=str, default="catalog_generated.catalog", help="优先读取的 catalog 文件名；不存在时回退到解析 images 文件名")
    ap.add_argument("--fixed-throttle", type=float, default=0.5, help="catalog 缺少 throttle 或文件名模式下使用的固定 user/throttle")
    ap.add_argument("--force-fixed-throttle", action="store_true", help="忽略 catalog 中的 user/throttle，强制使用 --fixed-throttle")
    ap.add_argument("--angle-scale", type=float, default=1.0, help="对标签 angle 的缩放（可用于翻符号或调整幅度）")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1, help="批大小(须固定，与构图一致)")
    ap.add_argument("--max-samples", type=int, default=0, help="最多用多少条(0=不限制，用全部可解析样本)")
    ap.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="与 --max-samples 联用：按该种子无放回随机抽 N 条；不设则仍取排序后前 N 条",
    )
    ap.add_argument("--val-ratio", type=float, default=0.0, help="验证集比例；默认 0 表示不划分")
    ap.add_argument("--test-ratio", type=float, default=0.0, help="测试集比例；默认 0 表示不划分")
    ap.add_argument("--val-size", type=int, default=0, help="验证集固定条数；优先于 --val-ratio")
    ap.add_argument("--test-size", type=int, default=0, help="测试集固定条数；优先于 --test-ratio")
    ap.add_argument("--split-seed", type=int, default=42, help="train/val/test 划分随机种子")
    ap.add_argument("--split-out", type=str, default=None, help="保存本次 split JSON，供评估复用")
    ap.add_argument("--split-file", type=str, default=None, help="复用已有 split JSON")
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="计算设备：auto=有 GPU 则用 CUDA，否则 CPU；需 pip install cupy-cuda12x 等",
    )
    ap.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"), help="训练计算 dtype（float32 通常更快）")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0, help="L2 正则系数；默认关闭")
    ap.add_argument("--l1-coeff", type=float, default=0.0, help="L1 正则系数；默认关闭")
    ap.add_argument("--regularize-bias-bn", action="store_true", help="正则化 bias/BN 参数；默认跳过")
    ap.add_argument("--dropout", type=float, default=0.0, help="ResNet FC 前 dropout 概率；默认关闭")
    ap.add_argument("--early-stopping", action="store_true", help="按验证 loss 启用早停；无验证集时按训练 loss")
    ap.add_argument("--patience", type=int, default=5, help="early stopping 容忍 epoch 数")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="early stopping 最小改善幅度")
    ap.add_argument("--initializer", type=str, default="kaiming_normal", help="权重初始化：kaiming_normal/xavier_uniform/xavier_normal/kaiming_uniform/normal/constant")
    ap.add_argument("--out", type=str, default=DEFAULT_RESNET_CHECKPOINT_OUT, help="训练断点输出基名，会生成 .json + .npz")
    ap.add_argument("--best-out", type=str, default=DEFAULT_RESNET_BEST_OUT, help="最佳模型输出基名，会生成 .json + .npz")
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="为 --out / --best-out 追加后缀 _<run-id>，全新训练时避免覆盖旧模型（续训请配合 --resume 指向对应路径）",
    )
    ap.add_argument(
        "--unique-run",
        action="store_true",
        help="全新训练时自动用时间戳作为 run-id（与日志时间戳一致）；与 --resume 互斥逻辑：续训不会改路径",
    )
    ap.add_argument("--log-dir", type=str, default="mycar/logs", help="训练日志目录")
    ap.add_argument("--log-file", type=str, default=None, help="训练日志文件；默认按时间生成")
    ap.add_argument("--resume", action="store_true", help="从 --out 对应训练断点续训")
    ap.add_argument("--checkpoint-every", type=int, default=100, help="每多少 step 保存一次训练断点(0=只在 epoch 末保存)")
    ap.add_argument("--stop-file", type=str, default="mycar/logs/STOP_TRAINING", help="检测到该文件时保存断点并退出")
    ap.add_argument("--export-onnx", action="store_true", help="训练结束后导出 ONNX 推理图")
    ap.add_argument("--onnx-out", type=str, default=None, help="ONNX 输出路径，默认跟 --best-out 同名 .onnx")
    ap.add_argument("--stem", type=str, default="cifar", choices=("cifar", "imagenet"), help="小分辨率建议 cifar")
    ap.add_argument("--graph-opt", action="store_true", help="打开构图期优化(可能略快)")
    ap.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="TensorBoard 日志目录；默认 mycar/logs/tensorboard/<时间戳>",
    )
    ap.add_argument(
        "--tb-log-interval",
        type=int,
        default=10,
        help="每多少 step 写入 TensorBoard 标量（0=仅 epoch 末）",
    )
    ap.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="禁用 TensorBoard 日志",
    )
    ap.add_argument(
        "--augment",
        action="store_true",
        help="启用数据增强（RandomCrop/Rotation/ColorJitter）",
    )
    ap.add_argument(
        "--mixup",
        action="store_true",
        help="batch 内 MixUp（需 --augment 或单独开启）",
    )
    ap.add_argument(
        "--cutmix",
        action="store_true",
        help="batch 内 CutMix",
    )
    ap.add_argument(
        "--tb-image-interval",
        type=int,
        default=200,
        help="每多少 step 向 TensorBoard 记录原图/增强对比（0=关闭）",
    )
    ap.add_argument(
        "--save-png-plots",
        action="store_true",
        help="训练结束后额外保存 matplotlib PNG 曲线（默认仅用 TensorBoard）",
    )
    ap.add_argument("--tb-grad-interval", type=int, default=100, help="每多少 step 记录梯度范数/梯度分布（0=关闭）")
    ap.add_argument("--tb-param-interval", type=int, default=200, help="每多少 step 记录参数范数/参数分布（0=关闭）")
    ap.add_argument("--tb-activation-interval", type=int, default=200, help="每多少 step 记录激活统计和特征图（0=关闭）")
    ap.add_argument("--tb-max-hist-params", type=int, default=24, help="每次最多记录多少个参数/梯度 histogram")
    ap.add_argument("--tb-feature-channels", type=int, default=16, help="每个激活层最多记录多少个 feature map 通道")
    ap.add_argument("--check-shape", action="store_true", help="首个 batch 后打印图节点 shape 检查")
    ap.add_argument("--check-content", action="store_true", help="首个 batch 后检查 NaN/Inf/全零输出")
    ap.add_argument("--summary-once", action="store_true", help="训练开始前打印一次模型参数 summary")
    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="数据加载 worker 数（>0 时用 MyFlows MultiprocessDataLoader 预取；见 benchmark/dataloader_bench.py）",
    )
    ap.add_argument(
        "--plot-prefix",
        type=str,
        default=None,
        help="历史 JSON/PNG 文件名前缀；默认与日志时间戳一致",
    )
    args = ap.parse_args()

    data_dir = (ROOT / args.data).resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"数据目录不存在: {data_dir}")

    log_dir = (ROOT / args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = (ROOT / args.log_file).resolve() if args.log_file else (
        log_dir / f"myflows_train_{run_stamp}.log"
    )
    plot_prefix = args.plot_prefix or f"myflows_train_{run_stamp}"
    tb_logdir = (ROOT / args.logdir).resolve() if args.logdir else (
        log_dir / "tensorboard" / f"resnet18_{run_stamp}"
    )

    if args.resume and args.unique_run:
        raise SystemExit("--unique-run 仅用于全新训练；续训请去掉该参数并显式指定 --out。")
    run_id = (args.run_id or "").strip() or None
    if args.unique_run and not args.resume:
        run_id = run_stamp
    if run_id:
        args.out = _with_run_id(args.out, run_id)
        args.best_out = _with_run_id(args.best_out, run_id)
        if args.onnx_out:
            onnx_p = Path(args.onnx_out)
            args.onnx_out = str(
                onnx_p.parent / f"{onnx_p.stem}_{run_id}{onnx_p.suffix or '.onnx'}"
            )
        print(f"[paths] run-id={run_id}（本次 checkpoint/best 不会覆盖未带此后缀的旧文件）")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    tee = TeeLogger(log_file)
    sys.stdout = tee
    sys.stderr = tee

    index = _load_donkey_index(
        data_dir,
        fixed_throttle=args.fixed_throttle,
        angle_scale=args.angle_scale,
        catalog_name=args.catalog,
        force_fixed_throttle=bool(args.force_fixed_throttle),
    )
    if not index:
        raise SystemExit("未找到训练样本：请确认 images/*.jpg 存在。")
    if args.max_samples and args.max_samples > 0:
        n_take = min(int(args.max_samples), len(index))
        if args.sample_seed is not None:
            rng = np.random.default_rng(int(args.sample_seed))
            pick = rng.choice(len(index), size=n_take, replace=False)
            index = [index[i] for i in pick]
            print(f"[data] 随机子集: {n_take} 条 (seed={args.sample_seed})")
        else:
            index = index[:n_take]
            print(f"[data] 取排序后前 {n_take} 条")

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
    test_index = []
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

    # 用第一条样本定输入分辨率
    first_p = data_dir / index[0][0]
    img0 = cv2.imread(str(first_p), cv2.IMREAD_COLOR)
    if img0 is None:
        raise SystemExit(f"首帧读取失败: {first_p}")
    h0, w0 = img0.shape[:2]
    print(f"使用输入尺寸 H×W = {h0}×{w0} (与首帧一致)")

    dtype = np.float32 if args.dtype == "float32" else np.float64
    B = int(args.batch)
    h, w = h0, w0

    device = resolve_myflows_device(args.device)
    print_myflows_device(device, args.device)

    # 变量与模型（只构图一次，批大小固定为 B；须在 set_device 之后创建）
    x_var = ms.Variable(
        np.zeros((B, 3, h, w), dtype=dtype),
        name="X",
    )
    y_var = ms.Variable(
        np.zeros((B, 2), dtype=dtype),
        name="y",
    )
    model = ms.ResNet18(
        in_channels=3,
        output_dim=2,
        stem=args.stem,
        base_width=64,
        name="resnet18_donkey",
        dropout=args.dropout,
        initializer=args.initializer,
    )
    logits = model(x_var)
    loss_node = ms.MSELoss(logits, y_var, name="loss")
    graph = ms.Graph(
        loss_node,
        optimize=bool(args.graph_opt),
    )
    opt = ms.Adam(graph, learning_rate=args.lr)
    reg_config = RegularizationConfig(
        l1=float(args.l1_coeff),
        l2=float(args.weight_decay),
        regularize_bias_bn=bool(args.regularize_bias_bn),
    )

    out_base = _checkpoint_stem((ROOT / args.out).resolve())
    best_base = _checkpoint_stem((ROOT / args.best_out).resolve())
    stop_file = (ROOT / args.stop_file).resolve()
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

    n = len(index)
    steps = (n + B - 1) // B
    print(f"日志文件: {log_file}")
    print(f"训练断点: {out_base}.json + {out_base}.npz")
    print(f"最佳模型: {best_base}.json + {best_base}.npz")
    print(f"安全暂停文件: {stop_file}")
    print(f"样本数: {n}, batch={B}, steps/epoch={steps}, lr={args.lr}")
    print(f"数据加载: num_workers={int(args.num_workers)}")
    if args.dropout:
        print(f"Dropout: p={float(args.dropout)}")
    if reg_config.enabled:
        print(f"正则化: l1={reg_config.l1} l2={reg_config.l2} regularize_bias_bn={reg_config.regularize_bias_bn}")
    if args.summary_once:
        print("[model-summary]")
        print(format_model_summary(model_summary(model)))
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
        print(f"TensorBoard 日志: {tb_logdir}")
        print(f"查看命令: tensorboard --logdir {tb_logdir.parent}")
        dashboard.log_run_config(args, {"samples": n, "input_h": h, "input_w": w, "steps_per_epoch": steps})
        labels = np.asarray([[angle, throttle] for _, angle, throttle in index], dtype=np.float64)
        dashboard.log_dataset_summary(labels, task="regression")
    train_transform = default_train_transforms(seed=42) if args.augment else None
    if args.augment:
        print("数据增强: RandomCrop + RandomRotation + ColorJitter")
    if args.mixup:
        print("MixUp: 已启用")
    if args.cutmix:
        print("CutMix: 已启用")
    last_mean_loss = None
    should_stop = False
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
        x_batch, y_batch = _pad_fixed_batch(x_batch, y_batch, B)
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
            load_fn=_load_training_sample,
        )
        load_start = time.perf_counter()
        for s, (x_items, y_items) in enumerate(loader):
            after_load = time.perf_counter()
            x_batch = np.asarray(x_items, dtype=dtype)
            y_batch = np.asarray(y_items, dtype=dtype).reshape(-1, 2)
            x_batch, y_batch = _pad_fixed_batch(x_batch, y_batch, B)

            if train_transform is not None or args.mixup or args.cutmix:
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
            if dashboard.active and dashboard.grad_interval > 0 and (global_step % dashboard.grad_interval == 0):
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
            running_mean_loss = total_loss / count
            dashboard.log_train_step(
                global_step,
                loss=step_loss,
                running_loss=running_mean_loss,
                data_load_ms=data_load_ms,
                train_step_ms=train_elapsed_ms,
                step_time_ms=step_elapsed_ms,
                batch_size=B,
                labels=y_batch,
                task="regression",
                force=s == steps - 1,
            )
            dashboard.log_activations(global_step, getattr(model, "_last_feature_nodes", {}), preferred_last="layer4")
            if s % 10 == 0 or s == steps - 1:
                print(
                    f"epoch {ep+1} step {s+1}/{steps} "
                    f"loss={myflows_scalar_float(loss_node.value):.6f}"
                )
            if args.checkpoint_every and count % int(args.checkpoint_every) == 0:
                json_path, npz_path = ms.save_checkpoint(
                    [model],
                    opt,
                    ep,
                    filepath=str(out_base),
                    loss=running_mean_loss,
                )
                print(
                    f"已保存训练断点(step {s+1}): {json_path} + {npz_path} "
                    f"(running_loss={running_mean_loss:.6f})"
                )
                dashboard.log_checkpoint("latest", f"step={global_step} epoch={ep+1} path={json_path} running_loss={running_mean_loss:.6f}", global_step)
            if stop_file.exists():
                json_path, npz_path = ms.save_checkpoint(
                    [model],
                    opt,
                    ep,
                    filepath=str(out_base),
                    loss=running_mean_loss,
                )
                print(
                    f"检测到暂停文件，已保存训练断点: {json_path} + {npz_path} "
                    f"(epoch={ep+1}, step={s+1}, running_loss={running_mean_loss:.6f})"
                )
                dashboard.log_checkpoint("stop_file", f"step={global_step} epoch={ep+1} path={json_path}", global_step)
                try:
                    stop_file.unlink()
                except OSError:
                    pass
                should_stop = True
                break
            load_start = time.perf_counter()
        if count:
            last_mean_loss = total_loss / count
            history.record_epoch(ep + 1, last_mean_loss)
            val_result = run_loss_validation(
                val_dataset,
                B,
                load_fn=_load_training_sample,
                step_fn=_validation_step,
                num_workers=0,
            )
            validation_loss = val_result.mean_loss if val_result is not None else None
            dashboard.log_epoch(
                ep + 1,
                loss=last_mean_loss,
                validation_loss=validation_loss,
                learning_rate=args.lr,
                epoch_time_s=time.perf_counter() - epoch_start,
            )
            print(f"epoch {ep+1} mean loss: {last_mean_loss:.6f}")
            if validation_loss is not None:
                print(f"epoch {ep+1} val loss: {validation_loss:.6f} ({val_result.samples} samples)")

            json_path, npz_path = ms.save_checkpoint(
                [model],
                opt,
                ep,
                filepath=str(out_base),
                loss=last_mean_loss,
            )
            print(f"已保存训练断点: {json_path} + {npz_path}")
            dashboard.log_checkpoint("latest_epoch", f"epoch={ep+1} path={json_path} mean_loss={last_mean_loss:.6f}", ep + 1)

            score_for_best = float(validation_loss if validation_loss is not None else last_mean_loss)
            if best_tracker.update(score_for_best):
                best_loss = float(best_tracker.best)
                best_json, best_npz = ms.save_checkpoint(
                    [model],
                    None,
                    ep,
                    filepath=str(best_base),
                    loss=best_loss,
                )
                print(f"已保存最佳模型: {best_json} + {best_npz} (loss={best_loss:.6f})")
                dashboard.log_checkpoint("best", f"epoch={ep+1} path={best_json} best_loss={best_loss:.6f}", ep + 1)
            if early_stopper is not None:
                stop_state = early_stopper.update(score_for_best)
                if stop_state.should_stop:
                    print(
                        f"early stopping: best={stop_state.best:.6f} "
                        f"bad_epochs={stop_state.bad_epochs}/{early_stopper.patience}"
                    )
                    should_stop = True
        if should_stop:
            break

    history_json = history.save_json(log_dir / f"{plot_prefix}_history.json")
    print(f"已保存训练历史: {history_json}")
    tb_was_active = dashboard.active
    dashboard.flush()
    dashboard.close()
    if tb_was_active:
        print(f"TensorBoard: tensorboard --logdir {tb_logdir.parent}")
    if args.save_png_plots and (history.step_losses or history.epoch_losses):
        plot_paths = plot_training_curves(history, log_dir, prefix=plot_prefix)
        for p in plot_paths:
            print(f"已保存训练曲线 PNG: {p}")

    if args.export_onnx:
        onnx_path = (ROOT / args.onnx_out).resolve() if args.onnx_out else best_base.with_suffix(".onnx")
        export_resnet18_onnx(
            best_base,
            output=onnx_path,
            batch_size=1,
            image_w=w,
            image_h=h,
            stem=args.stem,
            device=args.device,
            graph_opt=bool(args.graph_opt),
        )
        print(f"已导出 ONNX: {onnx_path} (batch=1)")

    tee.close()


if __name__ == "__main__":
    main()
