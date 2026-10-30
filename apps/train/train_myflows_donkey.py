#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 MyFlows 的 ResNet-18 在 DonkeyCar 图像数据上做转向回归（OpenCV 读取）：
- 数据目录：`--data`（默认 `mycar/data`），其中必须包含 `images/*.jpg`
- 优先读取 `--catalog` 指定的 catalog；不存在时回退到从文件名解析 angle
- catalog 缺少 throttle 或文件名模式下，`throttle` 固定为 `--fixed-throttle`（默认 0.5）
- 可单进程训练，也可用 `--distributed ps|ring` 在本机多进程数据并行训练

用法（在项目根目录下运行）:
  # 试跑
  python -m apps.train.train_myflows_donkey --max-samples 200 --epochs 1 --device auto

  # 正式训练（全量可解析样本；GPU=CUDA+CuPy；不覆盖旧模型）
  python -m apps.train.train_myflows_donkey --unique-run --max-samples 0 --epochs 20 --batch 2 \\
      --augment --graph-opt --checkpoint-every 500 --export-onnx --device auto

  # 续训（指向当次 --out 断点，不要加 --unique-run）
  python -m apps.train.train_myflows_donkey --resume --epochs 10 --batch 2 --device auto

  # 分布式数据并行：2 个 Worker 共享本机 GPU，gRPC Parameter Server（--distributed ring 为 Ring AllReduce）
  python -m apps.train.train_myflows_donkey --distributed ps --workers 2 --batch 8 --epochs 5 \\
      --val-size 200 --export-onnx --device cuda

说明:
  --max-samples 0 表示不限制条数（用全部可解析样本），不是 0 张图。
  --unique-run / --run-id 为 checkpoint 与 best 文件名追加后缀，避免覆盖。
  GPU 训练需 cupy-cuda12x（见 MyFlows/requirements-gpu.txt）；导出 ONNX 在训练结束后进行。
  分布式时 --batch 是全局 batch，按 Worker 切分，样本顺序与单进程相同。每步把各 Worker 的梯度与 BN 滑动统计量
  按样本数加权平均，所有 Worker 的参数始终一致并逐步核对；BN 前向用各 Worker 本地 batch 的统计量（与 PyTorch
  DDP 不开 SyncBN 相同）。验证集按 Worker 切分后汇总；断点、最佳模型与 ONNX 与单进程格式相同，可以互相续训。
  停止文件出现时所有 Worker 在同一步停下并保存断点；--resume 从断点记录的 epoch 与 step 继续。

  GPU 训练开始前先取得本机 GPU 锁（apps/common/gpu_lock.py），控制台、AutoPilot 与终端启动的训练依次排队；
  指定 --run-dir 时结束后写 result.json，退出码区分非有限 loss（3）、显存不足（4）与其他错误（1）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import MyFlows as ms
from apps.common.donkey_data import load_donkey_index as _load_donkey_index
from apps.common.image_preprocess import imread_nchw as _imread_nchw
from apps.common.image_preprocess import pad_fixed_batch as _pad_fixed_batch
from apps.common.gpu_lock import GpuLock
from apps.common.splits import entries_hash, resolve_splits, select_split
from apps.train.common import outcome
from apps.train.common.checkpoints import (
    DEFAULT_RESNET_BEST_OUT,
    DEFAULT_RESNET_CHECKPOINT_OUT,
    checkpoint_score_to_loss,
    checkpoint_stem as _checkpoint_stem,
    with_run_id as _with_run_id,
)
from apps.train.common.training_control import BestScore, EarlyStopping
from apps.train.common.logging import TeeLogger
from MyFlows.core.device import asnumpy, cuda_available, set_device, xp
from MyFlows.data.pipeline import MultiprocessDataLoader
from MyFlows.distributed.shards import split_global_batch
from MyFlows.train.regularization import RegularizationConfig, apply_regularization
from MyFlows.utils.checkpoint import named_state, read_checkpoint_metadata
from MyFlows.utils.initializers import make_initializer
from MyFlows.utils.model_inspector import format_inspection_report, format_model_summary, inspect_graph, model_summary
from tools.export_resnet_onnx import export_resnet18_onnx
from tools.device_runtime import myflows_scalar_float, print_myflows_device, resolve_myflows_device
from MyFlows.utils.training_observer import global_gradient_norm
from MyFlows.utils.training_dashboard import TrainingDashboard
from MyFlows.utils.transforms import augment_chw_batch, default_train_transforms
from MyFlows.utils.viz import TrainingHistory, plot_training_curves

# 并行方式 -> 可用的通信方式；Ring AllReduce 只有 gRPC 实现
DISTRIBUTED_TRANSPORTS = {"ps": ("grpc_proto", "socket_json"), "ring": ("grpc_proto",)}


def _resolve_conv_backend(requested: str, device: str, dtype) -> str:
    """auto：CUDA + float32 用 cuda_native_register（寄存器分块 GEMM），原生库缺失或过期时先自动编译，
    不改用 CuPy；原生算子只支持 float32，所以 float64 与 CPU 交给算子按设备选 CuPy/NumPy。"""
    if requested == "auto":
        requested = "cuda_native_register" if device == "cuda" and dtype == np.float32 else "auto"
    if requested.startswith("cuda_native"):
        from MyFlows.ops.cuda_native import ensure_library

        ensure_library()
    return requested


class _DeviceSpan:
    """量一段计算在设备上的耗时。

    GPU 上 kernel 异步执行，perf_counter 只能量到 CPU 发射 kernel 的时间；这里改用 CUDA event，
    在步末读 loss（已同步）之后再取差值，不额外插入同步点。CPU 上退化为 perf_counter。
    """

    def __init__(self, device: str):
        self._events = (xp.cuda.Event(), xp.cuda.Event()) if device == "cuda" else None
        self._start = self._stop = 0.0

    def start(self) -> None:
        if self._events:
            self._events[0].record()
        else:
            self._start = time.perf_counter()

    def stop(self) -> None:
        if self._events:
            self._events[1].record()
        else:
            self._stop = time.perf_counter()

    def elapsed_ms(self) -> float:
        if self._events:
            self._events[1].synchronize()
            return float(xp.cuda.get_elapsed_time(*self._events))
        return (self._stop - self._start) * 1000.0


PHASES = ("forward", "backward", "optimizer")


class _StepPhases:
    """一步里前向 / 反向 / 优化器三段的耗时，以及 CuPy 显存池的区间峰值。

    每 ``every`` 步用 CUDA event 分别量一次三段耗时（0 关闭）；每段结束时读一次显存池的已用字节数
    （只读主机端计数，不同步 GPU），取两次写出之间的最大值。WDDM 下 NVML 读不到单个进程的显存，所以显存以
    本进程的 CuPy 显存池为准：used 是张量实际占用，pool 是显存池向驱动申请的总量（训练中只增不减）。
    """

    def __init__(self, device: str, every: int):
        self.every = max(0, int(every))
        self.spans = {phase: _DeviceSpan(device) for phase in PHASES}
        self.pool = xp.get_default_memory_pool() if device == "cuda" else None
        self.device_total_bytes = None
        if self.pool is not None:
            _free, self.device_total_bytes = xp.cuda.runtime.memGetInfo()
        self.timing = False
        self.interval_used = self.peak_used = self.peak_pool = 0

    def begin(self, global_step: int) -> None:
        self.timing = self.every > 0 and global_step % self.every == 0

    def sample(self) -> None:
        if self.pool is None:
            return
        used, total = self.pool.used_bytes(), self.pool.total_bytes()
        self.interval_used = max(self.interval_used, used)
        self.peak_used = max(self.peak_used, used)
        self.peak_pool = max(self.peak_pool, total)

    @contextmanager
    def __call__(self, phase: str):
        if self.timing:
            self.spans[phase].start()
        try:
            yield
        finally:
            if self.timing:
                self.spans[phase].stop()
            self.sample()

    def row(self) -> dict:
        """写进本步 step 行的字段（读 event 时间要在步末的同步点之后调用），并开始新的显存区间。"""
        row = {}
        if self.timing:
            row.update({f"{phase}_ms": self.spans[phase].elapsed_ms() for phase in PHASES})
        if self.pool is not None:
            row.update(gpu_mem_used_mb=self.interval_used / 2**20, gpu_mem_pool_mb=self.pool.total_bytes() / 2**20)
            self.interval_used = 0
        return row

    def summary(self) -> dict:
        if self.pool is None:
            return {"source": None}
        return {"source": "cupy_memory_pool", "peak_used_mb": self.peak_used / 2**20,
                "peak_pool_mb": self.peak_pool / 2**20, "device_total_mb": self.device_total_bytes / 2**20}


# Arguments that do not change the numbers of a training trajectory: paths, logging, checkpoint cadence, data
# loading parallelism (sample order and augmentation do not depend on it), timeouts. Everything else is in
# the TrainingState fingerprint; --lr may differ on --resume-state (a rollback halves it) and is recorded.
_NON_TRAJECTORY_ARGS = {
    "out", "best_out", "run_dir", "log_file", "stop_file", "resume", "resume_state", "state_every", "state_dir",
    "keep_states", "checkpoint_every", "unique_run", "run_id", "export_onnx", "onnx_out", "no_tensorboard",
    "save_png_plots", "summary_once", "check_shape", "check_content", "num_workers", "gpu_lock_timeout",
    "phase_timing_every", "sync_timeout", "worker_memory_mb", "epochs", "lr", "log_dir", "logdir", "plot_prefix",
    "split_out",
}


# Arguments added after TrainingStates were first written: left out at their defaults, so older states still match.
_LATER_DEFAULTS = {"arch": "resnet18", "pretrained": None, "freeze_stages": 0, "image_size": None, "input_norm": "auto"}


def trajectory_config(args: argparse.Namespace) -> dict:
    return {k: v for k, v in sorted(vars(args).items())
            if k not in _NON_TRAJECTORY_ARGS and not k.startswith("tb_")
            and not (k in _LATER_DEFAULTS and v == _LATER_DEFAULTS[k])}


def trajectory_fingerprint(args: argparse.Namespace) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(trajectory_config(args), sort_keys=True, default=str).encode()).hexdigest()


def derive_seeds(seed: int | None, world: int) -> dict:
    """由 --seed 派生每个 Worker 的数据增强、dropout 与 MixUp/CutMix 种子；不设 --seed 时增强沿用 42 + rank、
    dropout 与混合用全局随机源（与旧版相同）。数据顺序由 epoch 决定，与 --seed 无关。"""
    if seed is None:
        return {"init": None, "augment": [42 + rank for rank in range(world)], "dropout": [None] * world,
                "mix": [None] * world, "derived_from": None}

    def stream(tag: int, rank: int) -> int:
        return int(np.random.SeedSequence([int(seed), tag, rank]).generate_state(1)[0])

    return {"init": int(seed), "augment": [stream(1, rank) for rank in range(world)],
            "dropout": [stream(2, rank) for rank in range(world)], "mix": [stream(3, rank) for rank in range(world)],
            "derived_from": "SeedSequence([seed, stream, rank])"}




def _load_training_sample(sample: tuple[str, float, float, int, int, str]) -> tuple[np.ndarray, np.ndarray]:
    path_str, angle, throttle, image_w, image_h, dtype_name = sample
    dtype = np.float32 if dtype_name == "float32" else np.float64
    x = _imread_nchw(Path(path_str), (int(image_w), int(image_h)), dtype)[0]
    y = np.array([float(angle), float(throttle)], dtype=dtype)
    return x, y


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MyFlows ResNet-18 训练 Donkey tub")
    ap.add_argument("--data", type=str, default="mycar/data", help="Donkey tub 根目录(含 images 与 catalog)")
    ap.add_argument("--catalog", type=str, default="catalog_generated.catalog", help="优先读取的 catalog 文件名；不存在时回退到解析 images 文件名")
    ap.add_argument("--fixed-throttle", type=float, default=0.5, help="catalog 缺少 throttle 或文件名模式下使用的固定 user/throttle")
    ap.add_argument("--force-fixed-throttle", action="store_true", help="忽略 catalog 中的 user/throttle，强制使用 --fixed-throttle")
    ap.add_argument("--angle-scale", type=float, default=1.0, help="对标签 angle 的缩放（可用于翻符号或调整幅度）")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1, help="批大小(须固定，与构图一致)；分布式时为全局 batch，按 Worker 切分")
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
    ap.add_argument("--seed", type=int, default=None, help="权重初始化的随机种子；不设则每次随机（分布式时各 Worker 仍从同一份初始权重出发）")
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
    ap.add_argument("--resume", action="store_true", help="从 --out 对应训练断点续训（从断点记录的 epoch 与 step 继续）")
    ap.add_argument("--checkpoint-every", type=int, default=100, help="每多少 step 保存一次训练断点(0=只在 epoch 末保存)")
    ap.add_argument("--stop-file", type=str, default="mycar/logs/STOP_TRAINING", help="检测到该文件时保存断点并退出")
    ap.add_argument("--export-onnx", action="store_true", help="训练结束后导出 ONNX 推理图")
    ap.add_argument("--onnx-out", type=str, default=None, help="ONNX 输出路径，默认跟 --best-out 同名 .onnx")
    ap.add_argument("--stem", type=str, default="cifar", choices=("cifar", "imagenet"), help="小分辨率建议 cifar")
    ap.add_argument("--arch", type=str, default="resnet18", choices=("resnet18", "mobilenet_v2", "alexnet"),
                    help="网络结构；mobilenet_v2 / alexnet（以及带 --pretrained 的 resnet18）与 torchvision 结构一致")
    ap.add_argument("--pretrained", type=str, default=None,
                    help="ImageNet 预训练权重：imagenet = runs/pretrained/converted/<arch>.npz，或转换后的 .npz 路径；"
                         "载入主干、替换回归头，结果里记录载入报告")
    ap.add_argument("--freeze-stages", type=int, default=0,
                    help="冻结前几个阶段（0..5；需 --pretrained）：参数不训练，BN 固定 running statistics 并保持 eval")
    ap.add_argument("--image-size", type=str, default=None,
                    help="输入尺寸 HxW（如 224x224）；默认与首帧一致")
    ap.add_argument("--input-norm", type=str, default="auto", choices=("auto", "none", "imagenet"),
                    help="输入归一化；auto = 有 --pretrained 时按 ImageNet 均值/标准差，否则不归一化")
    ap.add_argument("--base-width", type=int, default=64, help="ResNet-18 第一阶段通道数（后续阶段依次翻倍）；导出 ONNX 时沿用")
    ap.add_argument("--graph-opt", action="store_true", help="打开构图期优化(可能略快)")
    ap.add_argument(
        "--conv-backend",
        type=str,
        default="auto",
        choices=("auto", "numpy", "cupy", "cuda_native_register", "cuda_native_cublas"),
        help="卷积/池化后端；cuda_native_register / cuda_native_cublas 是原生库的两个 GEMM 引擎（寄存器分块 kernel / cuBLAS），"
        "原生库缺失时先自动编译；auto 在 CUDA + float32 时用 cuda_native_register，否则按设备用 CuPy/NumPy",
    )
    ap.add_argument(
        "--distributed",
        type=str,
        default="none",
        choices=("none", "ps", "ring"),
        help="并行方式：none=单进程；ps=Parameter Server 汇总梯度；ring=Ring AllReduce。Worker 共享本机 GPU",
    )
    ap.add_argument("--workers", type=int, default=2, help="分布式 Worker 进程数（--distributed 非 none 时生效）")
    ap.add_argument(
        "--transport",
        type=str,
        default="grpc_proto",
        choices=("grpc_proto", "socket_json"),
        help="PS 通信：grpc_proto（protobuf 二进制）或 socket_json（JSON 文本，只适合小模型）；Ring 只支持 gRPC",
    )
    ap.add_argument("--sync-timeout", type=float, default=300.0, help="分布式时等待其他 Worker 的最长秒数（含验证与保存断点）")
    ap.add_argument("--worker-memory-mb", type=int, default=0,
                    help="分布式时每个 Worker 的 GPU 显存池上限（MiB）；0=把启动时的空闲显存平分给各 Worker")
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
        help="数据加载 worker 数（>0 时用 MyFlows MultiprocessDataLoader 预取；见 benchmark/dataloader_bench.py）；分布式时每个 Worker 各自启动",
    )
    ap.add_argument(
        "--plot-prefix",
        type=str,
        default=None,
        help="历史 JSON/PNG 文件名前缀；默认与日志时间戳一致",
    )
    ap.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="可选：在该目录写 config.json、逐步 metrics.jsonl 与结束时的 result.json，供训练控制台与 AutoPilot 读取",
    )
    ap.add_argument("--phase-timing-every", type=int, default=20,
                    help="每多少 step 用 CUDA event 分别记录前向/反向/优化器耗时（0=关闭）")
    ap.add_argument("--gpu-lock-timeout", type=float, default=0.0,
                    help="等待本机 GPU 锁的最长秒数（0=一直等）；环境变量 MYFLOWS_GPU_LOCK=off 关闭加锁")
    ap.add_argument("--state-every", type=int, default=0,
                    help="每多少 step 保存一次完整 TrainingState（另在每个 epoch 末保存；0=不保存）；只用于单进程训练，"
                         "供 --resume-state 精确续训与 AutoPilot 回滚")
    ap.add_argument("--state-dir", default=None, help="TrainingState 目录；默认 <run-dir>/states")
    ap.add_argument("--keep-states", type=int, default=3, help="保留最近几个 TrainingState")
    ap.add_argument("--resume-state", default=None,
                    help="从这个 TrainingState 目录精确续训（与不中断的训练逐位一致）；除 --lr 外的训练配置必须与它相同")
    return ap


@dataclass
class TrainPlan:
    """一次训练的全部输入：主进程准备一次，单进程或每个分布式 Worker 都按它训练（可经 spawn 传给子进程）。"""

    args: argparse.Namespace
    run_id: str | None
    run_stamp: str
    log_dir: str
    log_file: str
    plot_prefix: str
    tb_logdir: str
    out_base: str
    best_base: str
    stop_file: str
    run_dir: str | None
    dataset: list
    val_dataset: list
    h: int
    w: int
    mode: str = "none"
    transport: str = "none"
    shard_sizes: list = field(default_factory=list)
    # 续训位置：从第 start_epoch 个 epoch 的第 start_step 步继续
    resume_found: bool = False
    start_epoch: int = 0
    start_step: int = 0
    start_global_step: int = 0
    resumed_loss_sum: float = 0.0
    resumed_loss_count: int = 0
    best_loss: float | None = None
    early_stopping_state: dict | None = None
    summary_path: str | None = None
    seeds: dict = field(default_factory=dict)
    data_hash: str | None = None
    val_hash: str | None = None
    split_file: str | None = None
    # TrainingState (P9): where states go, the one to resume exactly from, and the epoch the run ends before
    state_dir: str | None = None
    resume_state: str | None = None
    resume_meta: dict | None = None
    end_epoch: int | None = None
    fingerprint: str | None = None

    @property
    def world_size(self) -> int:
        return len(self.shard_sizes)

    @property
    def global_batch(self) -> int:
        return int(sum(self.shard_sizes))

    @property
    def steps_per_epoch(self) -> int:
        return (len(self.dataset) + self.global_batch - 1) // self.global_batch


def _resume_position(args, out_base: Path, best_base: Path) -> dict:
    """只读断点元数据（不载入数组）：续训从哪个 epoch/step 开始、已有的最佳 loss 与早停计数。"""
    meta = read_checkpoint_metadata(out_base)
    if meta is None:
        return {"resume_found": False}
    epoch = int(meta.get("epoch", -1))
    training = meta.get("training") or {}
    best_meta = read_checkpoint_metadata(best_base)
    # 最佳 loss 取最佳模型自己的分数；断点里的是当时的训练 loss，不能拿来和验证 loss 比较
    best = checkpoint_score_to_loss(best_meta.get("score_value", best_meta.get("best_loss"))) if best_meta else None
    position = {"resume_found": True, "start_epoch": epoch + 1, "start_step": 0,
                "start_global_step": int(training.get("global_step", 0)), "best_loss": best,
                "early_stopping_state": training.get("early_stopping")}
    if training and not training.get("epoch_complete", True):
        if int(training.get("global_batch", 0)) == int(args.batch):
            position.update(start_epoch=epoch, start_step=int(training.get("next_step", 0)),
                            resumed_loss_sum=float(training.get("loss_sum", 0.0)),
                            resumed_loss_count=int(training.get("loss_count", 0)))
        else:
            print(f"[resume] 断点记录的全局 batch={training.get('global_batch')} 与本次 --batch={args.batch} 不同，"
                  f"从头重新训练 epoch {epoch + 1}")
            position["start_epoch"] = epoch
    return position


def _apply_resume_state(args: argparse.Namespace, plan: "TrainPlan") -> None:
    """Position of an exact resume: verify the state, compare the trajectory configuration, set the cursor."""
    from MyFlows.utils.checkpoint import TrainingStateError, verify_training_state

    path = Path(plan.resume_state)
    try:
        verify_training_state(path)
    except TrainingStateError as exc:
        raise SystemExit(f"[resume-state] {exc}") from exc
    state = (read_checkpoint_metadata(path / "checkpoint") or {}).get("training") or {}
    if state.get("fingerprint") != plan.fingerprint:
        saved, now = state.get("config") or {}, trajectory_config(args)
        changed = sorted(k for k in set(saved) | set(now) if saved.get(k) != now.get(k))
        raise SystemExit("[resume-state] 训练配置与 TrainingState 不同，不能精确续训："
                         + ", ".join(f"{k}: {saved.get(k)!r} → {now.get(k)!r}" for k in changed))
    if state.get("data_hash") != plan.data_hash or state.get("val_hash") != plan.val_hash:
        raise SystemExit("[resume-state] 训练集或验证集的样本、标签已变化，不能精确续训")
    saved_seeds = state.get("seeds") or {}
    if (args.dropout and not (state.get("rng") or {}).get("exact")) or \
            ((args.mixup or args.cutmix) and any(s is None for s in saved_seeds.get("mix", [None]))):
        raise SystemExit("[resume-state] 旧 TrainingState 未保存完整随机数状态，不能精确续训")
    plan.seeds = saved_seeds or plan.seeds
    plan.resume_meta = state
    plan.end_epoch = int(state["end_epoch"])
    if state.get("epoch_complete"):
        plan.start_epoch, plan.start_step = int(state["epoch"]) + 1, 0
    else:
        plan.start_epoch, plan.start_step = int(state["epoch"]), int(state["next_step"])
        plan.resumed_loss_sum, plan.resumed_loss_count = float(state["loss_sum"]), int(state["loss_count"])
    plan.start_global_step = int(state["global_step"])
    plan.best_loss = state.get("best_loss")
    plan.early_stopping_state = state.get("early_stopping")
    plan.resume_found = True
    lr_note = "" if float(state.get("lr", args.lr)) == float(args.lr) else f"，lr {state.get('lr')} → {args.lr}"
    print(f"[resume-state] {path.name}：从 epoch {plan.start_epoch + 1} 第 {plan.start_step + 1} 步精确续训"
          f"（已训练 {plan.start_global_step} 步{lr_note}）")


def _rng_streams(train_transform, graph) -> dict:
    """Every random stream a training step draws from (data order and MixUp/CutMix are functions of the epoch
    and the global step): the augmentation transforms' and the dropout ops' generators."""
    from MyFlows.ops.dropout import Dropout_Op

    transforms = [t._rng for t in getattr(train_transform, "transforms", []) if getattr(t, "_rng", None) is not None]
    dropout = [n._rng for n in getattr(graph, "nodes", []) if isinstance(n, Dropout_Op)]
    return {"transforms": transforms, "dropout": dropout}


def _rng_state(streams: dict) -> dict:
    return {"transforms": [list(r.getstate()[1]) + [r.getstate()[0], r.getstate()[2]] for r in streams["transforms"]],
            "dropout": [r.bit_generator.state if r is not None else None for r in streams["dropout"]],
            "exact": all(r is not None for r in streams["dropout"])}


def _restore_rng_state(streams: dict, saved: dict) -> None:
    if len(saved.get("transforms") or []) != len(streams["transforms"]) or \
            len(saved.get("dropout") or []) != len(streams["dropout"]):
        raise SystemExit("[resume-state] 随机数流的数量与 TrainingState 不同")
    for rng, values in zip(streams["transforms"], saved["transforms"]):
        *internal, version, gauss = values
        rng.setstate((version, tuple(internal), gauss))
    for rng, value in zip(streams["dropout"], saved["dropout"]):
        if rng is not None and value is not None:
            rng.bit_generator.state = value


def load_rows(args: argparse.Namespace, data_dir: Path) -> tuple[list, dict | None, Path | None]:
    """The sample rows a run sees (catalog, the first or a seeded random ``--max-samples``) and their split.
    Shared with apps.train.evaluate_donkey, which must select exactly the same rows."""
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
        split_out=(ROOT / args.split_out).resolve() if getattr(args, "split_out", None) else None,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.split_seed,
    )
    return index, splits, split_path


def prepare(args: argparse.Namespace) -> tuple[TrainPlan, TeeLogger]:
    """主进程：解析路径、读数据索引与划分、确定输入尺寸与 Worker 分片，返回训练计划与日志。"""
    data_dir = (ROOT / args.data).resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"数据目录不存在: {data_dir}")
    if args.distributed != "none":
        if args.workers < 1:
            raise SystemExit("--workers 必须是正整数")
        if args.transport not in DISTRIBUTED_TRANSPORTS[args.distributed]:
            raise SystemExit(f"--distributed {args.distributed} 只支持 --transport "
                             f"{'/'.join(DISTRIBUTED_TRANSPORTS[args.distributed])}")
        if args.batch < args.workers:
            raise SystemExit(f"全局 batch（--batch {args.batch}）必须不小于 Worker 数（--workers {args.workers}）")
        if args.dtype != "float32":
            raise SystemExit("分布式训练按 FP32 交换梯度，只支持 --dtype float32")

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

    index, splits, split_path = load_rows(args, data_dir)
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

    # 用第一条样本定输入分辨率
    first_p = data_dir / index[0][0]
    img0 = cv2.imread(str(first_p), cv2.IMREAD_COLOR)
    if img0 is None:
        raise SystemExit(f"首帧读取失败: {first_p}")
    h, w = input_size(args, img0)
    print(f"使用输入尺寸 H×W = {h}×{w}" + (" (与首帧一致)" if not args.image_size else " (--image-size)"))

    out_base = _checkpoint_stem((ROOT / args.out).resolve())
    best_base = _checkpoint_stem((ROOT / args.best_out).resolve())
    out_base.parent.mkdir(parents=True, exist_ok=True)
    best_base.parent.mkdir(parents=True, exist_ok=True)
    distributed = args.distributed != "none"
    plan = TrainPlan(
        args=args, run_id=run_id, run_stamp=run_stamp, log_dir=str(log_dir), log_file=str(log_file),
        plot_prefix=plot_prefix, tb_logdir=str(tb_logdir), out_base=str(out_base), best_base=str(best_base),
        stop_file=str((ROOT / args.stop_file).resolve()),
        run_dir=str((ROOT / args.run_dir).resolve()) if args.run_dir else None,
        dataset=[(str(data_dir / rel), angle, throttle, w, h, args.dtype) for rel, angle, throttle in index],
        val_dataset=[(str(data_dir / rel), angle, throttle, w, h, args.dtype) for rel, angle, throttle in val_index],
        h=h, w=w, mode=args.distributed, transport=args.transport if distributed else "none",
        shard_sizes=split_global_batch(int(args.batch), int(args.workers) if distributed else 1),
        data_hash=entries_hash(index), val_hash=entries_hash(val_index) if val_index else None,
        split_file=str(split_path) if split_path is not None else None,
    )
    plan.seeds = derive_seeds(args.seed, plan.world_size)
    if args.seed is None and (args.state_every or args.resume_state):
        # Even an unseeded run needs private, saved streams for exact resume.
        streams = derive_seeds(random.getrandbits(32), plan.world_size)
        plan.seeds.update(dropout=streams["dropout"], mix=streams["mix"])
    plan.fingerprint = trajectory_fingerprint(args)
    plan.end_epoch = int(args.epochs)
    if args.state_every or args.resume_state:
        if distributed:
            raise SystemExit("TrainingState（--state-every / --resume-state）只用于单进程训练")
        default_dir = Path(args.run_dir) / "states" if args.run_dir else out_base.parent / "states"
        plan.state_dir = str(Path(args.state_dir).resolve() if args.state_dir else default_dir.resolve())
    if args.resume_state:
        if args.resume:
            raise SystemExit("--resume 与 --resume-state 不能同时使用")
        plan.resume_state = str(Path(args.resume_state).resolve())
        _apply_resume_state(args, plan)
    elif args.resume:
        for key, value in _resume_position(args, out_base, best_base).items():
            setattr(plan, key, value)
        if plan.resume_found:
            print(f"[resume] 从 epoch {plan.start_epoch + 1} 第 {plan.start_step + 1} 步继续"
                  f"（已训练 {plan.start_global_step} 步，最佳 loss={plan.best_loss}）")
        else:
            print("[resume] 未发现断点，从零开始训练")

    print(f"日志文件: {log_file}")
    print(f"训练断点: {out_base}.json + {out_base}.npz")
    print(f"最佳模型: {best_base}.json + {best_base}.npz")
    print(f"安全暂停文件: {plan.stop_file}")
    print(f"样本数: {len(plan.dataset)}, batch={plan.global_batch}, steps/epoch={plan.steps_per_epoch}, lr={args.lr}")
    print(f"数据加载: num_workers={int(args.num_workers)}")
    if distributed:
        print(f"分布式: {plan.mode} · {plan.transport} · {plan.world_size} 个 Worker，各 Worker 的 batch={plan.shard_sizes}")
    if args.dropout:
        print(f"Dropout: p={float(args.dropout)}")
    if plan.run_dir is not None:
        run_dir = Path(plan.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_config = {
            "kind": "single", "task": "donkey_resnet18_single", "run_id": run_id or run_stamp,
            "args": vars(args), "device": args.device, "samples": len(plan.dataset), "val_samples": len(val_index),
            "input_h": h, "input_w": w, "steps_per_epoch": plan.steps_per_epoch, "log_file": str(log_file),
            "tensorboard_dir": str(tb_logdir), "checkpoint": str(out_base), "best": str(best_base),
            "stop_file": plan.stop_file, "started_unix_s": time.time(),
            "parallel": {"mode": plan.mode, "transport": plan.transport, "workers": plan.world_size,
                         "shard_sizes": plan.shard_sizes},
            "seeds": plan.seeds, "data_hash": plan.data_hash, "val_hash": plan.val_hash,
            "split_file": plan.split_file, "pid": os.getpid(),
        }
        (run_dir / "config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2, default=str),
                                             encoding="utf-8")
        print(f"控制台运行目录: {run_dir}")
    return plan, tee


def input_size(args, first_image) -> tuple[int, int]:
    """(H, W)：--image-size 或首帧尺寸。"""
    if args.image_size:
        try:
            h, w = (int(v) for v in str(args.image_size).lower().split("x"))
        except ValueError:
            raise SystemExit(f"--image-size 应为 HxW，例如 224x224：{args.image_size}") from None
        if h < 32 or w < 32:
            raise SystemExit("--image-size 的高和宽至少为 32")
        return h, w
    return tuple(first_image.shape[:2])


def transfer_config(args) -> dict | None:
    """--arch / --pretrained / --freeze-stages 的有效配置；默认的 MyFlows ResNet18（旧结构）返回 None。"""
    if args.arch == "resnet18" and not args.pretrained:
        if args.freeze_stages:
            raise SystemExit("--freeze-stages 需要 --pretrained")
        if args.input_norm == "imagenet":
            raise SystemExit("--input-norm imagenet 只用于 torchvision 结构（--arch 或 --pretrained）")
        return None
    if args.freeze_stages and not args.pretrained:
        raise SystemExit("--freeze-stages 需要 --pretrained")
    if args.distributed != "none":
        raise SystemExit("迁移训练（--arch / --pretrained）目前只支持单进程")
    path = None
    if args.pretrained:
        path = (ROOT / "runs" / "pretrained" / "converted" / f"{args.arch}.npz" if args.pretrained == "imagenet"
                else (ROOT / args.pretrained).resolve())
        if not path.is_file():
            raise SystemExit(f"预训练权重不存在：{path}（先运行 tools/convert_torchvision_weights.py）")
    norm = args.input_norm if args.input_norm != "auto" else ("imagenet" if path else "none")
    return {"arch": args.arch, "pretrained": str(path) if path else None, "freeze_stages": int(args.freeze_stages),
            "input_norm": norm}


def _build_model(args, batch: int, h: int, w: int, dtype, conv_backend: str, dropout_seed: int | None = None,
                 load_pretrained: bool = True):
    """构图（批大小固定为 batch）。输入与标签不需要梯度：stem 卷积因此跳过 dX 的 GEMM 与 col2im。

    迁移训练（--arch / --pretrained）时按 torchvision 结构建模型，载入预训练主干、替换回归头并冻结前
    --freeze-stages 个阶段；载入报告与冻结摘要挂在 ``model.transfer`` 上。续训与评估随后会用 checkpoint
    覆盖参数，评估时传 load_pretrained=False 跳过载入。
    """
    x_var = ms.Variable(np.zeros((batch, 3, h, w), dtype=dtype), name="X", requires_grad=False)
    y_var = ms.Variable(np.zeros((batch, 2), dtype=dtype), name="y", requires_grad=False)
    initializer = make_initializer(args.initializer, seed=args.seed) if args.seed is not None else args.initializer
    transfer = transfer_config(args)
    if transfer is None:
        model = ms.ResNet18(
            in_channels=3,
            output_dim=2,
            stem=args.stem,
            base_width=int(args.base_width),
            name="resnet18_donkey",
            dropout=args.dropout,
            initializer=initializer,
            conv_backend=conv_backend,
            dropout_seed=dropout_seed,
        )
    else:
        from MyFlows.utils import pretrained

        model = pretrained.build_model(
            transfer["arch"], 2, dropout=float(args.dropout or 0.0), initializer=initializer,
            conv_backend=conv_backend, dropout_seed=dropout_seed,
            input_norm=None if transfer["input_norm"] == "none" else transfer["input_norm"],
            name=f"{transfer['arch']}_donkey")
        report = None
        if transfer["pretrained"] and load_pretrained:
            report = pretrained.summarize_load_report(
                pretrained.load_torchvision_weights(model, transfer["pretrained"], replace_head=True))
        model.transfer = dict(transfer, load_report=report,
                              freeze=pretrained.freeze_stages(model, transfer["freeze_stages"]))
    logits = model(x_var)
    loss_node = ms.MSELoss(logits, y_var, name="loss")
    graph = ms.Graph(loss_node, optimize=bool(args.graph_opt))
    return x_var, y_var, model, logits, loss_node, graph


def _state_arrays(params: dict, buffers: dict) -> dict:
    arrays = {name: asnumpy(node.value) for name, node in params.items()}
    arrays.update({name: asnumpy(getattr(layer, attr)) for name, (layer, attr) in buffers.items()})
    return arrays


def _assign_buffers(buffers: dict, arrays: dict) -> None:
    for name, (layer, attr) in buffers.items():
        target = getattr(layer, attr)  # BN 算子持有同一个数组，只能原地写
        target[...] = xp.asarray(arrays[name], dtype=target.dtype)


def _fingerprint(params: dict, buffers: dict, optimizer) -> str:
    """每步核对各 Worker 状态一致：参数、BN 统计量与 Adam 矩逐个做 FP64 求和后取哈希（只回读一个小向量）。"""
    values = [node.value for node in params.values()] + [getattr(layer, attr) for layer, attr in buffers.values()]
    for kind in ("v", "s"):
        state = getattr(optimizer, kind, {})
        values += [state[node] for node in params.values() if node in state]
    sums = xp.stack([xp.sum(value, dtype=xp.float64) for value in values])
    return hashlib.sha256(np.ascontiguousarray(asnumpy(sums)).tobytes()).hexdigest()


def _share_gpu_memory(world: int, limit_mb: int = 0) -> int:
    """所有 Worker 共享一张 GPU 时，给本进程的 CuPy 显存池设上限并返回字节数。

    超出上限时 CuPy 立即报显存不足；不设上限时 Windows WDDM 驱动会把显存换到系统内存，
    训练看起来像卡住（每步慢上千倍）。默认上限是启动时空闲显存的 90% 平分给各 Worker。
    """
    if limit_mb > 0:
        limit = int(limit_mb) * 2**20
    else:
        free, _total = xp.cuda.runtime.memGetInfo()
        limit = max(256 * 2**20, int(free * 0.9 / max(1, world)))
    xp.get_default_memory_pool().set_limit(size=limit)
    return limit


def _epoch_samples(dataset: list, epoch: int, shard_sizes: list, rank: int) -> list:
    """本 Worker 一个 epoch 里按步排列的样本。

    全局顺序与 MultiprocessDataLoader(shuffle=True, seed=epoch) 相同；最后一个不满的 batch 重复其末样本补齐
    （与 pad_fixed_batch 相同），再把每个全局 batch 依次切给各 Worker。
    """
    global_batch = int(sum(shard_sizes))
    order = list(range(len(dataset)))
    random.Random(epoch).shuffle(order)
    start = int(sum(shard_sizes[:rank]))
    stop = start + int(shard_sizes[rank])
    samples = []
    for offset in range(0, len(order), global_batch):
        batch = order[offset:offset + global_batch]
        batch += [batch[-1]] * (global_batch - len(batch))
        samples.extend(dataset[i] for i in batch[start:stop])
    return samples


def train_rank(plan: TrainPlan, rank: int = 0, sync=None) -> dict:
    """训练循环。单进程时 rank=0、sync=None；分布式时每个 Worker 进程各跑一份，经 sync 同步梯度与 BN 统计量。"""
    args = plan.args
    lead = rank == 0
    rank_tee = None
    if sync is not None and lead:
        rank_tee = TeeLogger(Path(plan.log_file))  # 分布式时训练日志由 rank 0 写
        sys.stdout = rank_tee
        sys.stderr = rank_tee
    say = print if lead else (lambda *_args, **_kwargs: None)

    dtype = np.float32 if args.dtype == "float32" else np.float64
    batch = int(plan.shard_sizes[rank])
    world = plan.world_size
    h, w = plan.h, plan.w
    steps = plan.steps_per_epoch
    device = resolve_myflows_device(args.device)
    if lead:
        print_myflows_device(device, args.device)
    # 层参数按全局默认 dtype 创建：必须与输入一致，否则 float32 输入会和 float64 权重混算成 float64
    ms.set_default_dtype(dtype)
    conv_backend = _resolve_conv_backend(args.conv_backend, device, dtype)
    if sync is not None and device == "cuda":
        limit = _share_gpu_memory(world, args.worker_memory_mb)
        say(f"GPU 显存: {world} 个 Worker 共享，每个 Worker 的显存池上限 {limit / 2**20:.0f} MiB")
    say(f"计算 dtype: {args.dtype}, 卷积后端: {conv_backend}")

    seeds = plan.seeds or derive_seeds(args.seed, world)
    x_var, y_var, model, logits, loss_node, graph = _build_model(args, batch, h, w, dtype, conv_backend,
                                                                 dropout_seed=seeds["dropout"][rank])
    opt = ms.Adam(graph, learning_rate=args.lr)
    reg_config = RegularizationConfig(
        l1=float(args.l1_coeff),
        l2=float(args.weight_decay),
        regularize_bias_bn=bool(args.regularize_bias_bn),
    )
    out_base, best_base = Path(plan.out_base), Path(plan.best_base)
    stop_file = Path(plan.stop_file)
    if plan.resume_state:
        from MyFlows.utils.checkpoint import load_training_state

        load_training_state(plan.resume_state, [model], opt, best_filepath=best_base)
        opt.learning_rate = float(args.lr)  # the state carries the old rate; a rollback resumes with a smaller one
    elif plan.resume_found:
        ms.load_checkpoint([model], opt, str(out_base))
    params, buffers = named_state(model)
    if sync is not None:
        # 所有 Worker 从同一份初始状态出发：主进程生成，续训时就是断点里的参数与 BN 统计量
        initial = sync.initialize(_state_arrays(params, buffers))
        for name, node in params.items():
            node.value = xp.asarray(initial[name], dtype=node.value.dtype)
        _assign_buffers(buffers, initial)

    transfer = getattr(model, "transfer", None)
    if transfer is not None:
        from MyFlows.utils.pretrained import frozen_fingerprint

        transfer["frozen_before"] = frozen_fingerprint(model)
        report = transfer.get("load_report") or {}
        say(f"迁移训练: {transfer['arch']} · 预训练 {report.get('loaded', 0)}/{report.get('expected', 0)} 个张量 · "
            f"冻结 {transfer['freeze']['frozen'] or '无'} · 可训练 {transfer['freeze']['trainable_values']:,} 个值 · "
            f"输入归一化 {transfer['input_norm']}")

    best_tracker = BestScore()
    best_tracker.best = plan.best_loss
    best_loss = plan.best_loss
    early_stopper = EarlyStopping(args.patience, args.min_delta) if args.early_stopping else None
    if early_stopper is not None and plan.early_stopping_state:
        early_stopper.best = plan.early_stopping_state.get("best")
        early_stopper.bad_epochs = int(plan.early_stopping_state.get("bad_epochs", 0))

    n = len(plan.dataset)
    if lead and reg_config.enabled:
        print(f"正则化: l1={reg_config.l1} l2={reg_config.l2} regularize_bias_bn={reg_config.regularize_bias_bn}")
    if lead and args.summary_once:
        print("[model-summary]")
        print(format_model_summary(model_summary(model)))
    tb_logdir = Path(plan.tb_logdir)
    run_dir = Path(plan.run_dir) if plan.run_dir else None
    dashboard = TrainingDashboard(
        tb_logdir,
        enabled=lead and not args.no_tensorboard,
        grad_interval=args.tb_grad_interval,
        param_interval=args.tb_param_interval,
        activation_interval=args.tb_activation_interval,
        image_interval=args.tb_image_interval,
        log_interval=args.tb_log_interval,
        max_hist_params=args.tb_max_hist_params,
        feature_channels=args.tb_feature_channels,
        jsonl_path=run_dir / "metrics.jsonl" if run_dir is not None and lead else None,
    )
    if dashboard.active:
        print(f"TensorBoard 日志: {tb_logdir}")
        print(f"查看命令: tensorboard --logdir {tb_logdir.parent}")
        dashboard.log_run_config(args, {"samples": n, "input_h": h, "input_w": w, "steps_per_epoch": steps,
                                        "workers": world, "distributed": plan.mode})
        labels = np.asarray([[angle, throttle] for _, angle, throttle, *_ in plan.dataset], dtype=np.float64)
        dashboard.log_dataset_summary(labels, task="regression")
    # 各 Worker 的增强用不同种子（由 --seed 派生），避免每个分片得到同样的随机变换
    train_transform = default_train_transforms(seed=seeds["augment"][rank]) if args.augment else None
    mix_seed = seeds["mix"][rank]
    if args.augment:
        say("数据增强: RandomCrop + RandomRotation + ColorJitter")
    if args.mixup:
        say("MixUp: 已启用")
    if args.cutmix:
        say("CutMix: 已启用")
    last_mean_loss = None
    train_wall_s, trained_samples, last_epoch_train_s = 0.0, 0, None
    should_stop = stopped_by_file = False
    history = TrainingHistory()
    global_step = plan.start_global_step
    inspect_done = False
    synced_params = None
    train_span = _DeviceSpan(device)
    phases = _StepPhases(device, args.phase_timing_every)
    epochs_completed = 0
    validation_history: list[dict] = []
    best_epoch = None
    end_epoch = plan.end_epoch if plan.resume_state else plan.start_epoch + int(args.epochs)
    rng_streams = _rng_streams(train_transform, graph)
    resumed = plan.resume_meta or {}
    first_loss = resumed.get("first_loss")
    lr_history = list(resumed.get("lr_history") or [{"global_step": 0, "lr": float(args.lr)}])
    if plan.resume_meta is not None:
        # Everything the uninterrupted run would hold at this point, so the trajectory continues bit for bit
        _restore_rng_state(rng_streams, resumed.get("rng") or {})
        trained_samples = int(resumed.get("trained_samples", 0))
        train_wall_s = float(resumed.get("train_wall_s", 0.0))
        epochs_completed = int(resumed.get("epochs_completed", 0))
        validation_history = list(resumed.get("validation_history") or [])
        best_epoch = resumed.get("best_epoch")
        last_mean_loss = resumed.get("last_mean_loss")
        if float(resumed.get("lr", args.lr)) != float(args.lr):
            lr_history.append({"global_step": plan.start_global_step, "lr": float(args.lr)})
    rank_steps = None
    if run_dir is not None and sync is not None:
        # 与分布式实验相同的 rank-*/ 记录：控制台据此显示拓扑、各 Rank 状态和 GPU 进程归属
        rank_dir = run_dir / f"rank-{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        (rank_dir / "prepare.json").write_text(json.dumps({"pid": os.getpid(), "rank": rank, "batch": batch}),
                                               encoding="utf-8")
        rank_steps = (rank_dir / "steps.jsonl").open("a", encoding="utf-8", buffering=1)

    def save_training_checkpoint(epoch, *, loss, epoch_complete, next_step=0, loss_sum=0.0, loss_count=0):
        """训练断点只由 rank 0 写（各 Worker 状态相同），元数据记下续训位置与早停计数。"""
        training = {"global_step": global_step, "global_batch": plan.global_batch, "epoch_complete": epoch_complete,
                    "workers": world, "distributed": plan.mode}
        if not epoch_complete:
            training.update(next_step=next_step, loss_sum=loss_sum, loss_count=loss_count)
        if early_stopper is not None:
            training["early_stopping"] = {"best": early_stopper.best, "bad_epochs": early_stopper.bad_epochs}
        return ms.save_checkpoint([model], opt, epoch, filepath=str(out_base), loss=loss, training=training)

    def save_state(epoch, *, epoch_complete, samples, wall_s, next_step=0, loss_sum=0.0, loss_count=0,
                   loss=None, running_loss=None):
        """Full TrainingState (P9) of a single-process run; ``samples`` and ``wall_s`` include the open epoch."""
        if not plan.state_dir or sync is not None:
            return None
        from MyFlows.utils.checkpoint import save_training_state

        state = {
            "epoch": epoch, "epoch_complete": epoch_complete, "next_step": next_step, "loss_sum": loss_sum,
            "loss_count": loss_count, "global_step": global_step, "end_epoch": end_epoch,
            "global_batch": plan.global_batch, "rng": _rng_state(rng_streams),
            "early_stopping": {"best": early_stopper.best, "bad_epochs": early_stopper.bad_epochs}
            if early_stopper is not None else None,
            "best_loss": best_tracker.best, "best_epoch": best_epoch, "validation_history": validation_history,
            "epochs_completed": epochs_completed, "trained_samples": samples, "train_wall_s": wall_s,
            "last_mean_loss": last_mean_loss, "loss": loss, "running_loss": running_loss, "first_loss": first_loss,
            "lr": float(opt.learning_rate), "lr_history": lr_history, "fingerprint": plan.fingerprint,
            "config": trajectory_config(args), "seeds": plan.seeds, "data_hash": plan.data_hash,
            "val_hash": plan.val_hash,
        }
        return save_training_state(plan.state_dir, [model], opt, global_step, state, keep=int(args.keep_states),
                                   best_filepath=best_base if best_tracker.best is not None else None)

    def validation_loss(epoch):
        """验证集按 Worker 交错切分，各自累加逐样本 MSE 与转向角绝对误差再求和；补齐用的重复样本不计入。
        返回 (val_loss, val_angle_mae, 样本数)。"""
        if not plan.val_dataset:
            return None, None, 0
        squared, angle_abs, count = 0.0, 0.0, 0
        model.eval()
        loader = MultiprocessDataLoader(plan.val_dataset[rank::world], batch, num_workers=0, shuffle=False,
                                        load_fn=_load_training_sample)
        for x_items, y_items in loader:
            real = len(x_items)
            x_batch = np.asarray(x_items, dtype=dtype)
            y_batch = np.asarray(y_items, dtype=dtype).reshape(-1, 2)
            x_batch, y_batch = _pad_fixed_batch(x_batch, y_batch, batch)
            x_var.value = x_batch
            y_var.value = y_batch
            graph.forward()
            pred = np.asarray(asnumpy(logits.value), dtype=np.float64)[:real]
            target = y_batch[:real].astype(np.float64)
            squared += float(np.sum((pred - target) ** 2)) / 2.0
            angle_abs += float(np.sum(np.abs(pred[:, 0] - target[:, 0])))
            count += real
        if sync is not None:
            totals = sync.reduce(f"val-{epoch}", {"squared": squared, "angle_abs": angle_abs, "count": count})
            squared, angle_abs = float(totals["squared"][0]), float(totals["angle_abs"][0])
            count = round(float(totals["count"][0]))
        if not count:
            return None, None, 0
        return squared / count, angle_abs / count, count

    if lead and args.state_every and plan.resume_meta is None and global_step == 0:
        # A too-large lr can blow up on the very first update: the initial weights are a rollback target too.
        save_state(plan.start_epoch, epoch_complete=False, samples=0, wall_s=0.0)
    for ep in range(plan.start_epoch, end_epoch):
        first_step = plan.start_step if ep == plan.start_epoch else 0
        total_loss = plan.resumed_loss_sum if first_step else 0.0
        count = plan.resumed_loss_count if first_step else 0
        epoch_start = time.perf_counter()
        loader = MultiprocessDataLoader(
            _epoch_samples(plan.dataset, ep, plan.shard_sizes, rank)[first_step * batch:],
            batch,
            num_workers=int(args.num_workers),
            shuffle=False,
            load_fn=_load_training_sample,
        )
        epoch_interrupted = False
        load_start = time.perf_counter()
        for s, (x_items, y_items) in enumerate(loader, start=first_step):
            after_load = time.perf_counter()
            x_batch = np.asarray(x_items, dtype=dtype)
            y_batch = np.asarray(y_items, dtype=dtype).reshape(-1, 2)

            if train_transform is not None or args.mixup or args.cutmix:
                x_orig = x_batch.copy()
                augment_chw_batch(
                    x_batch,
                    y_batch,
                    train_transform,
                    mixup=bool(args.mixup),
                    cutmix=bool(args.cutmix),
                    mix_seed=None if mix_seed is None else mix_seed + global_step,
                )
                dashboard.log_augmentation(global_step, x_orig, x_batch, batch_size=batch)

            x_var.value = x_batch
            y_var.value = y_batch

            model.train(True)
            phases.begin(global_step)
            train_span.start()
            opt.one_step(phases)
            train_span.stop()
            if reg_config.enabled:
                apply_regularization(opt, model.params, reg_config)
            if (args.check_shape or args.check_content) and not inspect_done:
                report = inspect_graph(graph, check_shape=args.check_shape, check_content=args.check_content)
                say("[graph-inspection]")
                say(format_inspection_report(report))
                if not report["ok"]:
                    raise SystemExit("graph inspection failed")
                inspect_done = True
            if dashboard.active and dashboard.grad_interval > 0 and (global_step % dashboard.grad_interval == 0):
                # 只在要记录时才算：全局范数要读回一个标量，会让 CPU 等 GPU 跑完整个反向
                dashboard.tb.log_scalar("gradients/global_norm", global_gradient_norm(model), global_step)
            dashboard.log_gradients(global_step, model)
            dashboard.log_parameters(global_step, model, args.lr)
            if sync is None:
                with phases("optimizer"):
                    opt.update()
                # 每步只在这里读回一次 loss（同步点），之后再取 CUDA event 计时
                step_loss = myflows_scalar_float(loss_node.value)
                stop_now = stop_file.exists()
            else:
                if synced_params is None:
                    # 第一步核对：所有带梯度的节点都在同步范围内，否则各 Worker 会悄悄分叉
                    unknown = set(opt.acc_gradient) - set(params.values())
                    if unknown:
                        raise RuntimeError(f"{len(unknown)} 个可训练节点不在模型参数表里，无法同步")
                    synced_params = {name: node for name, node in params.items() if node in opt.acc_gradient}
                # 本分片梯度（已含正则项）与前向更新过的 BN 统计量，按样本数加权平均后所有 Worker 相同
                local = {name: asnumpy(opt.get_gradient(node)) if name in synced_params
                         else np.zeros(node.value.shape, np.float32) for name, node in params.items()}
                local.update({name: asnumpy(getattr(layer, attr)) for name, (layer, attr) in buffers.items()})
                sync_start = time.perf_counter()
                means, step_loss = sync.allreduce(local, batch, myflows_scalar_float(loss_node.value))
                sync_s = time.perf_counter() - sync_start
                with phases("optimizer"):
                    opt.update({node: xp.asarray(means[name], dtype=node.value.dtype)
                                for name, node in synced_params.items()})
                _assign_buffers(buffers, means)
                confirm_start = time.perf_counter()
                stop_now = sync.commit(_fingerprint(params, buffers, opt), stop=stop_file.exists())
            if not math.isfinite(step_loss):
                # 不保存断点：最近一次断点仍是健康状态，回滚从那里继续
                raise outcome.NonFiniteLossError(global_step + 1, ep + 1, step_loss)
            phase_row = phases.row()
            if sync is not None and rank_steps is not None:
                ended = time.perf_counter()
                rank_steps.write(json.dumps({
                    "rank": rank, "epoch": ep, "step": global_step, "start_s": load_start, "end_s": ended,
                    "step_wall_s": ended - load_start, "data_s": after_load - load_start,
                    "gradient_sync_s": sync_s, "update_confirm_s": ended - confirm_start,
                    "n_samples": batch, "loss": step_loss, **phase_row}) + "\n")
            step_elapsed_ms = (time.perf_counter() - after_load) * 1000.0
            train_elapsed_ms = train_span.elapsed_ms()
            data_load_ms = (after_load - load_start) * 1000.0
            total_loss += step_loss
            count += 1
            global_step += 1
            if first_loss is None:
                first_loss = step_loss
            history.record_step(global_step, step_loss)
            running_mean_loss = total_loss / count
            dashboard.log_train_step(
                global_step,
                loss=step_loss,
                running_loss=running_mean_loss,
                data_load_ms=data_load_ms,
                train_step_ms=train_elapsed_ms,
                step_time_ms=step_elapsed_ms,
                batch_size=plan.global_batch,
                labels=y_batch,
                task="regression",
                force=s == steps - 1,
                epoch=ep + 1,
                extra=phase_row,
            )
            dashboard.log_activations(global_step, getattr(model, "_last_feature_nodes", {}), preferred_last="layer4")
            if s % 10 == 0 or s == steps - 1:
                say(f"epoch {ep+1} step {s+1}/{steps} loss={step_loss:.6f}")
            if stop_now:
                if lead:
                    json_path, npz_path = save_training_checkpoint(
                        ep, loss=running_mean_loss, epoch_complete=False, next_step=s + 1,
                        loss_sum=total_loss, loss_count=count)
                    print(
                        f"检测到暂停文件，已保存训练断点: {json_path} + {npz_path} "
                        f"(epoch={ep+1}, step={s+1}, running_loss={running_mean_loss:.6f})"
                    )
                    dashboard.log_checkpoint("stop_file", f"step={global_step} epoch={ep+1} path={json_path}", global_step)
                    try:
                        stop_file.unlink()
                    except OSError:
                        pass
                should_stop = stopped_by_file = epoch_interrupted = True
                break
            if lead and args.checkpoint_every and count % int(args.checkpoint_every) == 0 and s != steps - 1:
                json_path, npz_path = save_training_checkpoint(
                    ep, loss=running_mean_loss, epoch_complete=False, next_step=s + 1,
                    loss_sum=total_loss, loss_count=count)
                print(
                    f"已保存训练断点(step {s+1}): {json_path} + {npz_path} "
                    f"(running_loss={running_mean_loss:.6f})"
                )
                dashboard.log_checkpoint("latest", f"step={global_step} epoch={ep+1} path={json_path} running_loss={running_mean_loss:.6f}", global_step)
            if lead and args.state_every and global_step % int(args.state_every) == 0 and s != steps - 1:
                save_state(ep, epoch_complete=False, next_step=s + 1, loss_sum=total_loss, loss_count=count,
                           loss=step_loss, running_loss=running_mean_loss,
                           samples=trained_samples + (count - (plan.resumed_loss_count if first_step else 0))
                           * plan.global_batch, wall_s=train_wall_s + time.perf_counter() - epoch_start)
            load_start = time.perf_counter()
        # 训练吞吐只计前向、反向与同步，不含验证与保存断点
        epoch_train_s = time.perf_counter() - epoch_start
        train_wall_s += epoch_train_s
        trained_samples += (count - (plan.resumed_loss_count if first_step else 0)) * plan.global_batch
        if count and not epoch_interrupted:
            last_epoch_train_s = epoch_train_s
            last_mean_loss = total_loss / count
            history.record_epoch(ep + 1, last_mean_loss)
            validation, angle_mae, validation_count = validation_loss(ep)
            if not math.isfinite(last_mean_loss) or (validation is not None and not math.isfinite(validation)):
                raise outcome.NonFiniteLossError(global_step, ep + 1, validation if validation is not None
                                                 else last_mean_loss)
            epoch_extra = {"val_angle_mae": angle_mae, "val_samples": validation_count,
                           "samples_per_sec": (count - (plan.resumed_loss_count if first_step else 0))
                           * plan.global_batch / epoch_train_s if epoch_train_s else None}
            dashboard.log_epoch(
                ep + 1,
                loss=last_mean_loss,
                validation_loss=validation,
                learning_rate=args.lr,
                epoch_time_s=time.perf_counter() - epoch_start,
                extra=epoch_extra,
            )
            epochs_completed += 1
            validation_history.append({"epoch": ep + 1, "loss": last_mean_loss, "val_loss": validation,
                                       "val_angle_mae": angle_mae})
            say(f"epoch {ep+1} mean loss: {last_mean_loss:.6f}")
            if validation is not None:
                say(f"epoch {ep+1} val loss: {validation:.6f} val angle MAE: {angle_mae:.6f} "
                    f"({validation_count} samples)")

            # 每个 Worker 拿到相同的验证 loss，所以最佳模型与早停的判断处处一致
            score_for_best = float(validation if validation is not None else last_mean_loss)
            improved = best_tracker.update(score_for_best)
            if early_stopper is not None:
                stop_state = early_stopper.update(score_for_best)
                if stop_state.should_stop:
                    say(
                        f"early stopping: best={stop_state.best:.6f} "
                        f"bad_epochs={stop_state.bad_epochs}/{early_stopper.patience}"
                    )
                    should_stop = True
            if lead:
                json_path, npz_path = save_training_checkpoint(ep, loss=last_mean_loss, epoch_complete=True)
                print(f"已保存训练断点: {json_path} + {npz_path}")
                dashboard.log_checkpoint("latest_epoch", f"epoch={ep+1} path={json_path} mean_loss={last_mean_loss:.6f}", ep + 1)
            if improved:
                best_loss = float(best_tracker.best)
                best_epoch = ep + 1
                if lead:
                    best_json, best_npz = ms.save_checkpoint(
                        [model],
                        None,
                        ep,
                        filepath=str(best_base),
                        loss=best_loss,
                    )
                    print(f"已保存最佳模型: {best_json} + {best_npz} (loss={best_loss:.6f})")
                    dashboard.log_checkpoint("best", f"epoch={ep+1} path={best_json} best_loss={best_loss:.6f}", ep + 1)
            if lead and args.state_every:
                save_state(ep, epoch_complete=True, loss=last_mean_loss, running_loss=last_mean_loss,
                           samples=trained_samples, wall_s=train_wall_s)
        if should_stop:
            break

    if lead and plan.resume_state and global_step == plan.start_global_step:
        save_training_checkpoint(plan.start_epoch - 1, loss=last_mean_loss, epoch_complete=True)

    if lead:
        log_dir = Path(plan.log_dir)
        history_json = history.save_json(log_dir / f"{plan.plot_prefix}_history.json")
        print(f"已保存训练历史: {history_json}")
        tb_was_active = dashboard.active
        dashboard.flush()
        dashboard.close()
        if tb_was_active:
            print(f"TensorBoard: tensorboard --logdir {tb_logdir.parent}")
        if args.save_png_plots and (history.step_losses or history.epoch_losses):
            plot_paths = plot_training_curves(history, log_dir, prefix=plan.plot_prefix)
            for p in plot_paths:
                print(f"已保存训练曲线 PNG: {p}")
    if rank_steps is not None:
        rank_steps.close()
    memory = phases.summary()
    if run_dir is not None and sync is not None:
        (run_dir / f"rank-{rank}" / "memory.json").write_text(json.dumps(memory), encoding="utf-8")
    if rank_tee is not None:
        rank_tee.close()
    if transfer is not None:
        from MyFlows.utils.pretrained import frozen_fingerprint

        transfer["frozen_after"] = frozen_fingerprint(model)
        transfer["frozen_unchanged"] = transfer["frozen_after"] == transfer["frozen_before"]
        if not transfer["frozen_unchanged"]:
            say("[迁移训练] 警告：冻结的参数或 BN buffer 在训练中被改动")
    at_best = next((row for row in validation_history if row["epoch"] == best_epoch), None)
    finals = validation_history[-1] if validation_history else {}
    maes = [row["val_angle_mae"] for row in validation_history if row["val_angle_mae"] is not None]
    return {"steps": global_step - plan.start_global_step, "global_step": global_step,
            "last_mean_loss": last_mean_loss, "best_loss": best_loss,
            "stopped_early": bool(should_stop and not stopped_by_file), "stopped_by_file": stopped_by_file,
            "samples_per_s": trained_samples / train_wall_s if train_wall_s else None,
            "epoch_train_wall_s": last_epoch_train_s, "trained_samples": trained_samples,
            "train_wall_s": train_wall_s, "epochs_completed": epochs_completed, "best_epoch": best_epoch,
            "val_loss": finals.get("val_loss"), "val_angle_mae": finals.get("val_angle_mae"),
            "best_val_angle_mae": min(maes) if maes else None,
            "val_angle_mae_at_best": at_best.get("val_angle_mae") if at_best else None,
            "epochs": validation_history, "memory": memory, "device": device, "transfer": transfer,
            "conv_backend": ("numpy" if device == "cpu" else "cupy") if conv_backend == "auto" else conv_backend}


def _rank_main(rank: int, sync, plan: TrainPlan) -> None:
    """分布式 Worker 进程入口（由 SyncRun 以 spawn 启动）。"""
    try:
        summary = train_rank(plan, rank, sync)
    except Exception as exc:
        if type(exc).__name__ == "OutOfMemoryError":  # cupy.cuda.memory.OutOfMemoryError
            raise RuntimeError(f"显存不足（{exc}）。{plan.world_size} 个 Worker 共享一张 GPU，每个 Worker 只能用"
                               "其中一份；请减小 --batch、--base-width 或 --workers，或改用 --stem imagenet") from exc
        raise
    if rank == 0:
        Path(plan.summary_path).write_text(json.dumps(summary), encoding="utf-8")


def _write_initial_state(plan: TrainPlan, path: Path) -> None:
    """在主进程 CPU 上建同一个模型（续训时载入断点），把参数与 BN 统计量写成各 Worker 共同的初始状态。"""
    from MyFlows.distributed.sync import save_state

    args = plan.args
    set_device("cpu")
    ms.set_default_dtype(np.float32)
    _, _, model, _, _, _ = _build_model(args, plan.shard_sizes[0], plan.h, plan.w, np.float32, "auto")
    if plan.resume_found:
        ms.load_checkpoint([model], None, plan.out_base)
    params, buffers = named_state(model)
    save_state(path, _state_arrays(params, buffers))


def _prebuild_native(args) -> None:
    """要用原生卷积时先在主进程编好原生库，免得各 Worker 启动后排队等编译、耗掉同步等待时间。"""
    from MyFlows.ops.cuda_native import ensure_library

    if args.conv_backend.startswith("cuda_native"):
        ensure_library()
    elif args.conv_backend == "auto" and args.device != "cpu":
        try:
            ensure_library()
        except RuntimeError as exc:  # device=auto 且本机没有 GPU 时用不到原生库，由各 Worker 按实际设备决定
            print(f"[native] 预编译失败，交给 Worker 处理: {exc}")


def run_distributed(plan: TrainPlan, tee: TeeLogger) -> tuple[dict, TeeLogger]:
    from MyFlows.distributed.sync import SyncRun

    args = plan.args
    _prebuild_native(args)
    state_dir = Path(plan.run_dir or plan.log_dir)
    initial_state = state_dir / f"{plan.plot_prefix}_initial_state.npz"
    plan.summary_path = str(state_dir / f"{plan.plot_prefix}_rank0_summary.json")
    _write_initial_state(plan, initial_state)
    print(f"启动 {plan.world_size} 个 Worker（{plan.mode} · {plan.transport}），训练日志由 rank 0 写入 {plan.log_file}")
    tee.flush()
    tee.close()  # Worker 运行期间日志文件只由 rank 0 写
    started = time.perf_counter()
    try:
        with SyncRun(mode=plan.mode, transport=plan.transport, world_size=plan.world_size,
                     shard_sizes=plan.shard_sizes, initial_state=initial_state, timeout=args.sync_timeout,
                     run_id=plan.run_id or plan.run_stamp) as run:
            run.start(_rank_main, (plan,))
            finished = run.wait()
    except ValueError as exc:  # 启动前就能判断的配置问题，例如模型太大、Socket JSON 装不下
        finished = {"status": "failed", "error": str(exc)}
    finally:
        initial_state.unlink(missing_ok=True)
        tee = TeeLogger(Path(plan.log_file))
        sys.stdout = tee
        sys.stderr = tee
    elapsed = time.perf_counter() - started
    if finished["status"] != "passed":
        print(f"分布式训练失败（{elapsed:.1f} s）: {finished['error']}")
        if finished.get("traceback"):
            print(finished["traceback"])
        return {"status": "failed", "error": finished["error"],
                "failure_kind": outcome.classify(f"{finished['error']}\n{finished.get('traceback') or ''}")}, tee
    summary_path = Path(plan.summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_path.unlink(missing_ok=True)
    if plan.run_dir is not None:
        ranks = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(plan.run_dir).glob("rank-*/memory.json"))]
        if ranks and all(r.get("source") for r in ranks):
            summary["memory"] = {"source": "cupy_memory_pool", "per_rank": ranks,
                                 "peak_used_mb": sum(r["peak_used_mb"] for r in ranks),
                                 "peak_pool_mb": sum(r["peak_pool_mb"] for r in ranks),
                                 "device_total_mb": ranks[0]["device_total_mb"]}
    print(f"分布式训练完成: {plan.world_size} 个 Worker，{summary['steps']} 步，用时 {elapsed:.1f} s")
    return dict(summary, status="passed"), tee


def _uses_gpu(args) -> bool:
    return args.device == "cuda" or (args.device == "auto" and cuda_available())


def _result(args, plan: TrainPlan | None, summary: dict, *, started_unix_s: float, lock: GpuLock | None,
            failure: dict | None, exit_code: int) -> dict:
    """result.json：Executor 与 Verifier 读取的结束记录（指标、吞吐、显存峰值、样本数、后端、等锁时间）。"""
    finished = time.time()
    acquired = lock.acquired_unix_s if lock is not None else None
    status = "failed" if failure else ("stopped" if summary.get("stopped_by_file") else "succeeded")
    at_best, final = summary.get("val_angle_mae_at_best"), summary.get("val_angle_mae")
    result = {
        "status": status, "exit_code": exit_code, "failure": failure, "task": "donkey_resnet18_single",
        "config": {key: getattr(args, key) for key in ("lr", "batch", "num_workers", "epochs", "max_samples",
                                                       "seed", "dtype", "device", "conv_backend", "distributed",
                                                       "workers", "transport", "base_width", "stem", "weight_decay",
                                                       "dropout", "augment", "val_size", "split_seed", "arch",
                                                       "pretrained", "freeze_stages", "image_size", "input_norm")},
        "metrics": {"val_angle_mae": at_best if at_best is not None else final, "val_angle_mae_final": final,
                    "best_val_angle_mae": summary.get("best_val_angle_mae"), "val_loss": summary.get("val_loss"),
                    "best_loss": summary.get("best_loss"), "last_train_loss": summary.get("last_mean_loss"),
                    "best_epoch": summary.get("best_epoch")},
        "epochs": summary.get("epochs", []),
        "throughput": {"samples_per_s": summary.get("samples_per_s"),
                       "epoch_train_wall_s": summary.get("epoch_train_wall_s"),
                       "train_wall_s": summary.get("train_wall_s")},
        "memory": summary.get("memory") or {"source": None},
        "steps": summary.get("steps"), "global_step": summary.get("global_step"),
        "epochs_completed": summary.get("epochs_completed"),
        "stopped_early": bool(summary.get("stopped_early")), "stopped_by_file": bool(summary.get("stopped_by_file")),
        "transfer": summary.get("transfer"),
        "backend": {"requested": args.conv_backend, "actual": summary.get("conv_backend"),
                    "device": summary.get("device"), "dtype": args.dtype},
        "timing": {"started_unix_s": started_unix_s, "lock_wait_s": lock.wait_s if lock is not None else None,
                   "lock_acquired_unix_s": acquired, "finished_unix_s": finished,
                   "wall_s": finished - started_unix_s, "run_s": finished - acquired if acquired else None},
    }
    if plan is not None:
        expected = None
        if plan.resume_state:
            # An exact resume reports the whole trajectory, the steps before the state included
            expected = int(plan.end_epoch) * plan.steps_per_epoch * plan.global_batch
            meta = plan.resume_meta or {}
            result["resume"] = {"state": plan.resume_state, "from_global_step": meta.get("global_step"),
                                "lr_before": meta.get("lr"), "lr": float(args.lr)}
        elif not plan.resume_found:
            expected = int(args.epochs) * plan.steps_per_epoch * plan.global_batch
        result.update(
            samples={"train": len(plan.dataset), "val": len(plan.val_dataset),
                     "trained": summary.get("trained_samples"), "expected_trained": expected},
            data={"data_hash": plan.data_hash, "val_hash": plan.val_hash, "split_file": plan.split_file},
            seeds=plan.seeds,
            parallel={"mode": plan.mode, "transport": plan.transport, "workers": plan.world_size,
                      "shard_sizes": plan.shard_sizes},
            paths={"checkpoint": plan.out_base, "best": plan.best_base, "log": plan.log_file,
                   "states": plan.state_dir})
    return result


def main() -> None:
    args = build_parser().parse_args()
    started_unix_s = time.time()
    run_dir = (ROOT / args.run_dir).resolve() if args.run_dir else None
    plan = tee = lock = None
    summary: dict = {}
    failure, exit_code = None, outcome.EXIT_OK
    try:
        plan, tee = prepare(args)
        lock = GpuLock(label=f"apps.train {plan.run_id or plan.run_stamp}", run_dir=run_dir, say=print,
                       enabled=_uses_gpu(args))
        lock.acquire(timeout_s=args.gpu_lock_timeout or None)
        if plan.mode == "none":
            summary = dict(train_rank(plan), status="passed")
        else:
            summary, tee = run_distributed(plan, tee)
            if summary["status"] != "passed":
                kind = summary.get("failure_kind", "error")
                failure, exit_code = {"kind": kind, "message": summary.get("error")}, outcome.FAILURE_EXIT_CODES[kind]
        if summary["status"] == "passed" and args.export_onnx:
            _export_onnx(args, plan)
    except outcome.NonFiniteLossError as exc:
        print(f"训练终止: {exc}")
        failure = {"kind": "nan", "message": str(exc), "step": exc.step, "epoch": exc.epoch, "loss": str(exc.loss)}
        exit_code = outcome.EXIT_NONFINITE
    except SystemExit as exc:
        if exc.code in (None, 0):
            raise
        failure = {"kind": "error", "message": str(exc.code)}
        exit_code = exc.code if isinstance(exc.code, int) else outcome.EXIT_ERROR
        if not isinstance(exc.code, int):
            print(exc.code)
    except BaseException as exc:  # 包括 KeyboardInterrupt：同样留下 result.json 再退出
        traceback.print_exc()
        kind = outcome.classify(exc)
        failure = {"kind": kind, "message": f"{type(exc).__name__}: {exc}"}
        exit_code = outcome.FAILURE_EXIT_CODES[kind]
    finally:
        if lock is not None:
            lock.release()

    if run_dir is not None:
        result = _result(args, plan, summary, started_unix_s=started_unix_s, lock=lock, failure=failure,
                         exit_code=exit_code)
        outcome.write_result(run_dir, result)
        # results.json 是控制台运行列表沿用的旧格式
        legacy = {"status": "passed" if failure is None else "failed", "finished_unix_s": time.time(),
                  "steps": summary.get("steps"), "last_mean_loss": summary.get("last_mean_loss"),
                  "best_loss": summary.get("best_loss"), "stopped_early": bool(summary.get("stopped_early")),
                  "stopped_by_file": bool(summary.get("stopped_by_file")),
                  "samples_per_s": summary.get("samples_per_s"),
                  "epoch_train_wall_s": summary.get("epoch_train_wall_s"),
                  "val_angle_mae": result["metrics"]["val_angle_mae"],
                  "workers": plan.world_size if plan else None, "distributed": args.distributed}
        if failure is not None:
            legacy["error"] = failure["message"]
            legacy["failure_kind"] = failure["kind"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.json").write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")
    if tee is not None:
        tee.close()
    if exit_code != outcome.EXIT_OK:
        raise SystemExit(exit_code)


def _export_onnx(args, plan: TrainPlan) -> None:
    if transfer_config(args) is not None:
        print("ONNX 导出目前只支持默认的 MyFlows ResNet18 结构，跳过")
        return
    best_base = Path(plan.best_base)
    if not best_base.with_suffix(".json").exists():
        print(f"还没有最佳模型（第一个 epoch 结束前就停止了），跳过 ONNX 导出: {best_base}")
        return
    onnx_path = (ROOT / args.onnx_out).resolve() if args.onnx_out else best_base.with_suffix(".onnx")
    export_resnet18_onnx(
        best_base,
        output=onnx_path,
        batch_size=1,
        image_w=plan.w,
        image_h=plan.h,
        stem=args.stem,
        device=args.device,
        graph_opt=bool(args.graph_opt),
        base_width=int(args.base_width),
    )
    print(f"已导出 ONNX: {onnx_path} (batch=1)")


if __name__ == "__main__":
    main()
