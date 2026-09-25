"""Job presets and submission validation for the console."""

from __future__ import annotations

import copy
from pathlib import Path

from MyFlows.distributed.constants import KNOWN_TASKS, LEGAL_MODES

MODE_LABELS = {
    ("single", "none"): "单进程基线",
    ("ps", "socket_json"): "PS · Socket JSON",
    ("ps", "grpc_proto"): "PS · gRPC",
    ("ring", "grpc_proto"): "Ring AllReduce · gRPC",
}
TASK_LABELS = {
    "synthetic": "合成数据 MLP（冒烟）",
    "mnist_mlp": "MNIST MLP",
    "donkey_cnn": "DonkeyCar 小 CNN",
    "donkey_resnet18": "DonkeyCar ResNet18",
}
DISTRIBUTED_KEYS = {
    "task", "mode", "transport", "train_workers", "shard_sizes", "device", "optimizer", "learning_rate", "seed",
    "global_batch", "epochs", "steps", "timeout", "evaluate", "eval_batch", "final_test", "profile", "monitor",
    "prepared_dir", "init_checkpoint", "save_epoch_checkpoints", "base_width", "resnet_stem", "bn_mode",
    "bn_statistics", "heartbeat_interval_s", "heartbeat_timeout_s", "poll_interval_s", "collect_history",
    "numerical_snapshots", "conv_backend", "image_h", "image_w", "hidden", "data_dir", "record_local_gradients",
}
SINGLE_ARGS = {
    "max_samples": int, "epochs": int, "batch": int, "lr": float, "device": str, "dtype": str, "stem": str,
    "augment": bool, "graph_opt": bool, "checkpoint_every": int, "tb_log_interval": int, "weight_decay": float,
    "dropout": float, "early_stopping": bool, "patience": int, "val_size": int, "test_size": int,
    "sample_seed": int, "split_seed": int, "num_workers": int, "no_tensorboard": bool,
}
MAX_WORKERS = 8


class ValidationError(ValueError):
    pass


def modes() -> list[dict]:
    return [{"mode": m, "transport": t, "label": MODE_LABELS.get((m, t), f"{m}/{t}")}
            for m, t in sorted(LEGAL_MODES, key=lambda mt: list(MODE_LABELS).index(mt) if mt in MODE_LABELS else 99)]


def _task_config(archive_root: Path | None, task: str) -> dict:
    from benchmark.distributed_suite import base_config, task_config

    config = task_config(archive_root, task) if archive_root else base_config(task)
    if task == "synthetic":
        config.pop("prepared_dir", None)
    return config


def distributed_presets(repo_root: Path, archive_root: Path | None) -> list[dict]:
    presets = []

    def add(pid, task, label, description, overrides=None):
        config = _task_config(archive_root, task)
        config.update(overrides or {})
        prepared = config.get("prepared_dir")
        ready = prepared is None or Path(prepared).exists()
        bn = config.get("bn_statistics")
        if bn and not Path(bn).exists():
            ready = False
        presets.append({
            "id": pid, "type": "distributed", "task": task, "label": label, "description": description,
            "config": config, "data_ready": ready,
            "data_hint": None if ready else
            f"数据未准备：先运行 python -m benchmark.prepare_distributed_data --out .codex/distributed-data-v1 --tasks {task}",
        })

    add("synthetic-smoke", "synthetic", "合成数据冒烟（CPU）", "固定合成数据的小 MLP，几秒完成，用于检查进程与通信。",
        {"device": "cpu", "optimizer": "mbgd", "learning_rate": 0.01, "global_batch": 32, "steps": 200, "epochs": 1,
         "evaluate": False, "final_test": False, "profile": False, "timeout": 60.0})
    add("mnist-quick", "mnist_mlp", "MNIST MLP 快速（2 epoch）", "与正式实验同配置，只训练 2 个 epoch。", {"epochs": 2})
    add("mnist-formal", "mnist_mlp", "MNIST MLP 正式配置", "第四周正式实验的冻结配置。")
    add("resnet18-quick", "donkey_resnet18", "ResNet18 道路快速（1 epoch × 20 step）",
        "冻结配置下只跑 1 个 epoch、每 epoch 20 个 batch，用于检查 GPU 与原生算子。",
        {"epochs": 1, "steps": 20, "final_test": False, "save_epoch_checkpoints": False})
    add("resnet18-formal", "donkey_resnet18", "ResNet18 道路正式配置", "第四周正式实验的冻结配置（5 epoch）。")
    return presets


def single_presets(repo_root: Path) -> list[dict]:
    data_ready = (Path(repo_root) / "mycar" / "data" / "images").exists()
    hint = None if data_ready else "未找到 mycar/data/images"
    return [
        {"id": "single-resnet18-smoke", "type": "single", "task": "donkey_resnet18_single",
         "label": "单进程 ResNet18 小样本（200 张）", "description": "apps.train 主训练入口，200 个样本、1 个 epoch。",
         "args": {"max_samples": 200, "epochs": 1, "batch": 2, "lr": 0.001, "device": "auto", "dtype": "float32",
                  "checkpoint_every": 0, "tb_log_interval": 5}, "data_ready": data_ready, "data_hint": hint},
        {"id": "single-resnet18-1k", "type": "single", "task": "donkey_resnet18_single",
         "label": "单进程 ResNet18（1000 张 × 2 epoch）", "description": "带验证集的短训练。",
         "args": {"max_samples": 1000, "epochs": 2, "batch": 4, "lr": 0.001, "device": "auto", "dtype": "float32",
                  "val_size": 100, "checkpoint_every": 0, "tb_log_interval": 10}, "data_ready": data_ready,
         "data_hint": hint},
    ]


def validate_distributed(config: dict) -> dict:
    from MyFlows.distributed.launcher import normalize_config

    unknown = set(config) - DISTRIBUTED_KEYS
    if unknown:
        raise ValidationError(f"不支持的配置项：{', '.join(sorted(unknown))}")
    config = copy.deepcopy(config)
    task = config.get("task")
    if task not in KNOWN_TASKS:
        raise ValidationError(f"未知任务 {task}")
    mode, transport = config.get("mode"), config.get("transport")
    if (mode, transport) not in LEGAL_MODES:
        raise ValidationError(f"非法的模式/传输组合 {mode}/{transport}")
    try:
        workers = int(config.get("train_workers", 1))
    except (TypeError, ValueError):
        raise ValidationError("train_workers 必须是整数")
    if mode == "single" and workers != 1:
        raise ValidationError("单进程模式只能有 1 个 worker")
    if not 1 <= workers <= MAX_WORKERS:
        raise ValidationError(f"worker 数必须在 1 到 {MAX_WORKERS} 之间")
    config["train_workers"] = workers
    if config.get("device") not in ("cpu", "cuda"):
        raise ValidationError("device 只能是 cpu 或 cuda")
    if config.get("optimizer", "mbgd") not in ("mbgd", "adam"):
        raise ValidationError("optimizer 只能是 mbgd 或 adam")
    if float(config.get("learning_rate", 0.01)) <= 0:
        raise ValidationError("learning_rate 必须为正数")
    shards = config.get("shard_sizes")
    if shards is not None:
        if not isinstance(shards, list) or len(shards) != workers or any(int(s) <= 0 for s in shards):
            raise ValidationError("shard_sizes 必须是长度等于 worker 数的正整数列表")
        if sum(int(s) for s in shards) != int(config.get("global_batch", 32)):
            raise ValidationError("shard_sizes 之和必须等于 global_batch")
    if task != "synthetic":
        prepared = config.get("prepared_dir")
        if not prepared or not Path(prepared).exists():
            raise ValidationError(f"预处理数据目录不存在：{prepared}")
    if config.get("bn_statistics") and not Path(config["bn_statistics"]).exists():
        raise ValidationError(f"BN 统计量文件不存在：{config['bn_statistics']}")
    try:
        normalize_config(**config)
    except (ValueError, TypeError) as exc:
        raise ValidationError(str(exc)) from exc
    return config


def validate_single(args: dict, repo_root: Path) -> dict:
    out = {}
    for key, value in (args or {}).items():
        caster = SINGLE_ARGS.get(key)
        if caster is None:
            raise ValidationError(f"不支持的训练参数：{key}")
        if value is None:
            continue
        try:
            out[key] = bool(value) if caster is bool else caster(value)
        except (TypeError, ValueError):
            raise ValidationError(f"参数 {key} 的值无效：{value}")
    if out.get("device", "auto") not in ("auto", "cpu", "cuda"):
        raise ValidationError("device 只能是 auto、cpu 或 cuda")
    if out.get("dtype", "float32") not in ("float32", "float64"):
        raise ValidationError("dtype 只能是 float32 或 float64")
    if out.get("epochs", 1) < 1 or out.get("batch", 1) < 1:
        raise ValidationError("epochs 与 batch 必须为正整数")
    if not (Path(repo_root) / "mycar" / "data").is_dir():
        raise ValidationError("数据目录 mycar/data 不存在")
    return out


def single_cli_args(args: dict) -> list[str]:
    cli = []
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cli.append(flag)
        else:
            cli += [flag, str(value)]
    return cli
