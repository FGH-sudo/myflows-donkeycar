# -*- coding: utf-8 -*-
"""Shared helpers for MyFlows/testmyflow benchmark and gate runners."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def import_tree(tree: Path | str):
    """Put ``tree`` first on ``sys.path``, import ``MyFlows``, and verify it resolved from ``tree``."""
    resolved = str(Path(tree).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    elif sys.path[0] != resolved:
        sys.path.remove(resolved)
        sys.path.insert(0, resolved)
    import MyFlows as ms

    if not Path(ms.__file__).resolve().is_relative_to(Path(resolved)):
        raise RuntimeError(f"imported {ms.__file__}, expected a module under {resolved}")
    return ms


def build_resnet_workload(ms, args, device: str, dtype: str, backend: str):
    """Construct the standard benchmark ResNet18 + MSELoss + Graph workload."""
    from MyFlows.core.device import set_device, xp
    from MyFlows.core.node import Variable
    from MyFlows.layers.layer import Conv2D
    from MyFlows.layers.resnet import BatchNorm2d
    from MyFlows.ops.convolution import MaxPool2d_Op
    from MyFlows.utils.initializers import make_initializer

    set_device(device)
    model = ms.ResNet18(
        in_channels=3,
        output_dim=2,
        stem=args.stem,
        base_width=args.width,
        name="resnet18",
        initializer=make_initializer("kaiming_normal", seed=0),
    )
    if backend in ("cuda_tf32", "cuda_fp16", "cuda_bf16"):
        model.linear_backend = backend
    forced = {"float32": np.float32, "float64": np.float64}.get(dtype)
    for layer in model.sub_layers:
        if isinstance(layer, Conv2D) and backend != "default":
            layer.backend = backend
        if isinstance(layer, BatchNorm2d) and forced is not None:
            layer.running_mean = xp.asarray(layer.running_mean, dtype=forced)
            layer.running_var = xp.asarray(layer.running_var, dtype=forced)
    if forced is not None:
        for node in model.params:
            node.value = xp.asarray(node.value, dtype=forced)
    shape = (args.batch, 3, args.height, args.width_px)
    input_dtype = np.float64 if forced is np.float64 else np.float32
    try:
        x = Variable(xp.zeros(shape, input_dtype), name="X", requires_grad=False)
        y = Variable(xp.zeros((args.batch, 2), input_dtype), name="y", requires_grad=False)
    except TypeError:  # legacy Variable without requires_grad kwarg
        x = Variable(xp.zeros(shape, input_dtype), name="X")
        y = Variable(xp.zeros((args.batch, 2), input_dtype), name="y")
    loss = ms.MSELoss(model(x), y)
    graph = ms.Graph(loss, optimize=getattr(args, "graph_opt", False))
    for node in graph.nodes:
        if isinstance(node, MaxPool2d_Op) and backend != "default":
            node.backend = backend
    return model, x, y, loss, graph


def run_worker_json(module: str, worker_args: list[str], *, cwd: Path = ROOT, timeout: float = 1800.0):
    """Run ``python -X utf8 -m <module> <worker_args>`` and parse the trailing JSON line."""
    cmd = [sys.executable, "-X", "utf8", "-m", module, *worker_args]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
    if proc.returncode:
        raise RuntimeError(proc.stdout[-1000:] + proc.stderr[-2000:])
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{module} produced no JSON output")
    return json.loads(lines[-1])


def cuda_sync() -> None:
    """Synchronize the active CUDA stream when running on GPU."""
    from MyFlows.core.device import get_device, xp

    if get_device() == "cuda":
        xp.cuda.get_current_stream().synchronize()


def pool_stats_mib() -> dict[str, float]:
    """Return current CuPy default memory pool used/total bytes in MiB (zeros on CPU)."""
    from MyFlows.core.device import get_device, xp

    if get_device() != "cuda":
        return {"used_mib": 0.0, "total_mib": 0.0}
    pool = xp.get_default_memory_pool()
    return {
        "used_mib": float(pool.used_bytes()) / (1024.0 ** 2),
        "total_mib": float(pool.total_bytes()) / (1024.0 ** 2),
    }


def summarize_ms(samples: list[float]) -> dict[str, float]:
    """Compute robust summary statistics over a non-empty list of finite durations in ms."""
    values = [float(v) for v in samples]
    if not values:
        raise ValueError("samples must not be empty")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("samples must all be finite")
    ordered = sorted(values)
    idx_95 = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "count": len(values),
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.fmean(values)),
        "stdev_ms": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "p95_ms": float(ordered[idx_95]),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def add_common_worker_args(
    parser: argparse.ArgumentParser,
    *,
    default_warmup: int = 8,
    default_steps: int = 40,
) -> argparse.ArgumentParser:
    """Register standard ``--tree``, ``--warmup``, and ``--steps`` options."""
    parser.add_argument("--tree", type=Path, default=ROOT)
    parser.add_argument("--warmup", type=int, default=default_warmup)
    parser.add_argument("--steps", type=int, default=default_steps)
    return parser
