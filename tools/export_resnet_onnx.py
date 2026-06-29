#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export MyFlows Donkey checkpoints to ONNX with a chosen batch size."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import MyFlows as ms
from tools.device_runtime import print_myflows_device, resolve_myflows_device


def _checkpoint_stem(path: Path) -> Path:
    if path.suffix.lower() in {".json", ".npz", ".onnx"}:
        return path.with_suffix("")
    return path


def _default_output_path(checkpoint: Path, batch_size: int) -> Path:
    return checkpoint.with_name(f"{checkpoint.name}_b{int(batch_size)}.onnx")


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def export_resnet18_onnx(
    checkpoint: str | Path,
    *,
    output: str | Path | None = None,
    batch_size: int = 1,
    image_w: int = 160,
    image_h: int = 120,
    stem: str = "cifar",
    device: str = "auto",
    graph_opt: bool = True,
) -> Path:
    """Load a ResNet-18 checkpoint and export an ONNX graph for deployment."""
    checkpoint_path = _checkpoint_stem(_resolve_repo_path(checkpoint).resolve())
    json_path = checkpoint_path.with_suffix(".json")
    npz_path = checkpoint_path.with_suffix(".npz")
    if not json_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {json_path} + {npz_path}")

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    image_w = int(image_w)
    image_h = int(image_h)

    output_path = _resolve_repo_path(output).resolve() if output else _default_output_path(checkpoint_path, batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_myflows_device(device)
    print_myflows_device(resolved_device, device)

    x_var = ms.Variable(
        np.zeros((batch_size, 3, image_h, image_w), dtype=np.float32),
        name="X",
    )
    model = ms.ResNet18(
        in_channels=3,
        output_dim=2,
        stem=stem,
        base_width=64,
        name="resnet18_donkey",
    )
    logits = model(x_var)
    model.eval()
    ms.load_checkpoint([model], None, str(checkpoint_path))

    graph = ms.Graph(logits, optimize=bool(graph_opt))
    graph.forward()
    exported = ms.export_onnx(
        graph,
        str(output_path),
        input_nodes=[x_var],
        output_names=["control"],
    )
    return Path(exported)


def export_vgg11_onnx(
    checkpoint: str | Path,
    *,
    output: str | Path | None = None,
    batch_size: int = 1,
    image_w: int = 160,
    image_h: int = 120,
    device: str = "auto",
    graph_opt: bool = True,
) -> Path:
    """Load a VGG-11 Donkey regression checkpoint and export an ONNX graph for deployment."""
    checkpoint_path = _checkpoint_stem(_resolve_repo_path(checkpoint).resolve())
    json_path = checkpoint_path.with_suffix(".json")
    npz_path = checkpoint_path.with_suffix(".npz")
    if not json_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {json_path} + {npz_path}")

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    image_w = int(image_w)
    image_h = int(image_h)

    output_path = _resolve_repo_path(output).resolve() if output else _default_output_path(checkpoint_path, batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_myflows_device(device)
    print_myflows_device(resolved_device, device)

    x_var = ms.Variable(
        np.zeros((batch_size, 3, image_h, image_w), dtype=np.float32),
        name="X",
    )
    model = ms.VGG11(
        in_channels=3,
        output_dim=2,
        image_h=image_h,
        image_w=image_w,
        name="vgg11_donkey",
    )
    control = model(x_var)
    model.eval()
    ms.load_checkpoint([model], None, str(checkpoint_path))

    graph = ms.Graph(control, optimize=bool(graph_opt))
    graph.forward()
    exported = ms.export_onnx(
        graph,
        str(output_path),
        input_nodes=[x_var],
        output_names=["control"],
    )
    return Path(exported)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained MyFlows Donkey checkpoint to ONNX."
    )
    parser.add_argument("--model", choices=("resnet18", "vgg11"), default="resnet18")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="mycar/models/myflow_resnet18_best",
        help="Checkpoint stem or .json/.npz path. An .onnx path is treated as the same stem.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .onnx path. Defaults to <checkpoint>_b<batch-size>.onnx.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="ONNX input batch size.")
    parser.add_argument("--image-w", type=int, default=160, help="Input image width.")
    parser.add_argument("--image-h", type=int, default=120, help="Input image height.")
    parser.add_argument("--stem", choices=("cifar", "imagenet"), default="cifar")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-graph-opt", action="store_true", help="Disable graph optimization during export.")
    args = parser.parse_args()

    try:
        if args.model == "vgg11":
            exported = export_vgg11_onnx(
                args.checkpoint,
                output=args.output,
                batch_size=args.batch_size,
                image_w=args.image_w,
                image_h=args.image_h,
                device=args.device,
                graph_opt=not args.no_graph_opt,
            )
        else:
            exported = export_resnet18_onnx(
                args.checkpoint,
                output=args.output,
                batch_size=args.batch_size,
                image_w=args.image_w,
                image_h=args.image_h,
                stem=args.stem,
                device=args.device,
                graph_opt=not args.no_graph_opt,
            )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Exported ONNX: {exported}")
    print(f"Input shape: [{int(args.batch_size)}, 3, {int(args.image_h)}, {int(args.image_w)}]")


if __name__ == "__main__":
    main()
