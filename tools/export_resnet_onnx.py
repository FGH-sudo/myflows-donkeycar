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
from apps.common.image_preprocess import format_roi_crop, parse_roi_crop
from apps.train.common.regression_model import build_regression_resnet
from MyFlows.layers.resnet import BASIC_RESNETS
from tools.device_runtime import print_myflows_device, resolve_myflows_device


def _checkpoint_stem(path: Path) -> Path:
    if path.suffix.lower() in {".json", ".npz", ".onnx"}:
        return path.with_suffix("")
    return path


def _default_output_path(checkpoint: Path, batch_size: int, *, dynamic_batch: bool = False) -> Path:
    suffix = "_dynamic.onnx" if dynamic_batch else f"_b{int(batch_size)}.onnx"
    return checkpoint.with_name(f"{checkpoint.name}{suffix}")


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def export_resnet_onnx(
    checkpoint: str | Path,
    *,
    output: str | Path | None = None,
    batch_size: int = 1,
    image_w: int = 160,
    image_h: int = 120,
    stem: str = "cifar",
    device: str = "auto",
    graph_opt: bool = True,
    base_width: int = 64,
    regression_head: str = "global",
    arch: str = "resnet18",
    roi_crop: str | tuple[int, int, int, int] | None = None,
    dynamic_batch: bool = False,
) -> Path:
    """Load a MyFlows BasicBlock ResNet checkpoint and export its inference graph."""
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

    output_path = (
        _resolve_repo_path(output).resolve()
        if output
        else _default_output_path(checkpoint_path, batch_size, dynamic_batch=bool(dynamic_batch))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_myflows_device(device)
    print_myflows_device(resolved_device, device)

    x_var = ms.Variable(
        np.zeros((batch_size, 3, image_h, image_w), dtype=np.float32),
        name="X",
    )
    model = build_regression_resnet(
        arch, regression_head, image_h, image_w,
        in_channels=3,
        output_dim=2,
        stem=stem,
        base_width=int(base_width),
        name=f"{arch}_donkey",
    )
    logits = model(x_var)
    model.eval()
    ms.load_checkpoint([model], None, str(checkpoint_path))

    crop = parse_roi_crop(roi_crop)
    crop_str = format_roi_crop(crop)
    export_kwargs = {}
    if crop_str:
        export_kwargs["metadata"] = {"myflows.roi_crop": crop_str}
    if dynamic_batch:
        export_kwargs["dynamic_batch"] = True
    graph = ms.Graph(logits, optimize=bool(graph_opt))
    graph.forward()
    exported = ms.export_onnx(
        graph,
        str(output_path),
        input_nodes=[x_var],
        output_names=["control"],
        **export_kwargs,
    )
    return Path(exported)


# Existing callers of the ResNet-18 helper keep its default architecture.
export_resnet18_onnx = export_resnet_onnx


def export_vgg11_onnx(
    checkpoint: str | Path,
    *,
    output: str | Path | None = None,
    batch_size: int = 1,
    image_w: int = 160,
    image_h: int = 120,
    device: str = "auto",
    graph_opt: bool = True,
    roi_crop: str | tuple[int, int, int, int] | None = None,
    dynamic_batch: bool = False,
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

    output_path = (
        _resolve_repo_path(output).resolve()
        if output
        else _default_output_path(checkpoint_path, batch_size, dynamic_batch=bool(dynamic_batch))
    )
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

    crop = parse_roi_crop(roi_crop)
    crop_str = format_roi_crop(crop)
    export_kwargs = {}
    if crop_str:
        export_kwargs["metadata"] = {"myflows.roi_crop": crop_str}
    if dynamic_batch:
        export_kwargs["dynamic_batch"] = True
    graph = ms.Graph(control, optimize=bool(graph_opt))
    graph.forward()
    exported = ms.export_onnx(
        graph,
        str(output_path),
        input_nodes=[x_var],
        output_names=["control"],
        **export_kwargs,
    )
    return Path(exported)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained MyFlows Donkey checkpoint to ONNX."
    )
    parser.add_argument("--model", choices=(*BASIC_RESNETS, "vgg11"), default="resnet18")
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
    parser.add_argument("--roi-crop", type=str, default=None, help="Optional ROI crop margins TOP,BOTTOM[,LEFT,RIGHT].")
    parser.add_argument("--stem", choices=("cifar", "imagenet"), default="cifar")
    parser.add_argument("--base-width", type=int, default=64, help="ResNet first-stage width used in training.")
    parser.add_argument("--regression-head", choices=("global", "spatial"), default="global")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-graph-opt", action="store_true", help="Disable graph optimization during export.")
    parser.add_argument("--dynamic-batch", action="store_true", help="Export ONNX with symbolic batch dimension 'batch'.")
    args = parser.parse_args()

    try:
        dyn_kwargs = {"dynamic_batch": True} if args.dynamic_batch else {}
        if args.model == "vgg11":
            exported = export_vgg11_onnx(
                args.checkpoint,
                output=args.output,
                batch_size=args.batch_size,
                image_w=args.image_w,
                image_h=args.image_h,
                device=args.device,
                graph_opt=not args.no_graph_opt,
                roi_crop=args.roi_crop,
                **dyn_kwargs,
            )
        else:
            exported = export_resnet_onnx(
                args.checkpoint,
                arch=args.model,
                output=args.output,
                batch_size=args.batch_size,
                image_w=args.image_w,
                image_h=args.image_h,
                stem=args.stem,
                device=args.device,
                graph_opt=not args.no_graph_opt,
                base_width=args.base_width,
                regression_head=args.regression_head,
                roi_crop=args.roi_crop,
                **dyn_kwargs,
            )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Exported ONNX: {exported}")
    batch_label = "batch" if args.dynamic_batch else int(args.batch_size)
    print(f"Input shape: [{batch_label}, 3, {int(args.image_h)}, {int(args.image_w)}]")


if __name__ == "__main__":
    main()
