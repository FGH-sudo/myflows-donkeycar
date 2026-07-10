#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Grad-CAM TensorBoard visualizations for trained Donkey models."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import MyFlows as ms
from apps.common.donkey_data import load_donkey_index
from apps.common.image_preprocess import read_rgb
from apps.common.splits import load_split, select_split
from MyFlows.utils.gradcam import chw_float_image, gradcam_for_image, heatmap_chw
from MyFlows.utils.tensorboard_logger import TensorBoardLogger
from apps.train.common.checkpoints import checkpoint_stem
from tools.explain.model_factory import build_gradcam_model, select_feature_node
from tools.explain.reporting import append_gradcam_row, gradcam_report_header, write_report
from tools.explain.targets import select_target
from tools.device_runtime import myflows_asnumpy, print_myflows_device, resolve_myflows_device


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for MyFlows Donkey checkpoints.")
    parser.add_argument("--model-type", choices=("resnet", "vgg"), default="resnet")
    parser.add_argument("--checkpoint", type=str, default="mycar/models/myflow_resnet18_best")
    parser.add_argument("--data", type=str, default="mycar/data")
    parser.add_argument("--catalog", type=str, default="catalog_generated.catalog")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--fixed-throttle", type=float, default=0.5)
    parser.add_argument("--force-fixed-throttle", action="store_true")
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--split", type=str, default="all", choices=("train", "val", "test", "all"))
    parser.add_argument("--target-output", type=str, default="angle", help="Target regression output: angle/throttle/0/1")
    parser.add_argument("--layer", type=str, default="last", help="Feature layer name, e.g. layer4/stage5/last")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="docs/experiments/explainability")
    args = parser.parse_args()

    data_dir = (ROOT / args.data).resolve()
    checkpoint = checkpoint_stem((ROOT / args.checkpoint).resolve())
    if not checkpoint.with_suffix(".json").is_file() or not checkpoint.with_suffix(".npz").is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}.json + {checkpoint}.npz")

    index = load_donkey_index(
        data_dir,
        fixed_throttle=args.fixed_throttle,
        angle_scale=1.0,
        catalog_name=args.catalog,
        force_fixed_throttle=bool(args.force_fixed_throttle),
    )
    split_file = (ROOT / args.split_file).resolve() if args.split_file else None
    if split_file:
        payload = load_split(split_file)
        index = select_split(index, payload["splits"], args.split)
        print(f"[split] file={split_file} split={args.split} samples={len(index)}")
    elif args.split != "all":
        raise SystemExit("--split requires --split-file")
    if args.max_samples > 0:
        index = index[: args.max_samples]
    if not index:
        raise SystemExit("no samples found")

    first_rgb = read_rgb(data_dir / index[0][0])
    h, w = first_rgb.shape[:2]
    dtype = np.float32 if args.dtype == "float32" else np.float64
    device = resolve_myflows_device(args.device)
    print_myflows_device(device, args.device)

    x_var, model, logits = build_gradcam_model(args.model_type, h, w, output_dim=2, dtype=dtype)
    ms.load_checkpoint([model], None, str(checkpoint))
    if hasattr(model, "eval"):
        model.eval()
    logits_graph = ms.Graph(logits)
    layer_name, feature_node = select_feature_node(model, args.layer)

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"gradcam_{args.model_type}_{run_stamp}"
    logdir = (ROOT / args.logdir).resolve() if args.logdir else (ROOT / "mycar" / "logs" / "tensorboard" / f"gradcam_{args.model_type}_{run_stamp}")
    out_dir = (ROOT / args.out_dir).resolve() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tb = TensorBoardLogger(logdir, enabled=True)
    tb.log_text(
        "explain/gradcam/config",
        "\n".join(
            [
                f"model_type={args.model_type}",
                f"checkpoint={checkpoint}",
                f"layer={layer_name}",
                f"samples={len(index)}",
                f"split={args.split}",
                f"split_file={split_file}",
                f"fixed_throttle={args.fixed_throttle}",
                f"force_fixed_throttle={bool(args.force_fixed_throttle)}",
            ]
        ),
        0,
    )

    report_lines = gradcam_report_header(
        args.model_type,
        checkpoint,
        layer_name,
        logdir,
        split=args.split,
        split_file=split_file,
        fixed_throttle=args.fixed_throttle,
        force_fixed_throttle=bool(args.force_fixed_throttle),
        sample_count=len(index),
        root=ROOT,
    )

    for step, (rel, angle_true, throttle_true) in enumerate(index):
        image_path = data_dir / rel
        rgb = read_rgb(image_path)
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(dtype) / 255.0
        x_var.value = np.transpose(x, (2, 0, 1))[np.newaxis, ...]
        logits_graph.forward()
        out = np.asarray(myflows_asnumpy(logits.value)).reshape(-1)
        target_index, target_label = select_target(args.model_type, out, args.target_output)
        pred_angle = float(out[0]) if out.size > 0 else float("nan")
        pred_throttle = float(out[1]) if out.size > 1 else float("nan")
        angle_error = abs(pred_angle - float(angle_true))

        result = gradcam_for_image(
            rgb,
            x_var,
            logits,
            feature_node,
            target_index=target_index,
            image_w=w,
            image_h=h,
            dtype=dtype,
        )
        stem = f"{step:03d}_{Path(rel).stem}"
        overlay_path = out_dir / f"{stem}_overlay.png"
        heatmap_path = out_dir / f"{stem}_heatmap.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(result.overlay_rgb, cv2.COLOR_RGB2BGR))
        heatmap_rgb = np.transpose(heatmap_chw(result.heatmap, h, w), (1, 2, 0))
        cv2.imwrite(str(heatmap_path), cv2.cvtColor(np.uint8(255 * heatmap_rgb), cv2.COLOR_RGB2BGR))

        tb.log_image("explain/gradcam/original", chw_float_image(resized), step)
        tb.log_image("explain/gradcam/heatmap", heatmap_chw(result.heatmap, h, w), step)
        tb.log_image("explain/gradcam/overlay", chw_float_image(result.overlay_rgb), step)
        tb.log_text(
            "explain/gradcam/info",
            "\n".join(
                [
                    f"image={rel}",
                    f"target_output={target_label}",
                    f"score={result.score:.6f}",
                    f"true_angle={angle_true:.6f}",
                    f"pred_angle={pred_angle:.6f}",
                    f"angle_abs_error={angle_error:.6f}",
                    f"true_throttle={throttle_true:.6f}",
                    f"pred_throttle={pred_throttle:.6f}",
                    f"split={args.split}",
                    f"fixed_throttle={args.fixed_throttle}",
                    f"force_fixed_throttle={bool(args.force_fixed_throttle)}",
                    f"raw_outputs={out.tolist()}",
                ]
            ),
            step,
        )
        append_gradcam_row(
            report_lines,
            step=step,
            rel_path=rel,
            target_label=target_label,
            score=result.score,
            true_angle=float(angle_true),
            pred_angle=pred_angle,
            abs_error=angle_error,
            overlay_path=overlay_path,
            root=ROOT,
        )

    tb.flush()
    tb.close()
    report_path = write_report(report_lines, out_dir)
    print(f"Grad-CAM written to TensorBoard: {logdir}")
    print(f"Images written to: {out_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
