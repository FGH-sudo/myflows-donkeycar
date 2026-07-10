#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare FP32, dynamic INT8, and static INT8 ONNX inference on DonkeyCar data.

The script is intentionally a thin experiment runner:
- dataset/split handling stays in ``apps.common``;
- quantization stays in ``MyFlows.utils.quantize``;
- this file only orchestrates model variants, inference metrics, and reports.

Example:
  python scripts/run_quantize_eval.py \
      --fp32 mycar/models/myflow_resnet18_best.onnx \
      --data mycar/data \
      --split-file mycar/logs/resnet18_split.json --split test \
      --fixed-throttle 0.2 --force-fixed-throttle \
      --max-samples 0 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.common.donkey_data import load_donkey_index
from apps.common.image_preprocess import imread_nchw
from apps.common.splits import load_split, select_split
from tools.device_runtime import create_ort_inference_session, print_ort_device
from MyFlows.utils.metrics import angle_sign_accuracy
from MyFlows.utils.quantize import quantize_onnx_dynamic, quantize_onnx_static

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from calibration_reader import DonkeyCalibrationDataReader


def _resolve_project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_rows(
    data_dir: Path,
    max_samples: int,
    catalog: str,
    *,
    split_file: str | Path | None = None,
    split: str = "all",
    fixed_throttle: float = 0.5,
    angle_scale: float = 1.0,
    force_fixed_throttle: bool = False,
) -> list[tuple[Path, float, float]]:
    index = load_donkey_index(
        data_dir,
        fixed_throttle=float(fixed_throttle),
        angle_scale=float(angle_scale),
        catalog_name=catalog,
        force_fixed_throttle=bool(force_fixed_throttle),
    )
    if split_file:
        payload = load_split(split_file)
        index = select_split(index, payload["splits"], split)
        print(f"[split] file={Path(split_file).resolve()} split={split} samples={len(index)}")
    if max_samples > 0:
        index = index[:max_samples]
    return [(data_dir / rel, float(angle), float(throttle)) for rel, angle, throttle in index]


def _imread_chw(path: Path, w: int, h: int) -> np.ndarray:
    return imread_nchw(path, (w, h), dtype=np.float32)


def _eval_regression(session, rows, w: int, h: int, input_name: str) -> dict:
    preds: list[list[float]] = []
    trues: list[list[float]] = []
    latencies: list[float] = []
    for path, angle, throttle in rows:
        x = _imread_chw(path, w, h)
        t0 = time.perf_counter()
        out = session.run(None, {input_name: x})[0].reshape(-1)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if out.size < 2:
            raise ValueError(f"expected regression output [angle, throttle], got shape={out.shape}")
        preds.append([float(out[0]), float(out[1])])
        trues.append([float(angle), float(throttle)])

    pred_arr = np.asarray(preds, dtype=np.float64)
    true_arr = np.asarray(trues, dtype=np.float64)
    angle_mse = float(np.mean((pred_arr[:, 0] - true_arr[:, 0]) ** 2))
    throttle_mse = float(np.mean((pred_arr[:, 1] - true_arr[:, 1]) ** 2))
    overall_mse = float(np.mean((pred_arr - true_arr) ** 2))
    sign = angle_sign_accuracy(true_arr[:, 0], pred_arr[:, 0])
    return {
        "angle_mse": angle_mse,
        "throttle_mse": throttle_mse,
        "overall_mse": overall_mse,
        "angle_sign_acc": float(sign["angle_sign_accuracy"]),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p99": float(np.percentile(latencies, 99)),
        "n": len(rows),
    }


def _model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.is_file() else 0.0


def _display_repo_path(path: Path | None) -> str:
    if path is None:
        return ""
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _write_markdown(
    out_md: Path,
    *,
    data_dir: Path,
    split: str,
    split_file: Path | None,
    fp32_path: Path,
    rows_count: int,
    results: list[dict],
    out_json: Path,
    out_png: Path | None,
) -> None:
    split_text = f"`{split}` via `{_display_repo_path(split_file)}`" if split_file else "`all`"
    lines = [
        "# FP32 vs INT8 Inference Report",
        "",
        f"- Data: `{_display_repo_path(data_dir)}`",
        f"- Split: {split_text}",
        f"- Samples: {rows_count}",
        f"- FP32 model: `{_display_repo_path(fp32_path)}`",
        "",
        "| Variant | angle MSE | throttle MSE | overall MSE | angle sign acc | mean latency (ms) | P99 (ms) | size (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['variant']} | {r['angle_mse']:.6f} | {r['throttle_mse']:.6f} | "
            f"{r['overall_mse']:.6f} | {r['angle_sign_acc']:.4f} | "
            f"{r['latency_ms_mean']:.2f} | {r['latency_ms_p99']:.2f} | {r['size_mb']:.3f} |"
        )
    lines.extend(["", f"- Raw JSON: `{_display_repo_path(out_json)}`"])
    if out_png:
        lines.append(f"- Plot: `{_display_repo_path(out_png)}`")
    lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _write_plot(out_png: Path, results: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["variant"] for r in results]
    x = np.arange(len(labels), dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    width = 0.25
    axes[0].bar(x - width, [r["angle_mse"] for r in results], width, label="angle")
    axes[0].bar(x, [r["throttle_mse"] for r in results], width, label="throttle")
    axes[0].bar(x + width, [r["overall_mse"] for r in results], width, label="overall")
    axes[0].set_title("Regression Error")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    axes[1].bar(x - width / 2, [r["latency_ms_mean"] for r in results], width, label="mean")
    axes[1].bar(x + width / 2, [r["latency_ms_p99"] for r in results], width, label="p99")
    axes[1].set_title("Single-frame Latency")
    axes[1].set_ylabel("ms")
    axes[1].legend()

    axes[2].bar(x, [r["size_mb"] for r in results], width * 1.8, color="#4C78A8")
    axes[2].set_title("Model Size")
    axes[2].set_ylabel("MB")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("DonkeyCar ResNet-18 ONNX Quantization")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", type=str, required=True, help="FP32 ONNX path")
    ap.add_argument("--data", type=str, default="mycar/data")
    ap.add_argument("--catalog", type=str, default="catalog_generated.catalog")
    ap.add_argument("--max-samples", type=int, default=500, help="0 means all selected samples")
    ap.add_argument("--image-w", type=int, default=160)
    ap.add_argument("--image-h", type=int, default=120)
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--split-file", type=str, default=None)
    ap.add_argument("--split", type=str, default="all", choices=("train", "val", "test", "all"))
    ap.add_argument("--fixed-throttle", type=float, default=0.5)
    ap.add_argument("--force-fixed-throttle", action="store_true")
    ap.add_argument("--angle-scale", type=float, default=1.0)
    ap.add_argument("--calibration-samples", type=int, default=200)
    ap.add_argument("--out-json", type=str, default="docs/experiments/int8_metrics.json")
    ap.add_argument("--out-md", type=str, default="docs/experiments/int8_report.md")
    ap.add_argument("--out-png", type=str, default="docs/experiments/int8_report.png")
    ap.add_argument("--skip-static", action="store_true")
    args = ap.parse_args()

    fp32_path = _resolve_project_path(args.fp32).resolve()
    data_dir = _resolve_project_path(args.data).resolve()
    split_file = _resolve_project_path(args.split_file)
    rows = _load_rows(
        data_dir,
        args.max_samples,
        args.catalog,
        split_file=split_file,
        split=args.split,
        fixed_throttle=args.fixed_throttle,
        angle_scale=args.angle_scale,
        force_fixed_throttle=args.force_fixed_throttle,
    )
    if not rows:
        raise SystemExit("no evaluation samples found")

    variants: list[tuple[str, Path]] = [("fp32", fp32_path)]
    int8_dyn = fp32_path.with_name(fp32_path.stem + "_int8.onnx")
    quantize_onnx_dynamic(fp32_path, int8_dyn)
    variants.append(("int8_dynamic", int8_dyn))

    if not args.skip_static:
        int8_static = fp32_path.with_name(fp32_path.stem + "_int8_static.onnx")
        reader = DonkeyCalibrationDataReader(
            data_dir,
            image_w=args.image_w,
            image_h=args.image_h,
            max_samples=args.calibration_samples,
        )
        reader.input_name = create_ort_inference_session(fp32_path, "cpu")[0].get_inputs()[0].name
        quantize_onnx_static(fp32_path, int8_static, reader)
        variants.append(("int8_static", int8_static))

    results = []
    for label, path in variants:
        session, providers = create_ort_inference_session(path, args.device)
        print_ort_device(providers, args.device)
        in_name = session.get_inputs()[0].name
        metrics = _eval_regression(session, rows, args.image_w, args.image_h, in_name)
        metrics["variant"] = label
        metrics["path"] = str(path)
        metrics["size_mb"] = _model_size_mb(path)
        metrics["providers"] = providers
        results.append(metrics)
        print(metrics)

    out_json = _resolve_project_path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    out_md = _resolve_project_path(args.out_md).resolve()
    out_png = _resolve_project_path(args.out_png).resolve() if args.out_png else None
    if out_png:
        _write_plot(out_png, results)
    _write_markdown(
        out_md,
        data_dir=data_dir,
        split=args.split,
        split_file=split_file.resolve() if split_file else None,
        fp32_path=fp32_path,
        rows_count=len(rows),
        results=results,
        out_json=out_json,
        out_png=out_png,
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    if out_png:
        print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
