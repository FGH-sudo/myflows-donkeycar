#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FP32 / INT8 动态 / INT8 静态 ONNX 对比：精度 + 单帧延迟 + 模型体积。

示例（项目根目录）:
  python scripts/run_quantize_eval.py \\
      --fp32 mycar/models/myflow_resnet18_best.onnx \\
      --data mycar/data --max-samples 500 --device auto
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
from tools.device_runtime import create_ort_inference_session, print_ort_device
from MyFlows.utils.metrics import angle_sign_accuracy, mse
from MyFlows.utils.quantize import quantize_onnx_dynamic, quantize_onnx_static

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from calibration_reader import DonkeyCalibrationDataReader


def _load_rows(data_dir: Path, max_samples: int, catalog: str):
  index = load_donkey_index(data_dir, fixed_throttle=0.5, angle_scale=1.0, catalog_name=catalog)
  rows = [(data_dir / rel, angle) for rel, angle, _ in index]
  if max_samples > 0:
    rows = rows[:max_samples]
  return rows


def _imread_chw(path: Path, w: int, h: int) -> np.ndarray:
  return imread_nchw(path, (w, h), dtype=np.float32)


def _eval_regression(session, rows, w: int, h: int, input_name: str) -> dict:
  preds, trues = [], []
  latencies = []
  for path, angle in rows:
    x = _imread_chw(path, w, h)
    t0 = time.perf_counter()
    out = session.run(None, {input_name: x})[0].reshape(-1)
    latencies.append((time.perf_counter() - t0) * 1000.0)
    preds.append(float(out[0]))
    trues.append(float(angle))
  preds = np.asarray(preds)
  trues = np.asarray(trues)
  return {
      "mse": float(mse(preds, trues)),
      "angle_sign_acc": float(angle_sign_accuracy(trues, preds)["angle_sign_accuracy"]),
      "latency_ms_mean": float(np.mean(latencies)),
      "latency_ms_p50": float(np.percentile(latencies, 50)),
      "latency_ms_p99": float(np.percentile(latencies, 99)),
      "n": len(rows),
  }


def _model_size_mb(path: Path) -> float:
  return path.stat().st_size / (1024 * 1024) if path.is_file() else 0.0


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--fp32", type=str, required=True, help="FP32 ONNX 路径")
  ap.add_argument("--data", type=str, default="mycar/data")
  ap.add_argument("--catalog", type=str, default="catalog_generated.catalog")
  ap.add_argument("--max-samples", type=int, default=500)
  ap.add_argument("--image-w", type=int, default=160)
  ap.add_argument("--image-h", type=int, default=120)
  ap.add_argument("--device", type=str, default="auto")
  ap.add_argument("--out-json", type=str, default="docs/experiments/int8_metrics.json")
  ap.add_argument("--out-md", type=str, default="docs/experiments/int8_report.md")
  ap.add_argument("--skip-static", action="store_true")
  args = ap.parse_args()

  fp32_path = (ROOT / args.fp32).resolve()
  data_dir = (ROOT / args.data).resolve()
  rows = _load_rows(data_dir, args.max_samples, args.catalog)
  if not rows:
    raise SystemExit("无评估样本")

  variants: list[tuple[str, Path]] = [("fp32", fp32_path)]
  int8_dyn = fp32_path.with_name(fp32_path.stem + "_int8.onnx")
  quantize_onnx_dynamic(fp32_path, int8_dyn)
  variants.append(("int8_dynamic", int8_dyn))

  if not args.skip_static:
    int8_static = fp32_path.with_name(fp32_path.stem + "_int8_static.onnx")
    reader = DonkeyCalibrationDataReader(
        data_dir, image_w=args.image_w, image_h=args.image_h, max_samples=200
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
    metrics["size_mb"] = round(_model_size_mb(path), 3)
    results.append(metrics)
    print(metrics)

  out_json = (ROOT / args.out_json).resolve()
  out_json.parent.mkdir(parents=True, exist_ok=True)
  out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

  lines = [
      "# FP32 vs INT8 推理对比报告",
      "",
      f"- 数据: `{data_dir}`，样本数: {len(rows)}",
      f"- FP32 模型: `{fp32_path}`",
      "",
      "| 变体 | MSE | 符号准确率 | 平均延迟(ms) | P99(ms) | 体积(MB) |",
      "|------|-----|------------|--------------|---------|----------|",
  ]
  for r in results:
    lines.append(
        f"| {r['variant']} | {r['mse']:.6f} | {r['angle_sign_acc']:.4f} | "
        f"{r['latency_ms_mean']:.2f} | {r['latency_ms_p99']:.2f} | {r['size_mb']} |"
    )
  lines.extend(["", f"原始 JSON: `{out_json}`", ""])
  out_md = (ROOT / args.out_md).resolve()
  out_md.parent.mkdir(parents=True, exist_ok=True)
  out_md.write_text("\n".join(lines), encoding="utf-8")
  print(f"已写入: {out_json}\n{out_md}")


if __name__ == "__main__":
  main()
