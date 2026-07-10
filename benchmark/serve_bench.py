#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gRPC / FastAPI / 本地 ONNX 推理压测。

需先启动服务:
  python -m apps.serve.serve_grpc --model mycar/models/myflow_resnet18_best.onnx --port 50051
  python -m apps.serve.serve_fastapi --model mycar/models/myflow_resnet18_best.onnx --port 8000

示例:
  python benchmark/serve_bench.py --mode local --model mycar/models/myflow_resnet18_best.onnx
  python benchmark/serve_bench.py --mode grpc --host 127.0.0.1 --port 50051
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.serve.config import DEFAULT_FASTAPI_PORT, DEFAULT_GRPC_PORT, fastapi_base_url


def _sample_rgb(h: int = 120, w: int = 160) -> np.ndarray:
  rng = np.random.default_rng(0)
  return (rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))


def bench_local(model: Path, n: int, workers: int, device: str) -> tuple[list[float], int]:
  from apps.serve.onnx_predictor import OnnxPredictor

  pred = OnnxPredictor(model, image_w=160, image_h=120, device=device)
  rgb = _sample_rgb()

  def one():
    t0 = time.perf_counter()
    pred.predict(rgb)
    return (time.perf_counter() - t0) * 1000.0

  return _run_parallel(one, n, workers)


def bench_grpc(host: str, port: int, n: int, workers: int) -> tuple[list[float], int]:
  from apps.serve.grpc_client import GrpcInferenceClient

  rgb = _sample_rgb()
  client = GrpcInferenceClient(host, port)

  def one():
    t0 = time.perf_counter()
    client.predict_array(rgb)
    return (time.perf_counter() - t0) * 1000.0
  try:
    return _run_parallel(one, n, workers)
  finally:
    client.close()


def bench_fastapi(host: str, port: int, n: int, workers: int) -> tuple[list[float], int]:
  from apps.serve.fastapi_client import FastApiInferenceClient

  rgb = _sample_rgb()
  _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
  client = FastApiInferenceClient(fastapi_base_url(host, port))
  image_bytes = buf.tobytes()

  def one():
    t0 = time.perf_counter()
    client.predict_bytes(image_bytes, filename="img.jpg", content_type="image/jpeg")
    return (time.perf_counter() - t0) * 1000.0

  return _run_parallel(one, n, workers)


def _run_parallel(fn, n: int, workers: int) -> tuple[list[float], int]:
  latencies: list[float] = []
  errors = 0
  with ThreadPoolExecutor(max_workers=workers) as ex:
    futs = [ex.submit(fn) for _ in range(n)]
    for fut in as_completed(futs):
      try:
        latencies.append(fut.result())
      except Exception:
        errors += 1
  return latencies, errors


def _summarize(latencies: list[float], errors: int, requested: int) -> dict:
  latencies.sort()
  total_s = sum(latencies) / 1000.0
  def pct(q: float) -> float:
    if not latencies:
      return 0.0
    idx = min(len(latencies) - 1, max(0, int(q * (len(latencies) - 1))))
    return float(latencies[idx])
  return {
      "requests": int(requested),
      "success_count": len(latencies),
      "error_count": int(errors),
      "success_rate": len(latencies) / max(int(requested), 1),
      "qps": len(latencies) / total_s if total_s > 0 else 0.0,
      "p50_ms": statistics.median(latencies) if latencies else 0.0,
      "p95_ms": pct(0.95),
      "p99_ms": pct(0.99),
      "mean_ms": statistics.mean(latencies) if latencies else 0.0,
  }


def build_bench_stats(
    *,
    mode: str,
    workers: int,
    model: str,
    device: str,
    latencies: list[float],
    errors: int,
    requested: int,
) -> dict:
  stats = {"mode": mode, "workers": workers, "model": model, **_summarize(latencies, errors, requested)}
  if mode == "local":
    stats["device"] = device
  else:
    stats["client_device"] = "n/a"
    stats["server_device"] = "controlled_by_service"
  return stats


def _write_report(stats: dict, out_json: str | None, out_md: str | None) -> None:
  if out_json:
    path = (ROOT / out_json).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written: {path}")
  if out_md:
    path = (ROOT / out_md).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 服务压测报告",
        "",
        "| 指标 | 值 |",
        "|------|----|",
    ]
    for key, value in stats.items():
      lines.append(f"| `{key}` | `{value}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {path}")


def _write_tensorboard(stats: dict, logdir: str | None) -> None:
  if not logdir:
    return
  from MyFlows.utils.tensorboard_logger import TensorBoardLogger

  tb = TensorBoardLogger((ROOT / logdir).resolve(), enabled=True)
  for key in ("qps", "p50_ms", "p95_ms", "p99_ms", "mean_ms", "success_rate", "error_count"):
    tb.log_scalar(f"serve/{key}", float(stats.get(key, 0.0)), 0)
  tb.log_text("serve/summary", "```json\n" + json.dumps(stats, indent=2, ensure_ascii=False) + "\n```", 0)
  tb.close()


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=("local", "grpc", "fastapi"), default="local")
  ap.add_argument("--model", type=str, default="mycar/models/myflow_resnet18_best.onnx")
  ap.add_argument("--host", type=str, default="127.0.0.1")
  ap.add_argument("--port", type=int, default=None)
  ap.add_argument("--requests", type=int, default=100)
  ap.add_argument("--workers", type=int, default=4)
  ap.add_argument("--device", type=str, default="cpu")
  ap.add_argument("--out-json", type=str, default=None)
  ap.add_argument("--out-md", type=str, default=None)
  ap.add_argument("--tensorboard-logdir", type=str, default=None)
  args = ap.parse_args()
  port = args.port
  if port is None:
    port = DEFAULT_FASTAPI_PORT if args.mode == "fastapi" else DEFAULT_GRPC_PORT

  if args.mode == "local":
    lats, errors = bench_local((ROOT / args.model).resolve(), args.requests, args.workers, args.device)
  elif args.mode == "grpc":
    lats, errors = bench_grpc(args.host, port, args.requests, args.workers)
  else:
    lats, errors = bench_fastapi(args.host, port, args.requests, args.workers)

  stats = build_bench_stats(
      mode=args.mode,
      workers=args.workers,
      model=args.model,
      device=args.device,
      latencies=lats,
      errors=errors,
      requested=args.requests,
  )
  print(stats)
  _write_report(stats, args.out_json, args.out_md)
  _write_tensorboard(stats, args.tensorboard_logdir)


if __name__ == "__main__":
  main()
