#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI HTTP 推理服务（PPT 要求的 flask/FastAPI 部署，与 gRPC 并存）。

启动:
  python -m apps.serve.serve_fastapi --host 0.0.0.0 --port 8000 \\
      --model mycar/models/myflow_resnet18_best.onnx --device auto

或:
  python -m apps.serve.serve_fastapi --model mycar/models/myflow_resnet18_best.onnx --port 8000
"""
from __future__ import annotations

import argparse
import base64
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from apps.serve.config import (
    DEFAULT_BIND_HOST,
    DEFAULT_DEVICE,
    DEFAULT_FASTAPI_LOG_FILE,
    DEFAULT_FASTAPI_PORT,
    DEFAULT_IMAGE_H,
    DEFAULT_IMAGE_W,
    DEFAULT_MAX_UPLOAD_MB,
    DEFAULT_MODEL_TYPE,
    resolve_repo_path,
)
from apps.serve.logger import JsonlLogger
from apps.serve.metrics import ServeMetrics
from apps.serve.onnx_predictor import OnnxPredictor

_predictor: Optional[OnnxPredictor] = None
_metrics = ServeMetrics()


def _get_predictor() -> OnnxPredictor:
  if _predictor is None:
    raise RuntimeError("Predictor 未初始化，请通过 CLI 或环境变量启动服务")
  return _predictor


def create_app(
    model_path: Path,
    image_w: int = DEFAULT_IMAGE_W,
    image_h: int = DEFAULT_IMAGE_H,
    device: str = DEFAULT_DEVICE,
    model_type: str = DEFAULT_MODEL_TYPE,
    log_file: Path | None = None,
    max_upload_mb: float = DEFAULT_MAX_UPLOAD_MB,
):
  global _predictor
  _predictor = OnnxPredictor(model_path, image_w, image_h, device=device, model_type=model_type)

  app = FastAPI(title="MyFlows ONNX Inference", version="1.0")
  logger = JsonlLogger(log_file or resolve_repo_path(DEFAULT_FASTAPI_LOG_FILE))
  max_upload_bytes = int(float(max_upload_mb) * 1024 * 1024)

  @app.middleware("http")
  async def request_logger(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    status_code = 500
    error = None
    try:
      response = await call_next(request)
      status_code = int(response.status_code)
      response.headers["X-Request-ID"] = request_id
      return response
    except Exception as exc:
      error = f"{type(exc).__name__}: {exc}"
      raise
    finally:
      latency_ms = (time.perf_counter() - start) * 1000.0
      _metrics.record(latency_ms, status_code=status_code)
      logger.write({
          "request_id": request_id,
          "method": request.method,
          "path": request.url.path,
          "client": request.client.host if request.client else None,
          "status_code": status_code,
          "latency_ms": round(latency_ms, 3),
          "error": error,
      })

  class PredictJsonBody(BaseModel):
    image_b64: str = Field(..., description="JPEG/PNG 的 base64")
    width: int = 160
    height: int = 120

  @app.get("/healthz")
  def healthz():
    return {"status": "ok"}

  @app.get("/model_info")
  def model_info():
    p = _get_predictor()
    return p.info()

  @app.get("/metrics")
  def metrics():
    return _metrics.snapshot(_get_predictor().info())

  def _predict_rgb(rgb: np.ndarray, request_id: str | None = None) -> dict:
    t0 = time.perf_counter()
    predictor = _get_predictor()
    out = predictor.predict(rgb)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    result = predictor.format_prediction(out)
    result.update({
        "request_id": request_id,
        "status": "ok",
        "latency_ms": round(latency_ms, 3),
        "input_shape": list(rgb.shape),
        "device": predictor.device,
    })
    return result

  @app.post("/predict")
  async def predict_file(request: Request, file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "application/octet-stream"):
      raise HTTPException(400, f"unsupported content_type: {file.content_type}")
    data = await file.read()
    if not data:
      raise HTTPException(400, "空图像")
    if len(data) > max_upload_bytes:
      raise HTTPException(413, f"图像过大，限制 {max_upload_mb} MB")
    arr = np.frombuffer(data, dtype=np.uint8)
    rgb = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if rgb is None:
      raise HTTPException(400, "无法解码图像")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return _predict_rgb(rgb, request.state.request_id)

  @app.post("/predict_json")
  def predict_json(request: Request, body: PredictJsonBody):
    if not body.image_b64:
      raise HTTPException(400, "image_b64 为空")
    if body.width <= 0 or body.height <= 0 or body.width * body.height > 4096 * 4096:
      raise HTTPException(400, "width/height 非法")
    try:
      raw = base64.b64decode(body.image_b64)
    except Exception as exc:
      raise HTTPException(400, f"base64 解码失败: {exc}") from exc
    arr = np.frombuffer(raw, dtype=np.uint8)
    rgb = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if rgb is None:
      raise HTTPException(400, "无法解码图像")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != body.height or rgb.shape[1] != body.width:
      rgb = cv2.resize(rgb, (body.width, body.height))
    return _predict_rgb(rgb, request.state.request_id)

  return app


# 默认 app：需通过环境变量或 uvicorn 工厂模式；CLI 启动时设置
app = None


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", type=str, required=True)
  ap.add_argument("--image-w", type=int, default=DEFAULT_IMAGE_W)
  ap.add_argument("--image-h", type=int, default=DEFAULT_IMAGE_H)
  ap.add_argument("--port", type=int, default=DEFAULT_FASTAPI_PORT)
  ap.add_argument("--host", type=str, default=DEFAULT_BIND_HOST)
  ap.add_argument("--device", type=str, default=DEFAULT_DEVICE, choices=("auto", "cpu", "cuda"))
  ap.add_argument("--model-type", type=str, default=DEFAULT_MODEL_TYPE)
  ap.add_argument("--log-file", type=str, default=DEFAULT_FASTAPI_LOG_FILE)
  ap.add_argument("--max-upload-mb", type=float, default=DEFAULT_MAX_UPLOAD_MB)
  args = ap.parse_args()

  model_path = resolve_repo_path(args.model)
  global app
  app = create_app(
      model_path,
      image_w=args.image_w,
      image_h=args.image_h,
      device=args.device,
      model_type=args.model_type,
      log_file=resolve_repo_path(args.log_file),
      max_upload_mb=args.max_upload_mb,
  )

  import uvicorn

  uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
  main()
