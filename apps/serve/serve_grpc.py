#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gRPC 推理服务：加载 ONNX 对单张 RGB 图预测。

生成 stub（首次）:
  python -m grpc_tools.protoc -I proto --python_out=generated/grpc --grpc_python_out=generated/grpc proto/infer.proto

启动（GPU 推理需 onnxruntime-gpu，默认 --device auto）:
  python -m apps.serve.serve_grpc --model mycar/models/myflow_resnet18_best.onnx --port 50051 --device auto
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from apps.serve.config import (
    DEFAULT_BIND_HOST,
    DEFAULT_DEVICE,
    DEFAULT_GRPC_LOG_FILE,
    DEFAULT_GRPC_PORT,
    DEFAULT_IMAGE_H,
    DEFAULT_IMAGE_W,
    DEFAULT_MAX_PIXELS,
    exposure_warning,
    grpc_bind_address,
    resolve_repo_path,
)
from apps.serve.logger import JsonlLogger
from apps.serve.metrics import ServeMetrics
from apps.serve.onnx_predictor import OnnxPredictor

# 生成代码位于项目根 infer_pb2*
try:
    from generated.grpc import infer_pb2, infer_pb2_grpc
except ImportError:
    raise SystemExit(
        "请先运行: python -m grpc_tools.protoc -I proto "
        "--python_out=generated/grpc --grpc_python_out=generated/grpc proto/infer.proto"
    )


class InferServicer(infer_pb2_grpc.InferServiceServicer):
    def __init__(
        self,
        predictor: OnnxPredictor,
        log_file: Path,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        metrics: ServeMetrics | None = None,
    ):
        self.predictor = predictor
        self.logger = JsonlLogger(log_file)
        self.max_pixels = int(max_pixels)
        self.metrics = metrics or ServeMetrics()

    def Predict(self, request, context):
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        status = "ok"
        error = None
        w, h, c = request.width, request.height, request.channels or 3
        try:
            if w <= 0 or h <= 0:
                raise ValueError("width/height must be positive")
            if c != 3:
                raise ValueError("only RGB channels=3 is supported")
            if w * h > self.max_pixels:
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details("image too large")
                status = "error"
                error = "image too large"
                return infer_pb2.PredictResponse()
            expected = w * h * c
            if not request.image_rgb:
                raise ValueError("empty image payload")
            if len(request.image_rgb) != expected:
                raise ValueError(f"image size mismatch: got {len(request.image_rgb)}, expect {expected}")
            arr = np.frombuffer(request.image_rgb, dtype=np.uint8).reshape(h, w, c)
            out = self.predictor.predict(arr)
            return infer_pb2.PredictResponse(
                outputs=out.astype(np.float32).tolist(),
                model_type=self.predictor.model_type,
            )
        except ValueError as exc:
            status = "error"
            error = str(exc)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(error)
            return infer_pb2.PredictResponse()
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            self.metrics.record(latency_ms, ok=status == "ok")
            self.logger.write({
                "request_id": request_id,
                "status": status,
                "error": error,
                "width": w,
                "height": h,
                "channels": c,
                "payload_bytes": len(request.image_rgb),
                "latency_ms": round(latency_ms, 3),
                "model_type": self.predictor.model_type,
            })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--image-w", type=int, default=DEFAULT_IMAGE_W, help="ONNX 输入宽度为动态尺寸时使用")
    ap.add_argument("--image-h", type=int, default=DEFAULT_IMAGE_H, help="ONNX 输入高度为动态尺寸时使用")
    ap.add_argument("--port", type=int, default=DEFAULT_GRPC_PORT)
    ap.add_argument("--host", type=str, default=DEFAULT_BIND_HOST,
                    help="监听地址，默认只监听本机；对其他机器开放时传 0.0.0.0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-file", type=str, default=DEFAULT_GRPC_LOG_FILE)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=("auto", "cpu", "cuda"),
        help="ONNX Runtime 设备：auto/cuda 优先 GPU",
    )
    args = ap.parse_args()

    model_path = resolve_repo_path(args.model)
    predictor = OnnxPredictor(model_path, args.image_w, args.image_h, device=args.device)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.workers))
    infer_pb2_grpc.add_InferServiceServicer_to_server(
        InferServicer(predictor, resolve_repo_path(args.log_file), max_pixels=args.max_pixels),
        server,
    )
    addr = grpc_bind_address(args.host, args.port)
    warning = exposure_warning(args.host)
    if warning:
        print(warning, file=sys.stderr)
    if server.add_insecure_port(addr) == 0:
        raise SystemExit(f"无法监听 {addr}")
    server.start()
    print(f"gRPC InferService listening on {addr}, model={model_path}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
