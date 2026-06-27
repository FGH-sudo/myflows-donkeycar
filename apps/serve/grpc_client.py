#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gRPC 推理客户端 SDK（需先启动 apps.serve.serve_grpc）。

  python -m apps.serve.serve_grpc --model mycar/models/myflow_resnet18_best.onnx --device auto
  python -m apps.serve.grpc_client --image mycar/data/images/0_0.0000.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import grpc
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from apps.serve.config import DEFAULT_CLIENT_HOST, DEFAULT_GRPC_PORT, DEFAULT_TIMEOUT_S
from apps.serve.schema import PredictionResult
from generated.grpc import infer_pb2, infer_pb2_grpc


class GrpcInferenceClient:
    def __init__(
        self,
        host: str = DEFAULT_CLIENT_HOST,
        port: int = DEFAULT_GRPC_PORT,
        timeout: float = DEFAULT_TIMEOUT_S,
    ):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self.channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        self.stub = infer_pb2_grpc.InferServiceStub(self.channel)

    def close(self) -> None:
        self.channel.close()

    def predict_image(self, image_path: str | Path) -> PredictionResult:
        rgb = self._read_rgb(image_path)
        return self.predict_array(rgb)

    def predict_array(self, rgb: np.ndarray) -> PredictionResult:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expect RGB image with shape HxWx3, got {rgb.shape}")
        h, w, c = rgb.shape
        req = infer_pb2.PredictRequest(
            image_rgb=rgb.astype(np.uint8, copy=False).tobytes(),
            width=w,
            height=h,
            channels=c,
        )
        start = time.perf_counter()
        resp = self.stub.Predict(req, timeout=self.timeout)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return PredictionResult.from_outputs(
            list(resp.outputs),
            model_type=resp.model_type or "regression",
            latency_ms=round(latency_ms, 3),
            input_shape=[int(h), int(w), int(c)],
        )

    @staticmethod
    def _read_rgb(image_path: str | Path) -> np.ndarray:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> None:
    ap = argparse.ArgumentParser(description="gRPC inference client SDK CLI")
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--host", type=str, default=DEFAULT_CLIENT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_GRPC_PORT)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = ap.parse_args()

    client = GrpcInferenceClient(args.host, args.port, timeout=args.timeout)
    try:
        result = client.predict_image(args.image)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
