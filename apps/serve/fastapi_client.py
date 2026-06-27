#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI HTTP 推理客户端 SDK（需先启动 apps.serve.serve_fastapi）。

  python -m apps.serve.serve_fastapi --model mycar/models/myflow_resnet18_best.onnx --port 8000
  python -m apps.serve.fastapi_client --image mycar/data/images/0_0.0000.jpg --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import uuid
from pathlib import Path
from urllib import error, request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from apps.serve.config import DEFAULT_FASTAPI_PORT, DEFAULT_TIMEOUT_S, fastapi_base_url
from apps.serve.schema import PredictionResult


class FastApiInferenceClient:
  def __init__(self, base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT_S):
    self.base_url = (base_url or fastapi_base_url()).rstrip("/")
    self.timeout = float(timeout)

  def health(self) -> dict:
    return self._get_json("/healthz")

  def model_info(self) -> dict:
    return self._get_json("/model_info")

  def metrics(self) -> dict:
    return self._get_json("/metrics")

  def predict_image(self, image_path: str | Path) -> PredictionResult:
    path = Path(image_path)
    if not path.exists():
      raise FileNotFoundError(path)
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return self.predict_bytes(path.read_bytes(), filename=path.name, content_type=content_type)

  def predict_bytes(
      self,
      image_bytes: bytes,
      filename: str = "image.jpg",
      content_type: str = "image/jpeg",
  ) -> PredictionResult:
    if not image_bytes:
      raise ValueError("image_bytes is empty")
    boundary = f"----myflows-fastapi-client-{uuid.uuid4().hex}"
    body = b"\r\n".join([
        f"--{boundary}".encode("ascii"),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode("utf-8"),
        f"Content-Type: {content_type}".encode("ascii"),
        b"",
        image_bytes,
        f"--{boundary}--".encode("ascii"),
        b"",
    ])
    req = request.Request(
        f"{self.base_url}/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return PredictionResult.from_response(self._open_json(req))

  def predict_json(self, image_b64: str, width: int = 160, height: int = 120) -> PredictionResult:
    payload = json.dumps({"image_b64": image_b64, "width": int(width), "height": int(height)}).encode("utf-8")
    req = request.Request(
        f"{self.base_url}/predict_json",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return PredictionResult.from_response(self._open_json(req))

  def _get_json(self, path: str) -> dict:
    return self._open_json(request.Request(f"{self.base_url}{path}", method="GET"))

  def _open_json(self, req: request.Request) -> dict:
    try:
      with request.urlopen(req, timeout=self.timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return json.loads(resp.read().decode(charset))
    except error.HTTPError as exc:
      detail = exc.read().decode("utf-8", errors="replace")
      raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
      raise RuntimeError(f"FastAPI request failed: {exc}") from exc


def main() -> None:
  ap = argparse.ArgumentParser(description="FastAPI inference client SDK CLI")
  ap.add_argument("--image", type=str, required=True)
  ap.add_argument("--url", type=str, default=None, help="服务根地址，如 http://127.0.0.1:8000")
  ap.add_argument("--host", type=str, default="127.0.0.1")
  ap.add_argument("--port", type=int, default=DEFAULT_FASTAPI_PORT)
  ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
  ap.add_argument("--show-model-info", action="store_true")
  args = ap.parse_args()

  base_url = args.url or fastapi_base_url(args.host, args.port)
  client = FastApiInferenceClient(base_url, timeout=args.timeout)
  if args.show_model_info:
    print(json.dumps(client.model_info(), ensure_ascii=False, indent=2))
  result = client.predict_image(args.image)
  print(result.to_json())


if __name__ == "__main__":
  main()
