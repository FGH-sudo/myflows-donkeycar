# -*- coding: utf-8 -*-
"""Serving 模块共享配置。"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BIND_HOST = "0.0.0.0"
DEFAULT_CLIENT_HOST = "127.0.0.1"
DEFAULT_FASTAPI_PORT = 8000
DEFAULT_GRPC_PORT = 50051
DEFAULT_IMAGE_W = 160
DEFAULT_IMAGE_H = 120
DEFAULT_DEVICE = "auto"
DEFAULT_MODEL_TYPE = "regression"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_UPLOAD_MB = 5.0
DEFAULT_MAX_PIXELS = 4096 * 4096
DEFAULT_FASTAPI_LOG_FILE = "mycar/logs/serve_fastapi.log"
DEFAULT_GRPC_LOG_FILE = "mycar/logs/serve_grpc.log"


def resolve_repo_path(path: str | Path | None) -> Path | None:
  if path is None:
    return None
  value = Path(path)
  if value.is_absolute():
    return value
  return (ROOT / value).resolve()


def fastapi_base_url(host: str = DEFAULT_CLIENT_HOST, port: int = DEFAULT_FASTAPI_PORT) -> str:
  return f"http://{host}:{int(port)}"
