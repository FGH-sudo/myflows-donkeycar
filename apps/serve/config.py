# -*- coding: utf-8 -*-
"""Serving 模块共享配置。"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

# 推理服务没有鉴权：默认只监听本机，需要对其他机器开放时显式传 --host（容器入口传 0.0.0.0）。
DEFAULT_BIND_HOST = "127.0.0.1"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
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


def is_loopback(host: str) -> bool:
  return str(host).strip("[]").lower() in LOOPBACK_HOSTS


def grpc_bind_address(host: str, port: int) -> str:
  """gRPC 监听地址：0.0.0.0 用双栈的 [::]，IPv6 字面量加方括号。"""
  host = str(host).strip("[]")
  if host == "0.0.0.0":
    host = "::"
  return f"[{host}]:{int(port)}" if ":" in host else f"{host}:{int(port)}"


def exposure_warning(host: str) -> str | None:
  if is_loopback(host):
    return None
  return f"[warning] 推理服务没有鉴权，监听 {host} 会让网络中的其他机器也能调用它。"


def fastapi_base_url(host: str = DEFAULT_CLIENT_HOST, port: int = DEFAULT_FASTAPI_PORT) -> str:
  return f"http://{host}:{int(port)}"
