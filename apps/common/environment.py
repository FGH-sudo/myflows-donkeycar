"""Runtime provenance without importing PyTorch or compiling the native library."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


def environment(*, include_cuda: bool = True) -> dict:
    packages, missing = {}, {}
    for name in ("numpy", "cupy-cuda12x", "torch", "Pillow", "tensorboard", "psutil", "optuna",
                 "fastapi", "onnx", "onnxruntime-gpu", "grpcio", "nvidia-ml-py"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
            missing[f"packages.{name}"] = "not installed"
    gpu = None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=15, check=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        gpu = result.stdout.strip() if result.returncode == 0 else None
        if not gpu:
            missing["gpu"] = "nvidia-smi did not return device information"
    except (OSError, subprocess.TimeoutExpired) as exc:
        missing["gpu"] = type(exc).__name__
    cuda = {"runtime_version": None, "driver_version": None, "nvrtc_version": None}
    if include_cuda:
        try:
            from MyFlows.core.device import _lazy_cupy

            cp = _lazy_cupy()
            from cupy_backends.cuda.libs import nvrtc

            cuda.update(runtime_version=cp.cuda.runtime.runtimeGetVersion(),
                        driver_version=cp.cuda.runtime.driverGetVersion(), nvrtc_version=list(nvrtc.getVersion()))
        except Exception as exc:
            for key, value in cuda.items():
                if value is None:
                    missing[f"cuda.{key}"] = type(exc).__name__
    else:
        missing.update({f"cuda.{key}": "CUDA not requested for this run" for key in cuda})
    native_hash = None
    try:
        from MyFlows.ops.cuda_native.build import source_digest

        native_hash = source_digest()
    except (ImportError, OSError) as exc:
        missing["native_source_sha256"] = type(exc).__name__
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages,
            "gpu": gpu, "cuda": cuda, "native_source_sha256": native_hash, "missing": missing}
