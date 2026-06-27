# -*- coding: utf-8 -*-
"""
MyFlows (CuPy) 与 ONNX Runtime 的统一设备选择。

- MyFlows 训练/``eval_myflows_donkey.py``：``--device auto`` → CuPy CUDA（需 cupy-cuda12x）
- ONNX 评估/部署：``--device auto`` → CUDAExecutionProvider（需 onnxruntime-gpu + CUDA DLL 在 PATH）
"""

from __future__ import annotations

import MyFlows as ms
from MyFlows.core.device import configure_cuda_dll_path


def resolve_myflows_device(device_arg: str = "auto") -> str:
    """切换 MyFlows 全局设备，返回实际设备名 ``cpu`` 或 ``cuda``。"""
    name = str(device_arg).strip().lower()
    if name == "auto":
        return ms.use_cuda()
    if name in ("cuda", "gpu"):
        return ms.set_device("cuda")
    if name == "cpu":
        return ms.set_device("cpu")
    raise ValueError(f"不支持的 device: {device_arg!r}，请使用 auto、cpu 或 cuda")


def myflows_scalar_float(value) -> float:
    """从 loss 等取值；兼容 NumPy 与 CuPy。"""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        from MyFlows.core.device import asnumpy

        return float(asnumpy(value).item())


def myflows_asnumpy(value):
    from MyFlows.core.device import asnumpy

    return asnumpy(value)


def _wants_ort_gpu(device_arg: str) -> bool:
    return str(device_arg).strip().lower() in ("auto", "cuda", "gpu")


def ort_execution_providers(device_arg: str = "auto") -> list[str]:
    """ONNX Runtime 推理 Provider 请求列表（``auto``/``cuda`` 时优先 GPU）。"""
    if str(device_arg).strip().lower() == "cpu":
        return ["CPUExecutionProvider"]

    if _wants_ort_gpu(device_arg):
        configure_cuda_dll_path()
        try:
            import onnxruntime as ort

            available = set(ort.get_available_providers())
            for candidate in (
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
            ):
                if candidate in available:
                    return [candidate, "CPUExecutionProvider"]
        except ImportError:
            pass
    return ["CPUExecutionProvider"]


def create_ort_inference_session(model_path, device_arg: str = "auto", sess_options=None):
    """创建 ONNX Runtime ``InferenceSession``，返回 ``(session, 实际启用的 providers)``。"""
    import onnxruntime as ort

    if _wants_ort_gpu(device_arg):
        configure_cuda_dll_path()

    providers = ort_execution_providers(device_arg)
    kwargs: dict = {"providers": providers}
    if sess_options is not None:
        kwargs["sess_options"] = sess_options
    session = ort.InferenceSession(str(model_path), **kwargs)
    return session, session.get_providers()


def print_myflows_device(resolved: str, requested: str) -> None:
    if resolved == "cuda":
        print(f"计算设备: CUDA (CuPy, cuda_available={ms.cuda_available()})")
        return
    if requested in ("cuda", "gpu") and not ms.cuda_available():
        print("警告: 请求 CUDA 但不可用，已回退 CPU；请安装 cupy-cuda12x 等")
    print(f"计算设备: CPU (cuda_available={ms.cuda_available()})")


def print_ort_device(active_providers: list[str], requested: str) -> None:
    """``active_providers`` 应来自 ``session.get_providers()``。"""
    print(f"ONNX Runtime 实际启用: {active_providers} (请求 device={requested})")
    want_accel = _wants_ort_gpu(requested)
    if want_accel and active_providers and active_providers[0] == "CPUExecutionProvider":
        print(
            "警告: 已请求 GPU，但 Session 仅 CPU。请确认已安装 onnxruntime-gpu，且本机存在 "
            "cublasLt64_12.dll / cuDNN 9（可与 PyTorch 同目录，或安装 CUDA 12 工具包）。"
            " 脚本会自动把 PyTorch 的 torch\\lib 加入 DLL 搜索路径。"
        )
    elif want_accel and active_providers and active_providers[0] == "CUDAExecutionProvider":
        print("ONNX GPU (CUDAExecutionProvider) 已就绪。")
