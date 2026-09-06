# 原生 C/C++ 调度与 cuBLAS 对照实验

日期：2026-09-06。对应 [第二阶段 Spec](../../../stage2_cuda_optimized_spec.md)。本实验在游戏关闭、GPU 无计算任务的条件下进行，目标是区分 CuPy 封装调度与原生 C/C++ 调度的差异。

## 实现路径

`cuda_native_cublas` 的 Python 层只负责 CuPy 数组的校验、分配和设备指针传递。原生 DLL 由 CMake/MinGW 构建，使用 CUDA Driver API 加载 NVRTC 生成的 im2col、col2im 和 bias 归约 kernel，并在同一 CUDA stream 上调用 cuBLAS `Sgemm`。该路径不执行 CuPy `@`、`RawKernel` 或 Python 侧 kernel launch。

本机没有管理员安装的完整 CUDA Toolkit，但 Python 环境已有 NVIDIA CUDA 运行库和 cuBLAS DLL；原生调度 DLL通过动态加载这些库完成实验。CMake 已安装到 `C:\Program Files\CMake`，构建入口为 `tools/build_native_cuda.py`。

## 实验条件

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU。
- 固定 `seed=0`、FP32、相同输入 fixture、相同 CUDA stream。
- 每组预热 10 次，采样 3 组、每组 50 次；计时为 CUDA Event。
- 场景：P0、P1、P2；后端：CuPy、CUDA `im2col` 加 CuPy GEMM、`cuda_native_cublas`。
- 三个场景的正确性检查全部通过；原生后端额外回归测试和现有 CUDA 测试共 97 项通过。

## 完整组合耗时

单位为毫秒，数值越小越快。组合阶段包含前向和反向。

| 场景 | CuPy | CUDA `im2col` | `cuda_native_cublas` | 最快 |
|---|---:|---:|---:|---|
| P0 | 0.856 | 1.343 | 0.315 | 原生 C/C++ + cuBLAS |
| P1 | 0.704 | 1.312 | 0.282 | 原生 C/C++ + cuBLAS |
| P2 | 0.753 | 1.289 | 0.258 | 原生 C/C++ + cuBLAS |

传输包含阶段分别为：P0 `0.772 ms`、P1 `1.045 ms`、P2 `0.870 ms`；CuPy 分别为 `1.715 ms`、`2.027 ms`、`1.820 ms`。结果显示，在当前 fixture 下，原生路径同时减少了 CuPy 索引式 im2col 的开销，并避免了 Python 侧 GEMM/算子调度。

## 解释与限制

这不是“自写 GEMM 胜过 cuBLAS”的结果。矩阵乘法仍由 cuBLAS 执行，性能收益主要来自自写 im2col、原生 stream 调度和较少的 Python 封装开销。自写 `cuda_im2col_gemm` 仍作为独立失败尝试保留，不应与本实验的官方 GEMM 路径混为一谈。

本实验使用了用户态 CUDA 运行库和 NVRTC，不代表完整 CUDA Toolkit 已安装；后续若安装 `nvcc` 和 MSVC，可把同一 C++ 调度层切换到常规 CUDA 构建流程。当前结果足以证明课程要求的区分：两条路径都使用 GPU，但一条由 CuPy 封装调度，另一条由项目自己的 C/C++ 层调度官方 CUDA 库。

原始证据：`performance-native-P0-final-001`、`performance-native-P1-final-001`、`performance-native-P2-final-001` 下的 `results.json`、`timings.csv`、fixture 和输出快照。
