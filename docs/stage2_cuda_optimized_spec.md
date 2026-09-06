# 第二阶段 Spec：同算法路径的 CUDA 优化卷积对照

- 版本：v0.2，原生 C/C++ 调度方案，待工具链就绪后实施。
- 前置结果：[第一阶段报告](experiments/semester_2026_fall/stage1/README.md) 已保留直接卷积 `cuda_c`、CuPy im2col+GEMM 和对应 Nsight/性能数据。
- 本阶段目的：把现有 CuPy 的 im2col+GEMM/col2im 算法路径迁移为 CUDA 可观测路径，在相同数学分解和计时边界下比较 CuPy 与自写 CUDA；不预设任何一方必须更快。

## 1. 对照原则

本阶段永远保留三个标签：

| 路径 | 含义 |
| --- | --- |
| `cupy_im2col` | 现有 CuPy im2col、矩阵乘法和 col2im 参考路径 |
| `cuda_c` | 第一阶段直接卷积 baseline，保留历史报告和源码 |
| `cuda_im2col` | CUDA C im2col/col2im + CuPy GEMM；保留为 Python/CuPy 调度对照 |
| `cuda_native_cublas` | 自写 CUDA im2col/col2im + 原生 C/C++ 调度 + cuBLAS GEMM；验证去除 CuPy 调度后的官方库路径 |
| `cuda_im2col_gemm` | CUDA C im2col/col2im + 自写 CUDA GEMM；保留为自写矩阵乘法对照 |

`cuda_im2col` 不会覆盖 `cuda_c`，也不会修改第一阶段 run。所有新 run 使用新的目录，并在 manifest 中记录 Git 状态和源码 hash。只有在正确性、回归和受控性能实验完成后，才讨论默认 backend 或删除冗余代码。

第一步的控制变量是：FP32、NCHW/OIHW、groups=1、dilation=1、相同 stride/padding、相同输入 fixture、相同 weight/dy、相同 CUDA stream、相同输出与梯度定义。`cuda_im2col` 使用 CuPy GEMM，`cuda_native_cublas` 通过 C/C++ 主机端调度调用 cuBLAS `Sgemm`，`cuda_im2col_gemm` 使用自写 GEMM；三条路径分开报告，避免把 GEMM、调度和数据重排差异混成一个结论。后续可以在相同接口上替换为 cuBLASLt。

## 2. 实施任务

1. 新增 `MyFlows/ops/cuda/im2col.cu`，实现前向 im2col 和反向 col2im；每个线程写一个输出元素，边界按 zero padding，int32 索引，当前 stream，无隐式同步。
3. 新增 `MyFlows/ops/cuda_native/` 原生扩展：从 CuPy 数组取得设备指针和 shape/stride 元数据，在 C/C++ 主机端创建或复用 cuBLAS handle，绑定当前 CUDA stream，启动自写 im2col/col2im kernel 并调用 `cublasSgemm`。该路径不得使用 CuPy `@`、`RawKernel` 或 Python 侧 kernel launch；输出内存可以由 Python/CuPy 预分配，但计算调度必须由原生扩展完成。
4. 保留 `MyFlows/ops/cuda/gemm.cu`，实现前向 `A·Bᵀ`、输入梯度 `A·B`、权重梯度 `Aᵀ·B` 三种 16×16 shared-memory GEMM，作为自写 GEMM 对照，不把它误认为官方库路径。
5. 新增严格 wrapper，校验输入和连续列矩阵；提供 `cuda_im2col`、`cuda_native_cublas` 与 `cuda_im2col_gemm` 三条完整路径，bias、无 bias 和非连续 view 语义与当前 CUDA baseline 一致。
6. 在 `Conv2D_Op` 中加入显式 backend，forward 记录实际 backend，backward 沿用同一算法路径并用 `+=` 合并；不改变 `auto`、`cupy` 或 `cuda_c` 默认行为。
7. 扩展 CUDA 正确性测试：T0-T4、非均匀 dy、方向数值梯度、bias 有/无、view、共享参数、context shape 更新，并逐元素比较独立 FP64 参考。
8. 扩展 benchmark：单独记录 im2col、GEMM、col2im、完整 forward/backward、transfer-inclusive；至少重新测 P0/P1/P2，保留原始样本、P95、fixture hash、源码 hash 和编译时间。
9. 用 Nsight Systems 检查新路径的 kernel 数量、GEMM、空闲段和中间张量；Compute 选择 im2col 或 col2im 进行资源分析。Profiler 时间不混入普通 benchmark。
10. 根据数据决定后续方向：若数据重排占主导，优化连续写入/融合；若自写 GEMM 占主导，优化分块、访存和占用率；若官方 cuBLAS 路径与 CuPy 接近，则保留原生 C/C++ 调度作为课程实现和工程基线，不强行替换；后续可将 cuBLAS 换为 cuBLASLt 做同条件对照。

## 3. 验收

- `cuda_im2col`、`cuda_native_cublas` 与 `cuda_im2col_gemm` 都必须与独立 FP64 reference 在现有 Conv 正确性矩阵上通过；不能只与 CuPy 结果比较。
- 第一阶段所有原始测试和报告不变；新增 backend 不得让旧 `cuda_c` 或高级 groups/dilation 路径回归。
- P0/P1/P2 至少各有前向、反向、合计、传输计时的 CuPy、原生 C/C++ 调度和自写 CUDA 对照；拆分数据重排、GEMM、调度和内存成本。
- 报告分别回答：同算法路径下 CuPy 调度、原生 C/C++ 调度和自写 GEMM 哪一方在何种规模/stride/padding/批量下更好；差异来自数据重排、GEMM、kernel launch、内存或 wrapper 的哪一部分。
- 不使用“CUDA 全面更快”作为目标；若新路径没有收益，也必须保留真实结果和解释。

## 4. 暂不做

首轮不同时加入 groups/dilation、FP64、Pool、完整 ResNet、GPU PS、All-Reduce 或 Agent。常规部署可使用 CUDA Toolkit、`nvcc`、Windows MSVC C++ 工具链和 cuBLAS 开发库；当前实验使用 CMake/MinGW、NVRTC 和 Python 环境中的 cuBLAS DLL，避免依赖管理员安装。自写 tiled/implicit GEMM 已作为独立路径实现，但是否继续优化必须以空闲 GPU 的分解计时和 Nsight 证据为准。
