# 第二阶段 Spec：同算法路径的 CUDA 优化卷积对照

- 版本：v0.2，原生 C/C++ 调度方案；2026-09-06 已实施，结果见 [原生 cuBLAS 路径报告](experiments/semester_2026_fall/stage1/native_cublas_report.md)。
- 实施后状态：当前代码只保留 `cuda_native_cublas`（`MyFlows/ops/cuda_native/`，kernel 由原生 DLL 通过 NVRTC 编译）。下文第 2 节提到的 `MyFlows/ops/cuda/im2col.cu`、`gemm.cu` 以及自写 GEMM 路径是实施过程中的历史步骤，源码已删除，只在报告中保留失败基线。
- 前置结果：[第一阶段报告](experiments/semester_2026_fall/stage1/README.md) 已保留直接卷积 `cuda_c`、CuPy im2col+GEMM 和对应 Nsight/性能数据。
- 本阶段目的：把现有 CuPy 的 im2col+GEMM/col2im 算法路径迁移为 CUDA 可观测路径，在相同数学分解和计时边界下比较 CuPy 与自写 CUDA；不预设任何一方必须更快。

## 1. 对照原则

本阶段永远保留三个标签：

| 路径 | 含义 |
| --- | --- |
| `cupy_im2col` | 现有 CuPy im2col、矩阵乘法和 col2im 参考路径 |
| `cuda_native_cublas` | 自写 CUDA im2col/col2im/max-pool + 原生 C/C++ 调度 + cuBLAS GEMM；当前保留实现 |
| 历史失败基线 | `cuda_c`、`cuda_im2col`、`cuda_im2col_gemm` 的报告和原始数据；实现代码已删除 |

历史实验不覆盖第一阶段 run。当前代码只保留原生 C/C++ 调度路径；失败基线通过报告、配置、fixture 和原始 timings 保留。

控制变量为：FP32、NCHW/OIHW、groups=1、dilation=1、相同 stride/padding、相同输入 fixture、相同 weight/dy、相同 CUDA stream、相同输出与梯度定义。原生路径通过 C/C++ 主机端调度自写重排和池化 kernel，并调用 cuBLAS `Sgemm`；CuPy 作为 Python/CuPy 调度对照。失败路径只在历史报告中比较。

## 2. 实施任务

1. 新增 `MyFlows/ops/cuda/im2col.cu`（历史步骤，源码已删除），实现前向 im2col 和反向 col2im；每个线程写一个输出元素，边界按 zero padding，int32 索引，当前 stream，无隐式同步。
2. 新增 `MyFlows/ops/cuda_native/` 原生扩展：从 CuPy 数组取得设备指针和 shape/stride 元数据，在 C/C++ 主机端创建或复用 cuBLAS handle，绑定当前 CUDA stream，启动自写 im2col/col2im kernel 并调用 `cublasSgemm`。该路径不得使用 CuPy `@`、`RawKernel` 或 Python 侧 kernel launch；输出内存可以由 Python/CuPy 预分配，但计算调度必须由原生扩展完成。
3. 保留 `MyFlows/ops/cuda/gemm.cu`（历史步骤，源码已删除），实现前向 `A·Bᵀ`、输入梯度 `A·B`、权重梯度 `Aᵀ·B` 三种 16×16 shared-memory GEMM，作为自写 GEMM 对照，不把它误认为官方库路径。
4. 新增严格 wrapper，校验输入、连续列矩阵和池化上下文；提供 `cuda_native_cublas` 完整卷积/池化路径，bias、无 bias 和非连续 view 语义与参考路径一致。
5. 在卷积和池化 Op 中加入显式原生 backend，forward 记录实际 backend，backward 沿用同一算法路径并用 `+=` 合并；不改变 `auto` 和 `cupy` 默认行为。
6. 扩展 CUDA 正确性测试：T0-T4、非均匀 dy、方向数值梯度、bias 有/无、view、共享参数、context shape 更新，并逐元素比较独立 FP64 参考。
7. 扩展 benchmark：单独记录 im2col、GEMM、col2im、完整 forward/backward、transfer-inclusive；至少重新测 P0/P1/P2，保留原始样本、P95、fixture hash、源码 hash 和编译时间。
8. 用 Nsight Systems 检查新路径的 kernel 数量、GEMM、空闲段和中间张量；Compute 选择 im2col 或 col2im 进行资源分析。Profiler 时间不混入普通 benchmark。
9. 根据数据决定后续方向：若数据重排占主导，优化连续写入/融合；若自写 GEMM 占主导，优化分块、访存和占用率；若官方 cuBLAS 路径与 CuPy 接近，则保留原生 C/C++ 调度作为课程实现和工程基线，不强行替换；后续可将 cuBLAS 换为 cuBLASLt 做同条件对照。

## 3. 验收

- `cuda_native_cublas` 必须与独立 FP64 reference 在现有 Conv/Pool 正确性矩阵上通过；不能只与 CuPy 结果比较。
- 第一阶段所有原始测试和报告不变；新增 backend 不得让旧 `cuda_c` 或高级 groups/dilation 路径回归。
- P0/P1/P2 至少各有前向、反向、合计、传输计时的 CuPy、原生 C/C++ 调度和自写 CUDA 对照；拆分数据重排、GEMM、调度和内存成本。
- 报告分别回答：同算法路径下 CuPy 调度、原生 C/C++ 调度和自写 GEMM 哪一方在何种规模/stride/padding/批量下更好；差异来自数据重排、GEMM、kernel launch、内存或 wrapper 的哪一部分。
- 不使用“CUDA 全面更快”作为目标；若新路径没有收益，也必须保留真实结果和解释。

## 4. 暂不做

首轮不同时加入 groups/dilation、FP64、Pool、完整 ResNet、GPU PS、All-Reduce 或 Agent。常规部署可使用 CUDA Toolkit、`nvcc`、Windows MSVC C++ 工具链和 cuBLAS 开发库；实施时原生 DLL 由 CMake/MSVC 2022 配合 CUDA Toolkit 构建，运行时使用 NVRTC 和 Python 环境中的 CUDA 12.6 cuBLAS DLL（见 [原生 cuBLAS 路径报告](experiments/semester_2026_fall/stage1/native_cublas_report.md)）。自写 tiled/implicit GEMM 曾作为独立路径实现，源码已删除，只保留为报告中的失败基线。
