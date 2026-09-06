# 第二阶段 CUDA 卷积优化记录

日期：2026-09-06。对应 [第二阶段 Spec](../../../stage2_cuda_optimized_spec.md)。第一阶段的直接卷积失败基线仍以 [第一阶段报告](README.md) 和 `performance-final-004` 为准；本页同时记录空闲 GPU 下的阶段二复测结论。

## 当前实现

阶段二现在保留三条可切换路径：

- `cuda_c`：第一阶段直接卷积路径，历史上在主要 Conv 场景慢于 CuPy，作为失败基线保留。
- `cuda_im2col`：CUDA C 实现 `im2col/col2im`，矩阵乘仍调用 CuPy GEMM，用来隔离数据重排开销。
- `cuda_im2col_gemm`：CUDA C 实现 `im2col/col2im`，并使用自写 16×16 共享内存 GEMM，分别覆盖前向、dX 和 dW 三种矩阵乘形式。

同时，`cuda_c` 的 dW 和 db 已从单线程串行累加改为“每个输出元素一个 block、block 内并行归约”。`cuda_im2col_gemm` 的 dW 使用自写转置 GEMM，db 使用同一套并行归约内核。

## 正确性

自写 GEMM 的三种矩阵形式已经用独立小矩阵验证，并通过 T0-T4 卷积、T5 池化、bias、非连续 view、共享权重和图计算回归。当前隔离环境中的 CUDA 卷积测试为 9 项通过，图计算测试为 6 项通过；正确性矩阵中的 NumPy、CuPy、`cuda_c` 和 `cuda_im2col_gemm` 均通过。

## 性能记录规则

阶段二中间 run 不作为正式数据长期保存。GPU 运行游戏时，显存接近满载，GPU Event 会受到其他进程调度影响，因此当时的结果不能单独用于判断新 GEMM 是否胜出。

当前正式对照保留三类数据：

1. `performance-final-004`：第一阶段直接卷积失败基线，说明最初 CUDA 路径为什么慢。
2. `performance-final-idle-001`：游戏退出后的空闲 GPU 复测，包含 CuPy、`cuda_c`、`cuda_im2col` 和 `cuda_im2col_gemm` 的 144 组测量；36 组正确性检查全部通过。
3. 后续优化成功数据：只有在相同 fixture、相同顺序、重复运行并确认大部分目标场景优于 CuPy 后才新增。

空闲 GPU 复测没有改变原始结论。P0/P1/P2 三个卷积场景的组合耗时中，CuPy 分别为 `0.769/0.889/0.782 ms`，自写 GEMM 路径为 `2.130/2.102/2.258 ms`；自写 GEMM 目前没有在卷积场景胜出。组件数据表明，小矩阵下自写 GEMM 的启动、边界处理和访存开销超过了收益；因此游戏负载不是差距的主要原因，`cuda_im2col_gemm` 暂不设为默认路径。中文详细记录见本地实验目录的 `performance-final-idle-001/report.md`。

后续正式表统一使用中文列名，至少包括场景、前向、反向、合计、传输计时、中位数、P95 和结论。`P0/P1/P2`、CUDA、CuPy、GEMM、Nsight、FP32 等专有名词保留原写法。

## 下一步

下一步用 Nsight Compute 分析自写 GEMM 的占用率、全局内存访问、共享内存冲突和 kernel 启动开销；随后针对小矩阵增加专用路径或调整分块策略。若自写 GEMM 在多数场景仍慢，继续保留它作为失败尝试；只有在多数目标场景胜出后，才把成功数据和第一阶段失败基线并列展示，并讨论是否将 `cuda_im2col_gemm` 设为默认 CUDA 路径。
