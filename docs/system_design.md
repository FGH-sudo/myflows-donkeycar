# 当前系统设计

更新时间：2026-09-05。

本文只说明仓库当前已经存在的系统。第一阶段已实现 CUDA C/C++ Conv/Pool、CPU 同步 PS 和 Nsight 实验入口；阶段证据见 [第一阶段报告](experiments/semester_2026_fall/stage1/README.md)。AutoPilot Agent、All-Reduce、完整 GPU 监控和预训练模型库继续按 [semester_spec.md](semester_spec.md) 推进。

## 1. 项目边界

项目分为框架和应用两层：

- MyFlows/ 是自研深度学习框架。
- 根仓库是 DonkeyCar 应用、训练评估入口、部署和课程实验。

根仓库把 MyFlows/ 记录为独立 Git 仓库。阶段实验同时记录两个仓库的 HEAD、dirty/diff、源码 hash，并保存实际源码快照；恢复工具支持无 Git 历史的源码包。长期远程子仓库引用方式仍需在学期协作前统一。

## 2. 当前结构

~~~mermaid
flowchart LR
  data[DonkeyCar 图片和标签] --> common[数据读取与预处理]
  common --> train[ResNet18 训练]
  train --> checkpoint[JSON + NPZ checkpoint]
  train --> onnx[ONNX 模型]
  checkpoint --> eval[离线评估]
  onnx --> eval
  onnx --> pilot[DonkeyCar 驾驶]
  onnx --> serve[gRPC / FastAPI]
  framework[MyFlows 计算图和算子] --> train
  framework --> eval
~~~

## 3. 当前训练过程

1. apps/common/donkey_data.py 读取 DonkeyCar 图片路径、转向和油门标签。
2. apps/common/image_preprocess.py 完成 resize、RGB 转换和 NCHW 排列。
3. apps/train/train_myflows_donkey.py 构建 ResNet18、损失函数和优化器。
4. MyFlows 完成前向传播、反向传播和参数更新。
5. 训练脚本保存 checkpoint，并可导出 ONNX。
6. TensorBoard 当前可以记录 loss、梯度、参数、数据读取时间、单步训练时间和吞吐量。

当前计时只覆盖了部分训练阶段，也没有完整的 GPU 硬件指标。本学期会按照 Spec 补齐。

## 4. 当前推理过程

当前支持三种方式：

- MyFlows checkpoint 离线推理。
- ONNX Runtime 离线推理和 DonkeyCar 驾驶。
- gRPC 或 FastAPI 服务推理。

服务化代码作为已有基础保留，本学期只做必要的回归测试，不继续扩展为主要开发方向。

## 5. 当前 DonkeyCar 状态

- 当前本地数据有 10,000 条可读取记录，没有缺图。
- 当前数据的油门标签全部为 0.5，不具备变速学习条件。
- 当前驾驶代码使用固定油门，主要学习转向。
- 当前配置使用 donkey-generated-roads-v0 场景。
- 当前没有自动运行多次模拟驾驶并统计 CTE、出界次数和完成情况的工具。

因此，本学期首先建立固定油门下的闭环评估，再决定是否采集变速数据。

## 6. 第一阶段新增能力

- `Conv2D` / `MaxPool2d` 和对应 Op 接收 `backend=auto/numpy/cupy/cuda_c/cuda_im2col/cuda_im2col_gemm`；默认保留 NumPy/CuPy，显式 CUDA C 路径严格验证 FP32、设备和配置。后两条是阶段二实验后端。
- 自写 CUDA 前反向使用 CuPy RawModule/NVRTC 编译，支持 NCHW、OIHW、groups=1、dilation=1；Pool 为 padding=0。底层返回局部梯度，Op 用 += 合并。
- `cuda_c` 保留阶段一直接卷积基线；`cuda_im2col` 将 im2col/col2im 搬到 CUDA C，矩阵乘仍复用 CuPy GEMM；`cuda_im2col_gemm` 再将三种 GEMM 改为自写 CUDA kernel。两条阶段二路径用于拆分数据重排和矩阵乘的差异。
- FP32 小 CNN 完成同初值 CuPy/CUDA C 训练对照；完整 ResNet 的精度和 ImageNet stem 兼容仍在后续范围。
- CPU PS 使用 Windows spawn、一个 server、1/2/4 个 worker；稳定参数名、版本/hash 校验、样本加权均值、server 单次更新和有界超时清理。
- `benchmark/cuda_ops.py`、`ps_demo.py`、`profile_cuda.py` 分别记录三后端实验、PS 和真实 Nsight 报告；GPU Events、端到端 wall time、profile 计时分开。
- 统一测试入口补齐函数式测试，训练 smoke 显式传 seed；DataLoader 预先分批并按序返回，避免 worker 调度改变尾批数量。

## 7. 当前限制

- CUDA C 暂不支持 FP64、分组/空洞卷积和带 padding 的 Pool；这些高级卷积功能继续走原后端。阶段二两条 CUDA 路径当前同样限定 groups=1、dilation=1。
- PS 当前是小型 FP64 MLP 的 CPU 进程模拟，没有 GPU PS、All-Reduce 或 NCCL 多 GPU 实验。
- 没有训练调优 AutoPilot Agent。
- 没有完整 GPU 监控。
- 没有预训练参数导入和冻结训练功能。
- 当前跨框架性能测试的方法需要重做，旧结果不能直接用于本学期结论。
- 完整 ResNet 的 FP32 接入、通用模型状态与 checkpoint 精确恢复仍待后续完成。
