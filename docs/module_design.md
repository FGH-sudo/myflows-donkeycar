# 当前模块说明

本文列出本学期仍会使用的现有模块。计划新增的模块见 [semester_spec.md](semester_spec.md)，在实现前不会提前列为现有能力。

## MyFlows 框架

| 路径 | 当前职责 |
| --- | --- |
| MyFlows/core/ | 计算图、节点、Tensor 和 CPU/CUDA 设备切换 |
| MyFlows/ops/ | 当前 NumPy/CuPy 算子，包括卷积、池化、损失和激活函数 |
| MyFlows/ops/cuda_native/ | 原生 C/C++ 调度自写 FP32 Conv/Pool kernel、cuBLAS GEMM、严格数组 wrapper 和运行时 DLL 管理 |
| MyFlows/distributed/ | CPU 同步 PS 的模型映射、消息校验、server/worker/launcher 和事件监控 |
| MyFlows/examples/stage1_cnn.py | 同 seed、全链路 FP32 的 32 样本 CNN 训练 fixture |
| MyFlows/layers/resnet.py | 当前主模型 ResNet18 |
| MyFlows/train/ | SGD、Momentum、AdaGrad、RMSProp、Adam 和正则化 |
| MyFlows/utils/checkpoint.py | JSON + NPZ checkpoint 保存和恢复 |
| MyFlows/utils/onnx_exporter.py | ONNX 导出 |
| MyFlows/utils/training_dashboard.py | TensorBoard 训练指标记录 |
| MyFlows/utils/metrics_core/ | 分类、回归和 DonkeyCar 指标 |
| MyFlows/data/pipeline.py | 多进程图片读取，固定整批任务并按顺序返回；独立于 PS 训练 |
| MyFlows/tests/ | 框架单元测试和小型集成测试 |

## 根仓库应用

| 路径 | 当前职责 |
| --- | --- |
| apps/common/ | DonkeyCar 数据索引、划分和图像预处理 |
| apps/train/train_myflows_donkey.py | ResNet18 训练入口 |
| apps/train/common/ | 验证、早停、checkpoint 路径和训练日志 |
| apps/eval/eval_myflows_donkey.py | MyFlows checkpoint 评估 |
| apps/eval/eval_myflows_donkey_onnx.py | ONNX 模型评估 |
| apps/serve/ | ONNX predictor、gRPC、FastAPI 和客户端 |
| mycar/myflows_pilot.py | DonkeyCar 驾驶控制器；不是训练调优 Agent |
| tools/analyze_donkey_data.py | 数据数量和标签分布检查 |
| tools/export_resnet_onnx.py | 当前 ResNet18 ONNX 导出入口 |
| benchmark/cuda_ops.py | CUDA/NumPy/CuPy 正确性、性能、CNN 训练和短 profile 工作负载 |
| benchmark/ps_demo.py | CPU PS 单/多进程更新对照及故障注入 |
| benchmark/profile_cuda.py | Nsight 采集、报告存在性和真实 kernel 内容验证 |
| benchmark/stage1_common.py | 配置、环境、双仓库源码快照、错误状态与产物校验和 |
| benchmark/stage1_regression.py | 三个独立进程的完整回归与日志归档 |
| tools/run_tests.py | 同时收集 unittest 和函数式测试的统一入口 |
| tools/restore_stage1_source.py | 按运行 manifest 校验并恢复实际源码，不覆盖现有目录 |
| tests/ | 根仓库应用层测试 |

## 本学期预计新增位置

以下只是建议位置，需在 Spec 审核后再创建：

| 建议路径 | 计划内容 |
| --- | --- |
| MyFlows/distributed/ 的后续扩展 | All-Reduce、更多模型与训练状态接入 |
| MyFlows/monitoring/ | GPU、CPU 和训练阶段耗时监控 |
| MyFlows/layers/ | 补充 MobileNetV2、AlexNet，并继续使用现有 ResNet18 |
| MyFlows/utils/pretrained.py | 导入预训练参数、检查匹配情况、冻结和解冻参数 |
| apps/autopilot/ | 自动安排训练、选择参数、检查异常和生成报告 |
| tests/system/ | Agent、分布式训练、预训练和图像分类系统测试 |

这些目录名称可以在审核时调整。真正实现后再把本文件中的“预计新增”改成“当前已有”。
