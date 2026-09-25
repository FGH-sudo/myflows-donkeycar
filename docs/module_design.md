# 当前模块说明

本文列出本学期仍会使用的现有模块。计划新增的模块见 [semester_spec.md](semester_spec.md)，在实现前不会提前列为现有能力。

## MyFlows 框架

| 路径 | 当前职责 |
| --- | --- |
| MyFlows/core/ | 计算图、节点、Tensor 和 CPU/CUDA 设备切换 |
| MyFlows/ops/ | 当前 NumPy/CuPy 算子，包括卷积、池化、损失和激活函数 |
| MyFlows/ops/cuda_native/ | 原生 C/C++ 调度自写 FP32 Conv/Pool kernel、cuBLAS GEMM、严格数组 wrapper 和运行时 DLL 管理 |
| MyFlows/distributed/ | 单机回环同步 PS（Socket JSON / gRPC）与 Ring AllReduce；Worker 本地优化器更新，PS 只聚合梯度 |
| MyFlows/distributed/launcher.py | `run_training(**config)`：配置规范化、端口分配、spawn 启动、事件收集与清理 |
| MyFlows/distributed/ps.py、engine.py | PS 进程入口与同步轮次状态机：加权聚合、心跳与存活检查、重复/陈旧消息拒绝 |
| MyFlows/distributed/worker.py、session.py、schedule.py | Worker 入口、单 rank 前向反向与本地更新、批次游标和单进程基线 |
| MyFlows/distributed/ring.py、gradient_layout.py | Ring 两阶段规约、分块布局与 padding、元数据屏障 |
| MyFlows/distributed/transport_socket.py、transport_grpc.py | 带长度前缀的 Socket JSON；PS 与 Ring 的 gRPC 客户端和服务端 |
| MyFlows/distributed/measurement.py、monitor.py、metrics_reduce.py | 逐 rank 的 steps/epochs JSONL 与状态快照、CUDA Event 计时、流量统计、事件上报、指标归约 |
| MyFlows/distributed/tasks/ | synthetic、mnist_mlp、donkey_cnn、donkey_resnet18 任务适配与预处理数据读取 |
| MyFlows/examples/stage1_cnn.py | 同 seed、全链路 FP32 的 32 样本 CNN 训练 fixture |
| MyFlows/layers/resnet.py | 当前主模型 ResNet18 |
| MyFlows/train/ | SGD、Momentum、AdaGrad、RMSProp、Adam 和正则化 |
| MyFlows/utils/checkpoint.py | JSON + NPZ checkpoint 保存和恢复 |
| MyFlows/utils/onnx_exporter.py | ONNX 导出 |
| MyFlows/utils/training_dashboard.py | TensorBoard 训练指标记录；可选同时逐行写 metrics.jsonl 供控制台增量读取 |
| MyFlows/monitoring/gpu.py | `GpuSampler`（NVML 优先、nvidia-smi 兜底，缺测字段为 null 并记录原因）与 `ResourceRecorder`（后台线程，内存环形历史 + 可选 JSONL） |
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
| apps/console/server.py | 训练控制台入口 `python -m apps.console.server`：组装任务管理、运行索引、实时 GPU 采样并托管前端 |
| apps/console/api.py | REST 与 SSE 路由：GPU 实时流、运行列表/详情、step/epoch/资源增量序列、日志尾部、任务提交与停止 |
| apps/console/registry.py、readers.py | 发现控制台任务、分布式归档（run_index.json）与 TensorBoard 单进程日志；按游标增量读取 JSONL，统一 perf_counter 时间轴 |
| apps/console/jobs.py | 任务排队（默认串行）、子进程启动、停止（分布式发 CTRL_BREAK，单进程写暂停文件，超时后结束进程树）与重启后重新接管 |
| apps/console/presets.py | 任务预设与提交校验：只接受框架支持的模式/传输组合和白名单参数 |
| apps/console/web/ | React + Vite + Ant Design + ECharts 前端：总览、运行记录、运行详情（曲线/耗时/资源/拓扑/配置/日志）、新建训练、运行对比 |
| mycar/myflows_pilot.py | DonkeyCar 驾驶控制器；不是训练调优 Agent |
| tools/analyze_donkey_data.py | 数据数量和标签分布检查 |
| tools/export_resnet_onnx.py | 当前 ResNet18 ONNX 导出入口 |
| benchmark/cuda_ops.py | CUDA/NumPy/CuPy 正确性、性能、CNN 训练和短 profile 工作负载 |
| benchmark/ps_demo.py | 同步 PS 数值对照及故障注入（转发新训练入口） |
| benchmark/distributed_train.py | 单进程 / PS / Ring 统一训练入口 |
| benchmark/distributed_compare.py | 分布式 Spec 第 12.2 节七配置矩阵的对比入口 |
| benchmark/distributed_experiment.py | 正式实验单元：读 JSON 配置，训练并归档 config/manifest/results、逐 rank JSONL 和约 1 秒一次的资源采样 |
| benchmark/distributed_suite.py | 以子进程串行执行数值、状态、训练、性能、故障各阶段，维护 run_index.json |
| benchmark/complete_course_experiments.py | 完整课程实验编排（prepare/validate/train/performance/extras/report） |
| benchmark/prepare_distributed_data.py、prepare_resnet_comparison.py | 预处理 MNIST/道路数据为 npy，准备 ResNet 冻结 BN 统计量 |
| benchmark/report_distributed.py | 生成 comparison.csv、epoch_metrics.csv、加速比与收敛图 |
| benchmark/profile_distributed.py | 对分布式训练采集 Nsight Systems/Compute 并检查 kernel |
| benchmark/distributed_monitor_overhead.py | 资源监控开关各 3 次的开销对照 |
| benchmark/audit_distributed_artifacts.py、validate_resnet_steps.py、validate_ring_volume.py、diagnose_resnet_gradients.py | 归档审计、ResNet 逐步对照、Ring 通信量核对与梯度诊断 |
| benchmark/profile_cuda.py | Nsight 采集、报告存在性和真实 kernel 内容验证 |
| benchmark/stage1_common.py | 配置、环境、双仓库源码快照、错误状态与产物校验和 |
| benchmark/stage1_regression.py | 三个独立进程的完整回归与日志归档 |
| tools/run_tests.py | 同时收集 unittest 和函数式测试的统一入口 |
| tools/restore_stage1_source.py | 按运行 manifest 校验并恢复实际源码，不覆盖现有目录 |
| tools/build_native_cuda.py | CMake/MSVC 构建原生 CUDA DLL |
| tools/generate_distributed_proto.py | 从 proto/ 生成 generated/grpc/ 下的 gRPC 代码 |
| proto/、generated/grpc/ | PS、Ring、推理服务的 protobuf 定义与生成代码 |
| tests/ | 根仓库应用层测试 |

## 本学期预计新增位置

以下只是建议位置，需在 Spec 审核后再创建：

| 建议路径 | 计划内容 |
| --- | --- |
| MyFlows/layers/ | 补充 MobileNetV2、AlexNet，并继续使用现有 ResNet18 |
| MyFlows/utils/pretrained.py | 导入预训练参数、检查匹配情况、冻结和解冻参数 |
| apps/autopilot/ | 自动安排训练、选择参数、检查异常和生成报告 |
| tests/system/ | Agent、分布式训练、预训练和图像分类系统测试 |

这些目录名称可以在审核时调整。真正实现后再把本文件中的“预计新增”改成“当前已有”。
