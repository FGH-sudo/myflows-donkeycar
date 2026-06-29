# MyFlows 深度学习框架及应用 — 期末报告

## 1. 引言

本项目在上学期 NumPy 动态图框架基础上，围绕本学期任务书的框架完善、卷积验证、DonkeyCar 图像任务、跨框架对比、设计报告与结构优化要求展开。拓展项按课程要求纳入交付范围，不作为可有可无的附加内容。应用场景为 Donkeycar 模拟器道路图像的控制回归，ResNet-18 与 VGG-11 均输出 `[angle, throttle]`。

## 2. 系统设计

见 `docs/system_design.md`：应用层（公共数据、训练、评估、部署）+ MyFlows 核心 + 外部依赖三层架构；支持 gRPC 与 FastAPI 双部署通道。

## 3. 核心算法

- **im2col + GEMM 卷积**：见 `docs/algorithm_design.md` §1
- **数据增强**：RandomCrop、Rotation、ColorJitter、MixUp、CutMix
- **可选数据划分与验证闭环**：确定性 train/val/test split、split 文件复用、验证集 loss、best checkpoint 和 early stopping
- **正则化、Dropout 与初始化**：L1/L2/Elastic Net、Dropout 训练/评估态切换、Xavier/Kaiming/Normal/Constant 初始化
- **训练诊断**：model summary、层/节点 shape 检查、NaN/Inf 和全零输出检查
- **计算图优化**：常量折叠、Conv+ReLU 融合、**推理态 Conv+BN 折叠**（拓展 3）
- **INT8 量化**：动态/静态量化及 Donkey 校准集
- **训练可视化**：TensorBoard 五层 Dashboard，覆盖采集、存储、分析、展示和 Grad-CAM 解释算法层

## 4. 实现与模块

见 `docs/module_design.md`、`docs/detailed_design.md`。公共 Donkey 数据层复用于训练、评估、量化和解释；`apps/common/splits.py` 独立负责数据划分，`apps/train/common/` 独立负责验证、best selection 和 early stopping；Dropout、正则化、初始化和模型诊断放在 MyFlows 框架层。Checkpoint 为 JSON 描述 + NPZ 权重，`checkpoint.py` 与 `onnx_exporter.py` 分离实现并由 `serialization.py` 保持兼容 API。

## 5. 实验

### 5.1 ResNet 回归

脚本：`apps/train/train_myflows_donkey.py`、`apps/eval/eval_myflows_donkey.py`。指标：MSE、转向角符号准确率。

### 5.2 VGG 回归

脚本：`apps/train/train_vgg_donkey_regression.py`、`apps/eval/eval_vgg_donkey_regression.py`。指标：angle/throttle MSE、MAE、overall MSE、angle sign accuracy。本次快速验收使用 500 张 DonkeyCar 子集，train/val/test=300/100/100，结果记录于 `docs/experiments/vgg_framework_results.md`：test overall_mse=0.027843，angle_sign_accuracy=0.6200。

### 5.3 FP32 vs INT8

运行：

```bash
python scripts/run_quantize_eval.py --fp32 mycar/models/myflow_resnet18_best.onnx --data mycar/data --split-file mycar/logs/resnet18_split.json --split test --max-samples 0 --fixed-throttle 0.2 --force-fixed-throttle --device cuda --out-json docs/experiments/int8_metrics.json --out-md docs/experiments/int8_report.md --out-png docs/experiments/int8_report.png
```

本次使用 DonkeyCar test split 1000 张图像，固定油门 0.2，ONNX Runtime 实际启用 `CUDAExecutionProvider`。报告和图见 `docs/experiments/int8_report.md`、`docs/experiments/int8_report.png`。

| 模型 | angle MSE | throttle MSE | overall MSE | angle sign acc | mean latency | P99 | size |
|------|----------:|-------------:|------------:|---------------:|-------------:|----:|-----:|
| FP32 | 0.001698 | 0.000399 | 0.001048 | 0.9470 | 3.50 ms | 4.51 ms | 42.679 MB |
| INT8 dynamic | 0.001529 | 0.000451 | 0.000990 | 0.9590 | 397.14 ms | 519.95 ms | 10.767 MB |
| INT8 static | 0.004257 | 0.015295 | 0.009776 | 0.8480 | 5.29 ms | 7.24 ms | 10.772 MB |

结论：INT8 将模型体积压缩到约 25%，但当前 CUDA 部署下动态 INT8 延迟明显偏高，静态 INT8 精度下降更明显；最终实时推理推荐使用 FP32 ONNX，INT8 作为量化实验与体积压缩对比展示。

### 5.4 跨框架对比

运行：`python benchmark/compare_frameworks.py --data mycar/data --samples 64 --epochs 2 --batch 8 --device cuda`；绘图：`python benchmark/plot_compare.py`。对比 MyFlows、PyTorch、PaddlePaddle，且正式结果不混入 CPU 回退；图中保留训练耗时和 RSS 内存两项。结果生成到 `benchmark/results.csv` 和 `docs/experiments/framework_compare.png`，汇总见 `docs/experiments/vgg_framework_results.md`。

### 5.5 部署延迟

运行：`python benchmark/serve_bench.py --mode {local,grpc,fastapi}`。服务模块按 `config/logger/metrics/schema/predictor/server/client` 分层，记录结构化请求日志；FastAPI 提供 `/healthz`、`/model_info`、`/metrics`；gRPC/FastAPI 均提供稳定客户端 SDK；压测脚本可输出 JSON/Markdown 报告。

### 5.6 训练可视化与解释

TensorBoard 展示 `train/*`、`data/*`、`gradients/*`、`params/*`、`activations/*`、`augment/*`、`checkpoint/*` 和 `explain/gradcam/*`。训练侧由 `MyFlows/utils/observers/` 计算指标、`training_dashboard.py` 统一写入；Grad-CAM 基于 MyFlows checkpoint 离线生成，ResNet/VGG 均解释 `output_dim=2` 的 `[angle, throttle]` 回归输出，不修改 gRPC 推理协议。

### 5.7 计算图优化

`MyFlows/tests/test_graph_opt_bn_fold.py`：折叠前后输出一致、BN 节点数下降。

### 5.8 DataLoader 吞吐（拓展 4）

`python benchmark/dataloader_bench.py --data mycar/data --batch 8 --batches 500`：对比 `num_workers=0/2/4`。本轮 DonkeyCar 4000 张图像长跑结果显示，`num_workers=2` 吞吐最高（629.12 img/s），相对同步加载约 4.70x；产物见 `docs/experiments/dataloader_bench.md` 与 `docs/experiments/dataloader_bench.png`。

### 5.9 Split、正则化与诊断验收

训练脚本新增的 split、validation、regularization、Dropout、initializer 和 diagnostics 都是可选能力，默认关闭以保持旧命令兼容。正式结果仍需在 ResNet/VGG 训练完成后按 `docs/task_verification_guide.md` 的命令生成，不在本文档中预填指标。

## 6. 总结

| 任务书条目 | 完成情况 |
|------------|----------|
| (1) 可视化/增强/保存/推理/部署/指标 | Split、验证、正则化、Dropout、初始化、诊断、FP32/INT8 推理对比、FastAPI 已完成；服务压测报告需真实运行补齐 |
| (2) im2col+GEMM | 已完成 |
| (3) Donkey+VGG+对比 | 已完成 VGG 回归快速验收与 MyFlows/PyTorch/PaddlePaddle 跨框架 CUDA benchmark；结果见 `docs/experiments/vgg_framework_results.md` |
| (4) 设计文档+报告 | 本文档 + 四份设计 md |
| 结构解耦 | apps/common/splits、apps/train/common、MyFlows Dropout/regularization/initializers/inspector、utils observers/dashboard、metrics_core、checkpoint/onnx_exporter |
| 拓展 (3) 图优化 | Conv+BN 折叠 |
| 拓展 (4) 生产者消费者 | MultiprocessDataLoader |
| 拓展 (5) Docker/K8s | deploy/docker、deploy/k8s |

展望：可选接入更多 ONNX 算子，并继续扩展 DonkeyCar 场景下的跨框架 benchmark。

## 附录：常用命令

```bash
# 训练
python -m apps.train.train_myflows_donkey --max-samples 500 --epochs 5 --augment --export-onnx

# 量化评估
python scripts/run_quantize_eval.py --fp32 mycar/models/myflow_resnet18_best.onnx --data mycar/data --split-file mycar/logs/resnet18_split.json --split test --max-samples 0 --fixed-throttle 0.2 --force-fixed-throttle --device cuda --out-json docs/experiments/int8_metrics.json --out-md docs/experiments/int8_report.md --out-png docs/experiments/int8_report.png

# 服务
python -m apps.serve.serve_grpc --model mycar/models/myflow_resnet18_best.onnx
python -m apps.serve.grpc_client --image mycar/data/images/1042_0.0000.jpg
python -m apps.serve.serve_fastapi --model mycar/models/myflow_resnet18_best.onnx
python -m apps.serve.fastapi_client --image mycar/data/images/1042_0.0000.jpg --url http://127.0.0.1:8000

# 测试
cd MyFlows && python -m unittest discover -s tests -p "test_*.py"
```
