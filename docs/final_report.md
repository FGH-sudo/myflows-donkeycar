# MyFlows 深度学习框架及应用 — 期末报告

## 1. 引言

本项目在上学期 NumPy 动态图框架基础上，围绕本学期任务书的框架完善、卷积验证、DonkeyCar 图像任务、跨框架对比、设计报告与结构优化要求展开。拓展项按课程要求纳入交付范围，不作为可有可无的附加内容。应用场景为 Donkeycar 模拟器道路图像的转向角回归（ResNet-18）与角度分类（VGG-11）。

## 2. 系统设计

见 `docs/system_design.md`：应用层（公共数据、训练、评估、部署）+ MyFlows 核心 + 外部依赖三层架构；支持 gRPC 与 FastAPI 双部署通道。

## 3. 核心算法

- **im2col + GEMM 卷积**：见 `docs/algorithm_design.md` §1
- **数据增强**：RandomCrop、Rotation、ColorJitter、MixUp、CutMix
- **计算图优化**：常量折叠、Conv+ReLU 融合、**推理态 Conv+BN 折叠**（拓展 3）
- **INT8 量化**：动态/静态量化及 Donkey 校准集
- **训练可视化**：TensorBoard 五层 Dashboard，覆盖采集、存储、分析、展示和 Grad-CAM 解释算法层

## 4. 实现与模块

见 `docs/module_design.md`、`docs/detailed_design.md`。公共 Donkey 数据层复用于训练、评估、量化和解释；Checkpoint 为 JSON 描述 + NPZ 权重，`checkpoint.py` 与 `onnx_exporter.py` 分离实现并由 `serialization.py` 保持兼容 API。

## 5. 实验

### 5.1 ResNet 回归

脚本：`apps/train/train_myflows_donkey.py`、`apps/eval/eval_myflows_donkey.py`。指标：MSE、转向角符号准确率。

### 5.2 VGG 分类

脚本：`apps/train/train_vgg_donkey_classify.py`、`apps/eval/eval_vgg_donkey_classify.py`。指标：准确率、混淆矩阵。真实训练结果需后续运行命令生成并补入报告。

### 5.3 FP32 vs INT8

运行：`python scripts/run_quantize_eval.py --fp32 mycar/models/*.onnx`。报告：`docs/experiments/int8_report.md`。

### 5.4 跨框架对比

运行：`python benchmark/compare_frameworks.py`；绘图：`python benchmark/plot_compare.py`。对比 MyFlows、PyTorch、TensorFlow、PaddlePaddle。当前脚本已具备四框架分支，最终报告需使用安装完整依赖后的真实运行结果。

### 5.5 部署延迟

运行：`python benchmark/serve_bench.py --mode {local,grpc,fastapi}`。服务模块按 `config/logger/metrics/schema/predictor/server/client` 分层，记录结构化请求日志；FastAPI 提供 `/healthz`、`/model_info`、`/metrics`；gRPC/FastAPI 均提供稳定客户端 SDK；压测脚本可输出 JSON/Markdown 报告。

### 5.6 训练可视化与解释

TensorBoard 展示 `train/*`、`data/*`、`gradients/*`、`params/*`、`activations/*`、`augment/*`、`checkpoint/*` 和 `explain/gradcam/*`。训练侧由 `MyFlows/utils/observers/` 计算指标、`training_dashboard.py` 统一写入；Grad-CAM 基于 MyFlows checkpoint 离线生成，不修改 gRPC 推理协议。

### 5.7 计算图优化

`MyFlows/tests/test_graph_opt_bn_fold.py`：折叠前后输出一致、BN 节点数下降。

### 5.8 DataLoader 吞吐（拓展 4）

`python benchmark/dataloader_bench.py`：对比 `num_workers=0/2/4`。

## 6. 总结

| 任务书条目 | 完成情况 |
|------------|----------|
| (1) 可视化/增强/保存/推理/部署/指标 | FastAPI 已完成；INT8 与服务压测报告需真实运行补齐 |
| (2) im2col+GEMM | 已完成 |
| (3) Donkey+VGG+对比 | 脚本已就绪；VGG 分类结果、TensorFlow/PaddlePaddle 结果需运行补齐 |
| (4) 设计文档+报告 | 本文档 + 四份设计 md |
| 结构解耦 | apps/common、apps/train/common、utils observers/dashboard、metrics_core、checkpoint/onnx_exporter |
| 拓展 (3) 图优化 | Conv+BN 折叠 |
| 拓展 (4) 生产者消费者 | MultiprocessDataLoader |
| 拓展 (5) Docker/K8s | deploy/docker、deploy/k8s |

展望：可选接入更多 ONNX 算子，并继续扩展 DonkeyCar 场景下的跨框架 benchmark。

## 附录：常用命令

```bash
# 训练
python -m apps.train.train_myflows_donkey --max-samples 500 --epochs 5 --augment --export-onnx

# 量化评估
python scripts/run_quantize_eval.py --fp32 mycar/models/myflow_resnet18_best.onnx

# 服务
python -m apps.serve.serve_grpc --model mycar/models/myflow_resnet18_best.onnx
python -m apps.serve.grpc_client --image mycar/data/images/1042_0.0000.jpg
python -m apps.serve.serve_fastapi --model mycar/models/myflow_resnet18_best.onnx
python -m apps.serve.fastapi_client --image mycar/data/images/1042_0.0000.jpg --url http://127.0.0.1:8000

# 测试
cd MyFlows && python -m unittest discover -s tests -p "test_*.py"
```
