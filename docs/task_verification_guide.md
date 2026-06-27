# 本学期任务功能验证与演示指南

本指南用于按 `docs/本学期任务.md` 逐项运行和展示系统功能。它不是最终总结报告，而是答辩、验收和补实验结果时的操作手册。除特别说明外，所有命令均在项目根目录运行：

```bash
D:\DL\testmyflow
```

## 0. 验证原则

- 优先在 DonkeyCar 场景验证功能：数据、训练、评估、推理、量化、服务、压测尽量使用 `mycar/data` 和 `mycar/models`。
- `MyFlows/` 是自研深度学习框架源码，验证框架能力时可跑单元测试，但课程展示应尽量连接到 DonkeyCar 图像任务。
- `mycar/` 是 DonkeyCar 工程目录，不要移动；仿真驾驶命令必须从 `mycar/` 内运行。
- `--max-samples 0` 表示使用全部样本，不是 0 个样本；快速演示可用 `--max-samples 200` 或 `--max-samples 500`。
- 若已清空旧模型并准备从头训练，可使用默认输出路径；若要保留旧模型，新训练加 `--unique-run` 或 `--run-id`，续训时不要使用 `--unique-run`。
- 本指南中的 VGG、INT8、跨框架对比、压测命令会生成报告所需结果；如时间有限，先跑快速演示命令，再跑完整命令。

## 1. 环境与数据准备

### 1.1 依赖安装

部署推理依赖：

```bash
pip install -r requirements-deploy.txt
```

GPU 训练依赖，按本机 CUDA 版本选择：

```bash
pip install -r MyFlows/requirements-gpu.txt
```

TensorBoard 训练可视化依赖：

```bash
pip install -r MyFlows/requirements-tb.txt
```

benchmark 依赖：

```bash
pip install -r benchmark/requirements-bench.txt
```

跨框架对比还需要 PyTorch、TensorFlow、PaddlePaddle。Windows 原生环境若安装困难，可在 WSL/Linux 或单独 Python 环境中运行：

```bash
pip install torch tensorflow paddlepaddle psutil matplotlib
```

### 1.2 DonkeyCar 数据检查

训练和评估默认读取：

```text
mycar/data/images/*.jpg
mycar/data/catalog_generated.catalog
```

如果数据来自 generated-road-data，先转换：

```bash
python -m tools.convert_generated_road_to_tub_v2 --src mycar/generated-road-data --dst mycar/data --clear-dst
```

快速检查数据和训练入口是否可用：

```bash
python -m apps.train.train_myflows_donkey --max-samples 1 --epochs 0 --device cpu
```

分析当前 DonkeyCar 数据集质量：

```bash
python -m tools.analyze_donkey_data --data mycar/data --out-json docs/experiments/donkey_data_analysis.json
```

重点查看：样本数、直行占比、左右转是否平衡、throttle 是否恒定。当前课程验证以固定速度下的转向角学习为主，throttle 恒定不作为问题处理。

期望存在：

```text
mycar/data/images/*.jpg
mycar/data/catalog_generated.catalog
```

## 2. 任务 (1)：系统级框架能力完善

任务书要求包括训练可视化、数据增强、模型保存、推理量化、gRPC 服务化部署和指标评价。

### 2.0 模块分层自检

本项目已把重复数据读取、训练日志/checkpoint、可视化编排、Grad-CAM CLI 编排和 `MyFlows/utils` 内部职责拆层。可先做轻量自检：

```bash
python -m py_compile apps/common/donkey_data.py apps/common/image_preprocess.py
python -m py_compile apps/train/common/checkpoints.py apps/train/common/logging.py
python -m py_compile MyFlows/utils/training_dashboard.py MyFlows/utils/training_observer.py
python -m py_compile MyFlows/utils/checkpoint.py MyFlows/utils/onnx_exporter.py MyFlows/utils/metrics.py
python -m py_compile tools/explain/model_factory.py tools/explain/targets.py tools/explain/reporting.py
```

关键分层文件：

```text
apps/common/donkey_data.py
apps/common/image_preprocess.py
apps/train/common/checkpoints.py
apps/train/common/logging.py
MyFlows/utils/observers/*
MyFlows/utils/training_dashboard.py
MyFlows/utils/checkpoint.py
MyFlows/utils/onnx_exporter.py
MyFlows/utils/metrics_core/*
tools/explain/*
```

### 2.1 训练过程可视化

可视化模块按五层组织：数据采集层、日志存储层、分析层、展示层、解释算法层。展示层统一使用 TensorBoard；`MyFlows/utils/observers/` 负责分析层，`MyFlows/utils/training_dashboard.py` 负责训练期 TensorBoard 编排。

DonkeyCar ResNet 回归快速演示：

```bash
python -m apps.train.train_myflows_donkey --max-samples 500 --epochs 2 --augment --save-png-plots --tb-grad-interval 50 --tb-param-interval 100 --tb-activation-interval 100 --device auto
```

打开 TensorBoard：

```bash
tensorboard --logdir mycar/logs/tensorboard
```

展示内容：

- `train/*`：loss、accuracy、learning rate、step time、samples/sec。
- `data/*`：angle/throttle 分布、batch 标签统计、类别分布。
- `gradients/*`：全局梯度范数、分层梯度范数、梯度 histogram。
- `params/*`：参数范数、参数 histogram、近似更新比例。
- `activations/*`：激活均值/方差/稀疏度、feature map grid。
- `augment/*`：原图与增强图对比。
- `checkpoint/*` 和 `config/*`：训练配置和模型保存事件。
- 训练日志和 `*_history.json`。
- 可将截图放入 `docs/experiments/training_viz/`。

解释算法层 Grad-CAM（训练完成并生成 checkpoint 后运行）：

```bash
python -m tools.explain_donkey_gradcam --model-type resnet --checkpoint mycar/models/myflow_resnet18_best --data mycar/data --max-samples 8 --device auto
```

Grad-CAM 默认写入 `mycar/logs/tensorboard/gradcam_*`，PNG 和报告写入 `docs/experiments/explainability/`。`tools/explain/` 负责模型构建、目标选择和报告输出；它不经过 gRPC/FastAPI，也不修改 `proto/infer.proto`。

### 2.2 数据处理与增强

DonkeyCar ResNet 回归场景验证：

```bash
python -m apps.train.train_myflows_donkey --max-samples 500 --epochs 2 --augment --mixup --cutmix --device auto
```

DonkeyCar VGG 分类场景验证：

```bash
python -m apps.train.train_vgg_donkey_classify --max-samples 500 --epochs 2 --augment --mixup --cutmix --device auto
```

功能对应关系：

| 参数 | 功能 |
|------|------|
| `--augment` | RandomCrop、RandomRotation、ColorJitter |
| `--mixup` | MixUp batch 内混合 |
| `--cutmix` | CutMix 图像区域混合 |

框架级测试：

```bash
python -m pytest MyFlows/tests/test_transforms.py
```

如果没有 pytest：

```bash
python MyFlows/tests/test_transforms.py
```

代码位置：`MyFlows/utils/transforms.py`。

DonkeyCar 数据索引和图像预处理公共层：`apps/common/donkey_data.py`、`apps/common/image_preprocess.py`。训练、评估、量化、Grad-CAM 和 DataLoader benchmark 复用同一套 catalog/filename 解析逻辑。

### 2.3 模型保存与 ONNX 导出

ResNet 回归快速训练并导出 ONNX：

```bash
python -m apps.train.train_myflows_donkey --unique-run --max-samples 500 --epochs 2 --checkpoint-every 100 --export-onnx --device auto
```

完整训练建议：

```bash
python -m apps.train.train_myflows_donkey --unique-run --max-samples 0 --epochs 20 --batch 2 --augment --graph-opt --checkpoint-every 500 --export-onnx --device auto
```

期望产物：

```text
mycar/logs/myflow_resnet18_checkpoint*.json
mycar/logs/myflow_resnet18_checkpoint*.npz
mycar/models/myflow_resnet18_best*.json
mycar/models/myflow_resnet18_best*.npz
mycar/models/myflow_resnet18_best*.onnx
```

VGG 分类训练并导出 ONNX：

```bash
python -m apps.train.train_vgg_donkey_classify --max-samples 0 --epochs 10 --augment --device auto --export-onnx
```

期望产物：

```text
mycar/models/vgg11_classify_best.json
mycar/models/vgg11_classify_best.npz
mycar/models/vgg11_classify_best.onnx
```

框架级保存加载测试：

```bash
python -m pytest MyFlows/tests/test_serialization.py
```

如果没有 pytest：

```bash
python MyFlows/tests/test_serialization.py
```

实现分层：`MyFlows/utils/checkpoint.py` 负责 JSON+NPZ checkpoint，`MyFlows/utils/onnx_exporter.py` 负责 ONNX 导出，`MyFlows/utils/serialization.py` 保持兼容导出。

### 2.4 模型推理与 INT8 量化

ONNX FP32 离线评估：

```bash
python -m apps.eval.eval_myflows_donkey_onnx --checkpoint mycar/models/myflow_resnet18_best.onnx --max-samples 2000 --device auto
```

完整评估：

```bash
python -m apps.eval.eval_myflows_donkey_onnx --checkpoint mycar/models/myflow_resnet18_best.onnx --max-samples 0 --device auto
```

动态 INT8 量化：

```bash
python -m tools.quantize_onnx --input mycar/models/myflow_resnet18_best.onnx
```

FP32 / INT8 自动对比报告：

```bash
python scripts/run_quantize_eval.py --fp32 mycar/models/myflow_resnet18_best.onnx --max-samples 500 --device auto
```

如果静态量化失败，可先只展示动态量化：

```bash
python scripts/run_quantize_eval.py --fp32 mycar/models/myflow_resnet18_best.onnx --max-samples 500 --device auto --skip-static
```

期望产物：

```text
mycar/models/myflow_resnet18_best_int8.onnx
mycar/models/myflow_resnet18_best_int8_static.onnx
docs/experiments/int8_report.md
docs/experiments/int8_metrics.json
```

展示指标：MSE、转向角符号准确率、平均延迟、P99 延迟、模型大小。

### 2.5 gRPC 服务化部署

启动 gRPC 服务：

```bash
python -m apps.serve.serve_grpc --model mycar/models/myflow_resnet18_best.onnx --port 50051 --device auto
```

另开终端发送单图请求：

```bash
python -m apps.serve.grpc_client --image mycar/data/images/1042_0.0000.jpg --host 127.0.0.1 --port 50051
```

`apps/serve/grpc_client.py` 封装 `GrpcInferenceClient` SDK，负责 channel 生命周期、RGB 字节请求构造、超时控制和输出解析；命令行只是 SDK 的薄封装。

gRPC 压测：

```bash
python benchmark/serve_bench.py --mode grpc --host 127.0.0.1 --port 50051 --requests 100 --workers 4 --out-json docs/experiments/serve_bench_grpc.json --out-md docs/experiments/serve_bench_grpc.md
```

展示内容：服务启动日志、`mycar/logs/serve_grpc.log` 结构化请求日志、客户端预测输出、`qps`、`p50_ms`、`p95_ms`、`p99_ms`、`mean_ms`、成功率。

协议文件：`proto/infer.proto`。生成代码：`generated/grpc/infer_pb2.py`、`generated/grpc/infer_pb2_grpc.py`。

重新生成 gRPC 代码：

```bash
python -m grpc_tools.protoc -I proto --python_out=generated/grpc --grpc_python_out=generated/grpc proto/infer.proto
```

### 2.6 FastAPI 服务补充展示

任务书明确 gRPC，FastAPI 可作为 HTTP 部署补充。

```bash
python -m apps.serve.serve_fastapi --model mycar/models/myflow_resnet18_best.onnx --port 8000 --device auto
```

健康检查：

```text
http://127.0.0.1:8000/healthz
http://127.0.0.1:8000/model_info
http://127.0.0.1:8000/metrics
```

通过 FastAPI SDK 客户端发送单图请求：

```bash
python -m apps.serve.fastapi_client --image mycar/data/images/1042_0.0000.jpg --url http://127.0.0.1:8000 --show-model-info
```

`apps/serve/fastapi_client.py` 封装 `FastApiInferenceClient` SDK，负责 HTTP multipart 请求、超时、错误处理和统一响应解析，不依赖手写 `curl`。

压测：

```bash
python benchmark/serve_bench.py --mode fastapi --host 127.0.0.1 --port 8000 --requests 100 --workers 4 --out-json docs/experiments/serve_bench_fastapi.json --out-md docs/experiments/serve_bench_fastapi.md
```

展示内容：统一 JSON 响应、`X-Request-ID`、`mycar/logs/serve_fastapi.log`、`/metrics` 运行统计、FastAPI SDK 输出和压测报告。

服务模块分层文件：`apps/serve/config.py`、`apps/serve/logger.py`、`apps/serve/metrics.py`、`apps/serve/schema.py`、`apps/serve/onnx_predictor.py`、`apps/serve/serve_grpc.py`、`apps/serve/serve_fastapi.py`、`apps/serve/grpc_client.py`、`apps/serve/fastapi_client.py`。

### 2.7 指标评价模块

ResNet 回归指标：

```bash
python -m apps.eval.eval_myflows_donkey_onnx --checkpoint mycar/models/myflow_resnet18_best.onnx --max-samples 2000 --device auto
```

VGG 分类指标：

```bash
python -m apps.eval.eval_vgg_donkey_classify --checkpoint mycar/models/vgg11_classify_best --max-samples 2000 --device auto
```

展示指标：accuracy、confusion matrix、macro precision、macro recall、macro F1、classification report。实现分层位于 `MyFlows/utils/metrics_core/`，`MyFlows/utils/metrics.py` 保持兼容导出。

## 3. 任务 (2)：卷积和池化实现与准确度验证

框架级验证：

```bash
python -m pytest MyFlows/tests/test_convolution.py
```

如果没有 pytest：

```bash
python MyFlows/tests/test_convolution.py
```

该测试覆盖 Conv2D 前向、Conv2D 反向、MaxPool 前向反向、`im2col` / `col2im` 与 naive reference 对比。

DonkeyCar 场景关联验证：

```bash
python -m apps.train.train_myflows_donkey --max-samples 200 --epochs 1 --device auto
python -m apps.train.train_vgg_donkey_classify --max-samples 200 --epochs 1 --device auto
```

展示逻辑：先用单元测试证明底层卷积/池化正确，再用 DonkeyCar ResNet/VGG 训练证明卷积模块进入真实图像任务。

## 4. 任务 (3)：DonkeyCar VGG 分类与跨框架对比

### 4.1 VGG 分类训练

快速演示：

```bash
python -m apps.train.train_vgg_donkey_classify --max-samples 500 --epochs 2 --augment --device auto
```

正式结果：

```bash
python -m apps.train.train_vgg_donkey_classify --max-samples 0 --epochs 10 --augment --device auto --export-onnx
```

显存或内存紧张时：

```bash
python -m apps.train.train_vgg_donkey_classify --max-samples 0 --epochs 10 --batch 1 --augment --device cpu
```

### 4.2 VGG 分类评估

```bash
python -m apps.eval.eval_vgg_donkey_classify --checkpoint mycar/models/vgg11_classify_best --max-samples 2000 --device auto
```

完整评估：

```bash
python -m apps.eval.eval_vgg_donkey_classify --checkpoint mycar/models/vgg11_classify_best --max-samples 0 --device auto
```

应记录到报告：accuracy、confusion matrix、macro precision、macro recall、macro F1。

### 4.3 跨框架对比

当前可运行的基础 benchmark：

```bash
python benchmark/compare_frameworks.py --epochs 2 --samples 64 --device auto
python benchmark/plot_compare.py
```

期望产物：

```text
benchmark/results.csv
docs/experiments/framework_compare.png
```

`benchmark/results.csv` 是运行 `compare_frameworks.py` 后生成的结果文件。正式提交前应确认 MyFlows、PyTorch、TensorFlow、PaddlePaddle 不出现 `not installed`，并记录 `time_s`、`peak_mb` 和 FLOPs 估算行。

### 4.4 DonkeyCar 场景下的跨框架规划

当前 `benchmark/compare_frameworks.py` 使用同形状合成数据，适合控制变量做框架性能对比，但不完全等同 DonkeyCar 数据场景。为了满足“尽量在 DonkeyCar 场景下验证”，建议展示时采用两层说明：

- DonkeyCar 真实任务由 MyFlows ResNet/VGG 完整验证。
- 跨框架性能对比使用同规模合成输入控制变量；如时间允许，后续新增 DonkeyCar 数据版 benchmark。

建议后续增强脚本：`benchmark/compare_frameworks_donkey.py`。

目标能力：读取 `mycar/data/images`，统一 resize，按 angle 离散为 5 类，MyFlows / PyTorch / TensorFlow / PaddlePaddle 使用近似相同的小 CNN 或 VGG-lite，输出 `benchmark/results_donkey.csv` 和 `docs/experiments/framework_compare_donkey.png`。

## 5. 任务 (4)：设计文档与报告

已有文档：

```text
docs/system_design.md
docs/module_design.md
docs/algorithm_design.md
docs/detailed_design.md
docs/final_report.md
docs/project_structure.md
```

最终报告需要补入真实结果：ResNet 回归评估表、VGG 分类准确率和混淆矩阵、FP32 / INT8 对比表、跨框架 benchmark 表和图、部署压测结果、TensorBoard / PNG 训练曲线。

## 6. 拓展要求：算法优化、ResNet、图优化、DataLoader、部署

本课程中拓展要求按必做能力规划展示。

### 6.1 框架算法优化与图优化

验证命令：

```bash
python -m pytest MyFlows/tests/test_graph_opt.py
python -m pytest MyFlows/tests/test_graph_opt_bn_fold.py
```

DonkeyCar 训练中启用图优化：

```bash
python -m apps.train.train_myflows_donkey --max-samples 500 --epochs 2 --graph-opt --device auto
```

展示重点：`im2col + GEMM`、Conv + ReLU 融合、推理态 Conv + BN 折叠、折叠前后输出一致和 BN 节点数下降。

### 6.2 ResNet18、BatchNorm、残差模块

快速演示：

```bash
python -m apps.train.train_myflows_donkey --max-samples 500 --epochs 2 --augment --graph-opt --device auto
```

测试：

```bash
python -m pytest MyFlows/tests/test_resnet18_smoke.py
```

代码位置：`MyFlows/layers/resnet.py`、`MyFlows/ops/batchnorm.py`。

### 6.3 生产者消费者 DataLoader

DonkeyCar 图像加载吞吐 benchmark：

```bash
python benchmark/dataloader_bench.py --data mycar/data --batch 8 --batches 50
```

DonkeyCar 训练中启用多 worker：

```bash
python -m apps.train.train_myflows_donkey --max-samples 500 --epochs 2 --num-workers 2 --device auto
```

展示重点：`num_workers=0/2/4` 的 `img/s` 对比，以及训练脚本 `--num-workers` 多进程解码预取，证明数据处理与训练可并行。

### 6.4 Docker、Kubernetes、Kubeflow

Docker Compose：

```bash
docker compose -f deploy/docker/docker-compose.yml up --build
```

注意 `deploy/docker/docker-compose.yml` 默认将 `mycar/models` 挂载到 `/models`，默认模型路径为 `/models/model.onnx`。演示前可复制模型为 `mycar/models/model.onnx`，或修改 compose 的 `MODEL_PATH`。

Kubernetes：

```bash
docker build -f deploy/docker/Dockerfile -t myflows-infer:latest .
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
```

应用前必须把 `deploy/k8s/deployment.yaml` 中的 `hostPath.path` 改成本机 `mycar/models` 的绝对路径。Kubeflow 示例位于 `deploy/k8s/kubeflow_pipeline.py`；没有实际集群时，只作为调度设计和扩展示例展示。

## 7. 推荐演示顺序

### 7.1 课堂快速演示

1. 展示目录结构：`MyFlows/`、`apps/`、`tools/`、`generated/`、`benchmark/`、`deploy/`、`docs/`、`mycar/`。
2. 分析 DonkeyCar 数据：`python -m tools.analyze_donkey_data --data mycar/data`。
3. 跑卷积测试：`python MyFlows/tests/test_convolution.py`。
4. 跑 DonkeyCar ResNet 快速训练：`python -m apps.train.train_myflows_donkey --max-samples 200 --epochs 1 --device auto`。
5. 跑 ONNX 评估：`python -m apps.eval.eval_myflows_donkey_onnx --checkpoint mycar/models/myflow_resnet18_best.onnx --max-samples 200 --device auto`。
6. 启动 gRPC：`python -m apps.serve.serve_grpc --model mycar/models/myflow_resnet18_best.onnx --device auto`。
7. gRPC SDK 客户端请求：`python -m apps.serve.grpc_client --image mycar/data/images/1042_0.0000.jpg`。
8. FastAPI SDK 客户端请求：`python -m apps.serve.fastapi_client --image mycar/data/images/1042_0.0000.jpg --url http://127.0.0.1:8000`。
9. 展示 TensorBoard 或已导出的训练曲线截图。

### 7.2 正式提交前补结果

```bash
python -m apps.train.train_vgg_donkey_classify --max-samples 0 --epochs 10 --augment --device auto --export-onnx
python -m apps.eval.eval_vgg_donkey_classify --checkpoint mycar/models/vgg11_classify_best --max-samples 0 --device auto
python -m tools.analyze_donkey_data --data mycar/data --out-json docs/experiments/donkey_data_analysis.json
python scripts/run_quantize_eval.py --fp32 mycar/models/myflow_resnet18_best.onnx --max-samples 500 --device auto
python benchmark/compare_frameworks.py --epochs 2 --samples 64 --device auto
python benchmark/plot_compare.py
python benchmark/dataloader_bench.py --data mycar/data --batch 8 --batches 50
python benchmark/serve_bench.py --mode local --model mycar/models/myflow_resnet18_best.onnx --out-json docs/experiments/serve_bench_local.json --out-md docs/experiments/serve_bench_local.md
```

服务端压测需要先分别启动 gRPC 或 FastAPI 服务，再运行对应 `serve_bench.py` 命令。

## 8. 结果记录表

| 任务 | 命令/入口 | 产物或指标 |
|------|-----------|------------|
| ResNet 回归 | `apps.train.train_myflows_donkey` | loss、MSE、符号准确率、ONNX |
| VGG 分类 | `apps.train.train_vgg_donkey_classify` | accuracy、confusion matrix、F1 |
| 卷积验证 | `MyFlows/tests/test_convolution.py` | 测试通过 |
| 数据增强 | `--augment --mixup --cutmix` | 训练日志、增强图 |
| 可视化 | TensorBoard | 五层训练可视化、梯度/参数/激活、Grad-CAM |
| JSON+NPZ | 训练脚本输出 | `.json` + `.npz` |
| ONNX | `--export-onnx` | `.onnx` |
| INT8 | `scripts/run_quantize_eval.py` | MSE、延迟、P99、模型大小 |
| gRPC | `apps.serve.serve_grpc` + `apps.serve.grpc_client` | SDK 客户端输出、结构化日志、QPS、延迟 |
| FastAPI | `apps.serve.serve_fastapi` + `apps.serve.fastapi_client` | HTTP SDK、`/metrics`、结构化日志、QPS、延迟 |
| 跨框架 | `benchmark/compare_frameworks.py` | time、memory、FLOPs |
| DataLoader | `benchmark/dataloader_bench.py` | img/s |
| Docker | `docker compose` | 容器服务可访问 |
| K8s | `kubectl apply` | Deployment、Service |

## 9. DonkeyCar 场景覆盖矩阵

| 功能 | DonkeyCar 场景验证 | 推荐方式 |
|------|-------------------|----------|
| 数据处理/增强 | 是 | `mycar/data/images` + 训练脚本 |
| 可视化 | 是 | DonkeyCar 训练 + TensorBoard |
| JSON+NPZ 保存 | 是 | ResNet/VGG checkpoint |
| ONNX 导出 | 是 | ResNet/VGG ONNX |
| ONNX 推理 | 是 | `apps.eval.eval_myflows_donkey_onnx` |
| INT8 量化 | 是 | `scripts/run_quantize_eval.py` |
| gRPC | 是 | DonkeyCar ONNX 在线推理 |
| FastAPI | 是 | DonkeyCar ONNX HTTP 推理 |
| 指标评价 | 是 | 回归指标、分类指标 |
| 卷积正确性 | 间接 | 框架测试 + DonkeyCar CNN 训练 |
| 图优化 | 部分 | 图优化测试 + `--graph-opt` 训练 |
| DataLoader 并行 | 是 | `benchmark/dataloader_bench.py --data mycar/data` |
| 跨框架 benchmark | 当前部分不是 | 现有 synthetic benchmark；建议补 DonkeyCar 版 benchmark |
