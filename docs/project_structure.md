# 项目目录结构

本项目按“框架源码、应用入口、工具、部署、实验结果”分层管理。

| 路径 | 职责 |
|------|------|
| `MyFlows/` | 自研深度学习框架源码：计算图、算子、层、优化器、数据流水线、测试 |
| `apps/common/` | 应用共享工具：DonkeyCar 数据索引、图像预处理、batch padding |
| `apps/train/` | 课程训练入口：ResNet 回归、VGG 回归 |
| `apps/eval/` | 离线评估入口：MyFlows checkpoint、ONNX、VGG 回归 |
| `apps/serve/` | 在线推理服务：gRPC、FastAPI、客户端 SDK、ONNX predictor、日志/指标/schema/config 分层 |
| `tools/` | 数据转换、量化、设备选择等通用工具 |
| `generated/grpc/` | 由 `proto/infer.proto` 生成的 gRPC Python 代码 |
| `benchmark/` | 框架对比、部署压测、DataLoader 吞吐测试 |
| `scripts/` | 实验闭环脚本，如 FP32/INT8 量化评估报告生成 |
| `deploy/` | Docker Compose 部署配置 |
| `docs/` | 任务书、设计文档、报告、实验结果说明 |
| `video/` | ResNet-18 FP32 与静态 INT8 实际运行视频 |
| `mycar/` | DonkeyCar 工程、默认数据、模型、日志；不要随意移动 |
| `DonkeySimWin/` | DonkeyCar 模拟器运行资产；按大文件/外部资产处理 |

## 当前分层重点

| 层次 | 文件 | 说明 |
|------|------|------|
| 应用公共数据 | `apps/common/donkey_data.py` / `apps/common/image_preprocess.py` | 训练、评估、量化、Grad-CAM、DataLoader benchmark 共用数据读取与预处理 |
| 训练公共工具 | `apps/train/common/` | 日志、checkpoint stem、run-id 路径工具 |
| 可视化分析 | `MyFlows/utils/observers/` | 参数、梯度、激活、标签分布统计 |
| 可视化编排 | `MyFlows/utils/training_dashboard.py` | 统一训练 TensorBoard tag 与写入时机 |
| 保存导出 | `MyFlows/utils/checkpoint.py` / `onnx_exporter.py` / `serialization.py` / `tools/export_resnet_onnx.py` | checkpoint、ONNX lowering、ResNet/VGG 部署图导出 |
| 指标 | `MyFlows/utils/metrics_core/` / `metrics.py` | 回归、分类、Donkey 指标和兼容 API |
| 解释 CLI | `tools/explain/` / `tools/explain_donkey_gradcam.py` | Grad-CAM 模型构建、回归输出目标选择、报告与 CLI 编排 |

## 生成代码

修改 `proto/infer.proto` 后重新生成：

```bash
python -m grpc_tools.protoc -I proto --python_out=generated/grpc --grpc_python_out=generated/grpc proto/infer.proto
```

生成后检查 `generated/grpc/infer_pb2_grpc.py` 是否仍使用 `from generated.grpc import infer_pb2 as infer__pb2`。

## 重训与补实验结果

以下命令用于从原始 generated-road 数据重建训练目录，并补齐课程报告中的真实结果；当前整理不伪造实验数据：

```bash
python -m tools.convert_generated_road_to_tub_v2 --src mycar/generated-road-data --dst mycar/data --clear-dst
python -m apps.train.train_myflows_donkey --max-samples 0 --epochs 20 --batch 2 --augment --graph-opt --checkpoint-every 500 --export-onnx --device auto
python -m apps.train.train_vgg_donkey_regression --max-samples 0 --epochs 10 --device auto --export-onnx
python -m apps.eval.eval_vgg_donkey_regression --checkpoint mycar/models/vgg11_regression_best --max-samples 2000 --device auto
python benchmark/compare_frameworks.py --data mycar/data --epochs 2 --samples 64 --device cuda
python benchmark/plot_compare.py
python scripts/run_quantize_eval.py --fp32 mycar/models/myflow_resnet18_best.onnx --data mycar/data --split-file mycar/logs/resnet18_split.json --split test --max-samples 0 --fixed-throttle 0.2 --force-fixed-throttle --device cuda --out-json docs/experiments/int8_metrics.json --out-md docs/experiments/int8_report.md --out-png docs/experiments/int8_report.png
python benchmark/serve_bench.py --mode local --model mycar/models/myflow_resnet18_best.onnx
```
