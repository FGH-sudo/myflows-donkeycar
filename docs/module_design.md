# 模块设计

## core/

| 模块 | 职责 | 输入/输出 |
|------|------|-----------|
| `node.py` | `Node` / `Variable` | 动态图节点 |
| `graph.py` | 拓扑排序、`forward` / `backward` | `Graph(target, optimize=...)` |
| `graph_opt.py` | 常量折叠、Linear/Conv+Act/BN 融合 | `apply_graph_optimizations(..., mode=)` |
| `tensor.py` | `Tensor` 封装 | 与 `xp` 设备一致 |
| `device.py` | CPU/CUDA 切换 | `set_device` / `use_cuda` |

## ops/

| 模块 | 职责 |
|------|------|
| `convolution.py` | im2col、Conv2D_Op、池化、融合卷积+激活 |
| `batchnorm.py` | BatchNorm2d_Op、GlobalAvgPool2d_Op |
| `loss.py` | CrossEntropy、MSELoss |
| `activation.py` | ReLU、Softmax 等 |
| `dropout.py` | Dropout_Op；训练态随机置零并按 keep probability 缩放，评估态恒等输出 |

## layers/

| 模块 | 职责 |
|------|------|
| `layer.py` | Conv2D、Dense、MaxPool2d、Dropout |
| `resnet.py` | ResNet18、BasicBlock；`output_dim` 回归输出头、可选 Dropout 和初始化配置 |
| `vgg.py` | VGG11 回归网络；可选 Dropout 和初始化配置 |

## utils/

| 模块 | 职责 |
|------|------|
| `checkpoint.py` / `onnx_exporter.py` / `serialization.py` | checkpoint、ONNX 导出与兼容导出层 |
| `metrics_core/` / `metrics.py` | 回归、分类、Donkey 指标拆分实现与兼容导出层 |
| `transforms.py` | 五种增强 + MixUp/CutMix |
| `tensorboard_logger.py` | TensorBoard 封装 |
| `observers/` / `training_observer.py` | 梯度、参数、激活、标签分布等训练观测指标与兼容导出层 |
| `training_dashboard.py` | 训练脚本 TensorBoard 编排层 |
| `initializers.py` | Xavier/Kaiming/Normal/Constant 参数初始化工厂 |
| `model_inspector.py` | model summary、图节点 shape/content 检查、NaN/Inf/全零输出诊断 |
| `gradcam.py` | Grad-CAM 解释算法工具 |
| `viz.py` | TrainingHistory、PNG 曲线 |
| `quantize.py` | ONNX INT8 |

## train/

| 模块 | 职责 |
|------|------|
| `regularization.py` | L1、L2、Elastic Net 权重惩罚；默认跳过 bias/BN 参数，可由训练 CLI 打开 |

## data/

| 模块 | 职责 |
|------|------|
| `pipeline.py` | MultiprocessDataLoader（拓展 4） |

## 应用脚本（仓库根目录）

| 脚本 | 职责 |
|------|------|
| `apps/common/donkey_data.py` / `apps/common/image_preprocess.py` | DonkeyCar 数据索引、图像预处理和 batch padding |
| `apps/common/splits.py` | 确定性 train/val/test 划分、split JSON 保存/复用、评估子集选择 |
| `apps/train/common/` | 训练日志、checkpoint 路径、run-id、验证循环、best selection、early stopping |
| `apps/train/train_myflows_donkey.py` | ResNet-18 回归 + TB；通过 CLI 接入 split/validation/regularization/dropout/diagnostics |
| `apps/train/train_vgg_donkey_regression.py` | VGG-11 回归；通过 CLI 复用同一组训练公共 helper |
| `tools/export_resnet_onnx.py` | ResNet-18 / VGG-11 checkpoint 到固定 batch ONNX 部署图的独立导出入口 |
| `tools/explain/` / `tools/explain_donkey_gradcam.py` | checkpoint Grad-CAM 模型构建、回归输出目标选择、报告和 CLI 编排 |
| `apps/serve/config.py` / `logger.py` / `metrics.py` / `schema.py` | serving 配置、结构化日志、运行指标、响应结构 |
| `apps/serve/onnx_predictor.py` | ONNX Runtime 推理封装 |
| `apps/serve/serve_grpc.py` / `apps/serve/serve_fastapi.py` | gRPC / FastAPI 协议服务端 |
| `apps/serve/grpc_client.py` / `apps/serve/fastapi_client.py` | 可复用客户端 SDK 与 CLI |
| `scripts/run_quantize_eval.py` | FP32/INT8 推理评测报告；复用 split、固定油门配置并输出 Markdown/PNG/JSON |
| `benchmark/compare_frameworks.py` | 跨框架对比 |
| `benchmark/serve_bench.py` | 部署压测 |
