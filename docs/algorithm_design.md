# 算法设计

## 1. 卷积 im2col + GEMM

`Conv2D_Op` 将输入按 `kernel/stride/padding/dilation` 展开为 `cols`，计算 `cols @ kernel_cols.T`，输出 reshape 为 NCHW。

复杂度（单次卷积，忽略 groups）：约 `O(N · C_out · C_in · kH · kW · H · W)` 乘加。

反向传播通过 `col2im` 将梯度散射回输入。验证：`MyFlows/tests/test_convolution.py` 与 naive 实现对比。

## 2. 数据增强

DonkeyCar 样本索引和图像预处理由 `apps/common/donkey_data.py`、`apps/common/image_preprocess.py` 统一提供，训练、评估、量化、Grad-CAM 和 DataLoader benchmark 不再各自解析 catalog。

| 算法 | 说明 |
|------|------|
| RandomCrop | 随机比例裁剪后双线性 resize |
| RandomRotation | 仿射旋转，边界 REFLECT |
| ColorJitter | 亮度/对比度/饱和度扰动 |
| MixUp | `λ~Beta(α,α)`，图像与标签凸组合 |
| CutMix | 随机矩形区域替换，标签按面积加权 |

Batch 级 MixUp/CutMix 在 `apply_batch_pairwise_mix` 中随机配对。

## 3. 角度离散分类（VGG）

连续 `angle∈[-1,1]` 映射为 K 类（默认 5 档），见 `layers/vgg.py` `angle_to_class`。损失：`CrossEntropy`。

## 4. 计算图优化（拓展 3）

### 4.1 常量折叠

`Add(常量, 常量)` → 单个 `Variable(constant=True)`。

### 4.2 算子融合

- `MatMul + Add` → `Linear`
- `Conv2D + ReLU/LeakyReLU` → `Conv2D_ReLU_Op` 等

### 4.3 Conv + BN 折叠（推理态）

当 `BatchNorm2d_Op.training=False` 且父节点为 `Conv2D_Op` 时，将 BN 参数折入卷积权重：

```
BN(Conv(x)) = γ·(Conv(x)-μ)/√(σ²+ε) + β
            = (γ/√(σ²+ε))·Conv(x) + (β - γμ/√(σ²+ε))

W' = W · (γ/√(σ²+ε)).reshape(C_out,1,1,1)
b' = (b - μ) · γ/√(σ²+ε) + β
```

实现：`MyFlows/core/graph_opt.py` `fold_bn_into_conv`；仅 `apply_graph_optimizations(..., mode="inference")` 启用。

## 5. ONNX INT8 量化

- **动态量化**：`quantize_onnx_dynamic`，权 INT8、激活 FP32
- **静态量化**：`quantize_onnx_static` + `scripts/calibration_reader.py`（复用公共图像预处理）

对比脚本：`scripts/run_quantize_eval.py` → `docs/experiments/int8_report.md`

## 5.1 模型保存与导出分层

- `MyFlows/utils/checkpoint.py`：JSON+NPZ checkpoint 保存/加载、BN buffer、optimizer state。
- `MyFlows/utils/onnx_exporter.py`：ONNX graph lowering 与导出。
- `MyFlows/utils/serialization.py`：兼容导出层，保留 `save_checkpoint` / `load_checkpoint` / `export_onnx` 旧 API。

## 6. 推理部署

- **gRPC**：`apps/serve/serve_grpc.py`，`Predict` RPC 传 RGB 字节流
- **FastAPI**：`apps/serve/serve_fastapi.py`，`POST /predict` multipart 上传
- 共用：`onnx_predictor.py` `OnnxPredictor`，`config.py` / `logger.py` / `metrics.py` / `schema.py` 分离服务配置、日志、指标和响应解析
- 客户端 SDK：`grpc_client.py` 与 `fastapi_client.py` 分别封装协议细节，上层不直接依赖 curl 或手写请求

压测：`benchmark/serve_bench.py`（QPS / p50 / p99）

## 7. 跨框架 Benchmark

`benchmark/compare_frameworks.py`：合成小 CNN，对比 MyFlows / PyTorch / TensorFlow 训练耗时与峰值内存；`benchmark/plot_compare.py` 出图。

## 8. 生产者-消费者数据流水线（拓展 4）

`MyFlows/data/pipeline.py`：`MultiprocessDataLoader`，worker 进程 IO+解码，主进程 `Queue` 取 batch。吞吐对比：`benchmark/dataloader_bench.py`。
