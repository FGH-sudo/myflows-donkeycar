# ResNet18 低秩微调对照

取前 2,000 帧按时间段切分，训练 1535 帧，验证 200 帧。六种方式使用同一组 ImageNet 权重、任务头初值、样本顺序和预算。batch 8，学习率 0.001，训练 2 轮，seed 1。表中使用最后一轮验证集转向 MAE。

| 配置 | 可训练参数 | 初始验证 MAE | 最终验证 MAE |
| --- | ---: | ---: | ---: |
| head | 1,026 | 2.263466 | 0.228543 |
| lora-r4 | 77,832 | 2.263466 | 0.067657 |
| lora-r8 | 155,664 | 2.263466 | 0.058086 |
| lora-r16 | 311,328 | 2.263466 | 0.067794 |
| freeze-3 | 10,494,466 | 2.263466 | 0.040686 |
| full | 11,177,538 | 2.263466 | 0.087309 |

LoRA 固定原权重、偏置和 BN 统计量，更新 layer4 的五个卷积及回归头。缩放系数等于 1。各组均逐字节校验冻结状态。单个种子的结果用于比较参数预算和此次验证表现。

复现命令为 `python -m benchmark.transfer_study lora --out runs/stage3/lora-repro`。本目录 summary.json 和各 attempt 保存参数量、逐轮指标、训练命令与 checkpoint。

从秩 4 的最佳 checkpoint 合并并导出单输入 ONNX。在原验证清单上等间距取 8 帧，用未合并模型与 ONNX Runtime CPU 比较两路回归输出，最大绝对差 8.20×10⁻⁷。逐帧预测和模型校验和在 lora-r4/attempt-01/onnx-parity.json。复现入口为 `python -m benchmark.verify_lora_onnx --run <训练目录>`。
