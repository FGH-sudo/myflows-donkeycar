# 训练可视化截图资料

本目录保存答辩 PPT 和报告可引用的训练可视化图片。当前截图材料以 ResNet 为主；除文件名明确写有 `vgg` 的图片外，其余均按 ResNet 正式训练、评估或解释性材料解释。

## PPT 推荐引用图

优先使用下表中的图片。评价模块演示以 ResNet 为主；VGG 只作为辅助训练过程材料保留，不再单独放 test 指标图。训练摘要和 Grad-CAM 样本指标采用论文式简洁表格，不使用 Markdown 页面截图。

| 图片 | 内容 | 建议用途 |
|---|---|---|
| `resnet_loss_combined.png` | ResNet 训练 step loss 与 epoch mean loss | 展示训练过程和收敛趋势 |
| `resnet_val.png` | ResNet TensorBoard validation loss | 展示验证集监控与 early stopping 依据 |
| `resnet18_test_metrics.png` | ResNet test 指标柱状图 | 正式汇报 ResNet 离线评估结果 |
| `tensorboard_overview.png` | TensorBoard 总览页 | 展示训练可视化系统入口 |
| `resnet_gradients.png` | ResNet 梯度 histogram | 展示梯度监控能力 |
| `resnet_params.png` | ResNet 参数 histogram | 展示参数分布监控能力 |
| `activations_feature_grid.png` | ResNet 激活/feature grid | 展示中间特征可视化 |
| `diagnostics_log.png` | ResNet/VGG 训练诊断日志 | 展示 model summary 和 graph inspection |
| `001_100_-0.0667_overlay.png` | ResNet Grad-CAM 叠加图 | 展示模型关注区域 |
| `003_1027_-0.2222_overlay.png` | ResNet Grad-CAM 叠加图 | 展示另一个解释样本 |
| `gradcam_report_table.png` | ResNet Grad-CAM 样本指标表 | 展示真实角度、预测角度和误差 |
| `vgg_loss.png` | VGG 训练 loss TensorBoard 截图 | 展示 VGG 辅助实验结果 |
| `vgg_training_summary_table.png` | VGG 训练摘要表 | 展示 VGG early stopping 过程 |

## 保留的原始截图

以下图片保留用于备查，不建议作为正式表格直接放入 PPT：

| 图片 | 原因 |
|---|---|
| `gradcam_report.png` | 来自 Markdown 报告表格截图，已替换为 `gradcam_report_table.png` |
| `resnet18_test_metrics_powershell.png` | 终端输出截图，可作为运行证据；正式指标建议用 `resnet18_test_metrics.png` |

## 当前正式命令口径

ResNet 正式评估使用 ONNX test split：

```bash
python -m apps.eval.eval_myflows_donkey_onnx --checkpoint mycar/models/myflow_resnet18_best.onnx --split-file mycar/logs/resnet18_split.json --split test --max-samples 0 --fixed-throttle 0.2 --force-fixed-throttle --device cuda
```

VGG 正式训练和评估使用同一份 ResNet split，关闭数据增强，固定油门 0.2：

```bash
python -m apps.train.train_vgg_donkey_regression --max-samples 0 --epochs 20 --batch 2 --split-file mycar/logs/resnet18_split.json --dropout 0.2 --initializer xavier_uniform --weight-decay 1e-5 --early-stopping --patience 5 --min-delta 1e-4 --fixed-throttle 0.2 --force-fixed-throttle --summary-once --check-shape --num-workers 2 --device cuda --export-onnx
python -m apps.eval.eval_vgg_donkey_regression --checkpoint mycar/models/vgg11_regression_best --split-file mycar/logs/resnet18_split.json --split test --max-samples 0 --fixed-throttle 0.2 --force-fixed-throttle --device cuda
```

Grad-CAM 推荐使用 test split 和固定油门口径：

```bash
python -m tools.explain_donkey_gradcam --model-type resnet --checkpoint mycar/models/myflow_resnet18_best --data mycar/data --split-file mycar/logs/resnet18_split.json --split test --max-samples 8 --fixed-throttle 0.2 --force-fixed-throttle --target-output angle --device cuda
```

打开 TensorBoard：

```bash
tensorboard --logdir mycar/logs/tensorboard
```

## 口径说明

- `--summary-once`、`--check-shape`、`--check-content` 对应训练诊断模块，输出在训练日志中；当前不会写入 TensorBoard。
- TensorBoard 主要记录 `train/*`、`data/*`、`gradients/*`、`params/*`、`activations/*`、`checkpoint/*`，以及 Grad-CAM 的 image/text run。
- VGG 本轮训练因 early stopping 在 epoch 7 停止，结果汇总见 `docs/experiments/vgg_framework_results.md`。VGG 结果没有明显超过 zero-angle baseline，因此在答辩中只作为辅助对比，不再作为评价模块重点截图。
- ResNet 已有正式柱状图 `resnet18_test_metrics.png`，不再额外生成 ResNet 指标表格图。
- 数据增强截图只适合展示增强模块能力；当前 VGG 正式训练关闭增强，ResNet/VGG 正式结果不要用增强截图作为训练效果证据。

代码分层：`MyFlows/utils/observers/` 负责梯度、参数、激活、标签统计；`MyFlows/utils/training_dashboard.py` 负责 TensorBoard tag 与写入时机；`MyFlows/utils/model_inspector.py` 负责 model summary 和 graph inspection；`tools/explain/` 负责 Grad-CAM 模型构建、目标选择和报告输出。
