# 训练可视化资料目录

将以下产物放入本目录，供期末报告引用：

| 文件 | 来源 |
|------|------|
| `resnet_loss.png` | `python -m apps.train.train_myflows_donkey --save-png-plots` 或 TensorBoard 导出 |
| `vgg_accuracy.png` | `python -m apps.train.train_vgg_donkey_classify` + TensorBoard |
| `confusion_matrix.png` | `python -m apps.eval.eval_vgg_donkey_classify` |
| `tensorboard_scalars.png` | TensorBoard Scalars 截图 |
| `gradients_params.png` | TensorBoard Gradients / Params 截图 |
| `activations.png` | TensorBoard Activations 截图 |
| `gradcam_overlay.png` | `tools.explain_donkey_gradcam` 输出 |

代码分层：`MyFlows/utils/observers/` 负责梯度、参数、激活、标签统计；`MyFlows/utils/training_dashboard.py` 负责 TensorBoard tag 与写入时机；`tools/explain/` 负责 Grad-CAM 模型构建、目标选择和报告输出。

生成命令示例（项目根 `testmyflow`）:

```bash
python -m apps.train.train_myflows_donkey --max-samples 500 --epochs 2 --augment --save-png-plots
python -m apps.train.train_vgg_donkey_classify --max-samples 500 --epochs 2
python -m tools.explain_donkey_gradcam --model-type resnet --checkpoint mycar/models/myflow_resnet18_best --max-samples 8
python -m apps.eval.eval_vgg_donkey_classify --checkpoint mycar/models/...
tensorboard --logdir mycar/logs/tensorboard
```
