# 可解释性可视化资料目录

本目录保存 Grad-CAM 输出，供设计文档与实验说明引用。

生成示例：

```bash
python -m tools.explain_donkey_gradcam --model-type resnet --checkpoint mycar/models/myflow_resnet18_best --data mycar/data --split-file mycar/logs/resnet18_split.json --split test --max-samples 8 --fixed-throttle 0.2 --force-fixed-throttle --target-output angle --device cuda
```

当前保留一份代表性结果：`gradcam_resnet_20260708_220923/report.md`。

输出内容：

| 文件 | 说明 |
|------|------|
| `gradcam_<model>_<时间戳>/*_overlay.png` | 原图叠加 Grad-CAM 热力图 |
| `gradcam_<model>_<时间戳>/*_heatmap.png` | Grad-CAM 热力图 |
| `gradcam_<model>_<时间戳>/report.md` | 样本、目标输出、score、真实/预测角度、误差与相对路径 |

Grad-CAM 作为训练可视化模块的解释算法层实现，不经过 gRPC/FastAPI 推理服务，也不修改 `proto/infer.proto`。实现上 `MyFlows/utils/gradcam.py` 保留核心算法，`tools/explain/` 拆分模型构建、目标选择和报告输出，`tools/explain_donkey_gradcam.py` 保留 CLI 编排。`score` 是 `target-output` 对应的模型原始预测值，不是准确率或置信度；推荐复用 test split 和固定油门 0.2 口径生成材料。
