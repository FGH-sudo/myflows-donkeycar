# 可解释性可视化资料目录

本目录保存 Grad-CAM 输出，供期末报告引用。

生成示例：

```bash
python -m tools.explain_donkey_gradcam --model-type resnet --checkpoint mycar/models/myflow_resnet18_best --data mycar/data --max-samples 8 --device auto
```

输出内容：

| 文件 | 说明 |
|------|------|
| `*_overlay.png` | 原图叠加 Grad-CAM 热力图 |
| `*_heatmap.png` | Grad-CAM 热力图 |
| `report.md` | 样本、目标输出、score 与图片路径 |

Grad-CAM 作为训练可视化模块的解释算法层实现，不经过 gRPC/FastAPI 推理服务，也不修改 `proto/infer.proto`。实现上 `MyFlows/utils/gradcam.py` 保留核心算法，`tools/explain/` 拆分模型构建、目标选择和报告输出，`tools/explain_donkey_gradcam.py` 保留 CLI 编排。
