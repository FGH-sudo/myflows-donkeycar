# Grad-CAM 可视化报告

- model_type: `resnet`
- checkpoint: `D:\DL\testmyflow\mycar\models\myflow_resnet18_best`
- layer: `layer4`
- split: `test` via `D:/DL/testmyflow/mycar/logs/resnet18_split.json`
- fixed_throttle: `0.2` forced
- samples: `8`
- TensorBoard: `D:\DL\testmyflow\mycar\logs\tensorboard\gradcam_resnet_20260629_191017`
- score 是 target_output 对应的模型原始预测值，不是准确率或置信度。

| # | image | target_output | score | true_angle | pred_angle | abs_error | overlay |
|---|-------|---------------|-------|------------|------------|-----------|---------|
| 0 | `images\1005_-0.2444.jpg` | `angle` | -0.196087 | -0.244400 | -0.196087 | 0.048313 | `docs\experiments\explainability\gradcam_resnet_20260629_191017\000_1005_-0.2444_overlay.png` |
| 1 | `images\100_-0.0667.jpg` | `angle` | -0.043916 | -0.066700 | -0.043916 | 0.022784 | `docs\experiments\explainability\gradcam_resnet_20260629_191017\001_100_-0.0667_overlay.png` |
| 2 | `images\1014_-0.2000.jpg` | `angle` | -0.194104 | -0.200000 | -0.194104 | 0.005896 | `docs\experiments\explainability\gradcam_resnet_20260629_191017\002_1014_-0.2000_overlay.png` |
| 3 | `images\1027_-0.2222.jpg` | `angle` | -0.159444 | -0.222200 | -0.159444 | 0.062756 | `docs\experiments\explainability\gradcam_resnet_20260629_191017\003_1027_-0.2222_overlay.png` |
| 4 | `images\1030_-0.2444.jpg` | `angle` | -0.151814 | -0.244400 | -0.151814 | 0.092586 | `docs\experiments\explainability\gradcam_resnet_20260629_191017\004_1030_-0.2444_overlay.png` |
| 5 | `images\1033_-0.1778.jpg` | `angle` | -0.136664 | -0.177800 | -0.136664 | 0.041136 | `docs\experiments\explainability\gradcam_resnet_20260629_191017\005_1033_-0.1778_overlay.png` |
| 6 | `images\1043_0.0222.jpg` | `angle` | 0.001472 | 0.022200 | 0.001472 | 0.020728 | `docs\experiments\explainability\gradcam_resnet_20260629_191017\006_1043_0.0222_overlay.png` |
| 7 | `images\1051_0.1778.jpg` | `angle` | 0.193636 | 0.177800 | 0.193636 | 0.015836 | `docs\experiments\explainability\gradcam_resnet_20260629_191017\007_1051_0.1778_overlay.png` |
