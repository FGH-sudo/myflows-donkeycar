# 实际运行视频

本目录保存答辩用实际运行视频，作为 ResNet-18 推理与静态 INT8 量化推理的演示证据。

| 文件 | 内容 | 时长 | 分辨率 | 大小 |
|---|---|---:|---:|---:|
| `resnet18.mp4` | ResNet-18 FP32 ONNX 实际运行 | 24.60 s | 1280x684 | 11.40 MB |
| `resnet18_int8_static.mp4` | ResNet-18 静态 INT8 ONNX 实际运行 | 17.61 s | 1280x684 | 8.19 MB |

报告和 PPT 中建议与 `docs/experiments/int8_report.md`、`docs/experiments/int8_report.png` 搭配使用：表格/图展示 FP32、动态 INT8、静态 INT8 的指标对比，视频展示 FP32 与静态 INT8 的实际运行效果。
