# FP32 vs INT8 推理对比报告（未生成）

> 当前文件是占位说明，不包含真实实验结果。运行 `scripts/run_quantize_eval.py` 后会被自动覆盖：

```bash
python scripts/run_quantize_eval.py --fp32 mycar/models/myflow_resnet18_best.onnx --max-samples 500 --device auto
```

## 指标说明

| 列 | 含义 |
|----|------|
| MSE | 转向角回归均方误差 |
| 符号准确率 | `angle_sign_accuracy` |
| 平均延迟 | 单张 ONNX Runtime 推理 ms |
| P99 | 延迟 99 分位 |
| 体积 | 模型文件 MB |

运行后将覆盖本文件并生成 `int8_metrics.json`。
