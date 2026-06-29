# FP32 vs INT8 Inference Report

- Data: `D:\DL\testmyflow\mycar\data`
- Split: `test` via `D:\DL\testmyflow\mycar\logs\resnet18_split.json`
- Samples: 1000
- FP32 model: `D:\DL\testmyflow\mycar\models\myflow_resnet18_best.onnx`

| Variant | angle MSE | throttle MSE | overall MSE | angle sign acc | mean latency (ms) | P99 (ms) | size (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 0.001698 | 0.000399 | 0.001048 | 0.9470 | 3.50 | 4.51 | 42.679 |
| int8_dynamic | 0.001529 | 0.000451 | 0.000990 | 0.9590 | 397.14 | 519.95 | 10.767 |
| int8_static | 0.004257 | 0.015295 | 0.009776 | 0.8480 | 5.29 | 7.24 | 10.772 |

- Raw JSON: `D:\DL\testmyflow\docs\experiments\int8_metrics.json`
- Plot: `D:\DL\testmyflow\docs\experiments\int8_report.png`
