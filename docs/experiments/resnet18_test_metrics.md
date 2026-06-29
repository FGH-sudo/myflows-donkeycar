# ResNet-18 Test Evaluation Metrics

Command:

```powershell
python -m apps.eval.eval_myflows_donkey_onnx --checkpoint mycar/models/myflow_resnet18_best.onnx --split-file mycar/logs/resnet18_split.json --split test --max-samples 0 --fixed-throttle 0.2 --force-fixed-throttle --device cuda
```

Dataset split: `test`, samples=1000, batch=1.

| metric | value |
|---|---:|
| angle_mse | 0.001698 |
| throttle_mse | 0.000399 |
| angle_sign_accuracy | 0.9470 |
| near_zero_accuracy | 0.7075 |
| mean_abs_angle_true | 0.1413 |
| mean_abs_angle_pred | 0.1435 |
| runtime_s | 18.18 |

Log: `mycar/logs/resnet18_test_eval_20260629.log`
Figure: `docs/experiments/resnet18_test_metrics.png`
