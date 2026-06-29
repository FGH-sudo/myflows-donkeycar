# VGG 回归与跨框架 Benchmark 结果

生成时间：2026-06-29

## VGG-11 DonkeyCar 回归

命令：

```bash
python -m apps.train.train_vgg_donkey_regression --max-samples 500 --epochs 2 --batch 2 --val-size 100 --test-size 100 --split-seed 42 --split-out mycar/logs/vgg_split.json --dropout 0.2 --initializer xavier_uniform --weight-decay 1e-5 --early-stopping --patience 3 --min-delta 1e-4 --fixed-throttle 0.2 --device cuda --export-onnx
python -m apps.eval.eval_vgg_donkey_regression --checkpoint mycar/models/vgg11_regression_best --split-file mycar/logs/vgg_split.json --split test --max-samples 0 --fixed-throttle 0.2 --device cuda
```

数据划分：500 张子集，train=300，val=100，test=100。

训练摘要：

| epoch | train mean loss | val loss |
|---:|---:|---:|
| 1 | 0.0354 | 0.0294 |
| 2 | 0.0284 | 0.0306 |

best checkpoint 来自 epoch 1，已导出 `mycar/models/vgg11_regression_best.onnx`。

test 指标：

| metric | value |
|---|---:|
| angle_mse | 0.055677 |
| throttle_mse | 0.000009 |
| angle_mae | 0.200305 |
| throttle_mae | 0.002997 |
| overall_mse | 0.027843 |
| angle_sign_accuracy | 0.6200 |
| mean_abs_angle_true | 0.2016 |
| mean_abs_angle_pred | 0.0069 |

## VGG-11 跨框架 Benchmark

命令：

```bash
python benchmark/compare_frameworks.py --data mycar/data --samples 64 --epochs 2 --batch 8 --device cuda
python benchmark/plot_compare.py
```

产物：

- `benchmark/results.csv`
- `docs/experiments/framework_compare.png`

结果：

| framework | device | time_s | peak_mb | final_loss |
|---|---|---:|---:|---:|
| MyFlows | cuda | 56.5726 | 2824.68 | 0.032842 |
| PyTorch | cuda | 92.8103 | 4815.06 | 0.035160 |
| PaddlePaddle | gpu:0 | 29.7055 | 6703.97 | 0.066835 |

说明：本轮保留 MyFlows、PyTorch 与 PaddlePaddle，并使用严格 CUDA 口径，不混入 CPU 回退结果。当前 Windows Python 环境已安装 `paddlepaddle-gpu==3.3.1`（cu126 wheel），三方均在 RTX 4060 上完成同一 DonkeyCar VGG11 回归 benchmark；`framework_compare.png` 展示 time 与 RSS 两个维度。
