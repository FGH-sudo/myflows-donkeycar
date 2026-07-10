# VGG 回归与跨框架 Benchmark 结果

生成时间：2026-06-30

## VGG-11 DonkeyCar 回归

命令：

```bash
python -m apps.train.train_vgg_donkey_regression --max-samples 0 --epochs 20 --batch 2 --split-file mycar/logs/resnet18_split.json --dropout 0.2 --initializer xavier_uniform --weight-decay 1e-5 --early-stopping --patience 5 --min-delta 1e-4 --fixed-throttle 0.2 --force-fixed-throttle --summary-once --check-shape --num-workers 2 --device cuda --export-onnx
python -m apps.eval.eval_vgg_donkey_regression --checkpoint mycar/models/vgg11_regression_best --split-file mycar/logs/resnet18_split.json --split test --max-samples 0 --fixed-throttle 0.2 --force-fixed-throttle --device cuda
```

数据划分：复用 ResNet 正式划分文件 `mycar/logs/resnet18_split.json`，source=10000，train=8000，val=1000，test=1000。

训练配置：CUDA，固定油门 0.2，数据增强关闭，Dropout=0.2，L2 weight decay=1e-5，early stopping patience=5，min_delta=1e-4。

训练摘要：

| epoch | train mean loss | val loss |
|---:|---:|---:|
| 1 | 0.0164 | 0.0171 |
| 2 | 0.0162 | 0.0166 |
| 3 | 0.0162 | 0.0167 |
| 4 | 0.0162 | 0.0166 |
| 5 | 0.0162 | 0.0165 |
| 6 | 0.0162 | 0.0166 |
| 7 | 0.0162 | 0.0166 |

训练设置为最多 20 epoch；由于验证集 loss 达到 early stopping 条件，实际在 epoch 7 结束。best checkpoint 来自 epoch 5，checkpoint 记录的 best loss 为 0.016548，已导出 `mycar/models/vgg11_regression_best.onnx`。

test 指标：

| metric | value |
|---|---:|
| angle_mse | 0.031562 |
| throttle_mse | 0.000000 |
| angle_mae | 0.141485 |
| throttle_mae | 0.000000 |
| overall_mse | 0.015781 |
| angle_sign_accuracy | 0.5800 |
| near_zero_accuracy | 1.0000 |
| mean_abs_angle_true | 0.1413 |
| mean_abs_angle_pred | 0.0030 |
| samples | 1000 |

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
