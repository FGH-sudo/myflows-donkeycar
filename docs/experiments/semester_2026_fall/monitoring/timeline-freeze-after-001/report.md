# 训练时间线分析

完整 step 总时间 144.791 ms；空闲阈值 0.100 ms；未映射来源的空闲片段 0。

GPU 忙区间是 kernel、memcpy、memset 在所有 stream 上的并集。各阶段是 CPU NVTX 分区，没有阶段标记的部分明确记为 overhead；阶段墙钟总和等于 step 墙钟总和。GPU 与 CPU 异步，空闲段的主机源码关联用于定位发射/等待位置，不声称为某个 kernel 的因果归属。

| 阶段 | wall ms | GPU idle ms | 占全部 step 时间 | 阶段内 idle 占比 |
| --- | --- | --- | --- | --- |
| overhead | 6.802 | 2.358 | 4.70% | 34.67% |
| data | 82.181 | 82.181 | 56.76% | 100.00% |
| forward | 32.753 | 20.772 | 22.62% | 63.42% |
| backward | 20.047 | 12.085 | 13.85% | 60.28% |
| optimizer | 3.008 | 0.794 | 2.08% | 26.38% |

| step | kernel 数 | 平均 kernel µs | GPU idle ms |
| --- | --- | --- | --- |
| 5 | 373 | 13.323 | 17.381 |
| 6 | 373 | 13.344 | 17.370 |
| 7 | 373 | 13.264 | 17.252 |
| 8 | 373 | 13.709 | 34.885 |
| 9 | 373 | 14.375 | 31.302 |

最长空闲片段（完整表见 gaps.csv；分割后的片段共享 gap_id）：

| step | 阶段 | 时长 ms | 节点 | 源码 |
| --- | --- | --- | --- | --- |
| 8 | data | 26.146 | — | apps/train/train_myflows_donkey.py:1192 |
| 9 | data | 23.498 | — | apps/train/train_myflows_donkey.py:1192 |
| 7 | data | 10.254 | — | apps/train/train_myflows_donkey.py:1192 |
| 5 | data | 10.169 | — | apps/train/train_myflows_donkey.py:1192 |
| 6 | data | 10.150 | — | apps/train/train_myflows_donkey.py:1192 |
| 6 | data | 0.434 | — | apps/train/train_myflows_donkey.py:1162 |
| 8 | data | 0.406 | — | apps/train/train_myflows_donkey.py:1162 |
| 9 | data | 0.396 | — | apps/train/train_myflows_donkey.py:1162 |
| 5 | data | 0.374 | — | apps/train/train_myflows_donkey.py:1162 |
| 7 | data | 0.356 | — | apps/train/train_myflows_donkey.py:1162 |
| 8 | overhead | 0.355 | — | apps/train/train_myflows_donkey.py:1192 |
| 6 | overhead | 0.316 | — | apps/train/train_myflows_donkey.py:1192 |
| 7 | overhead | 0.300 | — | apps/train/train_myflows_donkey.py:1192 |
| 5 | overhead | 0.280 | — | apps/train/train_myflows_donkey.py:1192 |
| 9 | overhead | 0.262 | — | apps/train/train_myflows_donkey.py:1192 |
| 8 | backward | 0.256 | — | apps/train/train_myflows_donkey.py:178 |
| 5 | forward | 0.192 | node:forward:1:ChannelNormalize_Op | MyFlows/ops/pooling.py:80 |
| 7 | forward | 0.143 | node:forward:1:ChannelNormalize_Op | MyFlows/ops/pooling.py:80 |
| 8 | forward | 0.127 | node:forward:1:ChannelNormalize_Op | MyFlows/ops/pooling.py:80 |
| 8 | forward | 0.114 | node:forward:8:MaxPool2d_Op | MyFlows/ops/convolution.py:776 |

最长累计 kernel（完整表见 top_kernels.csv）：

| kernel | 次数 | 总 ms | 平均 µs |
| --- | --- | --- | --- |
| mf_adam_step | 160 | 6.513 | 40.707 |
| im2col_forward | 100 | 2.870 | 28.698 |
| gemm_register_s_nt_v1 | 50 | 2.108 | 42.163 |
| cupy_add__float32_float32_float32 | 200 | 1.801 | 9.007 |
| gemm_split_reduce | 60 | 1.633 | 27.215 |
| gemm_register_s_tn_v1 | 50 | 1.533 | 30.663 |
| gemm_register_w_nt_v1 | 25 | 1.428 | 57.104 |
| gemm_register_w_nn_v1 | 25 | 1.275 | 51.005 |
| gemm_register_h_nt_v1 | 20 | 1.017 | 50.869 |
| gemm_register_s_nn_v1 | 15 | 0.818 | 54.500 |
| mf_bn_grad_partials | 50 | 0.662 | 13.244 |
| mf_bn_stats_partials | 50 | 0.488 | 9.768 |
| mf_bn_normalize_hilo | 100 | 0.386 | 3.864 |
| conv_epilogue_nchw | 100 | 0.261 | 2.607 |
| gemm_register_h_nt_v0 | 5 | 0.250 | 49.932 |
| col2im_backward | 40 | 0.238 | 5.942 |
| cupy_maximum__float_float32_float32 | 85 | 0.227 | 2.669 |
| mf_bn_stats_finalize | 50 | 0.206 | 4.113 |
| cupy_copy__float32_float32 | 105 | 0.202 | 1.919 |
| cupy_mean | 10 | 0.185 | 18.496 |

本地可复现图：`timeline.png`（不纳入 Git，与 capture.sqlite 一同交付）。

原始采集位于 `docs/experiments/semester_2026_fall/monitoring/timeline-freeze-after-001/capture.sqlite`。manifest.json 保存双仓库源码状态、采集和导出命令。checksums.json 可用于核对 SQLite、来源映射、分析数据和图像。

本次使用 batch 2，样本数 32，冻结 3 个阶段，采集全局步 5 至 9。

复现采集可运行 `python -m benchmark.profile_train --out runs/stage3/profile-repro --profile-steps 5:9 --max-samples 32 --batch 2 --pretrained imagenet --freeze-stages 3`。
