# 训练时间线分析

完整 step 总时间 142.306 ms；空闲阈值 0.100 ms；未映射来源的空闲片段 0。

GPU 忙区间是 kernel、memcpy、memset 在所有 stream 上的并集。各阶段是 CPU NVTX 分区，没有阶段标记的部分明确记为 overhead；阶段墙钟总和等于 step 墙钟总和。GPU 与 CPU 异步，空闲段的主机源码关联用于定位发射/等待位置，不声称为某个 kernel 的因果归属。

| 阶段 | wall ms | GPU idle ms | 占全部 step 时间 | 阶段内 idle 占比 |
| --- | --- | --- | --- | --- |
| overhead | 15.004 | 3.583 | 10.54% | 23.88% |
| data | 50.390 | 50.269 | 35.41% | 99.76% |
| forward | 31.297 | 21.914 | 21.99% | 70.02% |
| backward | 42.302 | 25.541 | 29.73% | 60.38% |
| optimizer | 3.313 | 0.452 | 2.33% | 13.65% |

| step | kernel 数 | 平均 kernel µs | GPU idle ms |
| --- | --- | --- | --- |
| 5 | 494 | 13.961 | 18.564 |
| 6 | 494 | 14.157 | 20.357 |
| 7 | 494 | 17.133 | 29.068 |
| 8 | 494 | 19.233 | 19.300 |
| 9 | 494 | 14.076 | 14.470 |

最长空闲片段（完整表见 gaps.csv；分割后的片段共享 gap_id）：

| step | 阶段 | 时长 ms | 节点 | 源码 |
| --- | --- | --- | --- | --- |
| 7 | data | 18.941 | — | apps/train/train_myflows_donkey.py:1171 |
| 5 | data | 9.485 | — | apps/train/train_myflows_donkey.py:1171 |
| 6 | data | 9.431 | — | apps/train/train_myflows_donkey.py:1171 |
| 8 | data | 5.495 | — | apps/train/train_myflows_donkey.py:1171 |
| 9 | data | 5.051 | — | apps/train/train_myflows_donkey.py:1171 |
| 5 | backward | 0.635 | — | apps/train/train_myflows_donkey.py:178 |
| 7 | backward | 0.628 | — | apps/train/train_myflows_donkey.py:178 |
| 6 | backward | 0.628 | — | apps/train/train_myflows_donkey.py:178 |
| 8 | overhead | 0.617 | — | apps/train/train_myflows_donkey.py:1171 |
| 6 | backward | 0.616 | — | apps/train/train_myflows_donkey.py:178 |
| 6 | backward | 0.439 | — | apps/train/train_myflows_donkey.py:178 |
| 7 | data | 0.396 | — | apps/train/train_myflows_donkey.py:1141 |
| 6 | data | 0.359 | — | apps/train/train_myflows_donkey.py:1141 |
| 8 | data | 0.354 | — | apps/train/train_myflows_donkey.py:1141 |
| 6 | overhead | 0.353 | — | apps/train/train_myflows_donkey.py:1171 |
| 9 | data | 0.348 | — | apps/train/train_myflows_donkey.py:1141 |
| 7 | overhead | 0.340 | — | apps/train/train_myflows_donkey.py:1171 |
| 7 | overhead | 0.317 | — | apps/train/train_myflows_donkey.py:1171 |
| 8 | overhead | 0.303 | — | apps/train/train_myflows_donkey.py:1171 |
| 9 | overhead | 0.298 | — | apps/train/train_myflows_donkey.py:1171 |

最长累计 kernel（完整表见 top_kernels.csv）：

| kernel | 次数 | 总 ms | 平均 µs |
| --- | --- | --- | --- |
| mf_adam_step | 160 | 8.492 | 53.077 |
| im2col_forward | 100 | 2.875 | 28.755 |
| gemm_split_reduce | 90 | 2.850 | 31.667 |
| gemm_register_s_nn_v1 | 65 | 2.418 | 37.206 |
| cupy_add__float32_float32_float32 | 355 | 2.335 | 6.579 |
| gemm_register_s_nt_v1 | 50 | 2.118 | 42.357 |
| mf_bn_grad_partials | 100 | 2.050 | 20.497 |
| gemm_register_s_tn_v1 | 50 | 1.522 | 30.450 |
| col2im_backward | 100 | 1.444 | 14.436 |
| gemm_register_w_nt_v1 | 25 | 1.433 | 57.325 |
| mf_bn_stats_partials | 50 | 1.374 | 27.478 |
| gemm_register_w_nn_v1 | 25 | 1.268 | 50.701 |
| gemm_register_h_nt_v1 | 20 | 1.091 | 54.532 |
| gemm_register_w_tn_v0 | 5 | 1.047 | 209.401 |
| gemm_register_h_tn_v1 | 25 | 0.916 | 36.658 |
| gemm_register_w_tn_v1 | 20 | 0.799 | 39.952 |
| conv_grad_rows | 100 | 0.559 | 5.592 |
| mf_relu_backward_acc | 85 | 0.550 | 6.465 |
| mf_bn_dx_acc_hilo | 100 | 0.483 | 4.828 |
| mf_bn_normalize_hilo | 100 | 0.395 | 3.954 |

本地可复现图：`timeline.png`（不纳入 Git，与 capture.sqlite 一同交付）。

原始采集位于 `docs/experiments/semester_2026_fall/monitoring/timeline-001/capture.sqlite`。manifest.json 保存双仓库源码状态、采集和导出命令。checksums.json 可用于核对 SQLite、来源映射、分析数据和图像。

本次使用 batch 2，样本数 32，冻结 3 个阶段，采集全局步 5 至 9。

复现采集可运行 `python -m benchmark.profile_train --out runs/stage3/profile-repro --profile-steps 5:9 --max-samples 32 --batch 2 --pretrained imagenet --freeze-stages 3`。
