# 完整训练过程的资源与阶段指标

2000 条真实样本，其中 200 条用于验证，训练 3 轮。cifar stem、w64、batch 2、FP32、种子 1。两种数据加载方式的各轮训练 loss、验证 loss 与角度 MAE 逐值相同。

![完整训练过程的 12 项指标](trends.png)

虚线标明每轮结束。表中列出 workers=0 从启动到结束的全部有效样本，阶段耗时按选定步采样。

| 字段 | 单位 | 样本数 | 中位数 | P95 | 最大值 |
| --- | --- | --- | --- | --- | --- |
| gpu_utilization_pct | % | 357 | 74.000 | 94.000 | 100.000 |
| gpu_device_memory_mib | MiB | 357 | 4663.758 | 4737.789 | 4752.383 |
| cupy_pool_used_mib | MiB | 2700 | 1356.280 | 1356.280 | 1356.280 |
| cupy_pool_total_mib | MiB | 2700 | 1693.819 | 1693.819 | 1693.819 |
| gpu_device_memory_peak_mib | MiB | 357 | 4752.383 | 4752.383 | 4752.383 |
| used_peak_mib | MiB | 2700 | 1356.280 | 1356.280 | 1356.280 |
| gpu_temperature_c | deg C | 357 | 65.000 | 69.000 | 70.000 |
| gpu_power_w | W | 357 | 66.108 | 79.506 | 99.097 |
| gpu_power_limit_w | W | 357 | 80.000 | 80.000 | 80.000 |
| cpu_utilization_pct | % | 356 | 9.300 | 18.825 | 24.700 |
| step_total_ms | ms | 2700 | 46.115 | 72.418 | 296.390 |
| forward_ms | ms | 270 | 15.313 | 16.345 | 200.095 |
| backward_ms | ms | 270 | 25.051 | 26.402 | 90.830 |
| optimizer_ms | ms | 270 | 1.805 | 2.142 | 5.894 |
| data_load_ms | ms | 2700 | 2.059 | 29.050 | 43.356 |
| samples_per_sec_total | samples/s | 2700 | 43.369 | 45.101 | 46.787 |

流水线比较剔除前 10 个 step，数据读取占比为数据读取总耗时除以相同步的总耗时。

| loader workers | 稳态步数 | 数据占比 | 中位读取 ms | samples/s |
| --- | --- | --- | --- | --- |
| 0 | 2690 | 16.86% | 2.056 | 37.940 |
| 2 | 2690 | 2.08% | 0.238 | 43.040 |

原始目录为 `docs/experiments/semester_2026_fall/monitoring/trends-001/`，每组含 config、metrics、resources、result、TensorBoard 与格式检查。summary.json 的 SHA-256 为 `be909b2d7c397b6abf509721e9a72ebe29757cb8f39540e3d01950ac28f2d45b`。

复现命令为 `python -m benchmark.monitor_training --out runs/stage3/trends-repro --samples 2000 --val-size 200 --epochs 3 --warmup 10`。
