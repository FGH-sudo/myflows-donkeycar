# DataLoader Producer-Consumer Benchmark

| num_workers | batches | batch_size | total_images | train_ms | elapsed_s | img_s | ms_per_batch | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 500 | 8 | 4000 | 0.0 | 1.7474 | 2289.18 | 3.49 | 1.00 |
| 2 | 500 | 8 | 4000 | 0.0 | 3.0254 | 1322.14 | 6.05 | 0.58 |
| 4 | 500 | 8 | 4000 | 0.0 | 9.3523 | 427.70 | 18.70 | 0.19 |

说明：`num_workers=0` 为主进程同步读取；`num_workers>0` 使用 MyFlows `MultiprocessDataLoader`，worker 进程作为生产者执行图片 IO/解码/预处理，主进程作为消费者按 batch 取数。
