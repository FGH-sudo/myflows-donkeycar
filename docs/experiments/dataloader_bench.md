# DataLoader Producer-Consumer Benchmark

| num_workers | batches | batch_size | total_images | train_ms | elapsed_s | img_s | ms_per_batch | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 500 | 8 | 4000 | 0.0 | 29.8803 | 133.87 | 59.76 | 1.00 |
| 2 | 500 | 8 | 4000 | 0.0 | 6.3581 | 629.12 | 12.72 | 4.70 |
| 4 | 500 | 8 | 4000 | 0.0 | 9.0430 | 442.33 | 18.09 | 3.30 |

说明：`num_workers=0` 为主进程同步读取；`num_workers>0` 使用 MyFlows `MultiprocessDataLoader`，worker 进程作为生产者执行图片 IO/解码/预处理，主进程作为消费者按 batch 取数。
