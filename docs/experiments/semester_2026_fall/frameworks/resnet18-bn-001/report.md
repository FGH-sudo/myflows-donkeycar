# ResNet18 同条件跨框架比较

同一台 GPU 上，以 2000 帧道路图像、8 的 batch 和相同的 FP32 初值比较 MyFlows 与 PyTorch。固定时间段划分得到 1765 个训练样本和 200 个验证样本。每轮共享批次清单并共同丢弃训练尾批。

两端均为 ImageNet stem、w64、无卷积 bias、带 padding 的 maxpool、训练模式 BN 和两输出回归头。Adam 的 lr=0.001、β=(0.9, 0.999)、ε=1e−8。预处理、样本顺序与 NPZ 初值的 SHA-256 在 fixture/protocol.json。

| seed | 输出最大绝对差 | loss 绝对差 | 全部梯度相对 L2 | 梯度最大绝对差 | 一步后参数最大绝对差 |
| --- | --- | --- | --- | --- | --- |
| 1 | 3.13e-06 | 2.38e-07 | 0.000339 | 0.000839 | 0.002 |
| 2 | 2.74e-06 | 2.38e-07 | 3.49e-06 | 5.1e-06 | 0.0016 |
| 3 | 3.46e-06 | 2.38e-07 | 0.000414 | 0.000636 | 0.002 |

alignment-s*.json 保存每个张量的形状、最大绝对差、相对 L2 和带 1e−6 分母下限的逐元素最大相对差。

| 模式 | 框架 | step ms | samples/s | GPU % | 整卡峰值 MiB | 分配器已用峰值 MiB | CPU % | RSS 峰值 MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | myflows | 14.20 | 528.10 | 92.00 | 2166.39 | 820.02 | 8.67 | 599.09 |
| fixed | torch | 21.99 | 358.84 | 82.50 | 1710.39 | 419.26 | 10.00 | 1247.01 |
| fixed | torch-autotuned | 17.42 | 469.14 | 75.00 | 1554.39 | 302.71 | 11.27 | 1299.93 |
| e2e | myflows | 15.12 | 491.13 | 88.00 | 2172.39 | 820.02 | 8.63 | 703.88 |
| e2e | torch | 24.84 | 331.31 | 80.00 | 1710.39 | 419.26 | 10.87 | 1346.08 |
| e2e | torch-autotuned | 17.89 | 442.26 | 75.00 | 1554.39 | 302.27 | 9.40 | 1395.74 |

fixed 重用已传入设备的 batch。e2e 包含从共同 NPY 取 batch 与 H2D。每步在计时边界同步。各计时进程先预热，再从共同初值重建模型、BN 和优化器，并重置批次顺序。默认 torch 行启用 cuDNN deterministic，autotuned 行单独启用 cuDNN benchmark。所有 torch 行关闭 TF32。时间表对共同种子取中位数。

| 框架 | 实际训练样本/seed | 最终验证角度 MAE 各 seed | mean ± sample std | 末轮训练 / 验证 MSE 均值 | 初始化 / 训练墙钟均值 s |
| --- | --- | --- | --- | --- | --- |
| myflows | 5280 | 0.02531, 0.09848, 0.06813 | 0.06397 ± 0.03676 | 0.00428259 / 0.00571937 | 1.03 / 11.81 |
| torch | 5280 | 0.03279, 0.11511, 0.10433 | 0.08408 ± 0.04474 | 0.00466372 / 0.00633887 | 8.14 / 15.11 |

MyFlows 的卷积累加顺序、FP64 通道规约和显式舍入 Adam 与 cuDNN、PyTorch 的内核实现有差异。接近零的激活可能在 ReLU 产生不同的门控，微小的梯度符号差异也会被 Adam 首步放大。PyTorch 显存取分配器精确高水位，CuPy 已用峰值取阶段边界采样，两者与统一 NVML 设备采样分列。CPU 为主机利用率，RSS 为进程树采样峰值。

原始归档位于 `docs/experiments/semester_2026_fall/frameworks/resnet18-bn-001/`。独立进程的 result.json 包含初始化、预热、训练总时间和各轮质量。共同图像预处理耗时 1.88 s。summary.json SHA-256 为 `ec372e0e36acd4ef493dd8dcd6f61d4b9d1eba5363954fcf6e0beeb92ab3f89c`。

复现命令为 `python -m benchmark.compare_frameworks_resnet --out runs/stage3/frameworks-repro --samples 2000 --batch 8 --val-size 200 --epochs 3 --seeds 1,2,3 --timed-steps 80 --warmup 10 --graph-opt`。
