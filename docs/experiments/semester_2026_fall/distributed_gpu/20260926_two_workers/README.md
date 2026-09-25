# 双 Worker 能否超过单进程：2026-09-26 实测研究

结论：当前 Windows WDDM + 一张 RTX 4060 Laptop GPU、global batch=128、同步 FP32 Adam 条件下，这轮实测的 PS/Ring 双 Worker 均没有超过单进程。PS 的事件通知显著改善 MNIST，但 ResNet 没有得到相同收益，因此保留原轮询为默认，新增显式 `ps_wait_strategy="notify"` 选项。

## 主要结果：交错对照

以下为完整第 1 个 epoch 的 3 次独立进程重复，中位数及全部范围。每轮轮换配置顺序，所有 GPU 实验串行；未改变样本数、更新次数、精度、优化器或每步一致性检查。

|任务|配置|epoch 中位/s|范围/s|单进程时间 / 当前时间|
|---|---|---:|---:|---:|
|MNIST MLP|单进程|1.352|1.348–1.368|1.000×|
|MNIST MLP|PS 2，notify|3.979|3.959–4.011|0.340×|
|MNIST MLP|Ring 2|3.404|3.375–3.589|0.397×|
|MNIST MLP|PS 2，20 ms 轮询|11.813|11.807–11.864|0.114×|
|ResNet18 道路回归|单进程|9.643|9.623–9.763|1.000×|
|ResNet18 道路回归|PS 2，notify|15.700|15.631–15.979|0.614×|
|ResNet18 道路回归|Ring 2|13.994|13.927–14.085|0.689×|
|ResNet18 道路回归|PS 2，20 ms 轮询|14.929|14.909–15.170|0.646×|

加速比大于 1 才表示超过单进程。上述所有分布式配置均小于 1。

- MNIST MLP：通知方式相对旧轮询耗时下降 66.32%（11.813 → 3.979 s）。
- ResNet18 道路回归：通知方式相对旧轮询耗时增加 5.17%（14.929 → 15.700 s）。

## 公平性、环境与证据

- 硬件：一张 RTX 4060 Laptop GPU，8 GB，Windows WDDM，驱动 591.74。Worker 共享同一设备；PS 是独立 CPU 聚合进程；通信均为 gRPC/protobuf。
- 沿用 `20260919_optimized` 冻结配置、准备数据、初始 checkpoint、BN 文件、seed=0 和 FP32 Adam。MNIST 为 784→64→10，390 次更新、49,920 样本；ResNet base_width=16、冻结 BN moments，62 次更新、7,936 样本。global batch 都是 128。
- epoch 时间排除启动、准备、预热、评估和保存；三种架构都开启相同 CUDA Event 阶段同步、逐步 JSONL 和资源监控。阶段统计是各 rank 累计后取最大值，存在重叠，不能把列相加当端到端时间。
- 共 60 次完整性能运行：原代码基线 18 次；通知候选 18 次；交错对照 24 次。全部保留，没有剔除异常样本。这里的性能运行只有 1 epoch，没有重新执行原先的 MNIST 10 epoch / ResNet 5 epoch 收敛验收。
- 原始档案在仓库根目录 `runs/two_workers_20260926/`，包含 config、manifest/source hashes、逐 rank 步骤、最终 NPZ、资源记录和日志；大型原始档案继续留在本地。Git 收录本报告、紧凑汇总、校验结果和复现脚本。
- [summary.json](summary.json) 保留各阶段的中位数、范围与所有单次阶段时间；[resources_summary.json](resources_summary.json) 保留源代码哈希、状态摘要和资源统计。资源每秒采样，整机 CPU 与整卡 GPU 包括后台活动；很短的训练区间可能没有采样点。

## PS 改动及为何只作为可选项

`PSEngine._poll` 原先每次 WAITING 后休眠最多 20 ms；gRPC 服务端的 50 ms 长轮询窗口并没有消除内部休眠。本次增加共用 RLock 的 Condition，在梯度齐备、更新完成、abort 和 stop 时通知等待者。检查与等待共用同一把锁，防止丢失通知，并允许等待期间的心跳及其他 RPC 获取锁。原超时、请求重试、加权平均、版本控制和状态校验继续生效。

MNIST 单步计算很小，轮询空等成为主要负担。ResNet 的交错对照中，通知减少了同步/确认等待，但 forward/backward 的 CUDA Event 与主机耗时增大；共享 GPU 的调度和争用是待验证解释。本轮没有重新采集 Nsight 时间线，不能把调度原因写成已证明。结果不支持把 notify 设为所有任务的默认。

最终配置默认仍为 `ps_wait_strategy="poll"`。需要试用通知方式时，在实验 JSON 中加入 `"ps_wait_strategy": "notify"`，或统一入口使用 `--ps-wait-strategy notify`。MNIST 可以优先选择 notify；ResNet 应继续按同条件端到端时间选策略。

交错对照的 `ps-polling-grpc-2` 是 benchmark 专用旧等待函数消融；函数 AST 已与 MyFlows `6091f20` 的原 `_poll` 核对一致。它只替换等待函数，其他代码与 notify 相同，运行时入口与源码哈希保存在各 run 的 `benchmark_variant.json`。最终加入策略选择后，生产代码也保留相同旧等待逻辑。

## Ring 与共享单卡的剩余瓶颈

|任务/模式|前向+反向 wall/s|梯度同步/s|优化器 wall/s|梯度 D2H/s|送回设备/s|状态摘要/s|更新确认含摘要/s|
|---|---:|---:|---:|---:|---:|---:|---:|
|MNIST MLP / 单进程|0.603|0.000|0.456|0.000|0.063|0.000|0.000|
|MNIST MLP / PS 2，notify|0.723|1.425|0.555|0.057|0.127|0.357|0.805|
|MNIST MLP / Ring 2|0.740|0.824|0.546|0.059|0.163|0.360|0.798|
|MNIST MLP / PS 2，20 ms 轮询|0.811|6.574|0.620|0.056|0.128|0.358|6.014|
|ResNet18 道路回归 / 单进程|6.971|0.000|1.114|0.000|0.103|0.000|0.000|
|ResNet18 道路回归 / PS 2，notify|8.296|3.377|1.460|0.156|0.219|0.957|1.308|
|ResNet18 道路回归 / Ring 2|8.300|1.741|1.616|0.158|0.224|0.958|1.074|
|ResNet18 道路回归 / PS 2，20 ms 轮询|7.111|3.743|1.453|0.153|0.217|0.956|1.814|

Ring 已经采用条件变量等待；其两个邻居通信阶段、CPU 上的梯度归约、梯度 D2H/H2D、每个 rank 各自执行的 Adam，以及每步对参数/BN buffer/Adam v、s 做完整摘要仍有成本。两个 Worker 共用一张 GPU，拆成两个 64 样本分片没有增加 GPU 总算力，也不能预设计算耗时会减半。

下一步优先研究保留训练语义的优化：合并小张量传输与同步、复用扁平缓冲区、减少 CPU 拷贝、将状态传回 CPU 的操作批量化，并验证优化器融合。所有共用优化也必须应用到单进程控制组。只有最终训练完成的同步边界保持一致，降低逐阶段测量开销的比较才有效。以上尚未实现，不能把预计节省相加推导成已达成加速。

如果目标是可靠的 PS/Ring 2 Worker 强扩展，优先准备真正的两张 GPU：先实现每个 rank 的设备编号绑定，再测两个分片并行计算与归约。当前 `TrainSession` 将 cuda 字符串归一为 cuda，设备层没有完整 rank→GPU 编号映射，不能仅接入第二张卡就宣称已有多 GPU 支持。GPU 常驻归约/通信与计算重叠是后续方案，也要在硬件上验证。

NVIDIA 文档说明 MPS 可帮助未占满 GPU 的多进程工作负载，但系统支持范围不含原生 Windows；它不是本机可直接打开的加速开关。迁移环境后的收益也未在本轮验证。参见 [MPS 使用条件](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html) 与 [MPS 架构](https://docs.nvidia.com/deploy/mps/architecture.html)。

## 波动与正确性边界

顺序执行的候选阶段曾出现未改动的 MNIST Ring 达 9.520 s，同时整机 CPU 占用的全运行中位数达 79.3%；当时可观察到后台安装和系统扫描。候选 ResNet 也整体变慢。因此主要结论使用随后轮换旧/新策略的交错对照，原顺序实验仍完整公开如下，未将它们删除或替换。

|阶段|任务|模式|中位/s|范围/s|
|---|---|---|---:|---:|
|baseline|ResNet18 道路回归|ps-grpc-2|14.375|13.660–15.297|
|baseline|ResNet18 道路回归|ring-grpc-2|13.169|12.612–13.915|
|baseline|ResNet18 道路回归|single-1|8.785|8.221–11.087|
|baseline|MNIST MLP|ps-grpc-2|12.539|11.997–12.949|
|baseline|MNIST MLP|ring-grpc-2|3.383|3.244–3.514|
|baseline|MNIST MLP|single-1|1.179|1.167–1.536|
|notified|ResNet18 道路回归|ps-grpc-2|15.912|15.632–19.217|
|notified|ResNet18 道路回归|ring-grpc-2|14.085|14.077–16.002|
|notified|ResNet18 道路回归|single-1|9.793|9.773–15.634|
|notified|MNIST MLP|ps-grpc-2|4.467|4.359–4.975|
|notified|MNIST MLP|ring-grpc-2|3.734|3.286–9.520|
|notified|MNIST MLP|single-1|1.527|1.372–1.582|

- [notified_audit.json](notified_audit.json)：18 对原基线/候选运行；[interleaved_audit.json](interleaved_audit.json)：24 个交错运行分别匹配原基线。核对配置、数据、初始摘要、样本数、正常退出、rank 一致性、最终 NPZ 全部参数/BN buffer/Adam 状态以及逐 rank 每步 loss。最终要求逐元素完全相同，最大状态差为 0。
- 最终全量回归 236 项通过；通知专项包括条件唤醒、超时后重试、中止与停止，以及真实双 Worker 的策略等价检查。
- 上述比的是同一分片方式在等待机制改动前后的结果。原 ResNet 独立整批 vs 分片的严格梯度轨迹超差仍然保留；本报告没有修复或重新宣布通过该项。

## 复现

需要本地 `20260919_optimized` 的准备数据、初始 NPZ 与冻结 BN 文件。以下命令创建新目录，不覆盖本次证据：

```powershell
.\.venv\Scripts\python.exe -m benchmark.investigate_two_workers --out runs/two_workers_recheck --phase interleaved --include-polling-ablation
.\.venv\Scripts\python.exe tools/run_tests.py --scope all
```

已有本次完整档案时可以重建一致性审计：

```powershell
.\.venv\Scripts\python.exe -m benchmark.audit_two_worker_results --out runs/two_workers_20260926 --candidate notified
.\.venv\Scripts\python.exe -m benchmark.audit_two_worker_results --out runs/two_workers_20260926 --candidate interleaved
```
