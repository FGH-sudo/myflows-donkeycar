# MyFlows 课程项目实验报告

本报告以训练代码、统一回归、逐次运行记录和本机仿真遥测为证据。验证环境为 Windows 11、Python 3.11 和单张 RTX 4060 Laptop GPU（8 GB，24 SM），CPU 可移植性另在干净的 Ubuntu 22.04 环境测量。各工作包的验收记录见[第三阶段开发记录](stage3_progress.md)，命令与执行频率见[测试计划](test_plan.md)。

## 1. 系统设计

**框架。** MyFlows 负责张量、计算图、层、自动微分、优化器和 checkpoint，NumPy、CuPy 与原生 CUDA 后端共用层接口。计算图在构图后缓存每个节点的梯度需求，冻结参数和不需要梯度的分支既不清零也不反传；梯度在首次写入时分配，已消费的中间梯度随即释放。原生卷积把输入展开为矩阵，经寄存器分块 GEMM 或 cuBLAS 计算，再在融合尾处理中还原为 NCHW 并加 bias 与激活；反向按输入、权重和偏置各自的需求跳过 dX、dW 或 db。BN、残差加与 ReLU 可在图优化时融合，Adam 可在一次 launch 中更新多个 FP32 参数。默认 FP32，规约在 FP64 中按固定顺序累加，不用原子加，原生路径与 CuPy 路径各自逐位可复现。

**CUDA 与分布式。** 第 19 讲的寄存器分块 kernel 推广为三种布局、17 种 tile 几何，输出小而 K 长时按 K 切分并在 FP64 中归约。RTX 4060 Laptop GPU 上 ResNet18 的卷积形状使用实测计划，其他形状用解析规则，也可按形状在线实测。同步参数服务器（Socket JSON / gRPC）与 Ring All-Reduce 把本机多进程 Worker 接入同一训练入口。TF32、FP16/BF16 存储与 Winograd 只作实验后端。

**迁移学习。** `MyFlows/transfer/` 提供预训练模型注册、权重载入与名称/形状检查、按阶段或模块名冻结（参数与 BN 统计量分别控制）、任务头与特征适配器、参数组与分层学习率、按 epoch 的渐进解冻、冻结骨干的特征缓存和 LoRA。两个训练入口都通过 TransferLearningManager 构建迁移任务，result.json 记录冻结级别、来源哈希与冻结前后指纹。

**AutoPilot。** 在固定的数据划分和预算下执行试验，经监控、诊断、候选生成、验证和同种子确认后留下选择依据。TPE 与规则候选共享搜索空间，TrainingState 保存参数、矩估计、随机流、历史最佳和数据进度，故障后校验并恢复。失败按平台、代码、训练、配置四类分栏；算子子 Agent 读取 Nsight Compute 指标判断访存或计算受限；迁移子 Agent 由线性探测给出冻结级别与学习率倍率先验，并在配对种子下检查负迁移。

**可视化训练系统。** 训练按 v1 格式写 config、metrics、resources、rank steps 与 result，独立采样进程每 0.5 s 记录 GPU、CPU 与三种显存口径，TensorBoard 增量读取资源曲线。`--nvtx`、`--profile-steps` 与 `tools/timeline_gaps.py` 把 GPU 空闲段定位到 step、阶段和源码行。控制台通过 REST 与 SSE 显示资源、曲线、四段耗时占比和 12 项指标，Compare 页可加载跨框架归档。架构与模块边界见[系统设计](system_design.md)和[模块设计](module_design.md)。

## 2. 测试过程与结果

日常单元测试覆盖数值、冻结与解冻、断点续训、分布式一致性、监控格式与规则系统。正式实验先在小规模数据上跑通，再锁定协议和代码树；性能改动使用同机交替的独立进程和各自编译的原生库；FP32 数值对照在 6 个 batch 上与 FP64 比较，并单独统计 ReLU 输入异号。

| 项目 | 结果 |
| --- | --- |
| 本机统一回归（Windows，CPU + CUDA） | 阶段验收按测试计划运行 `--repeat 3`：根仓库 9db82b2、MyFlows f9344b5，每轮收集 530 项全部通过，无跳过，三轮耗时 557.7 s、546.4 s、567.8 s。阶段开工时为 443 项 |
| CPU 回归（WSL Ubuntu 22.04 独立 checkout，与 CI 相同依赖与环境变量） | 根仓库 8902daa、MyFlows f9344b5：收集 530 项，463 项通过，67 项带原因跳过（缺 CUDA/CuPy 46 项、缺 torch 12 项、缺固定版本的 dsh 克隆 8 项、缺本地 CIFAR-10 数据 1 项），测试 209.3 s，墙钟 230 s |
| 原生 CUDA 动态检查 | memcheck、racecheck、synccheck 各运行 24 项原生测试，错误与竞争告警均为 0。[报告](experiments/semester_2026_fall/operators/sanitizer-profile-001/report.md) |
| GPU 性能门槛 | 三个 ResNet18 配置与对照代码树交替 6 轮，步时比值 1.003、1.001、0.969，均在 1.10 以内。64 个卷积 GEMM 在 Nsight Compute 锁频计数下加权合计 register/cuBLAS 为 0.977，4 个每步只出现一次的小 dW GEMM 超过 1.20（1.204–1.466），逐形状门槛未全部通过。[门槛](experiments/semester_2026_fall/operators/gate-profile-003/report.md) |
| GitHub 托管 CI | 工作流 `.github/workflows/tests.yml` 已就绪，尚未推送到公开远端运行 |

## 3. 实验结果

| 工作内容 | 实测结果 |
| --- | --- |
| B1、E3 自动调优 | formal-full-002 完成 13 个 trial。人工基线三种子验证转向 MAE 为 0.03436 ± 0.01525。候选缩小学习率，在三个共同确认种子上与基线的配对差为 +0.00033 ± 0.02665，未超出噪声。[正式 campaign](experiments/semester_2026_fall/autopilot/formal-full-002/report.md) |
| B2、B3、E1 算子与分布式 | 原生 Conv2D、MaxPool 的前向与反向以共享输入对照独立 FP64 实现。历史道路任务在同一显卡上，单进程 9.052 s，优化后的 gRPC PS 两 Worker 14.073 s，Ring 两 Worker 13.187 s。多进程测试检查参数同步、奇数尾批、标准断点与异常退出。[算子对拍](experiments/semester_2026_fall/stage1/correctness-final-002/report.md) [PS 与 Ring 实测](experiments/semester_2026_fall/distributed_gpu/20260919_optimized/README.md) |
| B7、S1、S2 监控与冻结 | 9 次交替测量的 0.5 s 采样开销中位数为 0.601%，配对离散度 14.709 个百分点。真实 Nsight 采集中每个空闲段都定位到 step、阶段与源码行。冻结前三段后每步 kernel 数从 494 降至 373，40 组数值试验中 2,012 组梯度及更新逐字节一致。[监控与时间线](experiments/semester_2026_fall/monitoring/README.md) [冻结对照](experiments/semester_2026_fall/monitoring/freeze-001/report.md) |
| E2、S3 迁移组件 | ResNet18、MobileNetV2、AlexNet 的权重对齐通过。完整冻结且预处理确定时，缓存特征与原路径的逐步 loss 和任务头更新一致。LoRA 秩 8 训练 155,664 个参数，单种子小规模验证 MAE 0.05809，只训任务头为 0.22854，冻结 3 段微调为 0.04069；秩 4 合并 ONNX 在 8 个验证帧上的最大绝对输出差为 8.20×10⁻⁷。[权重来源](experiments/semester_2026_fall/e2_transfer/historical-001/report.md) [缓存](experiments/semester_2026_fall/transfer/cache-001/report.md) [低秩微调](experiments/semester_2026_fall/transfer/lora-001/report.md) |
| S4 分类与主任务迁移 | 24 个分类试验中，CIFAR-10 平衡子集的冻结特征提取验证准确率 77.70%，从零训练的小 CNN 31.65%；道路转向分档上分别为 87.50% 与 93.63%。全量回归沿用 7,680/1,000/1,000 划分，共 21 个试验。按各种子最佳验证轮，冻结 4 段为 0.02327 ± 0.00108，人工基线为 0.03436 ± 0.01525，配对差 −0.01109 的 95% 区间跨过零；按验证集选出的部署 checkpoint 测试转向 MAE 为 0.02478。末轮与最佳轮两种口径下均未检出负迁移。[分类递进](experiments/semester_2026_fall/transfer/classify-001/report.md) [全量回归](experiments/semester_2026_fall/transfer/donkey-001/report.md) |
| B4、S5 跨框架 | 相同初值、输入与 batch 索引下检查全部 62 个梯度及一次 Adam 更新。初次三种子试验端到端吞吐 MyFlows 444.46 samples/s，PyTorch 确定性模式 365.83，cuDNN 自动调优 444.49；三种子验证 MAE 为 MyFlows 0.06397 ± 0.03676、PyTorch 0.08408 ± 0.04474。之后每项主要优化都用同一命令复测，见第 4 节。[协议与资源明细](experiments/semester_2026_fall/frameworks/resnet18-001/report.md) |
| B4、S6 闭环 | 721 帧标定后先做三模型 30 次评估，再在同一 Unity 会话中轮换四个模型，共 40 次、24,011 帧，油门固定 0.2，全部达到 30 秒且无碰撞。部署中的 current-pilot 平均绝对 CTE 0.476、走廊出界 0 次；迁移模型平均绝对 CTE 27.012，每次越过走廊一次。离线 MAE 与闭环 CTE 在部分模型间排序相反。[配对结果](experiments/semester_2026_fall/closed_loop/transfer-001/report.md) |
| B5、S7 CIFAR-10 | 32 张真实图片达到 95% 过拟合门槛。45,000/5,000 训练与验证下，MyFlows 与同构 PyTorch 末轮验证准确率均为 60.22%；锁定 checkpoint 后官方 10,000 张测试图一次评分为 60.13%，多数类基线 10%。[分类报告](experiments/semester_2026_fall/classification/cifar10-001/report.md) |
| S8.1–S8.3 显存与优化器 | 首次写入分配梯度并释放中间梯度后，w16 的最大驻留 batch 从 37 提高到 46，步时未变差（三个配置的中位数缩短 11%–12%，两侧触及功耗墙的比例不同）；再重算 im2col 可到 93，池峰值降 47%–51%，步时增加 14%–24%，保留为选项。多张量 Adam 把 82 个参数的 CUDA 更新阶段从 3.435 ms 降到 2.337 ms，参数与两组矩逐位一致。[梯度分配](experiments/semester_2026_fall/operators/lazy-001/report.md) [列缓冲重算](experiments/semester_2026_fall/operators/recompute-001/report.md) [Adam 对照](experiments/semester_2026_fall/operators/adam-002/report.md) |
| S8.4 BN 融合 | 17 处 BN → (加残差) → ReLU 融合，三个配置步时缩短 2.7%–4.3%，中间输出显存降低 37%–40%；6 个 batch、4 个后端相对 FP64 的梯度误差统计与融合前相同。[BN 融合](experiments/semester_2026_fall/operators/bn-fusion-001/report.md) |
| S8.6 GEMM 调优 | 逐形状实测 tile 与 split-K 共 4,365 个组合，并采集 17 个占用率样本；在 Ada 上补充 12 种 tile 几何、窄输出并行归约与 64 个形状的实测计划表。CUDA Event 门槛同一形状的轮间极差中位数为 36.4%，改用 Nsight Compute 锁频周期后为 4.0%，据此修正 3 项计划（2 项输出逐位不变）。在计划表生效的 w16 batch 128 上，6 个 batch 的 FP64 对照中只有 register 后端的 120/498 个数组改变，最大相对变化 3.51×10⁻⁶，误差统计不变。一次分组读取的归约改动经测量变慢后撤回，流水线指标显示归约受 FP64 流水线限制。[调优](experiments/semester_2026_fall/operators/autotune-001/report.md) [锁频对比](experiments/semester_2026_fall/operators/gemm-ncu-001/report.md) [计划表对拍](experiments/semester_2026_fall/operators/plans-numerics-001/report.md) |
| S8.7、S8.8 精度与 Winograd | TF32/FP16 的 ReLU 输入误差约 1.7×10⁻³，BF16 约 1.4×10⁻²；2,000 帧三种子训练的验证 MAE 为 FP32 0.04631 ± 0.00691、TF32 0.04589 ± 0.00647、FP16 0.05520 ± 0.00364、BF16 0.06132 ± 0.00491，溢出跳步 0 次。Winograd 乘积数降到 44%–47%，梯度误差与 cuBLAS 相当，但由 CuPy 数组运算实现，训练步时为 72.02 ms（cuBLAS 20.72 ms）。均不作为默认后端。[精度](experiments/semester_2026_fall/operators/precision-001/report.md) [Winograd](experiments/semester_2026_fall/operators/winograd-001/report.md) |
| S10 诊断 | 在实际训练分区内取独立时间块做线性探测，验证转向 MAE 从中位数预测器的 0.17223 降至 0.06410，规则据此给出冻结级别与倍率候选；跨种子配对区间整体高于零时才报告负迁移。[诊断示例](experiments/semester_2026_fall/autopilot/diagnostics-001/report.md) |

## 4. 持续的跨框架复测

S5 要求每次重要优化后用同一条命令复测（`python -m benchmark.compare_frameworks_resnet`）。下表为各次复测的端到端吞吐（samples/s，含取 batch 与 H2D），每次都是同一会话内三个种子的中位数。PyTorch 的代码在各次之间没有变化，其吞吐的起伏（302–368）反映机器状态，比较 MyFlows 时应看同一行内与 PyTorch 的比值。

| 归档 | MyFlows 代码状态 | MyFlows | PyTorch 确定性 | PyTorch 自动调优 | MyFlows / 确定性 | MyFlows 三种子验证 MAE |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [resnet18-001](experiments/semester_2026_fall/frameworks/resnet18-001/report.md) | 初次（S2 之后） | 444.46 | 365.83 | 444.49 | 1.21 | 0.06397 ± 0.03676 |
| [resnet18-lazy-001](experiments/semester_2026_fall/frameworks/resnet18-lazy-001/report.md) | S8.1 首次写入分配梯度 | 458.07 | 325.99 | 411.31 | 1.41 | 0.06397 ± 0.03676 |
| [resnet18-recompute-001](experiments/semester_2026_fall/frameworks/resnet18-recompute-001/report.md) | S8.2 开启 im2col 重算 | 412.12 | 340.72 | 425.73 | 1.21 | 0.06397 ± 0.03676 |
| [resnet18-adam-001](experiments/semester_2026_fall/frameworks/resnet18-adam-001/report.md) | S8.3 多张量 Adam | 455.01 | 332.94 | 414.18 | 1.37 | 0.06397 ± 0.03676 |
| [resnet18-bn-001](experiments/semester_2026_fall/frameworks/resnet18-bn-001/report.md) | S8.4 开启图融合 | 491.13 | 331.31 | 442.26 | 1.48 | 0.06397 ± 0.03676 |
| [resnet18-autotune-001](experiments/semester_2026_fall/frameworks/resnet18-autotune-001/report.md) | S8.6 在线实测 tile 与切分 | 424.13 | 308.59 | 382.83 | 1.37 | 0.05458 ± 0.03454 |
| [resnet18-experimental-001](experiments/semester_2026_fall/frameworks/resnet18-experimental-001/report.md) | S8.7、S8.8 代码，默认后端 | 494.60 | 367.70 | 477.63 | 1.35 | 0.06397 ± 0.03676 |
| [resnet18-plans-001](experiments/semester_2026_fall/frameworks/resnet18-plans-001/report.md) | Ada 实测计划与锁频修正（当前代码） | 462.85 | 301.61 | 391.35 | 1.53 | 0.06397 ± 0.03676 |

S5 使用 batch 8，其 GEMM 形状不在 Ada 实测计划表中，最后一次复测的数值因此与之前相同。除切分数改变的在线调优外，各次 MyFlows 的三种子验证 MAE 逐位相同；PyTorch 三次质量运行均为 0.08408 ± 0.04474。切分数改变 FP64 归约前的部分和，舍入随之改变，经 Adam 放大后影响最终 MAE。

## 5. 限制与后续工作

**测量条件。** 本机是笔记本 GPU：桌面程序使空闲利用率就有 25%–48%，功耗上限在 60–80 W 之间变化，同一配置两次运行可差 10%–15%。步时对照都在同一会话内交替运行并报告离散度；单个 kernel 的比较改用 Nsight Compute 在锁定基频下的周期数。WDDM 在显存超过约 7.7 GB 时换页而不报 OOM，驻留容量以进程 Non Local Usage 和相邻整数 batch 判定。设备显存、CuPy 池与进程 RSS 三种口径分列。

**GEMM 门槛。** 逐形状 register/cuBLAS ≤ 1.20 的门槛仍有 4 个形状未通过：TN 128×64×2560（1.466）、512×256×640（1.327）、256×128×2560（1.254）、256×1152×2560（1.204），都是每步只出现一次的 dW GEMM，基频下合计多约 80 µs。三个 split-K 形状受固定顺序 FP64 归约限制，在 GeForce Ada 上 FP64 流水线活跃比例达 50%–83%；512×256×640 上所有 register 计划都慢于 cuBLAS 的多级流水 kernel。不清空缓存时比值基本不变。后续可为 register kernel 加入异步拷贝流水，或在保持结果不变的前提下减少 FP64 转换。实测计划表只覆盖本机 ResNet18 基准形状（w16 batch 128、w64 batch 32），其他形状使用解析规则或按需在线实测。

**迁移与闭环结论的范围。** 迁移对照只有 3 个种子，学习率 1e-3、batch 2 时多数配置在第 3 轮验证误差回升，各配置差值的 95% 区间都跨过零。生成道路的 Unity 接口只能控制客户端种子与默认 reset，同一模型（manual-t002）的平均绝对 CTE 在两次仿真会话中从 0.554 变为 29.823，模型排序只解释同一会话内的配对观测。分类准确率、离线回归 MAE 与闭环 CTE 回答的是不同问题，三者排序不一致。

**功能边界。** 原生 CUDA 路径不支持 FP64、分组与空洞卷积；TF32、FP16/BF16 与 Winograd 没有接入原生卷积调度器，本机上都比 FP32 原生卷积慢。分布式训练只在单机回环上验证，所有 Worker 共享一张 GPU，没有 SyncBN。CPU CI 只在本地 WSL 的独立 checkout 中验证过，GitHub 托管运行要在推送公开远端后进行。

**后续工作。** 迁移实验加入学习率衰减或早停并增加种子；闭环评估在固定的 Unity 会话中增加重复，或寻找可指定道路种子的场景；把混合精度的打包与 Winograd 变换写进原生调度器后再评估速度；推送后确认托管 CI 的耗时与 skip 列表。
