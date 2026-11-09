# 当前系统设计

更新时间为 2026-11-09。

本文只说明仓库当前已经存在的系统。第一阶段已实现 CUDA C/C++ Conv/Pool 与 Nsight 实验入口；分布式阶段已按 [stage1_distributed_gpu_spec.md](stage1_distributed_gpu_spec.md) 实现 GPU Worker 的同步 PS（Socket JSON / gRPC）与 Ring AllReduce。训练控制台（第 8 节）已把运行记录、实时曲线、GPU 占用和任务启停集中到一个 Web 界面。训练控制台也承担 AutoPilot 任务调度。训练入口支持 ImageNet 参数迁移、精确恢复、资源采样与 NVTX 诊断。

## 1. 项目边界

项目分为框架和应用两层：

- MyFlows/ 是自研深度学习框架。
- 根仓库是 DonkeyCar 应用、训练评估入口、部署和课程实验。

根仓库把 MyFlows/ 记录为独立 Git 仓库。阶段实验同时记录两个仓库的 HEAD、dirty/diff、源码 hash，并保存实际源码快照；恢复工具支持无 Git 历史的源码包。根仓库通过 .gitmodules 固定 MyFlows 子模块引用。

## 2. 当前结构

~~~mermaid
flowchart LR
  data[DonkeyCar 图片和标签] --> common[数据读取与预处理]
  common --> train[ResNet18 训练]
  train --> checkpoint[JSON + NPZ checkpoint]
  train --> onnx[ONNX 模型]
  checkpoint --> eval[离线评估]
  onnx --> eval
  onnx --> pilot[DonkeyCar 驾驶]
  onnx --> serve[gRPC / FastAPI]
  framework[MyFlows 计算图和算子] --> train
  framework --> eval
  native[原生 CUDA 库（Windows DLL / Linux .so）：寄存器分块 GEMM / cuBLAS] --> framework
  framework --> dist[PS 与 Ring 分布式训练]
  prepared[预处理数据 npy] --> dist
  dist --> archive[逐 rank 指标与资源采样归档]
  console[训练控制台 apps/console] -->|启动 / 停止| dist
  console -->|启动 / 停止| train
  archive --> console
  train -->|metrics.jsonl / TensorBoard| console
~~~

分布式训练的进程关系：

~~~mermaid
flowchart LR
  exp[distributed_experiment] --> launcher[Launcher]
  exp --> sampler[资源采样线程]
  launcher --> ps[PS 进程 CPU 聚合]
  launcher --> w0[Worker 0 GPU]
  launcher --> w1[Worker 1 GPU]
  w0 <-->|"Socket JSON 或 gRPC"| ps
  w1 <-->|"Socket JSON 或 gRPC"| ps
  launcher --> r0[Ring rank 0]
  launcher --> r1[Ring rank 1]
  r0 -->|gRPC| r1
  r1 -->|gRPC| r0
~~~

PS 模式和 Ring 模式二选一；Ring 模式没有 PS 进程。所有进程在同一台机器上，绑定 127.0.0.1，共享一张 GPU。

## 3. 当前训练过程

1. apps/common/donkey_data.py 读取 DonkeyCar 图片路径、转向和油门标签。
2. apps/common/image_preprocess.py 完成 resize、RGB 转换和 NCHW 排列。
3. apps/train/train_myflows_donkey.py 构建 ResNet18、损失函数和优化器。
4. MyFlows 完成前向传播、反向传播和参数更新。
5. 训练脚本保存 checkpoint，并可导出 ONNX。
6. TensorBoard 当前可以记录 loss、梯度、参数、数据读取时间、单步训练时间和吞吐量。

训练每步记录数据读取和计算墙钟，在指定步使用 CUDA Event 记录前向、反向与优化器时间，并通过独立采样器记录 GPU、CPU 与 RSS。字段见 [监控格式](monitoring_format.md)。

## 4. 当前推理过程

当前支持三种方式：

- MyFlows checkpoint 离线推理。
- ONNX Runtime 离线推理和 DonkeyCar 驾驶。
- gRPC 或 FastAPI 服务推理。

服务化代码作为已有基础保留，本学期只做必要的回归测试，不继续扩展为主要开发方向。

## 5. 当前 DonkeyCar 状态

- 当前本地数据有 10,000 条可读取记录，没有缺图。
- 当前数据的油门标签全部为 0.5，不具备变速学习条件。
- 当前驾驶代码使用固定油门，主要学习转向。
- 当前配置使用 donkey-generated-roads-v0 场景。
- 闭环评估通过 apps.eval.closed_loop 在相同场景清单下轮换 checkpoint 与 ONNX 模型，记录 CTE、出界事件、碰撞、距离、存活和推理时间，标定协议见 [遥测报告](experiments/semester_2026_fall/closed_loop/probe-001/report.md)。

因此，本学期首先建立固定油门下的闭环评估，再决定是否采集变速数据。

## 6. 第一阶段新增能力

- `Conv2D` / `MaxPool2d` 和对应 Op 接收 `backend=auto/numpy/cupy/cuda_native_register/cuda_native_cublas`；默认保留 NumPy/CuPy，显式原生 CUDA 路径严格验证 FP32、设备和配置。
- 两个原生后端由 C/C++ 动态库（Windows 为 `native_cublas.dll`，Linux 为 `native_cublas.so`）在当前 stream 上调度自写 im2col/col2im、max-pool kernel，只在卷积 GEMM 的引擎上不同：`cuda_native_register` 用第 19 讲的寄存器分块 kernel（2026-10-22 起为训练入口与分布式任务的默认），`cuda_native_cublas` 调用 cuBLAS；底层返回局部梯度，Op 用 `+=` 合并。
- 第一阶段的直接卷积、CUDA im2col 和自写 GEMM 仅作为历史失败基线保留在实验报告中。第 19 讲的寄存器分块 GEMM 是重新写的实现，现为原生卷积与 `Dense` 的默认 GEMM 引擎（见第 10 节）；课件的其余矩阵乘写法与补偿累加只作为对照实验，以及 `Linear` 的实验后端。
- FP32 小 CNN 完成同初值 CuPy/CUDA C 训练对照；完整 ResNet 的精度和 ImageNet stem 兼容仍在后续范围。
- `benchmark/cuda_ops.py`、`profile_cuda.py` 分别记录 CuPy/原生 CUDA 实验和真实 Nsight 报告；GPU Events、端到端 wall time、profile 计时分开。
- 统一测试入口补齐函数式测试，训练 smoke 显式传 seed；DataLoader 预先分批并按序返回，避免 worker 调度改变尾批数量。

## 7. 分布式训练（2026-09-19 交付）

- `MyFlows/distributed/launcher.py` 的 `run_training(**config)` 用 spawn 方式启动进程（Windows 与 Linux 相同）。合法组合为 single/none、ps/socket_json、ps/grpc_proto、ring/grpc_proto；任务为 synthetic、mnist_mlp、donkey_cnn、donkey_resnet18。
- PS 模式：1 个 CPU PS 进程分发初值、按样本数加权聚合并返回平均梯度；N 个 Worker 在 GPU 上前向、反向和本地 Adam/MBGD 更新，通过更新确认屏障保持一致。PS 不执行优化器更新。
- Ring 模式：N 个对等 rank，每个 rank 运行 gRPC 服务并连接右邻居，完成 Split、N−1 轮 ScatterReduce 与 N−1 轮 AllGather；启动器另起元数据屏障，只传摘要，不传梯度。
- 心跳、watchdog、限时等待、重复/陈旧消息拒绝和故障注入（crash、timeout、duplicate、retry、schema、heartbeat_loss、gather_crash）。
- `benchmark/distributed_experiment.py` 是正式实验单元：写 config/manifest，运行训练，并在后台线程约 1 秒一次采样 nvidia-smi 与 psutil，写 `resources.jsonl`；各 rank 通过 `Recorder` 写 `rank-*/steps.jsonl`、`epochs.jsonl` 与状态快照；结束后汇总为 `results.json`、`epochs.csv`、`events.jsonl`。
- `benchmark/distributed_suite.py` 与 `complete_course_experiments.py` 串行编排数值、状态、训练、性能、故障各阶段；`report_distributed.py` 生成对比表和图。
- 所有时间戳使用 `time.perf_counter()`；它在 Windows（QueryPerformanceCounter）和 Linux（CLOCK_MONOTONIC）上都是系统级计数器，不同进程的记录可以直接对齐。

### 7.1 正式训练中的数据并行（2026-10-23）

上面的 `run_training` 是测量用的实验框架：冻结的预处理数据、冻结 BN 统计量、逐步记录与故障注入。要训练能部署的模型，正式训练入口 `apps.train.train_myflows_donkey` 加上 `--distributed ps|ring --workers N` 即可，单进程和多 Worker 跑的是同一个训练循环 `train_rank`，同步部分由 `MyFlows/distributed/sync.py` 提供，复用同一套 PS 引擎、gRPC/Socket 传输与 Ring 分块交换。

~~~mermaid
flowchart LR
    main["主进程 prepare<br/>索引、划分、初始状态"] --> run["SyncRun"]
    run --> ps["PS 进程（ps 模式）<br/>或 Ring 元数据屏障（ring 模式）"]
    run --> w0["rank 0：train_rank<br/>日志、TensorBoard、断点"]
    run --> w1["rank 1..N-1：train_rank"]
    w0 <--> ps
    w1 <--> ps
~~~

- 初始状态：主进程在 CPU 上按同样的参数构建模型（续训时载入断点），把参数与 BN 统计量写成初始状态；PS 模式由 PS 分发，Ring 模式各 rank 读同一文件后用屏障核对摘要。
- 数据：全局顺序与单进程的 `MultiprocessDataLoader(shuffle=True, seed=epoch)` 相同，最后一个不满的 batch 重复末样本补齐，再把每个全局 batch 依次切给各 Worker；`--batch` 是全局 batch。各 Worker 只读自己的图片。
- 每步同步：每个 Worker 发送本分片的梯度（已含 L1/L2 正则项）、前向更新后的 BN 滑动统计量和 loss，按样本数加权平均后，所有 Worker 用同一个平均梯度做本地 Adam 更新，并把 BN 统计量设为平均值。BN 前向用各 Worker 本地 batch 的统计量（与 PyTorch DDP 不开 SyncBN 相同）。Ring 先乘以 FP32 的 n_i/N 再求和，与 PS 的算法一致：1 个 Worker 时与单进程逐位相同，2 个 Worker 时 PS 与 Ring 逐位相同。
- 一致性：每步提交时比较参数、BN 统计量与 Adam 矩逐个 FP64 求和后的哈希，不同则整次运行失败；任一 Worker 退出、超时或心跳中断也会让整次运行失败。
- 停止与续训：任何 Worker 看到停止文件都会在提交时投票，所有 Worker 在同一步停下，由 rank 0 保存断点。断点元数据记录 epoch、下一步、全局 step、本 epoch 已累计的 loss 与早停计数，`--resume` 从中间步继续；单进程断点格式相同，可以和多 Worker 互相续训。
- 验证与最佳模型：验证集按 Worker 交错切分，各自累加逐样本 MSE 后经 `reduce` 求和；所有 Worker 得到同一个验证 loss，最佳模型与早停判断一致，由 rank 0 保存。ONNX 导出在主进程中进行。
- 资源：所有 Worker 共享本机一张 GPU，每个 Worker 的 CuPy 显存池默认上限为启动时空闲显存的 90% 除以 Worker 数，超出时报显存不足；否则 Windows WDDM 驱动会把显存换到系统内存，训练看起来像卡住。Socket JSON 每条消息有 64 MB 上限，只能用于宽度不超过 32 的 ResNet。

## 8. 训练控制台（2026-09-23）

`python -m apps.console.server` 启动本机 Web 控制台（默认 http://127.0.0.1:8790），把训练可视化、GPU 监控和任务管理放在一个界面里。

~~~mermaid
flowchart LR
    Browser["浏览器 React 前端"] -- "REST / SSE" --> API["apps/console/api.py"]
    API --> Registry["RunRegistry 运行索引"]
    API --> Jobs["JobManager 任务队列"]
    API --> Live["ResourceRecorder 实时 1 s"]
    Live --> Sampler["GpuSampler NVML / nvidia-smi"]
    Jobs -- "子进程" --> Dist["benchmark.distributed_experiment"]
    Jobs -- "子进程" --> Single["apps.train.train_myflows_donkey（单进程或 PS / Ring Worker）"]
    Jobs --> JobRec["每任务 ResourceRecorder 0.5 s"]
    Registry -- "增量读取" --> Files["rank-*/steps.jsonl、epochs.jsonl、metrics.jsonl、resources.jsonl、results.json"]
    Registry --> Archive["docs/experiments/.../run_index.json"]
    Registry --> TB["mycar/logs/tensorboard"]
~~~

- 数据来源：控制台任务目录 `runs/console/`、分布式正式归档（按 `run_index.json` 发现）和 TensorBoard 单进程日志。控制台只读这些文件，不改动训练进程。
- 实时更新：前端先用 REST 取全量（降采样到 3000 点），再订阅 `/api/runs/{id}/stream`，服务端按游标每秒推送新增的 step、epoch 和资源行；运行结束时发送 `end` 事件。
- 时间对齐：step、epoch、资源采样都用 `perf_counter`，以最早一条记录为原点，因此 loss 曲线、阶段耗时和 GPU 利用率可以放在同一时间轴上。
- 任务：只接受框架支持的模式/传输组合与白名单参数；默认串行执行。分布式实验停止时发送 CTRL_BREAK，正式训练（单进程或多 Worker）写 `STOP_TRAINING` 让训练在同一步保存断点后退出；20 秒内未退出则结束整个进程树。正式训练用到原生后端而原生库缺失时，任务启动后先自动编译，新建页面会提示这一点。控制台重启后按 PID 与创建时间重新接管仍在运行的任务。
- Worker 视图：分布式实验按 rank 显示曲线、阶段耗时占比、最新 step 与梯度同步占比；各 rank 在 `prepare.json` 中记录 PID，控制台据此把 GPU 进程和进程内存对应到 rank。多 Worker 的正式训练也写 `rank-*/prepare.json` 与 `steps.jsonl`，拓扑页显示各 Worker 的 step 耗时与同步占比，曲线仍取 rank 0 的 `metrics.jsonl`。
- 单进程训练：`TrainingDashboard` 在 `--run-dir` 下同时写 `metrics.jsonl`，与 TensorBoard 记录内容一致。

运行详情的耗时图按数据、前向、反向、优化器四段展示，阶段占比按共同的阶段采样步累计并除以含读取的总步时。分母在后端降采样之前计算。硬件资源页显示整卡显存与采样峰值。Compare 页可选择 S5 框架归档，分别查看固定张量吞吐、含数据传输的吞吐和逐种子验证曲线。

### AutoPilot、迁移训练与恢复

控制台将 campaign 交给 AutoPilot 服务。Planner 生成配置，Executor 通过 JobManager 串行执行 trial，Verifier 读取运行证据与在线异常，Scheduler 管理预算和恢复。Reporter 汇总人工基线、配对种子和独立测试集结果。持久化的 campaign、trial 和 attempt 记录支持重启后接续。TPE 使用已验证的历史结果搜索，回滚从通过校验的 TrainingState 恢复。[AutoPilot 规格](autopilot_agent_spec.md)

ResNet18、MobileNetV2 与 AlexNet 可读取按 torchvision 名称保存的 NPZ 权重。载入时核验骨干名称和形状并记录任务头替换。冻结阶段同时控制参数和 BN 统计量。分类验证使用转向分档任务，主任务使用 DonkeyCar 回归和固定划分。训练结果记录来源哈希、冻结等级与冻结前后指纹。

## 9. 当前限制

- CUDA C 暂不支持 FP64、分组/空洞卷积和带 padding 的 Pool；这些高级卷积功能继续走原后端。当前原生路径限定 groups=1、dilation=1。
- 分布式训练只支持单机回环；所有 Worker 共享一张 GPU，没有多 GPU、多机或 NCCL 实验，也不能按 Worker 指定设备。分布式实验中 ResNet18 独立整批与分片轨迹的逐步梯度等价未通过。正式训练的多 Worker BN 用各 Worker 本地 batch 的统计量，没有 SyncBN。
- 在 WDDM 上，NVML 设备采样、CuPy 池统计和进程 RSS 分列展示。历史 distributed_experiment 自身的 nvidia-smi 采样只覆盖第一张卡。
- 跨框架基准使用同构 torchvision ResNet18、相同的 FP32 初值及批序。比较覆盖固定张量和含 H2D 两种吞吐，聚合后同时保留各个种子的结果。[S5 协议和结果](experiments/semester_2026_fall/frameworks/resnet18-001/report.md)
- 普通 checkpoint 保存模型、BN buffer 和优化器状态。单进程精确续训使用 TrainingState，另存数据游标、增强与 dropout RNG、最佳模型、早停历史和指纹。多 Worker 继续使用普通 checkpoint。
- GPU 上的端到端梯度误差由卷积 GEMM 的 float32 累加决定（ResNet18 单步梯度相对 FP64 的中位误差 3e-6–4e-6）。自写的寄存器分块 GEMM 是原生卷积与 `Dense` 的默认引擎，cuBLAS 可选：ResNet18 每步耗时两者相同，寄存器分块的梯度中位误差高约 5%，在窄输出的前向 GEMM 上误差最多为 cuBLAS 的 6.3 倍。补偿累加（kahan/dot2）只作为实验后端，没有接入卷积。
- 计算图缓存梯度需求，只对可训练路径和显式保留的中间梯度执行反向。梯度在首次写入时分配，已消费的中间梯度随即释放。反向重算 im2col 默认关闭：w16、cifar stem 的最大驻留 batch 可由 46 提高到 93，但步时增加 14%–24%。
- TF32、FP16/BF16 存储与 Winograd F(2×2,3×3) 只作实验后端：它们走 CuPy 的通用卷积路径，本机小规模训练都比原生 cuBLAS 卷积慢，不作为默认值。

## 10. 数值精度与 CUDA 第 19 讲改进（2026-10-22）

实测与修改前代码（上一提交的 git worktree，含其自己编译的 DLL）的逐项对照见 [CUDA 第 19 讲实验报告](experiments/semester_2026_fall/cuda_lecture19/README.md)。

- 默认 dtype：层参数与 BN 统计量默认 float32（`MyFlows.set_default_dtype` 可改为 float64）。此前默认 float64，`apps/train` 的 ResNet18 因此整网以 FP64 计算。
- 规约：BN 统计量、BN/卷积/Linear 的 bias 梯度、交叉熵的 log-sum-exp、MSE、全局平均池化与全局梯度范数都在 FP64 中按固定顺序累加，再舍入一次。BN 的均值以两个 float 的和保存，`x − mean` 按 `(x − hi) − lo` 计算，均值远大于标准差时 x̂ 不再丢失精度。
- BN 用 FP64 统计与 hi/lo 均值计算归一化。开启图优化可将归一化、残差加和 ReLU 合并，反向在规约及 dX 内应用 ReLU 门控。softmax、log-sum-exp 与逐行损失共用一个 kernel。Adam 可在一次 CUDA launch 内更新多个参数及两个矩，支持各组自己的学习率，CPU 回退沿用逐步显式舍入。
- Adam 改为 Kingma & Ba 的标准形式（矩估计从 0 开始、偏差修正）。旧实现首步令 v = g、s = g² 后仍做修正，首步步长约为标准值的 0.32 倍，且 g₁² 在 s 中被放大 1000 倍、上千步才衰减；此前的实验结果均基于旧实现。
- 原生库：卷积尾处理用 32×33 共享内存 tile 把 NHWC 行转成 NCHW，同时加 bias 与 ReLU/LeakyReLU；反向在同一 kernel 内做激活门控、dY 转置和 FP64 bias 部分和（warp shuffle）；输入 `requires_grad=False` 时跳过 dX 的 GEMM 与 col2im。
- 训练入口：`apps/train` 按 `--dtype` 设置默认 dtype，`--conv-backend auto` 在 CUDA + float32 时选 `cuda_native_register`（`cuda_native_cublas` 可显式指定，控制台“新建训练”页同样提供这两项）；输入与标签不求梯度；全局梯度范数只在需要记录时计算；`train_step_ms` 改用 CUDA event，每步只在读回 loss 时同步一次。
- 确定性：原生路径与 CuPy 的 col2im（改为 gather kernel，不再用原子加的 `add.at`）结果逐位可复现。
- GEMM 引擎：第 19 讲的寄存器分块 kernel 推广为 NN/NT/TN 三种布局（`native.gemm`）。`gemm_tiles.inc` 编译 20 种输出块、线程寄存器块与 K 块组合，其中 5 种为原有基础 tile；另有 3 种（名称以 `/s<级数>` 结尾）的 TN float4 kernel 用 cp.async 多级流水直接把 tile 从全局内存拷到共享内存，共享 tile 与 FMA 顺序不变，结果与同几何的寄存器 kernel 逐位相同（sm_80 以下退回寄存器双缓冲）。输出小而 K 长时按 K 切分，归约结果等于按固定顺序在 FP64 中求和再舍入一次：2 份切分用 FP32 计算（二次舍入定理），3–32 份在指数跨度允许时用 64 位整数精确求和后舍入一次，其余情况与 NaN/Inf 仍走 FP64 循环，三者逐位相同；输出不超过 32,768 个元素且切分不少于 48 份时，改由 8 个 warp 交错累加后按固定顺序合并，结果仍可逐位复现。RTX 4060 Laptop GPU 上 ResNet18 的卷积形状使用 `ada_plans.py` 中实测的计划，其他设备或形状用解析规则；`MYFLOWS_GEMM_AUTOTUNE=1` 时逐形状实测并按设备、运行库与原生源码哈希缓存。不切分时累加顺序与 tile 无关；改变切分数会改变舍入。`cuda_native_register` 后端用它做卷积与 `Dense` 的 GEMM，`cuda_native_cublas` 则调用 cuBLAS。Triton 扩展决定不做。

### 训练监控记录（2026-11-05）

训练图在首次收到贡献时写入梯度，多条反向路径汇合时继续累加。每个算子反传后释放已消费的中间梯度，Grad-CAM 通过 retain_grad 显式保留特征梯度。设置 MYFLOWS_RECOMPUTE_IM2COL=1 可在反向按原输入重建卷积展开矩阵。默认选择保存矩阵，实测的显存与速度取舍见第三阶段记录。

`python -m apps.train --run-dir <目录>` 默认启动独立资源采样进程，每 0.5 s 写 resources.jsonl；控制台/AutoPilot 已有记录器时通过环境标记避免重复写入。采样进程校验训练 PID 的创建时间，训练异常消失后自动退出。`--monitor off` 关闭训练自身采样，`--monitor-interval-s` 调整间隔。

训练按 v1 写 config、metrics、rank steps 与 result；资源记录也带版本和缺测原因。旧字段继续保留，新字段明确区分整卡显存、CuPy 池和进程 RSS，并补充含数据读取的 step_total_ms 与同分母吞吐。result.environment 保存依赖、设备、驱动、CUDA/NVRTC 与原生源码哈希。格式和时钟定义见 [monitoring_format.md](monitoring_format.md)。

TensorBoard 直接写事件 protobuf，训练进程不再为日志加载 torch；资源曲线增量读取 JSONL，不再次采硬件。当前小规模 CUDA 训练已验证事件回读；监控开销、时间线与优化闭环的阶段验收继续见 [stage3_progress.md](stage3_progress.md)。
