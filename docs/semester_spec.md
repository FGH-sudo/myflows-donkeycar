# 本学期开发 Spec

- 状态：v0.4 进度更新；第一阶段 CUDA/PS/Nsight 已实现并形成验收证据，后续学期方案继续审核
- 日期：2026-09-05
- 依据：[深度学习框架-16 综合项目III.pdf](<深度学习框架-16 综合项目III.pdf>)
- 当前应用：DonkeyCar 自动驾驶
- 当前主模型：ResNet18
- 审核证据：[semester_spec_review.md](semester_spec_review.md)
- 当前阶段：[第一阶段 Spec：CUDA 算子、PS 模拟与 Nsight](stage1_cuda_ps_spec.md)
- 阶段成果：[实验报告、完整复现命令与交付包](experiments/semester_2026_fall/stage1/README.md)
- 规划边界：按用户说明，之前学期的进度目标已经完成；不因本地缺少往期交付材料重新列缺口。

## 1. 最终目标

在现有 MyFlows 基础上，先完成课程基础项，再按剩余容量推进拓展项。下表区分课程要求与本组建议，验收数值是本组拟议标准，不是 PDF 原文规定。

| ID | PDF 依据 | 本组验收范围 | 层级 |
| --- | --- | --- | --- |
| B1 | 第 3、11 页 | 训练调优 Agent：六个核心组件、专业任务适配器、自动实验、诊断、选择与报告 | 基础 |
| B2 | 第 3、9 页 | CUDA C/C++ Conv2D、MaxPool2D 的前向和反向；NumPy/CuPy/CUDA C 误差和性能比较 | 基础 |
| B3 | 第 3、8 页 | 单机多进程同步 PS，包含 PS、Worker、Launcher、Monitor | 基础 |
| B4 | 第 3 页 | 持续优化 DonkeyCar；同条件比较其他框架的训练时间和资源 | 基础 |
| B5 | 第 4 页 | Agent 自动调优、新算子可用性与效率、图像分类的系统测试 | 基础 |
| B6 | 第 4、7 页 | 更新系统/Agent/可视化设计和报告；第 4、8、12 个汇报节点 | 基础 |
| B7 | 第 10 页，关联第 5 页 | 完整 GPU/CPU/训练阶段监控，并完成一次有前后对照的瓶颈改进 | 纳入基础 |
| E1 | 第 5 页 | All-Reduce 模式的单机多进程模拟 | 拓展 |
| E2 | 第 5 页 | 至少三种 ImageNet 预训练 CNN；参数导入、冻结/解冻、迁移训练 | 拓展 |
| E3 | 第 5 页 | TPE 多超参搜索、实时异常检测和自动回滚 | 拓展 |

第 5 页将 GPU 优化列在拓展中，但第 10 页明确要求采集指标，所以 B7 纳入基础。PDF 未指定 CNN 名单、分类数据集、worker 数、试验次数或硬性加速比；这些由本组方案确定。

本学期不复刻上学期交付材料。已有代码可以继续使用，但所有本学期结论都要由当前代码重新运行得到。

### 完成顺序

PDF 把任务分成基础部分和拓展优化。本草案按下面的顺序推进：

- 必须先完成：当前项目稳定运行、CUDA C/C++ 卷积和池化、Parameter Server、GPU 与训练耗时监控、AutoPilot Agent 基础版、同条件跨框架比较、DonkeyCar 优化、系统测试和报告。
- 完成基础验收后继续：All-Reduce、三种 CNN 的预训练和迁移训练、TPE 多参数搜索、实时异常检测和自动回滚。恢复接口可以提前设计，拓展实现不占用基础任务验收时间。
- 本学期不安排：MCTS。

拓展内容仍列入完整计划，但不能为了赶拓展功能而省略正确性测试、原始结果或基础部分。

### 当前执行阶段

按 PDF 第 7 页的汇报节点推进：第四次课先关注 GPU 编程、分布式计算与 Nsight，第八次课再讨论 Agent。当前阶段执行细节以 [stage1_cuda_ps_spec.md](stage1_cuda_ps_spec.md) 为准；本文继续维护学期范围、后续依赖和最终验收。

- 已实现：测试收集/seed 修复、FP32 算子基线、自写 CUDA Conv/Pool 前反向、原图小 CNN、三后端性能与 Nsight、CPU 最小 PS。阶段结果与原始证据见报告。
- Agent 的进一步设计与实现暂缓；本阶段只保留普通配置、运行结果与 run_id，不要求先完成通用 TrainRunner、TrialResult 或完整 ModelState。
- 第四次课展示目标与阶段完成标准分开记录；前向演示不能代替反向/池化/集成验收。
- PS 已验证小 MLP 的 1/2/4 worker 和 20+12 分片；后续接无 BN 小 CNN/分类任务。GPU PS 不属于当前基础要求。
- W0/W1 按实际依赖分步完成，独立 CUDA kernel 不等待整个框架重构。正式学期实验仍需完整的精度、状态和复现条件。

## 2. 开发起点与最新进展

### 2.1 第一阶段开始前的审核快照

以下结论来自 2026-09-05 开发前的代码检查与运行，保留用于理解修复原因；已经解决的事项以第 2.2 节和阶段报告为准，不能将这些历史条目继续当作当前缺口。

- 根仓库应用层测试共 21 项，本次全部通过。
- MyFlows 的 unittest discovery 本次 68 项全部通过；额外直接执行的 7 个 ResNet/BN/GAP 测试和 3 个序列化/ONNX 测试也通过。后 10 个函数式测试不在默认 discovery 的 68 项中。
- digits 图像分类用例本次隔离重跑三次通过；但其 np.random.seed(0) 不能控制默认初始化器的 default_rng(None)，复现性问题仍在，不将这次通过写为已经修复。
- CuPy 可以使用当前 NVIDIA GPU，MyFlows 的 CUDA 基础测试可以运行。
- 当前没有 CUDA C/C++ 自定义算子源码。
- 当前没有 Parameter Server、All-Reduce 或训练调优 Agent。
- 当前训练日志只记录部分耗时，没有完整 GPU 监控。
- 当前有 ResNet18、trainable 标记、优化器外部梯度入口，以及含参数/BN buffer/优化器状态的 checkpoint；缺少稳定命名的状态接口、预训练映射和阶段冻结工作流。
- checkpoint 尚未保存完整 RNG、数据游标、调度器和早停状态；现有 resume 是 epoch 级入口，不能承诺精确的 step 级恢复。
- 训练 --dtype float32 目前只设置输入/标签；模型初始化和 BN buffer 仍默认 FP64。本次小型 ResNet 探针中 82 个参数和模型输出均为 FP64。
- 当前 benchmark/compare_frameworks.py 仍使用 VGG11，未形成 ResNet18 同条件基线；不能直接用于本学期报告。
- 当前 DonkeyCar 数据共 10,000 条，油门标签全部为 0.5；驾驶端使用固定油门 0.2。
- 根仓库和 MyFlows 都有未提交改动，且根仓库缺少完整的子仓库配置。
- 系统默认 Python 3.14 没有项目依赖；仓库内 Python 3.11 虚拟环境可以运行项目。
- 本机为单张 RTX 4060 Laptop GPU，显存 8188 MiB；驱动 591.74。虚拟环境 Python 3.11.7 开启了 include-system-site-packages，尚不是隔离环境锁定。

这些是本学期新增能力的接入前提，不推翻用户确认的往期完成状态。现有自动求导、CNN、训练评估、ONNX、服务与本地驾驶资产继续复用。

### 2.2 第一阶段实施结果（2026-09-05）

- tools/run_tests.py 已收集框架类测试及原先遗漏的 10 个函数式测试；初始化显式传 seed，保留原训练断言。当前统一入口共 127 项，连续三轮结果归档在阶段报告。
- 回归另外暴露了多进程 DataLoader 把样本按 worker 分别组批的问题，偶尔产生额外尾批。已改为完整 batch 派发并按 batch_id 返回，增加顺序、非均匀负载、单尾批及异常传播检查。
- MyFlows/ops/cuda/ 已提供六个自写 CUDA kernel，严格 FP32 分派并保留 NumPy/CuPy；小 CNN 参数/梯度/Adam 状态的 FP32 路径已经验证。
- MyFlows/distributed/ 已提供 CPU 同步 PS/Worker/Launcher/事件监控，1/2/4 worker 的 20 步参数更新与单进程参考等价，四类故障均能失败退出并清理。
- 三后端完整 P0-P3 矩阵、Systems/Compute 小中规模报告、dX 优化前后普通计时均已归档。性能有规模依赖，CUDA C 并未全面快于 CuPy。
- requirements-stage1-lock.txt 锁定 Python 3.11 阶段依赖；不继承系统包、不安装 PyTorch 的环境通过框架回归和导出源码复现。每次实验保存两个仓库的实际源码快照、状态与哈希；仓库发布方式后续另行规范。
- 完整 ResNet FP32/BN、统一 ModelState/TrainRunner、全程资源监控、正式 CIFAR-10/三框架/驾驶比较、Agent/All-Reduce/迁移训练仍未完成；本阶段未运行 DonkeyCar 长训练或闭环实验。

## 3. 本学期不做什么

- 不再开发或使用 VGG 训练、评估和模型对比。
- 不补写或重新制作上学期的答辩截图和实验报告。
- 不继续扩展 gRPC、FastAPI、INT8、Grad-CAM 等已有功能，除非本学期系统测试需要。
- 不把 DAgger、PPO 等强化学习方向列入本学期主线。
- 本学期不安排 MCTS。
- 不在没有统一测试方法的情况下宣称某个实现更快。

## 4. 开发内容

### 4.0 分阶段建立共享接口

以下名称为学期拟议接口，尚未实现。保留现有 CLI 和旧 checkpoint 的兼容入口；不将这张表作为当前 CUDA 开发必须先全部完成的前置任务。

| 契约 | 最小内容 | 使用方 |
| --- | --- | --- |
| ModelState | 稳定且唯一的层级名、shape、dtype、trainable；参数与 BN buffer 分开；严格缺失/多余项报告 | PS、checkpoint、预训练、跨框架转换 |
| TrainConfig | task、model、device、dtype、op_backend、seed、global_batch、loader_workers、train_workers、预算、split/data hash、输出目录 | Executor、单进程/PS runner、benchmark |
| TrainRunner | build、train_step、validate、save/load；返回实际样本数、loss、梯度、阶段时间；不由多个组件重复 forward/backward | Agent、Monitor、PS |
| TrialResult | schema_version、run_id、trial_id、attempt、status、resolved_config、验证指标、资源、路径、失败原因 | Memory、Verifier、Reporter |
| TrainingState | ModelState、优化器、epoch/global_step、RNG、采样/增强状态、数据游标、早停/调度器状态、配置指纹 | 精确续训与拓展回滚 |

第一阶段只需要数组级 Conv/Pool 接口、显式 backend、固定 fixture，以及 PS 小模型的唯一参数映射和消息协议。完整 ModelState/TrainRunner/TrialResult 在后续集成时统一；TrainingState 的精确恢复内容在 E3 实施。

关键约束：

- 参数名从模块路径产生，例如 layer2.0.conv1.weight；不直接用目前重复的 kernel/bias/gamma/beta 作字典键。共享参数按对象去重，遍历顺序稳定。
- dtype 必须贯穿参数、buffer、梯度、优化器状态、输入和输出；类别索引保持整数。恢复 FP64 老模型必须显式保留或转换，禁止默默当作 FP32 实验。
- 随机性由明确的 seed 派生并传入初始化器、NumPy/CuPy、数据采样和增强；记录每个 trial 的 seed。不能只调用 np.random.seed。
- device 表示 cpu/cuda；op_backend 表示 numpy/cupy/cuda_c。正式 CUDA C 测试遇到不支持形状应报错，不可静默退回 CuPy；混合运行单独记录实际后端及覆盖率。
- loader_workers 是数据读取进程数，train_workers 是 PS/All-Reduce 训练进程数。两者不可混用，Agent 的 worker 搜索必须指明是哪一个。
- 每次运行独立保存配置、日志、结果和 checkpoint；主训练入口目前的全局 STOP_TRAINING 文件也要改为运行级路径，避免一个 trial 误停另一个。

### 4.1 按依赖稳定当前项目

以下保留 W0 的完整工作范围。阶段测试收集、seed、小 CNN dtype 和独立环境已完成；整个 ResNet 精度和通用状态接口仍需在相应集成之前完成。

缺陷修复与通用重构分开安排：漏收集测试、已有初始化随机性和偶发测试问题在第一阶段 C0 处理，建立完整回归基线后再进行正式 CUDA 对比；当前路径的 dtype、梯度正确性和计时问题不得后移。其余修复在依赖它的功能首次接入前完成，具体放行条件见 [第一阶段 Spec 第 2.1 节](stage1_cuda_ps_spec.md)。完整 ResNet 精度、预训练结构兼容、三框架协议和精确回滚分别绑定对应任务，不统一推迟到学期末。

- 整理根仓库和 MyFlows 的未提交改动，分别确认哪些需要保留。
- 补齐 MyFlows 的仓库引用方式，保证新电脑可以取得正确版本。
- 固定 Python 3.11 和依赖版本，统一运行命令。
- 找出图像分类测试偶发失败的原因并修复。
- 为模型参数建立稳定且不重复的名称，方便预训练参数导入和跨进程传递梯度。
- 统一测试收集，覆盖现有 68 个 unittest 与 10 个函数式框架测试；保留原命令作为兼容检查，新增统一入口的测试数要可核对。
- 修正 seed 传播和 dtype 传播，用小模型证明同 seed 同初值、全链路 FP32。
- 后续接入 TrainRunner/TrialResult 最小接口，先跑一个分类和一个 DonkeyCar 小任务；第一阶段使用独立算子入口和小模型演示。

完成标准：

- 新环境可以按文档安装并运行。
- 应用层和框架层测试连续运行三次都通过。
- 同一随机种子下，小模型训练结果可以重复。
- 三次完整通过指统一入口，不是只重复某一个用例；没有 GPU 的测试环境可以标明 skip，但课程 GPU 验收不能用 skip 代替。
- checkpoint v1 保持可读；新状态格式必须包含版本、模型配置和严格校验。实验可使用干净提交，或同时归档根仓库与 MyFlows 的 diff、未跟踪源码和哈希，不把 HEAD 等同于工作区全部代码。

### 4.2 CUDA C/C++ 卷积和池化

本项的第一阶段范围已实现，线程映射、接口、测试矩阵与验收见 [第一阶段 Spec](stage1_cuda_ps_spec.md)。NumPy/CuPy 参考路径继续保留，完整 ResNet 接入属于后续工作。

第一阶段实现课程明确要求的两个算子：

- Conv2D 前向和反向。
- MaxPool2D 前向和反向。

保留三种运行方式：

- NumPy：CPU 参考结果。
- CuPy：当前 GPU 实现。
- CUDA C/C++：本学期新增实现。

当前使用 CuPy RawModule/NVRTC 编译 .cu 并调用其中的自写 kernel，CuPy 管理数组和显存。若老师明确要求独立编译的 C++ 扩展，再增加编译版本。

第一版支持 contiguous NCHW、FP32、OIHW 权重、bias、stride 和 padding，groups=1、dilation=1。Conv 需要覆盖 1x1、3x3、7x7；Pool 覆盖 2x2/s2 和 3x3/s2。分组/空洞算子保留现有后端，后续按需增加 CUDA C 支持。

先实现便于检查的直接计算版本，后根据 Nsight 证据选择 tiling/shared memory 等优化；不得把调用已有 CuPy 卷积包装为自写 CUDA C。MaxPool 明确相同最大值选择首个索引，重叠窗口梯度必须累加；第一阶段保持现有 padding=0，后续增加 padding 时使用负无穷并排除该区域回传。

性能测试必须做到：

- 三种实现使用完全相同的输入。
- 先预热，再正式计时。
- GPU 计时前后进行同步，避免只测到任务提交时间。
- 测试多种 batch、通道数和图片尺寸。
- 记录平均时间、中位数、P95、加速比、最大绝对误差和相对误差。
- 使用 Nsight 检查至少一个小规模和一个大规模输入。

完成标准：

- 前向结果和输入、权重梯度均通过数值比较。
- Conv 还要比较 bias 梯度；小规模 FP64 数值梯度使用现有参考路径，CUDA C 的 FP32 暂定 atol=1e-4、rtol=1e-3，逐算子记录；误差调整必须给出归约长度和参考误差依据。
- 正确性用例覆盖负值、零值、非方形输入、stride/padding、池化并列最大值与重叠窗口；非法形状和 dtype 有明确错误。
- 每个性能案例固定输入和上游梯度，预热 10 次、测量 50 次，独立重复 3 组；分别记录 forward/backward、编译、传输和端到端时间。过大输入可缩减次数，但必须预先记录理由。
- CPU、CuPy、CUDA C 三种实现可以用同一条测试命令运行。
- 报告说明哪些输入规模有收益，哪些情况下 CuPy 已经足够。

### 4.3 GPU 和训练耗时监控

第一阶段只实现算子计时、PS 通信/等待记录及 Nsight 分析所需标注。以下完整训练监控模块安排在阶段后续，不作为 CUDA kernel 首版的前置条件；Nsight 报告也不替代长期 GPU 资源采样。

需要采集 PDF 中要求的指标：

- GPU 利用率。
- 当前显存和显存峰值。
- GPU 温度和功耗。
- CPU 利用率。
- 单步总时间。
- 数据读取或准备时间。
- forward 时间。
- backward 时间。
- optimizer update 时间。
- samples/sec。

实现要求：

- 硬件数据优先通过 NVML 读取；不可用时可以调用 nvidia-smi。
- CUDA 阶段计时必须同步，或使用 CUDA Event。
- 监控结果同时保存为机器可读文件和 TensorBoard 曲线。
- 每次训练记录 GPU 型号、驱动、Python 和依赖版本。
- 硬件采样初始设为 500 ms；记录单调时钟、PID、GPU、run/trial/step 关联。硬件采样峰值、进程分配峰值和 CuPy 内存池占用分开，不混称“显存峰值”。
- 温度/功耗等接口不支持时保存 null、原因和采集状态；不能写 0 冒充测量。优先尝试 NVML，再探测 nvidia-smi；最终报告明确缺测项和替代机器验证需求。
- 区分包含等待的 wall time 和 CUDA Event 的设备执行时间；step 总时间不强制等于各异步阶段之和。诊断同步模式与正式性能模式分开标识。

完成标准：

- 一次训练结束后能看到所有必需字段、采集成功状态和缺测原因；核心时间/吞吐/GPU 利用率/显存不可缺测，硬件不支持的温度/功耗项不能标记为完整验收。
- 可以指出训练主要时间花在数据读取、前向、反向还是参数更新。
- 根据监控结果完成至少一项优化，并重新测试证明变化。
- 同配置比较监控开/关各 3 次，量化监控开销；目标低于 5%，超出则降低采样频率或单列诊断运行，不能带着未知开销宣称训练加速。

### 4.4 Parameter Server

单机启动以下进程：

- Parameter Server：保存全局参数，接收梯度，聚合后更新参数。
- Worker：读取各自的数据，完成前向和反向，上传梯度，再取得新参数。
- Launcher：启动和关闭所有进程。
- Monitor：记录每个 worker 的状态和耗时。

第一版采用同步训练：所有 worker 完成当前轮次后，Parameter Server 才更新参数。固定合成数据的小 MLP 已完成 1/2/4 worker 更新协议验证；下一步接无 BN/Dropout 的小 CNN 和图像分类任务，最后考虑较大模型。第一阶段细节见 [阶段 Spec](stage1_cuda_ps_spec.md)。

实现约定：

- 第一版使用 Windows spawn、CPU worker 与 NumPy IPC，先实现真实的参数服务器流程。本机只有一张 GPU；多进程共享 GPU 是单独的可选实验，不等同于多 GPU 加速。
- PS 是唯一更新参数和优化器状态的进程；worker 只计算梯度。复用 Optimizer.update(var_gradients=...)，加入稳定参数名到 Variable 的适配，并确认不会二次平均。
- 协议携带 run_id、step_id、parameter_version、worker_id、n_samples、参数 schema/hash、梯度 dtype/shape；每步每 worker 恰好接受一次，拒绝重复、陈旧和不匹配消息。
- 若 worker 的 loss 是本地样本均值，PS 使用 g = sum(n_i * g_i) / sum(n_i)。不得不加权平均不同大小的尾批；空 shard 显式报错或按协议跳过。
- 比较时固定 global_batch 和批次清单；建议小模型 global_batch=32 对应 1/2/4 worker 的 32/16/8 样本，并另测 5+3 等非均匀划分。
- 首轮等价性测试关闭 Dropout，采用无 BN 的小 CNN；带 BN 的 ResNet 先固定统计量并广播 buffer。局部 BN 与全局 BN 不等价，若采用局部 BN 必须单列算法差异，不承诺参数轨迹一致。
- 队列读写有超时，异常传到 Launcher；停止时清理所有子进程。失败恢复留给后续 Agent 拓展，基础 PS 先做到明确失败、无死锁。

需要记录：

- 每个 worker 的计算时间。
- 梯度上传时间。
- 梯度聚合和参数更新时间。
- 参数拉取时间。
- 每轮总时间和整体训练时间。
- 传输的数据量。

完成标准：

- 支持 1、2、4 个 worker。
- 固定随机种子和总 batch 后，单进程与 Parameter Server 的结果在允许误差内一致。
- worker 异常退出或超时时，Launcher 能结束剩余进程并给出明确错误。
- 报告如实说明多进程是否带来加速，以及瓶颈在哪里。
- FP64 CPU 小模型以单进程同一批次为参照，单步及连续 20 步参数暂定 atol=1e-8、rtol=1e-6；FP32 使用另列容差。加入 worker 崩溃、超时、重复消息、尾批样本计数测试，超时阈值到达后 5 秒内完成清理。

### 4.5 All-Reduce

All-Reduce 在 Parameter Server 正确后实现。

第一版在单机多进程中完成：

- 每个 worker 持有相同模型。
- 每个 worker 计算自己的梯度。
- worker 之间完成梯度求和或平均。
- 每个 worker 使用相同的平均梯度更新参数。

优先实现环形 All-Reduce，用进程间队列或共享内存传递分块梯度。

复用 PS 的 ModelState、批次和监控协议；验证 reduce-scatter + all-gather，分块不能整除时 padding 后裁剪。无中央梯度聚合者，各 worker 独立持有一致的优化器状态；同样按样本数加权，避免平均次数错误。CPU IPC 结果称为模拟性能，不与 NCCL 多 GPU 性能混同。

完成标准：

- 1、2、4 个 worker 的训练结果可重复。
- 各 worker 每次更新后的参数保持一致。
- 与单进程和 Parameter Server 比较训练、通信和等待时间。

### 4.6 预训练模型和迁移训练

本学期拟实现三种不包含 VGG 的 CNN：

1. ResNet18：先补齐官方结构兼容模式，再做权重映射。
2. MobileNetV2：用于验证深度可分离卷积，也适合后续轻量驾驶模型。
3. AlexNet：作为第三种候选；全连接层资源开销需先测，不能仅因结构直观就估计为低成本。

三个模型保持候选状态。基础图像分类不依赖三模型迁移学习。ResNet18 导入前必须处理：

- 当前训练默认 cifar stem；ImageNet 权重需要独立的兼容配置。当前 imagenet stem 的池化缺 padding=1，本次 224 输入验证得到 55x55，而本机 torchvision 为 56x56。
- 当前 Conv 默认有 bias，而 torchvision ResNet 卷积无 bias；增加显式结构配置，不静默增加可训练偏置。FC 权重需要从 [out,in] 转换为 [in,out]。
- 导入 BN gamma/beta/running_mean/running_var，明确 num_batches_tracked 的处理；对齐 eps、momentum 与 running variance 的更新语义。
- 先保留 1000 类头验证来源模型输出一致，再替换任务头。旧 DonkeyCar checkpoint 使用原结构模式，不能因兼容修正变为不可用。
- MobileNetV2 需补 ReLU6、倒残差和深度卷积的完整兼容配置；AlexNet 需核对池化/分类头尺寸。逐个通过来源输出比对后再进入迁移训练。

预训练参数优先使用 PyTorch 官方 ImageNet 权重。每个模型需要支持：

- 按名称载入卷积、全连接和 BatchNorm 参数。
- 检查参数名称、形状和载入数量。
- 替换最后的分类层或回归层。
- 按网络阶段冻结参数。
- 只训练未冻结的参数。
- 保存新的 checkpoint。

验证方法：

- 使用相同图片和相同预处理，比较 MyFlows 与来源模型的输出。
- 输出预训练参数载入成功率和未载入参数清单。
- 训练前后比较冻结参数，确认其没有变化。
- 冻结阶段必须同时规定 BN buffer 行为；冻结参数不自动冻结 running statistics，外层 model.train() 不能覆盖冻结阶段的 eval 状态。
- 本草案建议 CIFAR-10；训练集内固定划分 train/validation，官方 test 只用于最终一次评估。迁移验证遵循来源权重的输入尺寸和归一化，不能直接把原 32x32 训练方案与 ImageNet 预处理混用。
- 在图像分类任务上验证迁移训练确实可以降低 loss。

完成标准：

- 三种模型都能载入官方预训练参数。
- 主干参数及 buffer 必须完全匹配，只有明示的任务头替换/兼容字段可例外；记录来源版本、权重校验和、映射清单。来源输出 FP32 暂定 atol=1e-4、rtol=1e-3，不能以“载入比例高”代替输出验证。
- 三种模型都能替换输出层并完成一次迁移训练。
- 冻结和解冻行为有自动测试。

### 4.7 MyFlows AutoPilot Agent（后续阶段）

这里的 AutoPilot Agent 是训练调优程序，不是 DonkeyCar 驾驶控制器。

本节保留 v0.2 的候选范围供后续讨论，当前不继续细化或实现。待 CUDA/PS/Nsight 有实际接口和结果后，第八次课前再审定 Agent 设计；本节内容不阻塞第一阶段。

第一版完成 PDF 中要求的六部分：

- Planner：根据训练目标、可用时间和允许尝试次数，确定本轮要试哪些参数。
- Scheduler：安排训练顺序，确保同一张 GPU 不会同时启动冲突任务。
- Executor：调用现有训练命令并获取退出状态、日志和结果。
- Memory：保存每次尝试的参数、结果、耗时和模型路径。
- Critic / Verifier：判断本次结果是否有效，停止无效 trial；拓展版接入训练回滚。
- Reporter：汇总所有尝试，输出最佳参数、失败原因和对比图表。

同时补全第 11 页的专业任务适配器：GPU Monitor、Operator Optimization、PS Training、Transfer Learning、Benchmark。GPU Monitor、Operator Optimization、PS Training、Benchmark 四个基础适配器负责读取对应实验结果、提出有证据的候选建议并交给 Verifier；Transfer Learning 在 E2 实现前登记为 unavailable，不伪造可运行能力。基础架构采用规则和结构化结果，不要求 LLM、独立对话或自动修改算子源码。

第一版搜索以下参数：

- learning rate。
- batch size。
- loader_workers，基础值为 0 或 2。
- train_workers 只在 PS runner 验收后开放；冻结阶段数只在 E2 验收后开放，搜索空间由 capability 检查过滤。

基础版先支持按给定的候选范围自动完成多次训练。拓展版使用 TPE 根据历史结果选择下一组参数，可以直接使用 Optuna，不自行重写贝叶斯优化算法。

基础版必须形成“读取指标 → 判断瓶颈/配置有效性 → 选择下一组候选 → 验证并报告”的闭环。仅批量启动脚本不算完整的 B1。建议首轮配置：lr={1e-4,3e-4,1e-3}、batch={1,2,4}、loader_workers={0,2}，采样其中最多 6 个候选；正式预算在基线实测后锁定，禁止默认执行整个笛卡尔积。

每轮同时限定 max_trials、单 trial 超时、总 wall-time、GPU 并发数 1、重试次数最多 1。以固定验证集的 angle_MAE/分类 accuracy 为主指标，资源超预算或 NaN trial 不参与排名。保留“人工基线未被超越”的合法结论，不预设调优一定获胜。

基础版必须识别非有限结果、OOM、超时、异常退出并记录失败；拓展版再进行趋势诊断与回滚。完整异常范围：

- loss 为 NaN 或 Inf。
- loss 连续明显上升。
- CUDA OOM。
- 训练进程超时或异常退出。
- GPU 长时间空闲。

调度恢复与训练回滚分开：

- 基础版持久化 pending/running/succeeded/failed/cancelled 状态；中断后核验遗留进程和结果，只续排未完成 trial。新 attempt 使用独立目录，配置、数据和代码指纹参与去重。
- checkpoint 回滚属于 E3，依赖 4.0 的完整 TrainingState；校验模型、优化器、RNG、数据游标和配置后恢复。采用临时写入、校验和及最后发布 manifest，避免读取只写了一半的 JSON/NPZ。
- 如果 OOM 后改变 batch 或 worker 数，记为从 checkpoint 派生的新配置试验，不声称与原轨迹精确续训相同。
- 回滚后必须修改策略或终止，限制重试次数，防止反复恢复同一失败状态。

完成标准：

- 可以自动完成多次训练，不需要人工逐条启动。
- 中断后可以继续已有任务，不重复已完成的尝试。
- 最终报告包含每次尝试的命令、参数、结果和最佳选择理由。
- 必须展示至少一项由监控数据触发的诊断和可检验建议；最佳配置重新独立运行，不能只挑搜索中的偶然最好数值。
- 自动测试覆盖预算、进程退出、错误/缺失结果、重复 trial、重启去重和 GPU 串行；用可控故障 fixture 验证，无需真的占满显存。
- 拓展版中，TPE 能根据历史结果提出下一组参数。
- 拓展版中，能实际演示一次异常检测和 checkpoint 恢复。

### 4.8 DonkeyCar 优化

DonkeyCar 分两步处理。

第一步：固定油门基线。

- 沿用固定执行油门 0.2；保留原始数据中的 0.5，它是采集标签而非执行速度命令。离线主要比较转向指标，油门误差单列；不为“对齐”篡改原始标签或直接提速。训练监督是否屏蔽油门头作为一个单变量消融实验。
- 在同一场景、同一速度和同一随机设置下重复运行。
- 自动记录 CTE、最大偏离、出界次数、行驶时间、完成情况和推理延迟。
- 建立当前 ResNet18 的闭环基线。
- 优先按采集 session/连续片段划分，减少相邻帧泄漏；若只有单个序列，采用带间隔的时间块并记录局限。现有按行随机划分可用于旧流程回归，不单独证明未见道路泛化。
- 新增评估入口前先探测模拟器 telemetry：CTE、位置/距离、碰撞、重置和随机种子是否可用；不保证提供的能力不写成已经支持。

第二步：比较优化结果。

- 对比人工选择的训练参数与 Agent 选择的训练参数。
- 比较离线误差和模拟驾驶结果是否一致。
- 若固定油门下已经稳定，再采集不同油门数据并训练变速模型。
- 若尝试 MobileNetV2 驾驶模型，必须与 ResNet18 使用同一数据和同一评估方式。

完成标准：

- 每种配置至少 5 次配对运行，同一场景/起点/种子清单，固定油门、超时和终止规则；随机种子接口不可控时如实记录并增加重复，不声称完全确定性。
- 报告同时给出平均值和最差结果。
- 是否提升由 CTE、出界和完成情况决定，不能只看离线 MSE。
- 预先在场景标定中锁定出界阈值、连续越界帧数、一次事件的去抖规则、完成路线/时限；没有路线完成信号时报告定时存活和距离，不能称为完赛率。
- 给出平均/最大绝对 CTE、越界事件数、完成或存活指标、推理 P50/P95；结论同时报告配对差值，优先要求完成情况不退化，再讨论 CTE 改进。

### 4.9 同条件跨框架比较

主要使用固定划分的 DonkeyCar 数据和 ResNet18，比较 MyFlows、PyTorch 和 PaddlePaddle。三种实现必须使用：

- 同一份训练集、验证集和样本顺序。
- 相同的图像预处理和数据增强。
- 相同的 batch、epoch、优化器、学习率、随机种子和数值精度。
- 同一台 GPU，以及相同的预热和正式计时范围。

先做三框架 GPU 依赖烟测；本机包已存在不等于三套 GPU 后端已验证。使用独立环境/独立进程按序运行，防止显存池、全局设备和初始化时间相互污染。

比较协议还必须固定：

- ResNet stem、width、bias、Pool padding、BN epsilon/统计规则、输出头、loss reduction、优化器 epsilon/衰减。正式 FP32 关闭 AMP 和 TF32，并记录实际精度。
- 从共同 NPZ 初值转换到三框架；同一个 seed 不能保证不同框架初值相同。先对齐单批前向、loss、梯度和更新，再做计时。
- 使用相同 batch 索引清单及尾批策略。现有 MyFlows 会补齐尾批，其他框架未同样补齐；新协议选择显式 mask/样本权重或共同 drop_last，不能悄悄训练重复样本。
- 多进程读取返回顺序必须可追踪并重排；第一阶段已按 batch_id 重排，正式数据增强还须单独固定 RNG/采样状态。
- 分别做固定设备张量的计算测试、含数据准备/传输的端到端测试；预热后重置模型/优化器/采样状态。GPU 同步边界相同，禁止以未同步的 Python 时间代表设备完成时间。

记录最终 loss、验证指标、总训练时间、单步时间、samples/sec、GPU 利用率、显存峰值和 CPU 利用率。框架初始化、数据准备和正式训练分别计时，不能只挑对某个框架有利的数字。

完成标准：

- 一条命令可以依次运行三种框架并保存原始结果。
- 报告列出未能完全对齐的实现差异及其影响。
- 只说明实测结果，不预设 MyFlows 一定更快。
- 至少 3 个共同种子，输出 mean/std、运行失败、实际样本数和参数配置；标注 RSS 是进程当前值还是采样峰值、显存是分配峰值还是硬件采样峰值。当前 _peak_mb() 返回瞬时 RSS，需改为真实测量并更名。

### 4.10 系统测试

测试分为以下几类：

- 框架基础测试：现有计算图、算子、优化器和 checkpoint 不退化。
- CUDA 测试：前向、反向、不同输入规模和错误输入。
- 分布式测试：参数一致性、梯度平均、worker 超时和异常退出。
- 预训练测试：参数对应、输出对比、冻结和解冻。
- Agent 测试：任务恢复、异常判断、结果排序和报告生成。
- 图像分类测试：暂定 CIFAR-10，小数据用于日常测试，完整数据用于阶段验收。
- DonkeyCar 测试：数据检查、离线评估和模拟器闭环评估。

日常测试应在几分钟内完成。耗时较长的 GPU、完整分类数据和模拟器测试单独运行，并保存结果。

图像分类在 B5 内独立交付：digits 仅作快速 smoke；新增 CIFAR-10 数据加载、十分类训练/验证入口，复用现有 CrossEntropy 和小型 CNN/ResNet18 CIFAR stem。日常可用固定平衡小子集，阶段运行固定 45,000/5,000 train/validation，锁定配置后在官方 10,000 test 上最终评估。若采用其他数据集，保留相同隔离规则。

建议分类门槛：32 个样本的过拟合正确率至少 95%；完整流程产生独立验证/测试指标，并超过多数类基线。与同构 PyTorch 模型在相同预算下验证收敛质量，初始允许差距 3 个百分点，超出先诊断实现和训练条件；完整训练预算在首轮基线后锁定，不承诺未经实测的最终精度。

系统测试按 B1-B7 映射到对应产物。单元测试成功、一次模型训练成功、模拟器实际闭环成功分别记录，不能相互替代。

## 5. 时间安排

| 时间 | 主要工作 | 汇报时应展示 |
| --- | --- | --- |
| 当前至第 2 次课 | 第一阶段 C0/C1：修复测试收集与 seed/偶发测试，固定 FP32 fixture 和三次完整回归；建立 NumPy/CuPy 基线、探测工具、实现 Conv 前向 | 可信的修复基线及第一个自写 kernel 的正确性证据 |
| 第 2～4 次课 | C2-C6：Conv 反向、Pool、原图小 CNN、三后端比较/Nsight；CPU PS 独立推进 | 第 4 次课：CUDA 实现及已有误差/耗时、真实 profile、PS 1/2 worker 演示；按阶段 Spec 区分展示目标与完整验收 |
| 第 5～6 次课 | 复核第一阶段证据，W3 接小 CNN/分类；完成 W0/W1 剩余精度/状态/训练接口，启动 W4/W5/W6 | 可复用的训练/评估入口；完整监控、分类/比较和驾驶基线按依赖逐项落地 |
| 第 7～8 次课 | W7：基于实际 CUDA/PS/监控能力开展 Agent 调研与设计，接口就绪后做小预算自动实验 | 第 8 次课以设计和已有原型为主；不要求 TPE/自动回滚 |
| 第 9～12 次课 | W8：Agent 与 CUDA/PS/监控整合；分类、比较、驾驶评估第二轮；故障测试和优化对照 | 第 12 个汇报节点：基础 B1-B7 形成可演示的集成结果和缺口清单 |
| 第 13 次课至结项 | 基础问题修复、正式重复实验、设计/报告；基础通过且有容量才执行 E1/E2/E3 | 最终可复现实验包；拓展逐项验收，不挤占基础验收与报告 |

第一阶段路径：最小 fixture/环境 → CUDA Conv → Pool/计算图集成 → benchmark/Nsight；CPU PS 用固定初值和显式参数映射独立推进。C1 通过后即可开始 profile，不等待全部反向完成。

后续路径：第一阶段结果 + W0/W1 剩余工作 → W4/W5/W6 监控与应用基线 → W7 Agent → W8 集成验收。驾驶与跨框架基线仍在 Agent 正式调优前建立。

PDF 第 7 页分别写第 4 次课、第 8 次课和第 12 次周；这里保留其相对汇报节点，实际日期以课表为准。当前课次、人数与工时尚未确认，不把相对安排解释为已承诺日期；已过节点改成最近可交付里程碑。

## 6. 每个阶段必须留下的内容

每次正式实验都要保存：

- 当前 Git commit。
- 根仓库和 MyFlows 各自的 HEAD、dirty 状态、diff/未跟踪源码清单及哈希；仅记录根 commit 不能复现本地代码。
- 完整运行命令。
- Python、CUDA、GPU 和主要依赖版本。
- 使用的数据和样本数量。
- 数据 manifest、内容/索引指纹、train/val/test split、batch 顺序、尾批策略、预处理与有效配置。
- 随机种子。
- 原始 JSON 或 CSV 结果。
- 自动生成的图表和 Markdown 总结。
- 失败实验及失败原因。
- 运行耗时口径、预热/同步策略、实际设备和 dtype、指标 schema、数据缺测、baseline 对应关系及产物校验和。

新实验统一放在 docs/experiments/semester_2026_fall/。旧实验文件不混入本学期结果。

拟议目录为 run_id/ 下的 manifest.json、config.json、metrics.jsonl、resources.csv、result.json、report.md、artifacts/。当前 .gitignore 忽略该目录下 JSON/CSV/图片和全部 NPZ，因此报告需要附可访问的实验包位置与 checksum、manifest 索引和环境复现命令；不得认为文件放进该目录就随 Git 交付。大型数据/权重不强行入库，采用资产清单和外部交付包。

## 7. 最终验收清单

### 必须完成

- [x] 当前基础测试稳定通过；127 项统一回归连续三轮，见阶段报告。
- [ ] 默认测试入口覆盖框架函数式测试，seed/dtype/稳定状态接口通过验证。
- [x] CUDA C/C++ Conv2D 和 MaxPool2D 前向、反向正确（第一阶段 FP32 支持范围）。
- [x] NumPy、CuPy、CUDA C 性能和误差报告完成（固定算子矩阵）。
- [x] Nsight 分析完成（小/中规模与一次 dX 优化，不代替完整训练监控）。
- [x] Parameter Server 支持 1、2、4 个 worker（CPU 小 MLP；小 CNN/分类集成继续）。
- [ ] GPU、CPU 和训练阶段监控指标齐全。
- [ ] AutoPilot Agent 基础版可以自动运行多次训练并生成报告。
- [ ] Agent 包含六组件和专业任务适配器，能根据指标诊断、在预算内选配置，并恢复调度记录。
- [ ] MyFlows、PyTorch、PaddlePaddle 同条件对比完成。
- [ ] 图像分类系统测试完成。
- [ ] DonkeyCar 闭环评估和对比完成。
- [ ] 系统设计、测试过程、实验结果和最终报告完成。

### 完成基础部分后继续

- [ ] All-Reduce 支持 1、2、4 个 worker。
- [ ] ResNet18、MobileNetV2、AlexNet 三种预训练模型可用。
- [ ] 参数载入、冻结和迁移训练通过测试。
- [ ] AutoPilot Agent 使用 TPE、处理训练异常并自动恢复。

任何一项只有设计或截图、没有可运行代码和原始结果，都不算完成。

## 8. 方案建议与待确认项

| 决策 | 本稿建议 | 需要进一步确认的部分 |
| --- | --- | --- |
| 当前顺序 | 已确定先做 CUDA/PS/Nsight，Agent 暂缓 | 阶段具体安排见 stage1_cuda_ps_spec.md |
| 基础与拓展 | 按 PDF 保持 E1/E2/E3 为拓展；完整监控纳入基础 | 不再将 All-Reduce 是否属于基础留为悬而未决 |
| CUDA 接入 | 自写 .cu 源码，通过 CuPy RawKernel/NVRTC 编译调用 | 第 4 次课前核对教师是否额外要求独立 C++ 扩展；本机编译与 Nsight 权限尚需实测 |
| 分类 | CIFAR-10 为阶段任务，digits 为 smoke | 数据下载和完整训练预算 |
| 驾驶 | 固定执行油门 0.2，主指标是转向和闭环 | 有额外需求再采变速数据，不阻塞当前规划 |
| 预训练 | ResNet18 → MobileNetV2 → AlexNet 候选 | E2 启动前测内存/耗时；必要时单独审核第三模型替换 |
| 资源 | 单 GPU 串行 trial；PS 首版 CPU 进程模拟 | 人数、每人周工时、当前课次和结项日期 |

第一阶段已实施并形成实验包，剩余学期范围继续按依赖推进；此处的候选数据集、Agent 与拓展接口仍是后续讨论内容。

## 9. 可执行任务拆分

工时为熟悉 Python、尚需学习 CUDA/分布式的团队的初始估算，含编码和模块测试，不含无人值守训练时间；不是经过历史速度校准的承诺。人员未知，以“框架/应用/实验”角色描述，允许同一人顺序承担。

| ID / 角色 | 具体产物与建议位置（均为待开发） | 依赖 | 人时估算 | 退出检查 |
| --- | --- | --- | --- | --- |
| W0 / 框架 | 统一测试入口、seed/dtype、环境清单；先 fixture/小模型，后完整 ResNet 路径 | 无，按接入范围逐步完成 | 12-20 | 78 个现有框架检查被收集；21 个应用检查；同 seed 同权重；FP32 全链路 |
| W1 / 框架+应用 | 先做 PS 小模型显式参数映射；阶段后统一 ModelState/runner/config/result，兼容旧 checkpoint | 最小映射可独立；完整接口依赖 W0 相关工作 | 16-24 | 两次构图相同 key；命名与 shape 错误拒绝；小任务返回规范结果 |
| W2 / 框架 | MyFlows/ops/cuda/ 的 .cu、编译/分派；benchmark/cuda_ops.py；误差与 Nsight 证据 | 独立 fixture/环境；小模型集成依赖其 seed/dtype，不依赖完整 W1 | 32-48 | Conv/Pool fwd/bwd 与参考一致；不静默 fallback；三后端规模比较 |
| W3 / 框架 | MyFlows/distributed/ 的 protocol/ps/worker/launcher；小模型 PS 入口 | 显式小模型参数映射、固定初值；不依赖 CUDA 或完整 W1 | 24-36 | 小 MLP 1/2/4 worker 与 20 步等价已实现；后续接分类；保留分片/重复消息/异常清理回归 |
| W4 / 框架+实验 | MyFlows/monitoring/ 的 sampler/timer；训练/PS hooks；瓶颈对照 | 首阶段基本计时独立；完整监控与 W1/W2/W3 整合 | 16-24 | B7 指标、通信等待可追踪；量化开销；一项优化前后数据 |
| W5 / 应用+实验 | CIFAR-10 数据与分类入口；benchmark/compare_frameworks.py 的 ResNet18 三框架协议 | W0/W1；监控依赖 W4 | 24-36 | 无测试集泄漏；单步对齐；独立进程基线、真实精度/峰值 |
| W6 / 应用+实验 | apps/eval/ 的模拟器评估入口与 telemetry adapter；场景配置、配对运行清单 | W0，独立于 Agent | 20-32 | 固定油门；至少 5 次基线；CTE/越界/完成定义及原始数据 |
| W7 / 应用 | apps/autopilot/ 六组件与专业任务适配器；持久 trial registry；小预算搜索 | W1/W4/W5；PS 适配依赖 W3 | 24-40 | 6 个以内候选自动执行；诊断/排序/预算/重启去重/故障测试 |
| W8 / 全组 | tests/system/；Agent 联动 CUDA/PS；优化驾驶对比、正式实验包、设计报告 | W2-W7 | 24-40 | B1-B7 逐项证据；失败和限制可见；新环境复现检查 |

基础合计约 192-300 人时；按 25% 集成/返工余量，容量约 240-375 人时。可用容量 = 人数 × 每人每周项目小时 × 剩余周数；例如 3 人 × 8 小时 × 12 周 = 288 人时，仅可作为基础目标的中间估算，不能再默认塞入全部拓展。

上述估算沿用 v0.2 的学期总工作量，尚未按实际开发速度校准，不代表当前剩余工时。第一阶段 C0-C7 属于这些 W 任务的子集，不能另加一遍工时。

| 拓展 | 新增工作 | 依赖 | 额外人时估算 |
| --- | --- | --- | --- |
| E1 | Ring All-Reduce、rank 一致性、通信对比 | 基础通过、W3 | 20-32 |
| E2 | 三模型结构兼容、权重映射/冻结、迁移数据与训练 | 基础通过、W1/W5 | 44-72 |
| E3 | TPE 条件搜索、完整 TrainingState、事务 checkpoint、异常回滚 | 基础通过、W7；冻结搜索依赖 E2 | 28-44 |

拓展按可用容量单独选择；E1→E2→E3 是默认候选顺序，不要求同时开工。完整目标另需 92-148 人时，仍需预留返工。

第一阶段 C0-C7 的实现、验收与交付见 [阶段报告](experiments/semester_2026_fall/stage1/README.md)。下一步先确定完整 ResNet FP32/BN 与状态接口的最小集成范围，再接 PS 小 CNN、完整监控和正式分类/跨框架/驾驶基线；Agent 设计仍按用户确定的后续阶段推进。

若容量不足，先削减超参候选/模型规模/拓展数量，保留 B1-B7 的可运行闭环、正确性和基本重复实验；不能以旧学期资产缺失为理由增加补材料任务。
