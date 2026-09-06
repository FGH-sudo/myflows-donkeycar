# 第一阶段开发 Spec：CUDA 算子、PS 模拟与 Nsight

- 版本：v0.2，阶段已完成；C0-C7 实现、逐项验收与实验包见阶段报告。
- 日期：2026-09-05。
- 最近汇报节点：第 4 次课，主题为分布式计算、GPU 编程实现及 CUDA Nsight。
- 总体范围：[semester_spec.md](semester_spec.md)；现状证据：[semester_spec_review.md](semester_spec_review.md)。
- 课程依据：[课程 PDF](<深度学习框架-16 综合项目III.pdf>) 第 3、4、7、8、9 页。
- 实施结果：[阶段报告与复现入口](experiments/semester_2026_fall/stage1/README.md)。本文件保留验收约定，报告记录实际 run_id、结果与限制。

## 1. 阶段目标与边界

本阶段先将现有卷积和池化计算接入自写 CUDA C/C++，建立正确性与性能证据；同时用小模型完成单机多进程 PS 模拟，用 Nsight 分析实际执行。Agent 的需求细化、架构设计和实现均暂缓。

当前代码收缩为 `cuda_native_cublas`：卷积和池化均由原生 C/C++ 层调度自写 CUDA kernel，卷积矩阵乘调用 cuBLAS；直接卷积、CUDA im2col 和自写 GEMM 仅保留历史失败报告。

| 目标 | 对应课程内容 | 本阶段交付 |
| --- | --- | --- |
| G-CUDA | 第 3 页卷积/池化 CUDA C/C++；第 4 页新增算子测试 | 自写 Conv2D 与 MaxPool2D 的前向、反向；接入原计算图的小 CNN |
| G-BENCH | 第 9 页三种实现比较 | NumPy、CuPy、CUDA C 的同输入误差、耗时及输入规模对比 |
| G-PROFILE | 第 7 页 GPU 编程与 Nsight | Systems 时间线、Compute 内核分析、至少一项有依据的优化尝试及前后对照 |
| G-PS | 第 8 页 PS/Worker/Launcher/Monitor | CPU 单机同步 PS，1/2 worker 更新等价性、通信/等待计时和异常清理 |

当前不展开：Agent/LLM/TPE、完整 TrainingState/精确回滚、All-Reduce、预训练模型、完整 CIFAR-10 训练、三框架 ResNet 正式比较、DonkeyCar 长训练/闭环评估、GPU PS、多 GPU。学期任务仍保留，由总体 Spec 安排。

本阶段 PS 完成门槛为 1/2 worker；已额外完成小 MLP 的 4 worker 等价验证。后续 W3 继续接入小 CNN/分类任务。性能不预设 CUDA C 或多进程一定加速；正确实现和能够解释实测结果都属于有效成果。

### 第四次课展示与阶段完成分别记录

PDF 规定汇报主题，没有规定以下逐项完成门槛；这是本组的准备目标。

- 汇报准备目标：真实运行的 CUDA Conv2D 前向、至少两种规模的误差/耗时、一个真实 Nsight 分析案例、1/2 worker PS 小演示；如实展示反向/池化的已完成部分及剩余任务。
- 阶段完成标准：本文件第 9 节 G-BASE/G-CUDA/G-BENCH/G-PROFILE/G-PS/G-REPORT 全部通过，包括前置修复、反向、池化和计算图集成。只达到汇报准备目标时，状态仍是“阶段进行中”。
- 第四次课前优先形成可运行的完整小案例；已有正确结果先归档，再扩规模或优化。具体日期和人员工时按实际课表校准。

## 2. 现有实现与复用边界

本节表格保留 2026-09-05 开发前的接入依据；它不是当前缺陷清单。开发中已修复测试收集、显式 seed、小 CNN 的 FP32 传播，以及回归发现的多进程 DataLoader 批次拆分/返回顺序问题。完整回归与隔离环境证据见阶段报告，完整 ResNet 的精度/状态接口仍属后续工作。

| 位置 | 现有行为 | 本阶段处理 |
| --- | --- | --- |
| [ops/convolution.py](../MyFlows/ops/convolution.py) | Conv2D 为 im2col + 矩阵乘法；反向用矩阵乘法、col2im、bias 求和 | 保留作 NumPy/CuPy 参考，增加 CUDA C 分派 |
| [layers/layer.py](../MyFlows/layers/layer.py) | Conv2D/MaxPool 层构造节点；Conv 可融合激活 | 透传后端选择，保留旧调用默认行为 |
| [core/device.py](../MyFlows/core/device.py) | xp 在 NumPy/CuPy 间切换 | 继续负责数组设备，计算后端单独选择 |
| [core/graph.py](../MyFlows/core/graph.py) | backward 先清空梯度，再逆序调用各节点 | 局部 CUDA 梯度仍用 += 合并，保护多分支与共享参数 |
| [train/opt.py](../MyFlows/train/opt.py) | one_step 累积梯度，update 应用梯度并清缓存；支持外部梯度 | PS 复用外部梯度入口，只有 server 执行更新 |
| [utils/initializers.py](../MyFlows/utils/initializers.py) | 默认 default_rng(None) 不受 np.random.seed 控制 | 算子 fixture 显式固定 RNG；小模型初始化明确传入 seed/初值 |
| [tests/test_convolution.py](../MyFlows/tests/test_convolution.py) | 独立 NumPy 参考、前反向、池化并列最大值测试 | 复用参考逻辑并参数化后端，不删除旧测试 |

现有 MaxPool 采用 padding=0、行优先首个最大值、重叠窗口梯度累加。第一版保持这些语义。现有模型参数可能为 FP64，不能直接把指针交给 float* kernel；独立 fixture 先显式 FP32，模型 dtype 工作在集成前完成。

### 2.1 缺陷修复的前置门槛

当前审核发现的问题按影响范围处理。推迟通用接口重构，不等于把缺陷全部留到第一阶段结束后；影响当前输入、参考结果、梯度或计时可信度的问题，必须在对应实验之前解决。

| 问题 | 最迟处理节点 | 放行证据 |
| --- | --- | --- |
| unittest 漏收集 10 个函数式测试 | C0，建立回归基线时 | 统一入口实际执行已有 78 个框架检查及 21 个应用检查；后续新增检查另计 |
| 初始化 seed 未传入、digits 偶发失败 | C0，正式 CUDA 对比前 | 修复/查明已有测试初始化路径；同 seed 同初值；包含 digits 的完整回归连续 3 次通过，不能只重跑失败项或放宽断言刷通过 |
| CUDA 指针接收 FP64、输入/梯度精度不一致 | C0/C1 的算子入口；模型相关部分在 C4 前 | fixture 和 wrapper 严格校验 FP32；C4 小模型参数/梯度/优化器状态均符合精度约定 |
| 参考 Conv/Pool 或梯度累加出现真实错误 | 发现后立即阻塞受影响案例 | 先修参考实现并用独立参考/数值梯度确认，再把它作为 CUDA 对照 |
| GPU 未同步计时、瞬时 RSS 冒充峰值等旧 benchmark 问题 | C0 新基线入口与 C5 正式实验前 | 本阶段使用已校验的计时口径和真实字段；旧 VGG 脚本不作为阶段性能证据，整体跨框架改造后续完成 |
| PS 参数同名、梯度重复平均等通信接入问题 | C6 首次训练前 | 小模型唯一映射、schema/shape 校验、单步对照通过后再做多步/性能 |
| 双仓库 dirty 状态和依赖不隔离 | C0 记录；新环境复现/正式交付前补齐 | 记录实际源码与环境；正式实验包具备双仓库状态、依赖与取回方式，不以 HEAD 代替未提交代码 |

C0 的测试与 seed 修复属于当前阶段工作，完成后再进入 C1 的正式实现/对比。工具查找、资料阅读或独立编译小样例可以同时进行，但未通过基线时不发布正确性或性能结论。

完整 ResNet 的 FP32 修复在其首次 CUDA C 训练前完成，不能等该训练跑出问题再补；ImageNet 结构兼容在 E2 前处理；数据划分/三框架公平性在相应正式应用实验前处理；完整 checkpoint 精确回滚在 E3 前实现。暂缓项都绑定后续使用节点，不无限期搁置。

## 3. 技术路线与接口

### 3.1 三种路径

| device / backend | 计算方式 | 本阶段用途 |
| --- | --- | --- |
| cpu / numpy | 原 NumPy 实现 | 数值参考、CPU 性能、PS |
| cuda / cupy | 原 CuPy im2col + GEMM | GPU 对照 |
| cuda / cuda_native_cublas | 自写 CUDA kernel，由原生 C/C++ 调度并调用 cuBLAS | CUDA 实现与性能分析 |

Conv2D、MaxPool2d 及对应 Op 已增加末尾可选参数 backend，默认 auto 保持旧行为：CPU 走 numpy，GPU 走 cupy。只有显式 cuda_native_cublas 才启用原生实现；它在 CPU、非 FP32 或不支持配置上报错，禁止静默回退。

编译和显存管理继续使用 CuPy。kernel 从 .cu 文件读取，按源码 hash、编译选项、设备等记录缓存信息；首次编译与稳态运行分别测量。自写路径的卷积/池化计算不能委托现有卷积库完成，允许复用显存分配、连续化与结果梯度合并。

### 3.2 第一版支持范围

| 项目 | 第一版约定 |
| --- | --- |
| Conv 输入/权重/输出 | X[N,Cin,H,W]、W[Cout,Cin,kH,kW]、Y[N,Cout,Ho,Wo] |
| 数据类型 | CUDA C 输入、权重、bias、输出、dy、局部梯度均 FP32；类别标签保持整数 |
| Conv 参数 | 正整数或二元组 kernel/stride，非负对称 padding；groups=1、dilation=1 |
| Conv 大小 | 覆盖 1x1、3x3、7x7 及非方形 kernel；不支持配置在发射 kernel 前校验 |
| bias | 可选，存在时 shape=[Cout]；无 bias 不返回/合并 db |
| Pool | 2x2/s2、3x3/s2 与重叠窗口；padding=0，ceil_mode=False |
| 有效输入 | 非空、形状合法；有限数值是调用方前提，不逐次扫描/同步检测 NaN；NaN 传播不作为第一版已支持语义 |
| 连续性 | 底层 kernel 接收连续数组；Python wrapper 对合法非连续 view 连续化，禁止把 stride view 当连续指针 |
| 索引与发射 | int32 索引，数组元素数不得超过 int32 上限；默认 block_size=256，低层允许 64/128/256/512 |

Conv 输出尺寸为 Ho=floor((H+2*ph-kH)/sh)+1，Wo 同理；结果非正、通道不匹配或 bias 形状错误时拒绝执行。groups/dilation 高级能力保留在原后端，不能因新增后端破坏原功能。

Pool padding=1、完整 ImageNet stem 兼容放在后续总体模型兼容任务；本阶段不顺带改变现有 ResNet 的结构或 checkpoint。

### 3.3 已实现底层 API

以下是接口记法，当前实现位于 `MyFlows/ops/cuda_native/`；Op/Layer 接口使用固定默认值。

~~~text
conv2d_forward(x, weight, bias=None, *, stride=(1, 1), padding=(0, 0)) -> y
conv2d_backward(x, weight, grad_y, *, stride=(1, 1), padding=(0, 0),
                need_bias_grad=True) -> (grad_x, grad_weight, grad_bias)
maxpool2d_forward(x, *, kernel_size=(2, 2), stride=(2, 2)) -> (y, context)
maxpool2d_backward(grad_y, context) -> grad_x
~~~

- forward 缓存本次 backward 所需的输入引用/形状/参数和实际 backend；Pool context 保存 argmax、输入形状及窗口配置。每次 forward 更新 context，不能沿用过期形状。
- backward 返回新分配的局部梯度，Op 执行 x.grad += dx、weight.grad += dw、bias.grad += db；底层函数不清空计算图已有梯度。
- kernel 使用当前 CuPy stream；普通调用不强制全设备同步，错误检查/计时按各自边界同步。wrapper 拷贝和发射必须处于同一 stream。
- MVP 小模型显式关闭 Conv 激活融合和 Graph 优化。独立 ReLU 接在 Conv 后；后续允许融合节点复用基础 Conv，但不能丢失后端选择或伪装成完全融合内核。
- backend 在 forward 时确定，backward 必须沿用该路径；执行中切换全局 device/数组设备属于错误使用。

## 4. CUDA 算子实现顺序

### C1：直接 Conv2D 前向

一个线程负责一个输出元素 (n,oc,oh,ow)，按 ic/kh/kw 遍历有效输入位置，累加乘积后加 bias。边界位置按零 padding 处理，不创建完整 im2col 数组。

初版可用一维 grid/block，从线性输出索引还原坐标；block size 先取 128 或 256 中一个固定值并记录。正确前不做自动搜索。首先用 X=[1,1,5,5]、W=[1,1,3,3]、s1/p0 验证可手算结果，再测试多通道、batch 和边界。

### C2：Conv2D 反向

| 内核 | 初版安排 | 必须检查 |
| --- | --- | --- |
| dX | 每个输入元素由一个线程累加所有相关输出位置的贡献 | stride 对应位置的整除与边界、padding、跨通道累加 |
| dW | 每个权重元素由一个线程对 batch/输出空间做归约 | 与同一 dy 对应，不漏 batch/位置，不重复平均 |
| db | 每个输出通道对 dy 的 batch/空间维求和 | bias 可选、shape/dtype 正确 |

这是优先可验证的实现，不承诺大规模性能。先采用单写者归约减少原子写入的不确定性；Nsight 确认归约成为瓶颈后，再考虑块内归约等优化。验证 dy 使用固定随机非均匀数组，不能只测全 1 上游梯度。

### C3：MaxPool 前向和反向

每个输出元素扫描一个窗口，保存最大值及其原输入位置；相等时保留行优先首个索引。反向按 argmax 回传；重叠窗口的梯度必须求和。初版可采用每个输入元素收集相关窗口贡献的单写者方案，暂不引入 atomicAdd 优化。

先对齐当前 padding=0 语义，测试全负值、并列最大值和重叠窗口，不将填充值误当成最大值。

### C4：接回原计算图

1. 接入 backend 参数与严格分派，先对单 Conv/Pool Op 做 Graph 前反向测试。
2. 构建无 BN/Dropout 的小 CNN：Conv(1→4,k3,p1) → ReLU → Pool(k2,s2) → Flatten → Dense(16) → ReLU → Dense(2)。
3. 使用固定的 8x8 输入小样本，参数/buffer/梯度/优化器状态明确采用 FP32；从同一初值在 CuPy 和 CUDA C 两路径完成训练。
4. 增加共享 Conv 参数/两分支相加的用例，验证 += 语义；改变 batch/空间尺寸后再 forward/backward，检查 context 更新。

这个小 CNN 包含真实 Pool，用于验证两类算子的训练可用性。完整 ResNet18/DonkeyCar 的 FP32 接入和长训练留在学期后续集成；全量稳定命名、Agent Runner 和新 checkpoint 格式都不是 C1 的前置条件。

## 5. 正确性与性能实验

### 5.1 固定正确性矩阵

同一 fixture 先在 CPU 生成并固定 seed、X/W/b/dy，保存 hash；三后端使用相同数值，分别处理后端转换。GPU 数据加载不在纯算子计时内。

| ID | X / W 或 Pool | stride / padding | 目的 |
| --- | --- | --- | --- |
| T0 | X=[1,1,5,5]，W=[1,1,3,3] | 1 / 0 | 首个可手算 Conv |
| T1 | X=[2,3,7,9]，W=[4,3,3,3] | 2 / 1 | 多 batch/通道、非方形图像、边界 |
| T2 | X=[1,2,6,8]，W=[3,2,1,1] | 1 / 0 | 1x1 Conv |
| T3 | X=[1,3,16,16]，W=[4,3,7,7] | 2 / 3 | 7x7 Conv |
| T4 | X=[1,2,7,9]，W=[3,2,2,3] | (1,2) / (0,1) | 非方形 kernel 与二元参数 |
| T5 | Pool，X=[2,3,7,9]，k2/s2、k3/s2、k3/s1 | padding=0 | 池化、尾部截断、重叠累加 |
| T6 | 零/负值/并列最大值/非连续 view；bias 有/无 | 按对应算子 | 边界语义和 wrapper |
| T7 | 非法通道/shape/dtype/设备、groups≠1、dilation≠1 | 无 | CUDA C 明确拒绝，不静默 fallback |

- 三后端 FP32 比较暂定 atol=1e-4、rtol=1e-3，逐项检查 Y/dX/dW/db。另将相同 FP32 fixture 转成 FP64 用 NumPy 独立参考/数值梯度复核；Pool 数值梯度避开并列最大值，tie 单独按明确语义测试。
- 最大绝对误差定义为 max(abs(test-ref))；相对误差记录 max(abs(test-ref)/max(abs(ref),1e-6))。接近零时相对值可能大，验收使用上述 atol+rtol 联合规则，禁止直接删除困难元素。
- 发生误差超限先排查索引、归约和 dtype；修改容差需记录证据与影响范围，不能在看过正式结果后无说明放宽。
- 小 CNN 用固定 32 个可分样本验证过拟合能力，建议 100 个更新内准确率至少 95%；与 CuPy 同初值、同批次，检查早期梯度/更新及最终 loss。具体 fixture 和预算在 C4 开始前锁定。
- GPU 不可用时可以记录 skip，但 G-CUDA 不能由 skip 验收通过。

### 5.2 性能矩阵与计时口径

性能只在正确性通过的案例上报告。初始规模表不是完整 ResNet benchmark：

| ID | X / W | 用途 |
| --- | --- | --- |
| P0 | [1,3,16,16] / [8,3,3,3]，s1/p1 | 小算子与启动开销 |
| P1 | [4,16,32,32] / [32,16,3,3]，s1/p1 | 中等卷积，观察访存/归约 |
| P2 | [1,3,120,160] / [8,3,7,7]，s2/p3 | 接近驾驶图像尺度的合成输入，非真实驾驶实验 |
| P3 | Pool 输入 [4,16,32,32]、[1,3,120,160] | k2/s2、k3/s2 与 overlap |

先逐个运行一次估测时间和内存，再执行默认预热 10 次、测量 50 次、独立重复 3 组。单个案例默认最多 120 秒，整套性能试验默认最多 30 分钟；超预算明确记为 timeout/未完成。若需减少次数或规模，先更新配置，三后端使用一致的正式案例与次数，并保留不可运行项。

- CPU 使用 perf_counter；GPU 使用同一 stream 的 CUDA Events，并等待 stop event 完成。端到端 wall time 包含同步等待。
- 原始计时样本全部保存，报告平均、中位数、P95、样本数；加速比分别用 NumPy/CUDA C、CuPy/CUDA C 的同口径均值计算。
- 分开测 forward、backward 和 forward+backward；kernel-only 使用已分配连续缓冲区，operator-call 包含输出分配/连续化/发射，transfer-inclusive 另测。没有 kernel-only 路径时标记未测，不能给 wrapper 耗时换名。
- 编译、预热、host↔device 传输与数据生成单独记录；backward 测量前缓存准备一致，输出缓冲区初始化成本按口径明确包含或排除。
- GPU 上只运行一个基准任务；记录 GPU/驱动、CUDA/CuPy、源码 hash、编译选项、实际 kernel/backend、输入 shape/dtype/连续性和运行次序。
- NumPy/CuPy/CUDA C 的执行串行且分别建状态，避免全局 xp/缓存互相污染。需要时用独立子进程；同机负载和温度变化写入实验说明。

## 6. Nsight 分析与优化

Nsight Systems 用于观察 CPU/CUDA 调用、GPU kernel、传输和等待时间线；Nsight Compute 用于分析选定 kernel 的计算/访存、占用率和资源使用。工具分别回答“时间花在哪”与“该 kernel 为什么慢”。

实施步骤：

1. 在 C0 阶段探测 nsys/ncu 的实际安装路径、版本和可采集权限；不以 PATH 缺失直接断言没安装。
2. 先采 P0/P1 中一个短案例，确认能看到自写 kernel 名称；Python 外层可用 NVTX 标注 warmup/forward/backward。
3. 用 Systems 观察现有 CuPy 与直接 CUDA 版本的调用数量、中间操作和空闲段；选择最大耗时项作为分析对象。
4. 用 Compute 过滤一个预热后的目标 kernel，记录时长、访存/计算吞吐、占用与限制项；以实际版本可用字段为准。
5. 选一项有证据的改动，例如线程块大小、连续访存、重复读取或分块归约；保持输入、数学语义和正确性条件不变，重新测量。

正式 G-PROFILE 要有小/中两种规模的分析，至少一个自写 kernel 的 Systems 与 Compute 报告，以及一次优化前后的普通计时数据。没有收益也可以完成“优化尝试”，但必须解释观察结果；学期 B7 的真实瓶颈改进仍按总体目标继续推进。

Profiler 会引入采集开销，其时间不混入普通性能结果。若本机采集被工具/驱动权限阻塞，记录错误、已尝试路径和后续可用环境安排；第 4 次课如实展示阻塞，G-PROFILE 保持未完成，不用任务管理器截图代替。

技术参考：[CuPy RawKernel](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawKernel.html)、[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)、[NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)。这些用于理解工具与接口，实际命令在 C0 根据本机版本确定。

## 7. 最小 Parameter Server 方案

PS 与 CUDA 算子是两条独立验证路径。本阶段不要求它们在同一次训练中组合，也不要求多 worker 使用 GPU。

### 7.1 模型、数据和参数

- 第一轮使用 CPU FP64、无 BN/Dropout 的小 MLP：输入 4 维 → Dense(8) → ReLU → Dense(2)，MSELoss，复用 MBGD，lr=0.01。
- 固定 seed=0 的合成回归数据、32 个样本的 global_batch、20 个更新；单进程、1 worker、2 worker 使用同一初值和相同批次清单。
- 参数映射先明确为 fc1.weight、fc1.bias、fc2.weight、fc2.bias；显式映射到当前 Variable 并断言唯一/shape/dtype，不以对象 id 或重复默认名通信。
- 后续通用 ModelState 复用这个约定；不为最小演示提前重构所有模型/序列化。

### 7.2 进程与每步协议

采用 Windows spawn；进程入口为模块顶层函数，并使用 main guard。Launcher 创建一个 server 和若干 worker，使用有界 Queue/Event 和带超时的消息。

1. server 保存参数和唯一优化器状态，发布 step/version 对应的参数。
2. worker 加载自己的 shard，前向/反向得到本地 mean loss 的梯度，不执行 optimizer.update。
3. 上传 run_id、worker_id、step_id、parameter_version、schema/hash、n_samples、gradients。
4. server 检查同一步每 worker 恰好一份有效消息，按 g=sum(n_i*g_i)/sum(n_i) 聚合，然后更新一次。
5. 发布新参数、聚合 loss 和各阶段计时；worker 进入下一步。

固定每个输出维度的 MSE reduction，所有 worker 与单进程相同。显式测试 16+16、20+12 样本 shard；空 shard 第一版拒绝，不能静默改变有效 batch。重复、陈旧、shape/dtype 不符或非有限梯度消息触发失败并清理本次运行。

server 的 update(var_gradients=...) 前梯度缓存与 acc_no 必须干净，避免本地累积与外部均值再次平均。消息传递的是独立 NumPy 数组快照，不能在异步序列化完成前修改发送缓冲区。

### 7.3 Monitor 与故障处理

每个 worker 记录 compute、上传提交、上传确认、参数等待/接收时间；server 记录收集等待、聚合、更新、广播和整体 step。主机单调时钟统一记录关联 ID；消息字节数区分数组 payload 和实际未测的协议开销。

Queue.put 返回时间不等于接收端完成接收，所以用“提交”和“确认等待”命名；不得把排队/序列化/等待全部声称为网络传输时间。

默认消息超时 10 秒，小模型演示使用该值；大模型另配。检测到超时/worker 崩溃后，Launcher 发停止信号、回收队列并 join；必要时终止本轮创建的子进程，5 秒内完成清理。记录非零退出码和失败步骤，不遗留后台进程。

验收：1/2 worker 对单进程的单步及 20 步参数满足 atol=1e-8、rtol=1e-6；所有 step 的有效样本数、参数版本正确；故障注入覆盖崩溃、超时、重复与错误 schema。FP64 接近零时使用联合容差，不要求不同归约顺序逐 bit 相同。

4 worker 的小 MLP 等价验证已完成；无 BN 小 CNN 与更多通信优化在后续学期集成中继续。

## 8. 任务拆分与建议位置

以下路径已实现，阶段报告逐项关联验收证据。C0-C7 是总体 W0/W1/W2/W3 中的阶段子任务，不能重复计算工时。

| 顺序 | 任务与建议位置 | 依赖 | 可独立审核的结果 |
| --- | --- | --- | --- |
| C0 | 统一测试收集、已有 seed/偶发测试修复；环境/双仓库清单、FP32 fixture、工具探测及基线入口 | 无 | 第 2.1 节基线门槛通过；NumPy/CuPy 同输入结果；RawKernel 小样例与工具能力记录 |
| C1 | MyFlows/ops/cuda_native/native_cublas.cpp/native.py；卷积前向 | C0 | T0-T4 前向正确，能识别实际 kernel |
| C2 | 同目录 Conv dX/dW/db | C1 | 同一 dy 下三类梯度通过参考检查 |
| C3 | MyFlows/ops/cuda_native/native_cublas.cpp/native.py；Pool 前反向/context | C0；优先在 C2 后推进 | tie、负值、重叠回传正确 |
| C4 | ops/convolution.py、layers/layer.py 透传 backend；小模型 seed/dtype | C1-C3 | 原图小 CNN 训练、共享梯度与旧后端回归通过 |
| C5 | benchmark/cuda_ops.py 完整矩阵；Nsight 分析和优化记录 | C1 起可初测；完整验收依赖 C2-C4 | 三后端误差/耗时、两种规模 profile、优化尝试 |
| C6 | MyFlows/distributed/ 的 protocol.py/ps.py/worker.py/launcher.py；benchmark/ps_demo.py | CPU 环境和固定初值就绪；独立于 C1-C5 | 1/2 worker 等价、计时、故障清理 |
| C7 | docs/experiments/semester_2026_fall/stage1/ 的实验包与报告 | 已有结果逐步归档 | 第 4 次课可展示材料；完整阶段验收清单 |

测试位于 MyFlows/tests/test_cuda_native_cublas.py、test_cuda_graph_integration.py、test_ps_training.py；benchmark 产物/错误状态与源码恢复测试由根仓库 tests/test_stage1_artifacts.py 承担。tools/run_tests.py 统一收集 unittest 和函数式测试。

单人执行建议：C0 → C1 → C2 → C3 → C4，C5 从 C1 开始随结果推进；C6 在第 4 次课前单独安排，不能一直等 GPU 优化结束。多人可将 C6 独立分工，但本 Spec 不假设具体组员人数。

### 运行命令

以下命令已实现，必须使用新的输出目录。隔离环境安装、Nsight 完整命令与源码恢复流程见阶段报告。

~~~powershell
.\.venv\Scripts\python.exe -m benchmark.cuda_ops --suite correctness --backends numpy,cupy,cuda_native_cublas --seed 0 --out-dir docs/experiments/semester_2026_fall/stage1/run-001
.\.venv\Scripts\python.exe -m benchmark.cuda_ops --suite performance --backends numpy,cupy,cuda_native_cublas --seed 0 --out-dir docs/experiments/semester_2026_fall/stage1/run-002
.\.venv\Scripts\python.exe -m benchmark.ps_demo --workers 2 --global-batch 32 --steps 20 --seed 0 --out-dir docs/experiments/semester_2026_fall/stage1/ps-002
~~~

入口必须拒绝覆盖已存在的正式 run 目录，提供有效配置、成功/失败状态和明确退出码。GPU 被要求但不可用时非零退出；CPU smoke 可以独立运行。Nsight 使用短 case 过滤入口，具体 profile 参数在工具探测后补入运行说明。

## 9. 阶段验收与汇报产物

### 阶段完成清单

- [x] G-BASE：测试收集和已有 seed/偶发测试问题已处理，127 项完整回归连续 3 次通过；当前路径的 dtype/参考结果/计时可信。
- [x] G-CUDA：自写 Conv Y/dX/dW/db 和 Pool 前反向通过固定矩阵；dtype/shape/实际 backend 可追踪。
- [x] G-CUDA：原计算图小 CNN 能训练，梯度累加和 context 更新正确，原 NumPy/CuPy/高级卷积功能的回归通过。
- [x] G-BENCH：三后端同输入、P0-P3 完整规模矩阵，均值/P95/加速比/绝对和相对误差有原始记录，无案例缩减。
- [x] G-PROFILE：Systems/Compute 原始报告、瓶颈证据、一次优化尝试及普通计时前后对照。
- [x] G-PS：1/2 worker 单步及 20 步等价、非均匀 shard 加权、状态/时间/字节记录、失败清理；另完成 4 worker。
- [x] G-REPORT：运行命令、隔离环境与源码恢复、两个仓库的实际源码状态、数据/fixture 指纹、结果与限制可复现。

逐条核对见 [acceptance.md](experiments/semester_2026_fall/stage1/acceptance.md)，原始数据与交付包见 [阶段报告](experiments/semester_2026_fall/stage1/README.md)。

阶段结果不等于整个学期验收：小 MLP 已覆盖 4 worker；全量 GPU 监控、完整分类、PS 小 CNN 集成、跨框架/驾驶评估和 Agent 继续由总体 Spec 跟踪。

### 每次正式运行保存

run 目录至少有 config.json、manifest.json、results.json、timings.csv、stdout.log、report.md；按任务增加 gradients/fixture 文件、PS events.jsonl、.nsys-rep/.ncu-rep 和图表。只保存本任务实际产生的字段，不为 CPU PS 伪造 GPU 指标。

manifest 记录两个仓库 HEAD/dirty/diff 及未跟踪源码清单、源码 hash、环境版本、设备、fixture hash、seed、计时口径和产物校验和。历史版本通过 Git 提交和工作树状态恢复，不再保存 source-at-run.zip。原始性能行包含场景、后端、阶段、形状、精度、重复编号、耗时、误差、状态/失败原因。

当前 .gitignore 忽略实验 JSON/CSV/图片和权重；正式报告需包含实验目录、Git 版本和复现命令，不能只提交 Markdown 后假定结果已交付。保留本地旧模型、数据和历史资产，不重新制作往期材料。

第 4 次课建议按“现有卷积 → CUDA 线程映射与结果 → Nsight 观察 → PS 参数更新流程 → 下一步”汇报。每张图都关联 run_id，明确已做、未做和当前结论。

## 10. 风险与阶段结束后的衔接

| 风险 | 本阶段处理 | 后续衔接 |
| --- | --- | --- |
| CUDA C 慢于 CuPy | 保留结果，用 profile 定位；先维持直接版本作为正确性基线 | 再研究分块、共享内存或自写 im2col/col2im + 库 GEMM 混合路径，报告清楚自写边界 |
| 现有 FP64/融合路径影响接入 | fixture FP32 起步，小 CNN 集成前统一精度并显式关闭融合 | W0 完成完整 ResNet/优化器精度；融合和导出兼容继续回归 |
| Nsight 不可采集 | C0 提前发现；记录具体阻塞和替代验证环境安排 | 保留未完成项，不以截图代替 profiling |
| 最小演示被通用框架重构拖住 | C6 显式小模型参数映射，C1 直接用数组接口 | W1 再统一 ModelState/TrainRunner/TrialResult |
| 第 4 次课前时间不足 | 优先可手算的 Conv、首个 profile、最小 PS；保留反向/池化的真实进度 | 未完成 G 项继续收尾，不提前宣告阶段通过 |

本阶段产生可调用的算子、可解释的 profile 和可运行 PS 后，再细化 Agent 如何调用这些能力。这里只约定普通配置、结果和 run_id，不提前设计 Agent 的六组件、搜索策略或记忆系统。
