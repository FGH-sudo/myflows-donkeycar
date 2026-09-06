# 本学期 Spec 审核记录

- 审核日期：2026-09-05
- 对象：2026-09-02 初稿；本记录对应 v0.2 审核时点。后续阶段顺序以最新 [semester_spec.md](semester_spec.md) 和 [第一阶段 Spec](stage1_cuda_ps_spec.md) 为准。
- 依据：[课程 PDF](<深度学习框架-16 综合项目III.pdf>) 共 12 页、当前工作区代码、实际测试与小规模探针。
- 范围：用户已确认之前学期进度目标完成；不重新审计往期交付是否齐全。以下问题均针对本学期新增开发及其验收。

## 1. 审核发现

结论：原稿的大方向和基础/拓展划分基本正确，但尚不足以直接指导并行模块开发。最需要修正的是先后顺序、共享训练状态、可重复实验与验收方法，而不是增加更多模型或算法。

### R1 [P1] 实际精度与测试覆盖未进入前置任务

原稿 §4.1 仅要求修复偶发测试和统一参数名，§4.2/§4.9 直接假定可以开展 FP32 CUDA 和跨框架实验。当前训练入口只为输入/标签选择 dtype，Conv/Dense 初始化及 BN buffer 仍默认 FP64。小型 ResNet 实测 82 个参数与输出均为 FP64；按现状启动 FP32 RawKernel 或性能对比会出现错误解释、类型不匹配或不公平的比较。

此外，默认 unittest discovery 不收集 [test_resnet18_smoke.py](../MyFlows/tests/test_resnet18_smoke.py) 的 7 个函数式测试和 [test_serialization.py](../MyFlows/tests/test_serialization.py) 的 3 个函数式测试。“68 项全部通过”不能代表这些核心测试也被执行。

修订：W0 明确纳入全链路 dtype、统一收集和可复现初始化；参数、buffer、梯度、优化器状态的精度都要验证。保留全部既有测试，不为得到稳定结果而删除困难用例。

证据：[train_myflows_donkey.py](../apps/train/train_myflows_donkey.py)、[layer.py](../MyFlows/layers/layer.py)、[resnet.py](../MyFlows/layers/resnet.py)。

### R2 [P1] 排期让拓展项挤占基础验收，分类入口缺少明确任务

原稿 §5 在第 7-12 次课集中安排 TPE、自动恢复、三模型和 All-Reduce，基础的跨框架比较及驾驶闭环却到第 13-14 次课才开始；§4.10 虽提 CIFAR-10，但没有单独的分类训练/评估入口任务。这样容易得到多个拓展演示，却来不及验证 PDF 第 3-4 页的基础目标。

PDF 第 7 页要求第 8 次课汇报 Agent 的调研和设计，并未要求该节点完成 TPE 和回滚。监控指标在第 10 页有明确要求，应尽早成为基线和 Agent 的输入。

修订：第 5-6 次课前形成监控、分类、跨框架和驾驶基线；第 8 次课展示 Agent 设计及小预算闭环；第 12 个节点整合 B1-B7。基础通过后再选拓展。W5 独立负责分类流程，不依赖预训练库。

### R3 [P1] Agent 缺少专业任务职责与精确的恢复边界

原稿 §4.7 有六组件，但遗漏 PDF 第 11 页列出的专业子 Agent 的职责；也未定义指标如何驱动下一次配置选择。只运行候选列表不足以验证“性能诊断、优化建议生成”。原稿基础搜索冻结阶段，而冻结工作流又到拓展预训练阶段才实现，形成依赖倒置。

现有 [checkpoint.py](../MyFlows/utils/checkpoint.py) 已保存模型、BN buffer 和优化器，不需要推倒重做；但没有 RNG、数据游标和完整训练控制状态。训练 resume 按 epoch+1 启动，step 内保存后不能据此承诺精确回滚。

修订：明确专业任务适配器、共享 TrainConfig/TrialResult、预算与验证目标；按能力开放 PS worker 和冻结搜索。基础支持调度记录恢复，完整 TrainingState 与训练回滚属于 E3，分别测试。规则型 Agent 可以满足首版，不强行增加 LLM 服务依赖。

### R4 [P1] 现有 benchmark 不能直接承担“同条件比较”

[compare_frameworks.py](../benchmark/compare_frameworks.py) 当前仍使用 VGG11，且：

- 三框架没有统一初始权重，只有样本选择 seed。
- MyFlows 补齐尾批，PyTorch/Paddle 使用真实尾批大小。
- GPU 计时没有一致的显式同步与预热协议。
- Paddle 在计时外建立设备数据，PyTorch 在 batch 内传输；计时边界不同。
- _peak_mb() 返回调用时的 RSS，未采样真实峰值；同一进程依次运行也会污染资源统计。

这些不否定旧脚本作为开发工具的价值，但不能直接作为新学期 ResNet18 的公平性证据。原稿只列 batch/lr/seed 等条件，还未定义结构、BN、初值、尾批和测量口径。

修订：W5 在正式计时前先做同构模型单步等价性，统一初值和批次清单、真实 dtype、优化器细节，并用独立进程区分纯计算与端到端测试。

### R5 [P2] PS 一致性缺少样本加权、版本协议和 BN 前提

原稿 §4.4 的“固定总 batch 后结果一致”未说明本地均值梯度、不同大小 shard、陈旧消息、BN 与 Dropout。当前硬件只有一张 8 GB 级别 GPU，多进程并不自动带来训练加速。

现有 [opt.py](../MyFlows/train/opt.py) 已有 apply_gradients/update 入口，可复用，但梯度缓存以进程内 Variable 为 key；需要稳定名字映射，不能直接跨进程传对象身份。[pipeline.py](../MyFlows/data/pipeline.py) 只有数据加载并行，不是 PS。

修订：首版 CPU spawn，同步 PS；唯一 server 更新，按样本数加权，携带 step/version/schema。先用无 BN/Dropout 小模型验证 1/2/4 worker，后续 ResNet 明确 buffer 和 BN 策略，异常时有限时间清理。

### R6 [P2] ResNet18 预训练被低估成单纯“载入参数”

当前模型不完全等同于本机 torchvision ResNet18：imagenet stem 的 MaxPool 没有 padding，224 输入经 stem 池化得到 55x55，torchvision 同阶段为 56x56；当前 Conv 默认有 bias，FC 存储方向也不同。当前 DonkeyCar 默认 cifar stem 不能直接载入标准 ImageNet stem 权重。

即便设置 trainable=False，BN 的 running_mean/running_var 仍需独立冻结；外层 model.train(True) 也会覆盖子层状态。MobileNetV2 的 ReLU6/倒残差和 AlexNet 的分类头、资源成本需要单独工作包。

修订：E2 先做兼容结构和完整主干/BN 映射，在原 1000 类头上比较来源输出，再替换任务头。保持旧模型模式与 checkpoint 可用。第三模型仍为候选，不把三模型一次性训练设为基础阻塞。

证据：[resnet.py](../MyFlows/layers/resnet.py)、[MaxPool2d_Op](../MyFlows/ops/convolution.py)、[batchnorm.py](../MyFlows/ops/batchnorm.py)；来源参考为本机 torchvision 0.26.0+cu128 的 models/resnet.py，而不是未验证的远程最新版。

### R7 [P2] 数据、资源和实验包验收过于含糊

原稿没有隔离验证/测试集的调优角色，也没说明驾驶“多次”“出界”“完成”如何计算。当前 [splits.py](../apps/common/splits.py) 按行随机划分、存索引，未校验源数据指纹；连续驾驶帧存在跨集合近邻泄漏风险，需要检查采集结构后决定分组。

油门标签 0.5 与驾驶执行 0.2 承担不同用途。两者不同不意味着必须修改标签或提升驾驶速度；首版固定执行 0.2、比较转向和闭环即可。

当前 .venv 继承系统依赖；.gitignore 忽略实验 JSON/CSV/图片与权重。只记录根仓库 commit、把产物放入 docs/experiments，并不能保证其他成员可复现。

修订：固定验证集选择配置、测试集最终评估；至少 5 次配对驾驶；预定义出界/完成口径。监控缺测与峰值类型显式记录；归档双仓库版本、dirty 源码、数据清单、实验包位置和校验和。

## 2. 当前能力与开发增量

| 领域 | 可复用的现有能力 | 本学期需要补的部分 |
| --- | --- | --- |
| 框架 | 动态图/自动求导、NumPy/CuPy、Conv/Pool/BN、优化器 | seed/dtype 状态契约、自写 CUDA C、性能与误差证据 |
| 模型与训练 | ResNet18、DonkeyCar 训练/验证/早停、分类算子 | 独立分类入口、稳定 ModelState、统一 TrainRunner、预训练兼容 |
| 保存与恢复 | JSON+NPZ、BN buffer、优化器状态、epoch resume | 原子发布、RNG/游标/控制状态、调度恢复与精确回滚区分 |
| 数据并行 | 多进程 DataLoader | PS/All-Reduce 的真实梯度通信、同步与进程管理 |
| 可视化 | TensorBoard、loss/参数/梯度、部分耗时 | GPU 硬件采样、设备/墙钟分阶段计时、瓶颈诊断输入 |
| 应用 | 固定油门驾驶入口、ONNX、离线评估、服务资产 | 自动闭环量化、训练改进前后对照、同条件三框架实验 |
| Agent | 尚无训练调优系统；MyFlowsPilot 是驾驶控制器 | 六组件、专业任务适配器、自动选择/诊断/校验/报告 |

这不是重新评定往期项目；已有能力是本学期的起点。缺少新学期新增模块不等于此前目标未完成。

## 3. 本次验证

所有项目命令在根目录使用 .venv/Scripts/python.exe，日期均为 2026-09-05。

| 检查 | 实际结果 | 结论边界 |
| --- | --- | --- |
| -m unittest discover -s tests -v | 21/21 通过，测试耗时 0.926 s | 含已有服务/导出回归，不代表闭环驾驶 |
| -m unittest discover -s MyFlows/tests -v | 68/68 通过，测试耗时 7.209 s；CUDA dense 用例通过 | 只统计 unittest 收集范围 |
| MyFlows/tests/test_resnet18_smoke.py | 7 个函数检查通过 | 本次手工补跑，不在上述 68 项 |
| MyFlows/tests/test_serialization.py | 3 个函数检查通过 | 参数/BN roundtrip 与 ONNX 导出；不是完整训练回滚 |
| -m unittest MyFlows.tests.test_digits_conv -v | 3 个独立进程重跑均通过 | 不证明随机初始化问题已经消除 |
| 初始化器探针 | 重置 np.random.seed 后两次默认初始化不相同；显式 seed=0 的初始化相同 | 证明 seed 传播缺口；尚未修改代码或证明全部偶发原因 |
| 精度探针 | base_width=4 的 ResNet18 CIFAR：82 个参数 FP64，FP32 输入产生 FP64 输出 | 证明默认模型路径未遵守输入精度 |
| stem 池化探针 | MyFlows 55x55；PyTorch padding=1 的池化为 56x56 | 证明 ImageNet 兼容前置工作 |
| -m tools.analyze_donkey_data --data mycar/data | 10,000 条，缺图 0，油门 min/max/mean=0.5，std=0 | 未改动或新采数据 |
| nvidia-smi 查询 | RTX 4060 Laptop GPU，8188 MiB，驱动 591.74 | 未跑性能 benchmark |
| 工具可见性 | PATH 未找到 nvcc/nsys/ncu；本机 CuPy RawKernel 接口存在 | 不能据此断言工具没安装；未验证编译/profile 权限 |

本机主要版本：Python 3.11.7、NumPy 2.2.6、CuPy CUDA12x 14.0.1、torch 2.11.0+cu128、torchvision 0.26.0+cu128、paddlepaddle-gpu 3.3.1、psutil 7.2.2。Paddle 包存在尚未等同于其 GPU 路径已验收。FastAPI 测试出现上游弃用警告，不影响本次通过结果。

原始测试输出保存在本次任务工具记录中；本文件是审核摘要，不冒充正式训练实验包。没有开展完整训练、三框架速度比较、Nsight profiling 或模拟器闭环，因此没有新的精度/加速比/驾驶性能结论。

文档修改后已检查本次三份 Markdown 的本地链接和尾随空格，均通过；根仓库和 MyFlows 的 git diff --check 通过，仅有已有文件的 LF/CRLF 提示。

## 4. Git 与落地安排

- 根仓库 HEAD：76fa7b264b37b407e8223cc9bfe8238c316dab08。
- MyFlows HEAD：50b4f4891aa0471ac69b3582eb44910ce4d4d3ee。
- 两个仓库在审核开始前已有未提交代码、文档修改及删除；根仓库保存 MyFlows 的 gitlink，但没有完整 .gitmodules 映射。以上 HEAD 不包含现有工作区改动。
- 本次仅修订 semester_spec.md、新增本审核记录，并更新 docs/README.md 入口；不提交、不回滚、不更改框架/应用源码或本地资产。

v0.2 审核时建议先开展 W0/W1。后续讨论已调整为独立 CUDA fixture/kernel 和最小 PS 优先，通用接口分阶段补齐；本记录保留代码证据，实际开发顺序以最新总体 Spec §5/§9 与第一阶段 Spec 为准。

人数、每人周工时和当前课次尚待补充；Spec 的 192-300 基础人时及 25% 余量是初始估算，应在 W0/W1 完成后用实际速度校准。三模型、TPE、All-Reduce 的额外工时单列，不默认全部挤进课程剩余时间。
