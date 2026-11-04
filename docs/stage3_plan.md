# 第三阶段开发规划：课程任务二、迁移学习与遗留任务

- 状态：v0.2（2026-11-05），已记入用户对第 6 节各事项的决定，交给开发者执行
- 依据：[semester_spec.md](semester_spec.md) v0.6（第 1、4.3、4.6、4.8–4.10、7 节）；[深度学习框架-19 CUDA 编程（下）.pdf](<深度学习框架-19 CUDA 编程（下）.pdf>) 第 47–82 页；[深度学习框架-24迁移学习.pdf](<深度学习框架-24迁移学习.pdf>) 第 15–48 页；[深度学习框架-16 综合项目III.pdf](<深度学习框架-16 综合项目III.pdf>) 第 3–6 页
- 代码基线：根仓库 aaedaf2，MyFlows ce63df5；统一回归 446 项通过（2026-11-04）。文中的行号以这两个提交为准
- 范围：第 19 讲（下）的课程任务二（监控与瓶颈分析）与优化清单；第 24 讲的迁移学习组件、方法与实验；此前遗留的基础项 B4（跨框架比较、闭环评估）、B5（图像分类测试、测试计划）、B6（文档与报告）
- 约定：本文给出目标、改动位置、验收标准与顺序，实现细节由开发者决定，但不能降低验收标准。与 semester_spec 冲突时，以 semester_spec 的验收口径为准

## 1. 工作包总览

| ID | 内容 | 对应 | 层级 | 依赖 |
| --- | --- | --- | --- | --- |
| S1 | 课程任务二收尾：落盘格式定版、训练入口自带采样、监控开销、时间线定位、同口径优化、交付报告 | B7；第 19 讲（下）第 70–81 页 | 基础 | 无 |
| S2 | 冻结阶段的反向截断 | E2；兼作 S1.5 的同口径优化 | 基础（性能缺陷） | S1.2–S1.4 提供测量手段 |
| S3 | 迁移学习组件与微调方法 | E2；第 24 讲第 44–48 页 | 拓展 | S2 |
| S4 | 迁移学习实验：分类递进实验、DonkeyCar 主任务正式对照 | E2；第 24 讲第 23–42 页 | 拓展 | S3 |
| S5 | 跨框架同条件比较（MyFlows 与 PyTorch，ResNet18） | B4；semester_spec 4.9 | 基础 | S1.1（记录格式） |
| S6 | DonkeyCar 闭环评估 | B4；semester_spec 4.8 | 基础 | 无（S4.2 的模型可后加） |
| S7 | 图像分类系统测试（CIFAR-10） | B5；semester_spec 4.10 | 基础 | 无（数据集已下载，见 S7） |
| S8 | 算子与显存优化 | 第 19 讲（下）第 55–78 页 | 基础 + 拓展 | S2 之后串行 |
| S9 | 测试计划、性能门槛与 CI | B5；第 16 讲第 6 页 | 基础 | 无 |
| S10 | AutoPilot 增量：失败归类、算子瓶颈分类、迁移策略先验 | B1 增量 | 拓展 | S1.4、S3 |
| S11 | 文档与报告 | B6 | 基础 | 随各项更新 |

每个工作包的通用完成标准：

- 代码有单元测试，统一回归全部通过；
- 涉及实验的项有原始结果与 `report.md`，归档在 `docs/experiments/semester_2026_fall/` 下；
- 涉及数值的改动重跑与参考实现的对拍（2.2 节）；
- 先用小规模数据跑通，再运行全量（2.4 节）；
- 相关文档同步（S11.1）。

## 2. 开发约定与环境注意事项

开工前必读。这些都来自本学期实际踩过的问题。

### 2.1 运行环境

- Windows 11 + 仓库内的 Python 3.11 虚拟环境 `.venv`，命令用 `.venv/Scripts/python.exe`（Linux/WSL 用 `.venv/bin/python`）。依赖用 `uv pip install` 安装，不要运行 `uv sync`：它会移除单独安装的 torch cu128 wheel。新增依赖先征得用户同意。
- 统一回归：`.venv/Scripts/python.exe -X utf8 -m tools.run_tests --scope all`（446 项，约 9 分钟；`--scope framework` 或 `--scope apps` 只跑一部分）。dsh 守卫的 node 测试由 `tests/system/test_autopilot_dsh.py` 调用。
- 原生库：`MyFlows/ops/cuda_native/build.py` 在源码哈希变化时自动重新编译（文件锁保证并发进程只编一次），产物在 `build/native-cuda/`；`MYFLOWS_NATIVE_AUTOBUILD=0` 关闭自动编译。
  - DLL 内嵌的 kernel 源码必须拆成每段小于 16 KB 的原始字符串，否则 MSVC 报 C2026。
  - NVRTC 冷编译约 7.5 s，只在首次运行或源码变化时发生，之后命中缓存。
  - `MYFLOWS_NATIVE_SYNC=1` 让每次 launch 后同步，用于把异步错误定位到 kernel。
  - 查看寄存器与溢出：把各段源码拼接后调用 `cupy.cuda.compiler.compile_using_nvrtc(src, options=('--std=c++17', '--ptxas-options=-v'), arch='89', log_stream=...)`，不要自己传 `--gpu-architecture`（CuPy 会加）。
- 不要在 MyFlows 的运行时模块里 `import torch`。torch 自带的 NVRTC 12.8 与 CuPy 使用的 12.6 文件名相同，先加载会让 CuPy 的 CUB kernel 编译失败；`MyFlows/core/device.py` 用 `find_spec` 规避，不能改回 import。torch 只在测试与对照脚本中使用；对照实验里 MyFlows 与 torch 各在独立进程运行。
- CuPy 在导入时缓存 `CUDA_PATH`。单独的 CuPy 脚本要先调用 `MyFlows.core.device._lazy_cupy()`，否则规约 kernel 会用系统 CUDA 13 的头文件编译并失败。CuPy 的磁盘缓存会掩盖这类编译问题，验证编译路径时先清缓存。
- `core.autocrlf=true`，工作区出现 CRLF 属正常。

### 2.2 数值与确定性

- 原生路径与 CuPy 路径目前逐位可复现：规约不用原子加，按固定顺序累加；Adam 的 CPU 与 GPU 结果逐位一致。新代码保持这一点，做不到时在报告中说明原因。
- 用 FP64 参考评估 float32 梯度时，要在多个 batch 上统计（`benchmark/resnet_before_after.py --seeds`，默认 1–6），ReLU 输入的异号单独统计。单个 batch 的结论不可靠；约 1% 的梯度误差通常来自 ReLU 处的异号，而不是累加误差。
- 涉及数值的改动（融合、重算、换算法、混合精度）都要重跑对拍。前后对照的方法见 [CUDA 第 19 讲实验报告](experiments/semester_2026_fall/cuda_lecture19/README.md) 第 4 节：用 git worktree 检出改动前的两个仓库，编译其自己的 DLL，再运行同一对照脚本。

### 2.3 测量方法

- 本机为 RTX 4060 Laptop GPU（8 GB，24 SM，sm_89）。
  - 桌面程序让空闲时的 GPU 利用率就有 25–48%。
  - 功耗上限在 60–80 W 之间变化。
  - 同一配置两次运行相差可达 10–15%。

  因此前后对照要在同一会话内交替运行（A/B/A/B），报告中位数与离散度；比较单个 kernel 用 Nsight Compute 的 `sm__cycles_elapsed.max`，不用跨次运行的墙钟时间。
- 设备显存超过约 7.7 GB 时，WDDM 会把显存换到内存而不是报 OOM，step 慢约 25 倍，看起来像卡死。显存以 CuPy 显存池为准：WDDM 下 NVML 读不到单个进程的显存。
- Nsight 工具用 `benchmark/nsight.py` 的 `find_tool()` 查找。采集结果保留 `capture.sqlite`，用 sqlite3 只读查询：`CUPTI_ACTIVITY_KIND_KERNEL` 按 `shortName` 关联 `StringIds`，另有 `CUPTI_ACTIVITY_KIND_MEMCPY`、`CUPTI_ACTIVITY_KIND_RUNTIME` 与 `NVTX_EVENTS`。

### 2.4 训练与机器使用

- 开发与验证先用小规模数据（`--max-samples`、少量 epoch）跑通，再运行全量。2026-11-05 用户确认：第三阶段期间电脑全程空闲，全量数据训练、正式 campaign 与模拟器可以直接运行，不必逐次征求同意；每次较长的运行记入第 8 节。
- 机器是用户自己的笔记本：不要留下开机自启或无人值守的训练；自己启动的控制台、服务与模拟器用完即停。
- 正式 AutoPilot campaign 运行期间不要修改 `MyFlows/` 与 `apps/train/`：trial 每次新起进程，代码指纹却在排队时记录。
- 不要动用户本机的 dsh（`~/.dsh`、全局安装、`npm link`）；`../dsh-myflows` 保持上游原样。

### 2.5 Git 与隐私

- `MyFlows/` 是子模块：先在 MyFlows 内提交，再在根仓库 `git add MyFlows` 并提交。建议每个工作包一个分支，合入方式由用户决定。
- 两个 GitHub 远端都是公开仓库。推送前先征得用户同意，并检查新增内容里没有本机绝对路径、用户名、邮箱、令牌或私网地址。受版本控制的文件里，项目路径一律写成相对仓库根目录的形式。
- 提交信息不加 AI 署名类的 trailer（Co-Authored-By 等）。
- `runs/`、`mycar/data/` 以及 `docs/experiments/` 下的 JSON/CSV/NPZ 不纳入 Git。实验只提交各运行目录的 `report.md`，原始数据的位置写在报告里。

### 2.6 文档

- `docs/semester_spec.md`、`docs/autopilot_agent_spec.md`、`docs/README.md` 与本文由用户审核和提交；开发中发现需要改动时，在报告中提出。
- `docs/system_design.md` 与 `docs/module_design.md` 只写当前代码事实，不能提前写成“已经实现”。

## 3. 当前基线（复测对照用）

以下数值在 2.3 节的噪声条件下测得。

| 项目 | 数值 | 来源 |
| --- | --- | --- |
| `apps.train` 默认配置（cifar stem、w64、batch 2）每步耗时 | 47.6 ms（register 引擎）、47.2 ms（cuBLAS） | CUDA 第 19 讲报告 5.5 节 |
| ResNet18 w16、batch 128、训练模式 BN 每步耗时 | 51.4 ms（register）、53.4 ms（cuBLAS） | 同上 |
| GPU 端到端梯度误差（相对 FP64） | 中位 3.0e-6–4.0e-6 | 同上 5.3、5.5 节 |
| 正式 campaign formal-full-002（全量、3 个 epoch、batch 2） | 约 35 samples/s；数据读取占比 2–4%；GPU 利用率约 95%；76% 的采样功耗达到上限；每个 trial 约 15 分钟 | 本地 `runs/autopilot/formal-full-002/report/report.md` |
| 人工基线 val_angle_mae（seed 1–3） | 0.0344 ± 0.0152 | 同上 |
| 测试集 angle MAE | 基线 0.0369，候选（lr 1e-4）0.0318 | 同上 |
| 显存 | w64、cifar stem、batch 16 时 layer1 每个卷积的 im2col 缓冲约 708 MB；w16、cifar stem、120×160 时 batch 42 开始换页 | CUDA 第 19 讲报告第 6 节；AutoPilot 演示 |
| 预训练权重等价 | 最大绝对差 ≤ 1.3e-5，top-5 一致 | `tools/pretrained_report.py` |
| 分类迁移验证（3000 个样本、2 个 epoch、冻结 3 个阶段） | 测试准确率 ResNet18 0.92、MobileNetV2 0.917、AlexNet 0.953 | 本地 `runs/transfer/` |

## 4. 工作包

### S1 课程任务二收尾（B7）

现状：

- `MyFlows/monitoring/gpu.py` 已有 `GpuSampler`（NVML 优先，nvidia-smi 兜底）与 `ResourceRecorder`。
- 控制台为每个任务每 0.5 s 采样一次（`apps/console/server.py:55`），写 `resources.jsonl`。
- `apps/train/train_myflows_donkey.py` 的 step 行有 `data_load_ms`、`train_step_ms`、`step_time_ms`、`samples_per_sec`；每 `--phase-timing-every` 步再加 `forward_ms`、`backward_ms`、`optimizer_ms`（CUDA Event，`:104`、`:137`）与 CuPy 显存池的区间峰值。

缺口：

- 没有格式版本与说明文档；
- 命令行直接运行时不采资源；
- 没有 TensorBoard 资源曲线；
- 正式训练路径没有监控开销数据；
- 训练循环没有 NVTX 标注（NVTX 只出现在 `benchmark/` 的脚本中）。

**S1.1 落盘格式定版**

- 为 `config.json`、`metrics.jsonl`、`resources.jsonl`、`rank-*/steps.jsonl`、`result.json` 定义 v1 格式：
  - JSON 对象带 `schema_version`，JSONL 每行带 `schema_version`；读取方把没有该字段的旧记录当作 v0。
  - 写明字段名、类型、单位与可空规则。硬件不支持的项为 null，并带缺测原因，不写 0。
  - 写明时钟：`perf_counter`、`monotonic_s`、`unix_s` 各用在哪里，如何对齐。
  - 写明采样频率。
  - 显存的三种口径分开命名：NVML 设备显存、CuPy 显存池（已用与申请总量）、进程 RSS。
- 给出 12 项指标与字段的对照表：GPU 利用率、显存、显存峰值、温度、功耗、CPU 利用率、单步时间、forward、backward、优化器更新、数据读取、samples/sec。
- `result.json` 增加 `environment`：GPU 型号、驱动、CUDA 与 NVRTC 版本、Python 与主要依赖版本、原生库源码哈希。复用 `apps/autopilot/provenance.py:83` 的 `environment()`；必要时移到 `apps/common/`，让训练入口不依赖 autopilot 包。
- 新增格式说明 `docs/monitoring_format.md`，以及校验工具 `tools/check_run_format.py <run-dir>`。
- 读取方同时接受 v0 与 v1：控制台的 `apps/console/registry.py`、`readers.py`，AutoPilot 的 `apps/autopilot/evidence.py`、`verifier.py`、`reporter.py`。
- 验收：
  - 新运行的全部文件通过校验工具；
  - 用一个旧运行目录作 fixture，确认旧运行仍能在控制台与 AutoPilot 报告中读取；
  - 格式说明覆盖全部字段。
  - 课件要求定版后不再改动，所以定版后字段只增不改。

**S1.2 训练入口自带资源采样与 TensorBoard 资源曲线**

- `apps.train` 增加 `--monitor {auto,on,off}` 与 `--monitor-interval-s`（默认 0.5）。
  - `auto` 在有 `--run-dir`、且不是由控制台或 AutoPilot 启动时开启。
  - 是否已有外部采样，用一个环境变量标识：控制台为任务启动 `ResourceRecorder` 时设置它，避免重复采样。
- 采样放在独立子进程中，与训练进程不争 GIL，例如 `python -m MyFlows.monitoring.record --pid <训练进程> --out <run-dir>/resources.jsonl --interval-s 0.5`。训练结束或异常退出时，子进程随之退出。
- TensorBoard：训练进程在写 step 日志的间隔里增量读取 `resources.jsonl`（不额外调用 NVML），写 `resources/*` 标量；`MyFlows/utils/training_dashboard.py` 增加对应方法。
- 验收：
  - 命令行运行 `apps.train --run-dir ...` 得到与控制台任务同格式的 `resources.jsonl`；
  - 由控制台启动时不重复采样；
  - TensorBoard 中有资源曲线；
  - 有使用假采样器的单元测试。

**S1.3 监控开销测量**

- 新增 `benchmark/monitor_overhead.py`。
  - 配置：正式配置的缩小版，cifar stem、w64、batch 2、`--max-samples 600`、1 个 epoch。
  - 三档监控（关闭、0.5 s、0.1 s 压力档）各跑 3 次，交替运行，每次独立进程。
  - 比较稳态 step 的中位耗时与 samples/s（去掉预热步），同时记录每次运行顶到功耗上限的采样比例。
- 验收：报告给出开销及其离散度。0.5 s 档的开销要低于 5%，否则降低频率或改为单独的诊断运行，并写进格式说明。
- 产物：`docs/experiments/semester_2026_fall/monitoring/overhead-001/report.md`。

**S1.4 训练时间线与定位工具**

- `apps.train --nvtx`：用 `cupy.cuda.nvtx.RangePush/RangePop`（用法同 `benchmark/cuda_ops.py:247`）标注每一步 `step <n>`，内部再标 `data`、`forward`、`backward`、`optimizer`、`sync`（多 Worker）、`validation`、`checkpoint`。
- `--nvtx-nodes`：再为计算图每个节点的前向与反向标注节点名，并输出 `nvtx_map.json`，记录每个 range 名称对应的源码位置（节点类的 `forward`/`backward` 所在文件与行号，用 `inspect` 取得）。
- `--profile-steps A:B`：只在第 A 到 B 步之间调用 `cupy.cuda.profiler.start()/stop()`，配合 `nsys profile --capture-range=cudaProfilerApi`，只采这一段。
- 新增 `benchmark/profile_train.py`：找到 nsys，运行一小段训练，导出 `capture.sqlite`。
- 新增 `tools/timeline_gaps.py <capture.sqlite>`：
  1. 由 kernel、memcpy、memset 活动求出 GPU 忙区间，找出超过阈值（例如 0.1 ms）的空闲段；
  2. 按时间把每个空闲段对应到 CPU 线程上包住它的 NVTX range，得到 step 序号、阶段与节点，再经 `nvtx_map.json` 给出文件与行号；
  3. 输出：空闲段表、各阶段的 GPU 空闲占比、每步 kernel 数与平均 kernel 时长（判断 kernel 是否太小太碎）、耗时最长的前 N 个 kernel；
  4. 从 sqlite 画出时间线图（step、阶段、kernel 三条泳道），作为可复现的“时间线截图”。
- 验收：
  - 在一次小规模运行上，每个空闲段都带 step、阶段与文件行号；
  - 各阶段时间之和与 step 时间对得上（同一分母）；
  - 有基于小型合成 sqlite 的单元测试。

**S1.5 一次同口径的优化与复测**

- 用 S1.2–S1.4 的数据确定瓶颈，只改一处，再用同一组指标复测。
- 首选 S2：阶段计时会显示冻结 k 个阶段时 `backward_ms` 基本不变，可以定位到 `MyFlows/core/graph.py:64` 与 `MyFlows/ops/convolution.py:407`。
- 如果时间线显示别的瓶颈更突出（例如每步大量清零梯度的小 kernel、优化器的逐参数 kernel），可以改选 S8 中的对应项，但仍然一次只改一项。
- 验收：报告给出完整链条（现象 → 时间线位置 → 文件与行号 → 改动 → 同口径复测），前后数据在同一会话内交替测得。

**S1.6 交付报告**

报告位置：`docs/experiments/semester_2026_fall/monitoring/README.md`。内容：

- 监控模块与落盘格式说明（链接 `docs/monitoring_format.md`）；
- 一次完整的小规模训练（例如 2000 个样本、3 个 epoch）的 12 项指标趋势图与指标表。课件要求看完整训练过程的趋势，不能只看单个采样点；
- 时间线图与空闲段表；
- 瓶颈判断（到文件与行号）与优化前后对比；
- 第 19 讲（下）五项自查的逐条结论；
- 生产者-消费者流水线（课件列为选做，本阶段纳入）：`MyFlows/data/pipeline.py` 在 `num_workers` 为 0 与 2 时的数据读取占比对照。

### S2 冻结阶段的反向截断

现状：冻结阶段仍执行完整反向，只省下了优化器更新。原因有四处：

- `freeze_stages` 只把参数的 `trainable` 设为 False（`MyFlows/utils/pretrained.py:195`）；
- `Variable.requires_grad` 在构造时就固定了（`MyFlows/core/node.py:70`）；
- `Graph.backward` 为全部节点清零梯度并逐个反向（`MyFlows/core/graph.py:64`）；
- 原生卷积反向总是计算 dW，dX 只对输入图像这类叶子跳过（`MyFlows/ops/convolution.py:407`，`MyFlows/ops/cuda_native/native.py:323` 没有 `need_weight_grad`）。

计算图在训练开始时构建一次并重复执行（`apps/train/train_myflows_donkey.py:830`）。

要做什么：

1. 计算每个节点是否需要梯度：参数节点取 `trainable`，其他叶子节点取 `requires_grad`，算子节点只要有一个父节点需要梯度就需要。
   - 由 `Graph` 在构图（以及图优化）之后计算并缓存；参数的 `trainable` 组合变化时（冻结、解冻、渐进解冻）重新计算。
   - 把结果写到算子节点的 `requires_grad` 属性上。各算子已有 `getattr(parent, "requires_grad", True)` 判断（`MyFlows/ops/basic.py:46`、`MyFlows/ops/pooling.py:87`、`MyFlows/ops/convolution.py:73`），写上之后，不需要的 dX 会自动跳过。
2. `Graph.backward` 只为需要梯度的节点清零梯度并调用 `backward()`。
3. 参数梯度：参数不训练时不计算 dW/db。
   - 原生卷积的反向接口增加 `need_weight_grad`，涉及 `native.py` 与 DLL 的 `mf_conv_backward`（`native_cublas.cpp:1171`）；dW 与 db 都不需要时，同时跳过 dY 转置。
   - NumPy、CuPy、深度卷积路径以及 `Dense` 同样处理；冻结的 BN 不计算 dγ/dβ。
4. 显存：前向时如果本节点不需要反向，就不保留反向用的中间结果（例如卷积的 im2col 缓冲，`MyFlows/ops/convolution.py:336`），用完立即释放。
5. 兼容：
   - 多 Worker 同步只涉及需要梯度的参数，现有代码已检查所有带梯度的节点都在参数表里；
   - 需要中间梯度的工具不受影响，或者显式声明保留，例如 Grad-CAM 读取特征节点的梯度（`MyFlows/utils/gradcam.py:97`）。

验收：

- 梯度逐位一致：
  - 范围：ResNet18 冻结 0–5 个阶段，MobileNetV2 与 AlexNet 各取两档冻结级别；CPU 与 CUDA（register、cublas）路径都要测。
  - 要求：未冻结参数的梯度与一步更新后的参数，和改动前逐位一致（同一输入、同一初值，改动前的结果用 2.2 节的 worktree 方法得到）。
  - 冻结参数没有梯度，训练前后指纹不变。
- 反向耗时随冻结阶段数下降：
  - 报告冻结 0/2/3/4/5 个阶段时的 `backward_ms`、每步耗时与显存池峰值（torchvision 结构 ResNet18，120×160 与 224 两种输入）。
  - 按各阶段的计算量估算，冻结 3 个和 4 个阶段时反向应分别减少约 55% 和 77%；实测与估算不符时要给出解释。
- 测试：新增测试覆盖各算子的跳过路径、解冻后的重新分析、渐进解冻场景，以及多 Worker 与单进程一致性不变；现有回归全部通过。

### S3 迁移学习组件与微调方法（第 24 讲第 44–48 页）

**S3.1 组件骨架 `MyFlows/transfer/`**

| 组件 | 职责 | 由现有代码迁移 |
| --- | --- | --- |
| PretrainedModelRegistry（`registry.py`） | 可用模型的架构、来源文件与 sha256、转换后的 NPZ、输入尺寸、均值与标准差、阶段划分、任务头名称 | `MyFlows/utils/pretrained.py` 的 `TORCHVISION_WEIGHTS`、`build_model` |
| CheckpointLoader（`loader.py`） | 载入 torchvision 转换权重与 MyFlows checkpoint，按名称与形状检查并出报告 | `load_torchvision_weights`、`ms.load_checkpoint` |
| ParameterFreezer（`freeze.py`） | 见 S3.3 | `freeze_stages`、`frozen_fingerprint` |
| HeadAdapterBuilder（`heads.py`） | 线性头；带 dropout 的 MLP 头（如课件中 512-512 两层、dropout 0.3 的例子）；DonkeyCar 回归头 | 各模型构造参数中的 `output_dim` |
| FeatureAdapter（`adapters.py`） | 位于骨干输出与新头之间：全局池化或展平、可选的 BN 或 dropout、1×1 卷积降维 | 新增 |
| TransferLearningManager（`manager.py`） | 读取配置并依次调用上述组件；生成参数组（S3.2）；按 epoch 执行解冻计划（S3.3）；提供特征缓存（S3.4）；输出 result.json 的 `transfer` 块 | `apps/train/train_myflows_donkey.py` 与 `apps/train/transfer_classify.py` 中的迁移逻辑 |

- `MyFlows/utils/pretrained.py` 保留为兼容层，重新导出新接口，现有测试与 checkpoint 不受影响。
- 验收：
  - 每个组件都有单元测试；
  - 两个训练入口改为通过 manager 构建迁移任务，行为与改动前一致：小规模对照中，同一 seed 的结果逐位一致。

**S3.2 参数组与分层学习率**

- `MyFlows/train/opt.py` 的优化器支持参数组：每组有自己的 lr，可选 weight_decay。Adam 等按组传入 lr；TrainingState 与 checkpoint 保存各组的 lr。
- `apps.train` 增加 `--backbone-lr-mult`：骨干各阶段的 lr 为 lr × 倍率，任务头为 lr。可选提供更细的 `--stage-lr-mults`。
- AutoPilot：把 `backbone_lr_mult` 作为有预训练权重时才开放的搜索键，可用性检查与 `freeze_stages` 相同（`apps/autopilot/spec.py:29–33`）。
- 验收：
  - 所有倍率为 1 时，与改动前逐位一致；
  - 按组 lr 的结果与手算的 Adam 一致；
  - 续训逐位一致。

**S3.3 ParameterFreezer：按名称冻结与渐进解冻**

- 按名称模式冻结与解冻。这需要模块路径形式的名称：
  - torchvision 兼容模型可以用 `torchvision_tensors()` 给出的官方名称；
  - 其他模型要先补一个由层级生成的路径名。`named_state` 现在的键是 `param_0000_<层名>`，见 semester_spec 4.0 节。
- 冻结模式三种：只冻结参数（BN 继续更新统计量）、只冻结 BN 统计量、两者都冻结。
- 解冻计划：
  - `--unfreeze-schedule "0:4,2:3,4:2"`，格式为“epoch:冻结阶段数”；
  - 新解冻阶段的 BN 是否立即进入训练模式，由 `--bn-unfreeze {follow,keep-frozen}` 决定；
  - 冻结级别变化时，触发 S2 的重新分析。
- 记录：epoch 行与 result.json 记录每个 epoch 的冻结级别，TrainingState 保存当前级别。
- 验收：
  - 解冻在指定 epoch 生效；
  - 仍冻结的阶段指纹不变；
  - 跨越解冻点的续训逐位一致。

**S3.4 特征提取缓存**

- 骨干全部冻结且没有随机增强时：
  - 对训练、验证、测试样本各算一次骨干输出（任务头之前的特征），保存在运行目录。缓存键是权重 sha256、预处理与样本清单的哈希。
  - 之后只训练任务头；评估与导出仍用完整模型。
- 验收：
  - 缓存与不缓存时，任务头的训练轨迹一致：用同一 kernel 时逐位一致，否则落在 FP32 舍入范围内并说明原因；
  - 报告首个 epoch 和之后每个 epoch 的耗时；
  - 权重或预处理变化时，缓存失效。

**S3.5 迁移任务的数据增强与曲线**

- `apps/train/transfer_classify.py` 增加 `--augment`（使用 `MyFlows/utils/transforms.py`）。
- 每个 epoch 记录训练与验证的 loss 和准确率，用来判断过拟合（课件第 26–28 页）。

**S3.6 LoRA（拓展，2026-11-05 决定纳入）**

- `MyFlows/transfer/lora.py` 实现 `LoRADense` 与 `LoRAConv2d`：
  - W₀ 冻结；A 随机初始化、B 置零，训练起点与基座一致；
  - 缩放系数为 α/r；
  - `merge()` 把 BA 合并回 W₀，用于推理与 ONNX 导出。
- 验收：
  - 初始化时输出与基座逐位一致；
  - 合并前后输出在 FP32 舍入内一致；
  - 小规模对照比较以下几种方式的可训练参数量与验证误差：只训任务头；LoRA（r = 4/8/16，挂在 layer4 与任务头）；冻结 3 个阶段后微调；全量微调。

**S3.7 Transfer Learning 子 Agent 的策略先验**

改动位置：`apps/autopilot/agents/transfer.py`。

- 策略先验：按课件第 41–42 页的四种情况（数据量 × 与 ImageNet 的相似度），由数据量和一个小规模的相似度估计，给出初始冻结级别与学习率倍率的候选。相似度可以用冻结骨干的线性探测分数，或各阶段 BN 统计量与目标数据激活统计的差异来估计。
- 负迁移：在同等预算、配对 seed 下，预训练不如从零训练且差值超出噪声时，给出“负迁移”Finding，建议减少冻结或改为从零训练。
- 验收：仿照 `tests/system/test_autopilot_m6.py` 写规则测试。

### S4 迁移学习实验（第 24 讲第 23–42 页）

**S4.1 分类任务上的递进实验（小规模）**

- 仿照课件第 24–38 页的递进，在 `transfer_classify` 的转向分档任务上依次比较以下 6 档：
  1. 从零训练的小 CNN；
  2. 加 dropout；
  3. 再加数据增强，lr 降为 1/10；
  4. 预训练骨干作特征提取（使用 S3.4 的缓存）；
  5. 特征提取 + 数据增强（不缓存）；
  6. 解冻最后两个阶段，用更低的学习率微调（S3.2、S3.3）。
- 每档跑 2 个 seed，报告验证准确率以及训练与验证的差距。
- 在 CIFAR-10（S7）的平衡子集上再做一次（课件用 3000/1000/1000），与课件的猫狗二分类示例更接近。
- 产物：`docs/experiments/semester_2026_fall/transfer/classify-001/report.md`。

**S4.2 DonkeyCar 主任务的正式迁移对照（全量运行已获准）**

- 先用 2000 个样本、2 个 seed 跑通全部配置，再按正式协议运行全量：
  - 全量数据，沿用 formal-full-002 的时间段划分文件（`runs/autopilot/splits/donkey-all-val1000-test1000-tb50g5-s42.json`，7680/1000/1000）；
  - 3 个 seed，3 个 epoch。
- 配置：
  1. 从零训练，cifar stem、w64（即现有人工基线，用当前代码重跑）；
  2. 从零训练，torchvision 结构（imagenet stem），120×160 输入；
  3. 预训练，冻结 0/2/3/4 个阶段；
  4. 最佳冻结级别，再加分层学习率（倍率 0.1）或渐进解冻。
- 报告：
  - 初始表现：训练前的验证 MAE；
  - 开发时间：达到配置 1 最终验证 MAE 所需的 epoch 或 step；
  - 最终表现：最终的验证与测试 MAE；
  - 每步耗时、samples/s、显存池峰值；
  - 训练与验证曲线、seed 间的离散度、是否出现负迁移；
  - 可视化：第一层卷积的特征图（预训练与从零训练对比，课件第 39–40 页），以及若干帧的 Grad-CAM（`MyFlows/utils/gradcam.py`）。
- 运行方式：可以用 AutoPilot（M6 已支持 `fixed.pretrained` 与冻结搜索），也可以写脚本按“配置 × seed”顺序调用 `apps.train --run-dir`。两种方式都要把原始结果与报告归档在 `docs/experiments/semester_2026_fall/transfer/donkey-001/`。
- 时间估算：formal-full-002 每个 trial 约 15 分钟。imagenet stem 在 120×160 输入下，layer1 的像素只有 cifar stem 的 1/16，预计会更快。6–7 个配置 × 3 个 seed 约需 3–5 小时，以小规模实测为准。

### S5 跨框架同条件比较（B4，semester_spec 4.9）

- 范围（2026-11-05 决定）：只比较 MyFlows 与 PyTorch，不安装 PaddlePaddle。
- 改写为 ResNet18 协议：新脚本 `benchmark/compare_frameworks_resnet.py`，配套新的回归测试。旧的 VGG 脚本 `benchmark/compare_frameworks.py` 及其测试 `tests/test_compare_frameworks_regression.py` 已于 2026-11-05 删除。`.gitignore` 里的 `benchmark/results.csv` 保留：本地还有旧脚本生成的结果文件。semester_spec 里提到旧脚本的地方是历史记录，不用改。
- 模型：
  - torchvision 结构的 ResNet18（imagenet stem），120×160 输入，2 个输出的回归头；
  - torch 端复用 `MyFlows/tests/torch_reference.py` 的 `TorchResNet18`，换成回归头；
  - 可选加测 cifar stem 版本：torch 端按 MyFlows 结构写一个带卷积 bias、不做 maxpool 的版本。
- 对齐：
  1. 初值：由 MyFlows 按官方名称导出 NPZ，torch 用 `strict=True` 载入。
  2. 数据：预处理后的张量缓存为 NPY，各框架读同一份数据、同一份 batch 索引清单；尾批策略一致（drop_last 或显式样本权重）。
  3. 精度：FP32，torch 关闭 TF32（`torch.backends.cuda.matmul.allow_tf32 = False`、`torch.backends.cudnn.allow_tf32 = False`）。对齐阶段设 `cudnn.deterministic = True`；计时阶段另报一行 `cudnn.benchmark = True` 的结果并注明。
  4. 先对齐单批：比较前向输出、loss、全部梯度与一步 Adam 后的参数，报告最大绝对差与相对差（可加 FP64 参考）。
- 计时：
  - 预热后再计时，各框架的设备同步边界一致；
  - 分别测固定设备张量的计算吞吐，以及含数据准备与 H2D 的端到端吞吐；
  - 记录：每步耗时与 samples/s；GPU 利用率与设备显存（所有框架都用 `MyFlows/monitoring` 的采样器，口径一致）；各框架分配器的峰值（`torch.cuda.max_memory_allocated`、CuPy 显存池）；CPU 利用率与进程 RSS 峰值（取采样峰值，不是瞬时值）。
- 训练质量：同样的小规模预算下（例如 2000 个样本、3 个 epoch），比较 3 个 seed 的验证 MAE，报告 mean ± std。
- 运行方式：一条命令依次在独立进程中运行两个框架：不共享显存池，MyFlows 与 torch 不在同一进程。
- 产物：`docs/experiments/semester_2026_fall/frameworks/resnet18-001/report.md`，并列出无法完全对齐的实现差异及其影响。
- 课件要求持续比较：之后每次重要优化，都用同一条命令复测。

### S6 DonkeyCar 闭环评估（B4，semester_spec 4.8）

**S6.1 telemetry 探测（`apps/eval/sim_probe.py`）**

- 启动或连接本地模拟器：路径为 `mycar/myconfig.py:18` 的 `DONKEY_SIM_PATH`，环境为 `donkey-generated-roads-v0`。
- 用固定动作跑若干秒，记录 info 各字段（`cte`、`pos`、`speed`、`hit`、`lap_count`、`last_lap_time` 等）的取值范围与频率。
- 检查 reset、起点与随机种子能否控制。
- 结果写入 `probe.json`，另附一页说明。

**S6.2 闭环评估入口（`apps/eval/closed_loop.py`）**

- 输入：
  - MyFlows checkpoint 或 ONNX 模型。推理与预处理复用 `mycar/myflows_pilot.py` 与 `apps/serve/onnx_predictor.py`，保证与驾驶栈一致。
  - 固定油门 0.2。
  - 场景清单（起点、种子），以及每个场景的时长上限与终止规则。
- 判定规则：
  - 按 S6.1 的结果标定出界阈值；
  - |cte| 连续 K 帧超过阈值时记一次出界事件，带去抖；
  - 碰撞即终止；
  - 没有完成路线的信号时，报告定时存活时间与行驶距离，不称为完赛率。
- 输出（格式遵循 S1.1）：
  - 逐帧 jsonl：时间、cte、速度、位置、转向、推理延迟；
  - 逐 episode 汇总与总汇总：平均与最大 |CTE|、出界事件数、存活时间与距离、圈数、推理延迟 P50/P95。
- 测试：用假环境测试指标计算与去抖逻辑，不需要模拟器。

**S6.3 基线与对比**

- 先对 ResNet18 当前的最佳模型做 5 次配对运行（同一份场景清单），报告平均值与最差结果。
- 再比较人工基线配置与 AutoPilot 的候选配置（campaign 保留了基线与前 2 名的最佳模型），以及 S4.2 中最好的迁移模型。检查离线 MAE 的排序与闭环结果是否一致。
- 模拟器已获准随时使用（2026-11-05）。它会占用 GPU 与屏幕，不要与训练或计时实验同时运行。
- 产物：`docs/experiments/semester_2026_fall/closed_loop/baseline-001/report.md`。

### S7 图像分类系统测试（B5，semester_spec 4.10）

- 数据（2026-11-05 获准下载）：
  - 来源：官方二进制版本 `cifar-10-binary.tar.gz`（官方页面 https://www.cs.toronto.edu/~kriz/cifar-10.html，170,052,171 字节，官方 MD5 `c32a1d4ab5d03f1284b67883e8d87530`）。官方站点在国内只有约 60 KB/s，2026-11-05 改从华为云上 MindSpore 教程使用的镜像下载（https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz），MD5 与官方一致。
  - 位置：已解压到 `runs/datasets/cifar10/cifar-10-batches-bin/`（不纳入 Git），压缩包保留在 `runs/datasets/cifar10/`。核对结果：训练集 50,000 条（每类 5,000），测试集 10,000 条（每类 1,000），共 10 个类别。
  - 不用 python 版本：它是 pickle 文件，反序列化来自网络的数据有执行代码的风险。
  - 格式：`data_batch_1.bin`–`data_batch_5.bin` 与 `test_batch.bin` 各 10,000 条记录，每条 3073 字节（1 字节标签 + 3×32×32 像素，按通道排列）；类别名在 `batches.meta.txt`。
- 读取与划分：
  - 新增 `apps/common/cifar10.py`，用 `np.fromfile` 读取，并校验文件大小；
  - 训练集内固定划分为 45,000/5,000；
  - 官方的 10,000 张测试集只在锁定配置后评估一次。
- 入口：新增 `apps/train/train_classify.py`，支持小型 CNN 与 ResNet18 cifar stem，复用融合的 softmax + 交叉熵与 Adam，运行目录格式与 S1.1 相同。
- 测试：
  - 日常测试用固定的平衡小子集；
  - 32 个样本的过拟合正确率 ≥ 95%（自动测试）；
  - 完整流程产生独立的验证与测试指标，并超过多数类基线；
  - 同构的 PyTorch 模型在相同预算下，准确率差距不超过 3 个百分点，超出时先诊断。

### S8 算子与显存优化（第 19 讲（下）第 55–78 页）

每一项都先测现状，一次只改一项，再用同一组指标复测，并重跑数值对拍。S8.1、S8.2、S8.4 和 S2 都要修改 `MyFlows/core/graph.py` 与 `MyFlows/ops/`，所以排在 S2 之后串行进行。

**S8.1 梯度按需分配与提前释放**

- 现状：`Graph.backward` 在反向开始时为所有节点分配并清零梯度（`MyFlows/core/graph.py:68`），之后以 `+=` 累加。峰值显存等于全部激活再加一整份梯度。CUDA 第 19 讲报告第 7 节已把它列为限制。
- 改动：
  - 梯度在第一次写入时直接赋值，之后再累加。注意不要与算子内部的缓冲共享内存。
  - 节点的反向完成后，释放非参数节点的梯度。需要保留中间梯度的场景要显式声明，例如 Grad-CAM（`MyFlows/utils/gradcam.py:97`）。
- 验收：
  - 参数梯度与改动前一致。首次赋值与 0 + g 只在 −0.0 与 +0.0 上不同，按数值相等比较；
  - 显存池峰值下降，并给出不换页时最大 batch 的变化；
  - 每步耗时不变差。

**S8.2 反向时重算 im2col**

- 现状：卷积前向把 im2col 缓冲保留到反向（`MyFlows/ops/convolution.py:336`）。3×3 卷积的缓冲是输入的 9 倍，w64、cifar stem、batch 16 时，layer1 每个卷积约 708 MB。
- 改动：
  - 前向 GEMM 之后释放缓冲；反向计算 dW 之前，由保存的输入重算 im2col。这对应课件第 55 页“访存受限时用计算换访存”。
  - 做成选项，默认值按实测决定。
  - 隐式 GEMM（在寄存器分块模板里直接按卷积窗口读取 A 操作数，不物化缓冲）留作后续拓展。
- 验收：
  - 结果逐位一致（重算的 im2col 与原缓冲相同）；
  - 报告显存峰值、每步耗时，以及不换页时的最大 batch。

**S8.3 多张量 Adam（课件第 77–78 页）**

- 现状：每个参数启动一次 kernel（`MyFlows/train/opt.py:229`、`MyFlows/ops/fused.py:400`），ResNet18 每步 62–82 次，另有逐参数的 Python 开销。
- 改动：
  - 一次 launch 处理全部参数。可以用指针数组的 RawKernel，也可以把参数、m、v 放在连续缓冲里、参数作为其视图。
  - 支持 S3.2 的按组 lr；逐元素的运算顺序不变。
- 验收：
  - 与逐参数版本逐位一致；
  - 报告优化器阶段耗时的变化。
- 预计收益不大，课件也说“测出来与排序不符以实测为准”。实测没有收益时，只记录结论，不合入。

**S8.4 BN + ReLU（+ 残差加）融合**

- 现状：卷积尾处理已经融合了 bias 与 ReLU；但 BasicBlock 里 BN → ReLU，以及 BN → 加残差 → ReLU，仍是分开的 kernel。
- 改动：
  - 在 `MyFlows/ops/fused.py` 用 ElementwiseKernel 融合前向与反向；
  - 在 `MyFlows/core/graph_opt.py` 增加对应的模式匹配，参照 `fuse_conv_activation_ops`。
- 验收：
  - 结果逐位一致；
  - 报告每步耗时与中间结果显存的变化。

**S8.5 compute-sanitizer（课件 19.8 节）**

- 新增 `tools/sanitize_native.py`：
  - 找到 CUDA Toolkit 自带的 compute-sanitizer；
  - 在其下运行原生 kernel 的测试（`MyFlows/tests/test_cuda_native_kernels.py`、`test_cuda_native_cublas.py`）；
  - 工具用 memcheck、racecheck 与 synccheck，可选 initcheck。CuPy 显存池可能造成误报，需要甄别。
- 验收：错误数为 0，或修正后归零；结果写进报告。

**S8.6 GEMM tile 自动调优（拓展，已纳入；课件第 17–19 页）**

- 现状：tile 与 split-K 的选择规则，是按本机 ResNet18 的卷积形状手工确定的（CUDA 第 19 讲报告第 7 节）。
- 改动：
  - 对每个（布局、M、N、K）实测五种 tile 与各个切分档位，结果按 GPU 与原生库哈希缓存；
  - 用 `MYFLOWS_GEMM_AUTOTUNE=1` 开启。
  - 数值：不切分时，累加顺序与 tile 无关，结果逐位不变；切分档位改变时舍入会变，需要记录。
- 验收：
  - 给出逐形状与端到端的耗时对比；
  - 附一张线程数与 block 数的占用率实验表。

**S8.7 TF32 与混合精度（拓展，已纳入；课件第 47、76 页）**

- 两步走：
  1. 为 cuBLAS 引擎加一个 TF32 实验后端，只作对照；
  2. 实现卷积与 Linear GEMM 的混合精度：FP16/BF16 存储、FP32 累加，加动态 loss scaling。
- 验收：在小规模上报告速度、显存、相对 FP64 的梯度误差与训练曲线。不作为默认后端。

**S8.8 Winograd（拓展，已纳入；课件第 52、68 页）**

- 为 stride 1 的 3×3 卷积实现 F(2×2, 3×3) 实验后端。
- 对比乘法次数、耗时与误差，并重跑对拍流程。

### S9 测试计划、性能门槛与 CI（B5；第 16 讲第 6 页）

**S9.1 测试计划（`docs/test_plan.md`）**

- 把 B1–B7、E1–E3 与本规划各项，对应到测试、命令、执行频率（每次提交、阶段验收前、手动）、通过标准与证据位置。
- 区分三类测试：日常测试（几分钟内）、GPU 测试、需要全量数据或模拟器的阶段测试。

**S9.2 性能门槛（`benchmark/perf_gate.py`，需要 GPU，手动执行）**

- 按 GPU 与代码版本保存本机基线，然后检查：
  - 寄存器分块引擎在各卷积 GEMM 形状上不慢于 cuBLAS 的 1.2 倍；
  - ResNet18 的几个固定配置，每步耗时不比基线慢 10% 以上；
  - 卷积与池化的单算子耗时。
- 测量按 2.3 节的方法交替进行。性能相关的改动合入前运行一次。

**S9.3 CI：GitHub Actions，只跑 CPU（2026-11-05 决定纳入）**

- 运行环境：GitHub 托管的 Ubuntu 机器（可选与已验证过的 WSL 一致的 22.04）、Python 3.11。不使用自托管 runner：那会让每次推送都在用户的笔记本上自动启动任务。
- 工作流：根仓库的 `.github/workflows/tests.yml`。
  - 触发：推送到 main、拉取请求与手动触发；同一分支上的新运行取消旧的运行。
  - `actions/checkout` 带 `submodules: recursive`（MyFlows 是公开子模块）；`actions/setup-python` 开启 pip 缓存；`actions/setup-node` 用于 dsh 守卫的 node 测试。
  - 执行 `python -X utf8 -m tools.run_tests --scope all`，测试日志作为 artifact 上传。
  - MyFlows 仓库可以另加一个只跑框架测试的工作流（检出到名为 `MyFlows` 的目录，以便 `import MyFlows`），由开发者判断是否需要。
- 依赖：新增 `requirements-ci.txt`，版本与 `uv.lock` 一致。
  - 保留：numpy、Pillow、scikit-learn、onnx、tensorboard、psutil、matplotlib、grpcio、grpcio-tools、fastapi、uvicorn、httpx、python-multipart、optuna、nvidia-ml-py。
  - 替换：`opencv-python` 换成 `opencv-python-headless`，`onnxruntime-gpu` 换成 `onnxruntime`。
  - 不装：CuPy、nvidia-* 的 CUDA 运行库、torch；donkeycar 相关的包按测试的实际需要决定。
- 跳过规则：缺少 CUDA/CuPy、torch、dsh 克隆、模拟器或本地数据（`mycar/data/`、`runs/`）时，相应测试以 skip 结束并写明原因，不能报错，也不能在导入时尝试编译原生库。
- Linux 上还没有跑过完整回归，预计要先修一批平台问题（路径、大小写、Windows 专用分支）。先在 WSL 上复现并修正。
- 验收：
  - 根仓库的工作流在 GitHub 上通过；
  - 日志列出全部 skip 及原因；
  - 单次运行不超过约 15 分钟；
  - 不需要任何密钥。
- 工作流文件随正常提交推送；推送仍按 2.5 节先征得用户同意。

**S9.4 定期执行**

- 在测试计划中写明执行节奏，例如：每个工作包结束时跑统一回归；阶段报告前跑 `--repeat 3` 与性能门槛。
- 不设置用户没有同意的后台定时任务。

### S10 AutoPilot 增量（B1）

- **S10.1 失败归类。** TrialResult 的 failure 增加 `category`，报告按类别分栏（第 19 讲（下）第 79 页要求代码问题与平台问题分开报）。类别如下：
  - `platform`：功耗墙与降频、显存换页、控制台被关闭、重启、等锁超时；
  - `code`：异常退出与断言失败；
  - `training`：NaN、发散；
  - `config`：参数无效。
- **S10.2 Operator Optimization 子 Agent。** 读取 Nsight Compute 的带宽占比与计算吞吐占比，把热点 kernel 分为访存受限或计算受限，再按课件第 69 页的顺序给出对应的手法类别：
  - 访存类：合并访问、tiling、私有化；
  - 调度类：占用率、线程分化；
  - 算法类：换数学方法。
- **S10.3 Transfer Learning 子 Agent。** 策略先验与负迁移判断，见 S3.7。
- 验收：有规则测试，并附报告示例。

### S11 文档与报告（B6）

- **S11.1 同步设计文档。** 只写当前代码事实。
  - `docs/system_design.md`：第 9 节限制中的“没有 AutoPilot”“没有预训练导入”已经过时；补上 AutoPilot、E2、E3、监控格式以及本阶段的新能力。
  - `docs/module_design.md`：补上 `apps/autopilot`、`MyFlows/utils/pretrained.py`、`MyFlows/layers/mobilenet.py`、`apps/common/splits.py`、`apps/train/transfer_classify.py`、`apps/train/evaluate_donkey.py`，以及本阶段新增的模块。
- **S11.2 整理只在 `runs/` 中的结果。** 把 AutoPilot 正式 campaign（formal-full-002）和 E2 的结果（权重等价、transfer_classify、M6 演示）整理成可提交的报告，分别放在 `docs/experiments/semester_2026_fall/autopilot/` 与 `docs/experiments/semester_2026_fall/e2_transfer/`。提交前做隐私检查（2.5 节）。
- **S11.3 可视化训练系统**（第 16 讲第 4 页）。同步 `system_design.md` 第 8 节。
  - 控制台适配 S1.1 的 v1 格式；
  - 运行详情页显示四段耗时占比（数据、前向、反向、优化器）与 12 项指标；
  - Compare 页可以加载 S5 的跨框架结果；
  - 如有余力，增加闭环评估的结果页。
- **S11.4 最终报告骨架（`docs/final_report.md`）。** 内容包括：
  - 系统设计：框架、CUDA 与分布式主线、AutoPilot、可视化训练系统；
  - 测试过程与结果（S9）；
  - 实验：B2–B4、B7、E1–E3、S4–S7；
  - 限制与后续工作。

## 5. 执行顺序

| 批次 | 框架线（`MyFlows/`、`apps/train/`） | 应用线（`apps/eval/`、`benchmark/`、`apps/common/`） |
| --- | --- | --- |
| 开工前 | 通读第 2 节与 semester_spec；跑统一回归，确认 446 项通过 | 同左 |
| 第一批 | S1.1 → S1.2 → S1.3 → S1.4 → S2（同时作为 S1.5）→ S1.6 | S6.1 → S6.2 → S6.3；S5（S1.1 定版后开始按新格式记录） |
| 第二批 | S3.1 → S3.2 → S3.3 → S3.4 → S3.5 | S4.1（小规模）→ S4.2（先小规模，经同意后全量）；S10.1 |
| 第三批 | S8.1 → S8.2 → S8.5 → S8.3、S8.4（先测再决定） | S9.1 → S9.2 → S9.3；S7（数据集获批后）；S10.2、S10.3 |
| 拓展 | S8.6、S8.7、S8.8、S3.6 | — |
| 贯穿 | S11.1 随各项更新 | S11.2–S11.4 |

- 两条线可以并行；同一条线内按顺序做。S2、S8.1、S8.2、S8.4 修改同一批文件，必须串行。
- GPU 同一时刻只运行一个训练或测量任务；需要计时的实验不要和测试、模拟器同时跑。
- 每完成一个工作包，在第 8 节追加一行进度记录。

## 6. 用户决定

| 事项 | 影响 | 决定（2026-11-05） |
| --- | --- | --- |
| 是否安装 PaddlePaddle | S5 | 不安装，只比较 MyFlows 与 PyTorch |
| 是否下载 CIFAR-10 | S7、S4.1 | 下载；2026-11-05 已下载并校验（见 S7） |
| 全量运行与模拟器的时段 | S4、S5、S6、S7 | 第三阶段期间电脑全程空闲，可以直接运行（2.4 节） |
| 拓展项是否纳入 | S3.6、S8.6–S8.8 | 全部纳入 |
| 旧的 VGG 跨框架脚本 | S5 | 删除（2026-11-05 已删除脚本与测试） |
| CI | S9.3 | 纳入，只跑 CPU：GitHub 托管的 Linux 机器，不用自托管 runner；GPU 测试继续在本地运行 |

## 7. 验收对照

| semester_spec 第 7 节条目 | 本规划 |
| --- | --- |
| GPU、CPU 和训练阶段监控指标齐全 | S1.1–S1.3 |
| 第 19 讲（下）课程任务二 | S1.4–S1.6、S2 |
| MyFlows 与 PyTorch 同条件对比（不比较 PaddlePaddle） | S5 |
| 图像分类系统测试 | S7（S4.1 补充） |
| DonkeyCar 闭环评估和对比 | S6 |
| 测试计划与性能回归门槛 | S9 |
| 系统设计、测试过程、实验结果和最终报告 | S11 |
| 冻结阶段的反向截断 | S2 |
| 第 24 讲的迁移组件与微调方法 | S3 |
| 主任务上的正式迁移对照 | S4.2 |

## 8. 进度记录

| 日期 | 工作包 | 提交（根仓库 / MyFlows） | 结论 | 证据 |
| --- | --- | --- | --- | --- |
| 2026-11-05 | S7 准备 | — | CIFAR-10 二进制版本已下载并解压；MD5 与官方一致；训练集每类 5,000 条、测试集每类 1,000 条 | `runs/datasets/cifar10/` |

## 9. 风险

| 风险 | 应对 |
| --- | --- |
| S2 改动计算图的反向调度，影响所有训练路径 | 先写逐位一致的对照测试，再改实现；在 CPU、CuPy、两种原生引擎上都跑一遍；分布式一致性测试必须通过 |
| 笔记本 GPU 的频率与功耗浮动掩盖小幅收益 | 交替测量，报告离散度；小于噪声的差异不下结论（2.3 节） |
| 模拟器不能控制起点或随机种子 | 如实记录，增加重复次数，不声称完全确定 |
| Linux CI 暴露平台问题 | 先在 WSL 上复现并修正，再启用 CI |
| 改动训练代码时恰好有正式 campaign 在运行 | 正式 campaign 期间冻结 `MyFlows/` 与 `apps/train/`（2.4 节） |
| 迁移实验的结论受 seed 噪声影响（基线 seed 间标准差 0.015） | 配对 seed、3 个 seed、独立测试集；只报告超出噪声的差异 |
