# 当前模块说明

本文列出本学期仍会使用的现有模块。计划新增的模块见 [semester_spec.md](semester_spec.md)，在实现前不会提前列为现有能力。

## MyFlows 框架

| 路径 | 当前职责 |
| --- | --- |
| MyFlows/core/ | 计算图、节点、Tensor 和 CPU/CUDA 设备切换；`dtype.py` 的全局默认 dtype（float32，可用 `set_default_dtype` 改为 float64）；`Variable(requires_grad=False)` 标记输入/标签，算子据此跳过 dX |
| MyFlows/ops/ | 当前 NumPy/CuPy 算子，包括卷积、池化、损失和激活函数；卷积/池化与 `Linear` 可选原生后端 `cuda_native_register`（寄存器分块 GEMM）/ `cuda_native_cublas`，`Linear` 另有实验用的 `cuda_native_kahan` / `cuda_native_dot2`，bias 梯度 FP64 累加；CuPy 的 col2im 为确定性的 gather kernel |
| MyFlows/ops/fused.py | 融合 kernel 与同算法 NumPy 回退：BatchNorm2d（FP64 平移统计、均值 hi/lo 双 float、反向重算 x̂）、softmax 交叉熵（FP64 log-sum-exp）、Adam（逐步显式舍入，CPU/GPU 逐位一致）、ReLU/LeakyReLU 反向累加、FP64 全局梯度范数 |
| MyFlows/ops/cuda_native/ | 原生 C/C++ 调度自写 FP32 Conv/Pool kernel，GEMM 引擎为第 19 讲的寄存器分块 kernel（NN/NT/TN，20 种 tile，其中 3 种的 TN kernel 为 cp.async 多级流水；长 K 小输出按 K 切分，归约结果与固定顺序的 FP64 求和逐位相同，`native.gemm`、`native.split_reduce`）或 cuBLAS；严格数组 wrapper 和运行时库管理（Windows 加载 DLL，Linux 加载 .so；`build.py` 在库缺失或与源码不一致时用 CMake 自动构建，文件锁保证多进程只构建一次）；NVRTC 按设备架构生成 SASS 并缓存到 `build/native-cuda/kernel-cache`；卷积尾处理融合 NHWC→NCHW 转置、bias 与 ReLU/LeakyReLU，bias 梯度 FP64；第 19 讲 basic/tiled/shared/register（寄存器分块）矩阵乘（fp32/kahan/dot2 累加）经 `native.matmul` 暴露；错误信息含源码行号与 CUDA/cuBLAS 错误名，`MYFLOWS_NATIVE_SYNC=1` 或 `set_debug_sync` 逐 kernel 同步 |
| MyFlows/distributed/ | 单机回环同步 PS（Socket JSON / gRPC）与 Ring AllReduce；Worker 本地优化器更新，PS 只聚合梯度 |
| MyFlows/monitoring/schema.py、record.py、gpu.py | v1 记录与增量 JSONL 游标；NVML/nvidia-smi/主机资源采样；训练 PID 生命周期绑定的独立采样进程，正常与异常退出清理 |
| MyFlows/utils/training_dashboard.py、tensorboard_logger.py | 逐 step/epoch 的 v1 JSONL，增量读取资源曲线；直接写 TensorBoard 事件，不导入 torch，支持标量、文本、图像与直方图 |
| MyFlows/distributed/sync.py | 供任意训练循环使用的同步层：`SyncRun` 在主进程启动 PS 进程或 Ring 元数据屏障与各 rank 进程，rank 内的 `PsSync` / `RingSync` 提供 `initialize`（统一初始状态）、`allreduce`（按样本数加权平均）、`commit`（核对状态指纹并汇总停止请求）、`reduce`（验证集等求和）；任一 rank 失败时整次运行失败 |
| MyFlows/distributed/launcher.py | `run_training(**config)`：配置规范化、端口分配、spawn 启动、事件收集与清理 |
| MyFlows/distributed/ps.py、engine.py | PS 进程入口与同步轮次状态机：加权聚合、心跳与存活检查、重复/陈旧消息拒绝、停止请求汇总与小张量求和（`reduce`） |
| MyFlows/distributed/worker.py、session.py、schedule.py | Worker 入口、单 rank 前向反向与本地更新、批次游标和单进程基线 |
| MyFlows/distributed/ring.py、gradient_layout.py | Ring 两阶段规约、分块布局与 padding、元数据屏障 |
| MyFlows/distributed/transport_socket.py、transport_grpc.py | 带长度前缀的 Socket JSON；PS 与 Ring 的 gRPC 客户端和服务端 |
| MyFlows/distributed/measurement.py、monitor.py、metrics_reduce.py | 逐 rank 的 steps/epochs JSONL 与状态快照、CUDA Event 计时、流量统计、事件上报、指标归约 |
| MyFlows/distributed/tasks/ | synthetic、mnist_mlp、donkey_cnn、donkey_resnet18 任务适配与预处理数据读取 |
| MyFlows/examples/stage1_cnn.py | 同 seed、全链路 FP32 的 32 样本 CNN 训练 fixture |
| MyFlows/layers/resnet.py、mobilenet.py | ResNet18、MobileNetV2、AlexNet 的 torchvision 兼容结构、张量名称与阶段划分 |
| MyFlows/transfer/registry.py、loader.py、manager.py | 注册三种 ImageNet 骨干，预检名称、形状与权重哈希，构建任务并保存迁移状态。MyFlows/utils/pretrained.py 保留兼容入口 |
| MyFlows/transfer/freeze.py、adapters.py、heads.py、cache.py | 按阶段或模块名冻结，独立控制 BN，适配特征与任务头，并按权重、预处理、样本清单校验离线特征 |
| MyFlows/ops/precision.py、winograd.py | 显式 cuBLAS TF32、FP16/BF16 操作数与 FP32 输出，以及 F(2×2,3×3) 卷积和反向变换 |
| MyFlows/ops/cuda_native/autotune.py | 按布局与形状实测 register tile 和 split-K。按设备、运行库、源码及调优协议保存测量结果 |
| MyFlows/ops/cuda_native/ada_plans.py、gemm_tiles.inc | RTX 4060 Laptop GPU 上 ResNet18 卷积 GEMM 的实测起始计划（布局与 M、N、K 精确匹配），以及编译进原生库的 20 种 register tile 几何（名称以 `/s<级数>` 结尾的 3 种在 TN 布局上用 cp.async 多级流水） |
| MyFlows/monitoring/nvtx.py | 按全局 step 控制 cudaProfilerApi，标注训练阶段与节点，保存相对源码位置 |
| MyFlows/train/ | 优化器与正则化、按组学习率与权重衰减、Adam 逐参数步数。独立连续的 FP32 CUDA 张量共用一次更新，loss_scaling.py 管理半精度训练的溢出跳步与恢复 |
| MyFlows/utils/checkpoint.py | JSON + NPZ checkpoint 保存和恢复。TrainingState 整目录发布并校验文件哈希，同一步重新保存会生成新版本，恢复时选最新的有效版本 |
| MyFlows/utils/onnx_exporter.py | ONNX 导出 |
| MyFlows/utils/metrics_core/ | 分类、回归和 DonkeyCar 指标 |
| MyFlows/data/pipeline.py | 多进程图片读取，固定整批任务并按顺序返回；独立于 PS 训练 |
| MyFlows/tests/ | 框架单元测试和小型集成测试 |

## 根仓库应用

| 路径 | 当前职责 |
| --- | --- |
| apps/common/ | DonkeyCar 数据索引、划分和图像预处理 |
| apps/train/train_myflows_donkey.py | ResNet18 正式训练入口：单进程与 `--distributed ps|ring --workers N` 数据并行共用同一个训练循环（`train_rank`）；断点记录续训位置（epoch、step、早停计数），验证集 loss 按真实样本精确计算，分布式时验证集按 Worker 切分后汇总 |
| apps/autopilot/ | campaign、trial、attempt 持久状态，规划、调度、执行、验证、证据与报告，TPE 条件搜索与 TrainingState 回滚 |
| apps/autopilot/agents/、dsh/ | GPU、算子、数据并行、基准与迁移规则，以及可选 dsh 工具适配 |
| apps/common/splits.py | 训练、离线评估和 AutoPilot 共用划分校验。时间块按文件声明的约定检查间隔。`contract` 2 隔开所有分区，旧划分按 `contract` 1 校验训练帧与留出帧。训练或 AutoPilot 新建划分时记录源图像路径及其顺序的哈希，同样条数的重新排序也会被发现 |
| apps/common/cifar10.py、apps/train/train_classify.py | 校验并读取 CIFAR-10 二进制档，训练小型 CNN 或 ResNet18，独立保存验证表现与锁定后的测试结果 |
| apps/train/transfer_probe.py | 在已有训练分区内部切分时间块和拟合线性探测，供迁移 Agent 生成候选 |
| apps/train/transfer_classify.py、evaluate_donkey.py | 转向分档分类迁移实验与固定测试集上的回归模型评估 |
| apps/train/common/ | 验证、早停、checkpoint 路径和训练日志 |
| apps/train/__main__.py | `python -m apps.train` 的短入口，等价调用 train_myflows_donkey |
| apps/common/environment.py | GPU/驱动、CUDA/NVRTC、依赖 metadata 和原生源码哈希；AutoPilot provenance 保留兼容导出 |
| tools/check_run_format.py | 校验 config、metrics、resources、rank steps 和 result 的 v1 格式，并兼容旧运行；字段口径见 monitoring_format.md |
| apps/eval/simulator.py、sim_probe.py、closed_loop.py | 管理本地 Unity 会话（`FrameStream.regenerate_road` 让 DonkeySim 按种子换一条路），核对实际发送的遥测字段，按锁定的阈值、去抖与场景清单配对评估 checkpoint、ONNX 与已部署的推理服务（模型清单里写 `endpoint`）；场景清单的 `environment` 决定用哪个模拟器 |
| apps/eval/environments.py | 闭环评估的两个环境：DonkeySim（固定油门、遥测 CTE、3D 位置算距离）与 Udacity（速度控制器保持目标车速；CTE 是前视图像估计的横向位置，距离由速度积分，失败为卡住或连续丢路）。`apps/eval/scenarios/udacity_lake.json` 是 Udacity 湖区赛道的场景清单 |
| apps/common/datasets.py | 数据集登记：默认的 `mycar/data` 与 `runs/datasets/<id>` 下采集的数据集；`normalize` 只接受这两类写法（拒绝绝对路径、`..` 和其他目录）；`list_datasets` / `describe` 汇总帧数、模拟器、图像尺寸、建议输入尺寸与采集自检；`TubWriter` 写出训练入口直接读取的 tub（images、逐行 JSON catalog 与 manifest、dataset.json）。控制台、AutoPilot（`fixed.data`）与 LLM 工具 `list_datasets` 都经过它 |
| apps/collect/ | 无人值守采集。`policy.py`：CTE 的 PD 专家、速度保持、随机符号与长度的转向扰动脉冲；`recorder.py`：逐段循环（标签是专家在该帧的输出，执行的是加了扰动的转向，可存左右镜像样本）、自检与报告，模拟器无法重启或不再响应时提前结束并保留已存的帧；`donkey_auto.py`：DonkeySim，每段用 `regen_road` 换一条路；`vision.py`：Udacity 的前视图像专家（车前路面颜色找边缘点，平地车道模型拟合两侧边缘、航向与弯度，沿右边缘的平行线行驶，过桥时的边缘切换需两帧确认）；`udacity_auto.py`：Udacity 采集入口，同时写模拟器格式的 driving_log.csv |
| apps/collect/merge.py | 逐条校验来源 catalog，复制原始 JPEG 并记录对应的来源。全部完成后发布数据集目录，普通写入失败后可重试同一个数据集 ID |
| apps/sim/ | Udacity 模拟器（term 1）。`udacity.py`：模拟器是 Socket.IO 1.x 客户端，这里是基于 websockets 的服务端，提供 reset / step / brake，按 20 Hz 节拍发送命令，`session` 负责启动并进入自动模式、重启场景；`udacity_launcher.py`：Windows 上用窗口消息完成配置对话框、菜单点击与 ESC 重启，窗口被最小化时先还原；`udacity_fake.py`：同一协议的替身（运动学小车与渲染的体育场赛道），供测试与 CI；`udacity_drive.py`：用 ONNX、checkpoint 或推理服务驾驶一段并输出摘要 |
| apps/serve/remote.py | 推理服务的客户端预测函数（FastAPI 上传 PNG，或 gRPC），让闭环评估与驾驶把已部署的服务放进回路 |
| apps/eval/eval_myflows_donkey.py | MyFlows checkpoint 评估 |
| apps/eval/eval_myflows_donkey_onnx.py | ONNX 模型评估 |
| apps/serve/ | gRPC、FastAPI、闭环和本地 pilot 共用 ONNX 单帧预测器。固定输入尺寸取自模型，动态尺寸使用调用方配置。固定批大小大于 1 时在批内复制输入，只返回当前图像的预测。服务默认监听 127.0.0.1（可用 `--host` 修改）。FastAPI 在解析前限制请求体并先核对图片文件头的像素数 |
| apps/console/server.py | 训练控制台入口 `python -m apps.console.server`：组装任务管理、运行索引、实时 GPU 采样并托管前端；中间件校验 Host 头并拒绝跨站的 `/api` 请求 |
| apps/console/api.py | REST 与 SSE 路由：GPU 实时流、运行列表/详情、step/epoch/资源增量序列、日志尾部、任务提交与停止、闭环评估归档的列表、详情与逐帧记录、可选数据集（`/api/datasets`，也随预设返回） |
| apps/console/closed_loop.py | 只读地读取归档的闭环评估（manifest、result、episodes、frames）：模型汇总与离线 MAE、逐场景相对基线的差值、按模型与场景降采样的逐帧记录（保留出界区间两端与碰撞帧）；逐帧文件解析一次后缓存 |
| apps/console/registry.py、readers.py | 发现控制台任务、分布式归档（run_index.json）与 TensorBoard 单进程日志；按游标增量读取 JSONL，统一 perf_counter 时间轴 |
| apps/console/jobs.py | 任务排队（默认串行）、子进程启动、停止（分布式实验发 CTRL_BREAK，正式训练写暂停文件、所有 Worker 在同一步保存断点，超时后结束进程树）与重启后重新接管 |
| apps/console/presets.py | 任务预设与提交校验：只接受框架支持的模式/传输组合和白名单参数（正式训练含并行方式、Worker 数与通信方式，以及经 `apps/common/datasets.py` 校验的数据集 `data`）；卷积后端选项（原生寄存器分块 / 原生 cuBLAS / CuPy / NumPy）及其与设备、dtype 的匹配；原生库状态（缺失时由任务自动编译） |
| apps/console/web/ | React + Vite + Ant Design + ECharts 前端：总览、运行记录、运行详情（曲线/耗时/资源/拓扑/配置/日志）、新建训练、运行对比（训练运行、跨框架实验、闭环评估三个视图） |
| mycar/myflows_pilot.py | DonkeyCar 驾驶控制器；不是训练调优 Agent |
| tools/analyze_donkey_data.py | 数据数量和标签分布检查 |
| tools/export_resnet_onnx.py | 当前 ResNet18 ONNX 导出入口 |
| benchmark/cuda_ops.py | CUDA/NumPy/CuPy 正确性、性能、CNN 训练和短 profile 工作负载 |
| benchmark/matmul_compare.py | 第 19 讲矩阵乘：课件格式的数据记录表（NumPy 为基线，1024/2048/4096 各测三轮）；自写 kernel × 三种累加与 NumPy/CuPy/cuBLAS 的精度（对 FP64 与 NumPy）、单独计时的 H2D/D2H、交叉点和可选 Nsight Compute 指标 |
| benchmark/gemm_engines.py | 原生卷积的两个 GEMM 引擎（寄存器分块 / cuBLAS）：ResNet18 一次训练步的逐形状 GEMM 耗时与误差、多 batch 单步梯度、交替多轮的每步耗时 |
| benchmark/ops_before_after.py、resnet_before_after.py | 以另一代码树（如上一提交的 git worktree）为“修改前”，逐进程对照算子精度、ResNet18 单步梯度与每步耗时 |
| benchmark/mlp_backends.py | MNIST MLP：各 Linear 后端的梯度精度、一个 epoch 的速度/准确率与 FP64 轨迹距离、Adam 修正前后对照 |
| benchmark/ps_demo.py | 同步 PS 数值对照及故障注入（转发新训练入口） |
| benchmark/distributed_train.py | 单进程 / PS / Ring 统一训练入口 |
| benchmark/distributed_compare.py | 分布式 Spec 第 12.2 节七配置矩阵的对比入口 |
| benchmark/config_paths.py | 归档配置中的数据与 BN 统计量路径按仓库根目录保存；旧归档里其他检出目录或其他系统的绝对路径会映射到当前检出 |
| benchmark/distributed_experiment.py | 正式实验单元：读 JSON 配置，训练并归档 config/manifest/results、逐 rank JSONL 和约 1 秒一次的资源采样 |
| benchmark/distributed_suite.py | 以子进程串行执行数值、状态、训练、性能、故障各阶段，维护 run_index.json |
| benchmark/complete_course_experiments.py | 完整课程实验编排（prepare/validate/train/performance/extras/report） |
| benchmark/prepare_distributed_data.py、prepare_resnet_comparison.py | 预处理 MNIST/道路数据为 npy，准备 ResNet 冻结 BN 统计量 |
| benchmark/report_distributed.py | 生成 comparison.csv、epoch_metrics.csv、加速比与收敛图 |
| benchmark/profile_distributed.py | 对分布式训练采集 Nsight Systems/Compute 并检查 kernel |
| benchmark/distributed_monitor_overhead.py | 资源监控开关各 3 次的开销对照 |
| benchmark/audit_distributed_artifacts.py、validate_resnet_steps.py、validate_ring_volume.py、diagnose_resnet_gradients.py | 归档审计、ResNet 逐步对照、Ring 通信量核对与梯度诊断 |
| benchmark/profile_cuda.py | Nsight 采集、报告存在性和真实 kernel 内容验证 |
| benchmark/nsight.py | 按平台查找 nsys/ncu：显式路径、`MF_NSYS`/`MF_NCU`、PATH，最后是 Windows / Linux 默认安装目录中的最新版本 |
| benchmark/stage1_common.py | 配置、环境、双仓库源码快照、错误状态与产物校验和 |
| benchmark/stage1_regression.py | 三个独立进程的完整回归与日志归档 |
| benchmark/monitor_overhead.py、profile_train.py | 训练资源监控的交替开销对照，以及指定步段的 Nsight Systems 采集与 SQLite 导出 |
| benchmark/transfer_study.py、transfer_visuals.py | 同一训练代码树上的分类、回归、低秩训练对照，以及在相同验证帧上的首层激活和 Grad-CAM |
| MyFlows/transfer/lora.py | 在 layer4 卷积和任务头训练 A、B 两个因子，合并到原有权重后导出单输入 ONNX |
| tools/timeline_gaps.py | 合并同设备各 stream 上的 kernel/拷贝/清零忙区间，将空闲段按 CPU NVTX 来源拆分，输出图表和分阶段占比 |
| tools/run_tests.py | 同时收集 unittest 和函数式测试的统一入口 |
| benchmark/compare_frameworks_resnet.py、framework_worker.py | 共用 FP32 初值、数据与批序，分别启动 MyFlows、确定性与自动调优 PyTorch 进程，汇总对齐、训练和资源证据 |
| benchmark/memory_study.py、adam_study.py、autotune_study.py、experimental_backends.py | 单因素交替测量、WDDM 边界、优化器、Nsight Compute 占用率、六种子 FP64 梯度与实验后端训练曲线 |
| benchmark/perf_gate.py、tools/sanitize_native.py | 手工 GPU 门槛与按设备/版本保存的基线，原生套件 memcheck、racecheck、synccheck 动态检查 |
| benchmark/gemm_ncu.py、qualify_register_plans.py | 在 Nsight Compute 锁频、清缓存条件下按 NVTX 区间比较 register 与 cuBLAS 的 GEMM 周期，并对超限形状实测候选计划；后者用生产调用与 CUDA Event 复核候选 |
| tools/restore_stage1_source.py | 按运行 manifest 校验并恢复实际源码，不覆盖现有目录 |
| tools/build_native_cuda.py | 手动构建原生 CUDA 库（调用 `MyFlows/ops/cuda_native/build.py`）：Windows 用 MSVC 生成 `native_cublas.dll`，Linux 用 GCC/Clang 生成 `native_cublas.so`；`--if-needed` 只在缺失或过期时构建 |
| tools/generate_distributed_proto.py | 从 proto/ 生成 generated/grpc/ 下的 gRPC 代码 |
| proto/、generated/grpc/ | PS、Ring、推理服务的 protobuf 定义与生成代码 |
| tests/ | 根仓库应用层测试 |
