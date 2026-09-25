# 当前系统设计

更新时间：2026-09-23。

本文只说明仓库当前已经存在的系统。第一阶段已实现 CUDA C/C++ Conv/Pool 与 Nsight 实验入口；分布式阶段已按 [stage1_distributed_gpu_spec.md](stage1_distributed_gpu_spec.md) 实现 GPU Worker 的同步 PS（Socket JSON / gRPC）与 Ring AllReduce。训练控制台（第 8 节）已把运行记录、实时曲线、GPU 占用和任务启停集中到一个 Web 界面。AutoPilot Agent、完整 GPU 监控和预训练模型库继续按 [semester_spec.md](semester_spec.md) 推进。

## 1. 项目边界

项目分为框架和应用两层：

- MyFlows/ 是自研深度学习框架。
- 根仓库是 DonkeyCar 应用、训练评估入口、部署和课程实验。

根仓库把 MyFlows/ 记录为独立 Git 仓库。阶段实验同时记录两个仓库的 HEAD、dirty/diff、源码 hash，并保存实际源码快照；恢复工具支持无 Git 历史的源码包。长期远程子仓库引用方式仍需在学期协作前统一。

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
  native[原生 CUDA DLL 与 cuBLAS] --> framework
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

当前计时只覆盖了部分训练阶段，也没有完整的 GPU 硬件指标。本学期会按照 Spec 补齐。

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
- 当前没有自动运行多次模拟驾驶并统计 CTE、出界次数和完成情况的工具。

因此，本学期首先建立固定油门下的闭环评估，再决定是否采集变速数据。

## 6. 第一阶段新增能力

- `Conv2D` / `MaxPool2d` 和对应 Op 接收 `backend=auto/numpy/cupy/cuda_native_cublas`；默认保留 NumPy/CuPy，显式原生 CUDA 路径严格验证 FP32、设备和配置。
- `cuda_native_cublas` 由 C/C++ DLL 在当前 stream 上调度自写 im2col/col2im、max-pool kernel，并调用 cuBLAS 完成卷积 GEMM；底层返回局部梯度，Op 用 `+=` 合并。
- 直接卷积、CUDA im2col 和自写 GEMM 仅作为历史失败基线保留在实验报告中，不再作为当前代码后端。
- FP32 小 CNN 完成同初值 CuPy/CUDA C 训练对照；完整 ResNet 的精度和 ImageNet stem 兼容仍在后续范围。
- `benchmark/cuda_ops.py`、`profile_cuda.py` 分别记录 CuPy/原生 CUDA 实验和真实 Nsight 报告；GPU Events、端到端 wall time、profile 计时分开。
- 统一测试入口补齐函数式测试，训练 smoke 显式传 seed；DataLoader 预先分批并按序返回，避免 worker 调度改变尾批数量。

## 7. 分布式训练（2026-09-19 交付）

- `MyFlows/distributed/launcher.py` 的 `run_training(**config)` 用 Windows spawn 启动进程。合法组合为 single/none、ps/socket_json、ps/grpc_proto、ring/grpc_proto；任务为 synthetic、mnist_mlp、donkey_cnn、donkey_resnet18。
- PS 模式：1 个 CPU PS 进程分发初值、按样本数加权聚合并返回平均梯度；N 个 Worker 在 GPU 上前向、反向和本地 Adam/MBGD 更新，通过更新确认屏障保持一致。PS 不执行优化器更新。
- Ring 模式：N 个对等 rank，每个 rank 运行 gRPC 服务并连接右邻居，完成 Split、N−1 轮 ScatterReduce 与 N−1 轮 AllGather；启动器另起元数据屏障，只传摘要，不传梯度。
- 心跳、watchdog、限时等待、重复/陈旧消息拒绝和故障注入（crash、timeout、duplicate、retry、schema、heartbeat_loss、gather_crash）。
- `benchmark/distributed_experiment.py` 是正式实验单元：写 config/manifest，运行训练，并在后台线程约 1 秒一次采样 nvidia-smi 与 psutil，写 `resources.jsonl`；各 rank 通过 `Recorder` 写 `rank-*/steps.jsonl`、`epochs.jsonl` 与状态快照；结束后汇总为 `results.json`、`epochs.csv`、`events.jsonl`。
- `benchmark/distributed_suite.py` 与 `complete_course_experiments.py` 串行编排数值、状态、训练、性能、故障各阶段；`report_distributed.py` 生成对比表和图。
- 所有时间戳使用 `time.perf_counter()`；Windows 上它是系统级计数器，不同进程的记录可以直接对齐。

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
    Jobs -- "子进程" --> Single["apps.train.train_myflows_donkey"]
    Jobs --> JobRec["每任务 ResourceRecorder 0.5 s"]
    Registry -- "增量读取" --> Files["rank-*/steps.jsonl、epochs.jsonl、metrics.jsonl、resources.jsonl、results.json"]
    Registry --> Archive["docs/experiments/.../run_index.json"]
    Registry --> TB["mycar/logs/tensorboard"]
~~~

- 数据来源：控制台任务目录 `runs/console/`、分布式正式归档（按 `run_index.json` 发现）和 TensorBoard 单进程日志。控制台只读这些文件，不改动训练进程。
- 实时更新：前端先用 REST 取全量（降采样到 3000 点），再订阅 `/api/runs/{id}/stream`，服务端按游标每秒推送新增的 step、epoch 和资源行；运行结束时发送 `end` 事件。
- 时间对齐：step、epoch、资源采样都用 `perf_counter`，以最早一条记录为原点，因此 loss 曲线、阶段耗时和 GPU 利用率可以放在同一时间轴上。
- 任务：只接受框架支持的模式/传输组合与白名单参数；默认串行执行。分布式任务停止时发送 CTRL_BREAK，单进程任务写 `STOP_TRAINING` 让训练保存断点后退出；20 秒内未退出则结束整个进程树。控制台重启后按 PID 与创建时间重新接管仍在运行的任务。
- Worker 视图：分布式运行按 rank 显示曲线、阶段耗时占比、最新 step 与梯度同步占比；各 rank 在 `prepare.json` 中记录 PID，控制台据此把 GPU 进程和进程内存对应到 rank。
- 单进程训练：`TrainingDashboard` 在 `--run-dir` 下同时写 `metrics.jsonl`，与 TensorBoard 记录内容一致。

## 9. 当前限制

- CUDA C 暂不支持 FP64、分组/空洞卷积和带 padding 的 Pool；这些高级卷积功能继续走原后端。当前原生路径限定 groups=1、dilation=1。
- 分布式训练只支持单机回环；所有 Worker 共享一张 GPU，没有多 GPU、多机或 NCCL 实验，也不能按 Worker 指定设备。ResNet18 独立整批与分片轨迹的逐步梯度等价未通过。
- 没有训练调优 AutoPilot Agent。
- GPU 监控仍不完整：`MyFlows/monitoring` 已支持 NVML 多卡采样，并由控制台接入分布式与单进程任务，但 `distributed_experiment` 内部的采样仍是 nvidia-smi 且只读第一张卡，资源曲线也没有写入 TensorBoard。Windows WDDM 驱动下读不到单个进程的显存，只能显示整卡显存和各进程 RSS。
- 没有预训练参数导入和冻结训练功能。
- 当前跨框架性能测试的方法需要重做，旧结果不能直接用于本学期结论。
- 主训练入口 `apps/train` 的完整 ResNet 仍未接入原生 CUDA 后端；通用模型状态与 checkpoint 精确恢复仍待后续完成。
