# MyFlows 深度学习框架与自动驾驶项目

本仓库包含两部分：

- MyFlows/：自研深度学习框架，负责计算图、算子、模型、优化器和 GPU 运行。
- 根仓库：负责 DonkeyCar 数据、训练、评估、模拟器驾驶、服务和课程实验。

本学期以 [课程目标 PDF（本地课件）](<docs/深度学习框架-16 综合项目III.pdf>) 为依据，开发重点是：

1. MyFlows AutoPilot Agent 自动安排和调优训练。
2. 使用 CUDA C/C++ 实现卷积和池化算子。
3. 实现单机多进程 Parameter Server，并进一步实现 All-Reduce。
4. 增加 GPU 资源监控和训练性能分析。
5. 建立至少三种 CNN 的预训练模型和迁移训练能力。
6. 继续优化 DonkeyCar，并用统一条件比较训练时间、资源占用和驾驶效果。
7. 完成相应的系统测试、设计文档和实验报告。

具体范围、排期和验收标准见 [本学期开发 Spec](docs/semester_spec.md)。后续学期规划与已验证的阶段成果分别记录。

已按 [第一阶段 Spec](docs/stage1_cuda_ps_spec.md) 实现 CUDA Conv/Pool 前反向、FP32 小 CNN、CPU PS 和 Nsight 入口；运行结果与阶段验收状态见 [第一阶段实验报告](docs/experiments/semester_2026_fall/stage1/README.md)。

已按 [分布式 GPU Spec](docs/stage1_distributed_gpu_spec.md) 实现单机同步 PS（Socket JSON / gRPC）与 Ring AllReduce（gRPC），Worker 在 GPU 上计算并本地更新，PS 只聚合梯度；MNIST MLP 与 DonkeyCar ResNet18 两个任务的 1/2/4 Worker 结果见 [PS / Ring 双任务实测报告](docs/experiments/semester_2026_fall/distributed_gpu/20260919_optimized/README.md)。同一套 PS / Ring 协议也接入了正式训练入口 `apps.train.train_myflows_donkey`（`--distributed ps|ring --workers N`）：用 `mycar/data` 的全部数据、正常的 BN 训练与验证集，产出与单进程格式相同、可续训和导出 ONNX 的模型，见下文“当前有效入口”。Agent 的进一步设计与实现安排到后续阶段。

按《CUDA 编程（上）》课件（第 19 讲）重新审查了数值实现与 CUDA 路径：修正了 Adam 首步步长偏小、`apps/train` 整网以 FP64 计算、BN 在均值远大于标准差时丢失精度、交叉熵 `log(p + 1e-10)` 截断等问题，加入第 19 讲的矩阵乘 kernel 与 Nsight Compute 分析；与修改前代码的逐项对照见 [CUDA 第 19 讲实验报告](docs/experiments/semester_2026_fall/cuda_lecture19/README.md)。

## 当前基础

仓库目前已经具备：

- NumPy/CuPy 动态计算图和自动求导。
- Conv2D、MaxPool2D、BatchNorm、Dropout 等基础算子。
- 显式 FP32 原生 CUDA Conv/Pool 后端（C/C++ 动态库调度自写 kernel，Windows 为 DLL、Linux 为 .so；卷积 GEMM 可选第 19 讲的寄存器分块 kernel 或 cuBLAS）、三后端实验与 Nsight Systems/Compute 采集；`apps/train`、分布式任务与控制台在 CUDA + float32 时默认使用寄存器分块版。
- 层参数默认 float32；BN 统计量、bias 梯度、交叉熵等规约在 FP64 中累加；BN、交叉熵、Adam 为融合 kernel，Adam 为标准实现。
- 第 19 讲矩阵乘 kernel（basic / tiled / shared / 寄存器分块 × fp32 / kahan / dot2 累加）与 `benchmark/matmul_compare.py`；寄存器分块版推广为原生卷积与 `Dense` 的 GEMM 引擎，与 cuBLAS 并列可选（`benchmark/gemm_engines.py`）。
- 单机同步 Parameter Server（Socket JSON / gRPC）与 Ring AllReduce（gRPC），1/2/4 个 GPU Worker 共享单卡；数值等价、状态恢复、故障注入与通信量检查。
- 分布式实验中的资源采样（nvidia-smi + psutil，约 1 秒一次）和 CUDA Event 阶段计时。
- MyFlows/monitoring：NVML 优先、nvidia-smi 兜底的整卡 GPU 采样与后台记录器（JSONL）。
- 正式训练的数据并行：`apps.train.train_myflows_donkey` 单进程与 PS / Ring 多 Worker 共用同一个训练循环；每步按样本数平均梯度与 BN 滑动统计量并核对各 Worker 状态一致，停止时所有 Worker 在同一步保存断点，可按断点中的 epoch 与 step 续训。
- 训练控制台 apps/console：浏览器中查看全部历史/实时运行的 loss、阶段耗时、GPU 占用和 PS/Ring 拓扑，并能发起、排队和停止正式训练（单进程或 PS / Ring 数据并行）与分布式实验。
- ResNet18 图像回归模型。
- DonkeyCar 数据读取、训练、验证和 ONNX 评估入口。
- TensorBoard 训练可视化、checkpoint、Grad-CAM、INT8 和推理服务基础。
- DonkeyCar 模拟器（按平台使用 Windows / Linux / macOS 版本）、本地数据和模型运行环境。

以下内容仍是本学期的待开发项：

- ResNet 分片与整批的逐步梯度严格等价（当前仍超差）。
- AutoPilot Agent。
- 框架级 GPU 监控的剩余部分：TensorBoard 资源曲线、分布式实验内部监控切换到 NVML；Windows WDDM 下无法读取单进程显存，控制台只显示整卡显存和各 Worker 的 RSS。
- 三种 CNN 的预训练参数导入、冻结和迁移训练。
- 自动化 DonkeyCar 闭环评估。

本学期不再把 VGG 作为训练、评估或模型对比方向。相关源码暂时保留，避免在文档整理阶段同时引入代码兼容性风险。

## 目录

| 路径 | 用途 |
| --- | --- |
| MyFlows/ | 自研框架源码和框架测试 |
| apps/common/ | DonkeyCar 数据读取和图像预处理 |
| apps/train/ | 当前 ResNet18 训练入口 |
| apps/eval/ | MyFlows 与 ONNX 评估入口 |
| apps/serve/ | gRPC、FastAPI 和 ONNX 推理 |
| apps/console/ | 训练控制台：FastAPI 后端 + React 前端（web/） |
| runs/console/ | 控制台发起的任务目录（不纳入 Git） |
| benchmark/ | CUDA 算子、PS/Ring 分布式实验、Nsight 与报告脚本 |
| proto/ | 分布式训练与推理服务的 protobuf 定义 |
| generated/grpc/ | 由 tools/generate_distributed_proto.py 生成的 gRPC 代码 |
| mycar/ | DonkeyCar 工程和驾驶入口 |
| tools/ | 数据分析、模型导出、量化、原生 CUDA 构建、统一测试等工具 |
| scripts/ | 量化评估辅助脚本 |
| deploy/ | 推理服务的 Docker / Kubernetes 配置 |
| docs/ | 当前文档、课程目标和本学期 Spec |

## 当前运行环境

项目使用仓库内的 Python 3.11 虚拟环境。不要直接使用系统默认的 python，因为当前系统默认 Python 3.14 没有项目依赖。Windows 在 PowerShell 中使用 `.\.venv\Scripts\python.exe`，Linux（含 WSL2）在 bash 中使用 `.venv/bin/python`，其余参数相同：

~~~powershell
.\.venv\Scripts\python.exe --version
~~~

~~~bash
.venv/bin/python --version
~~~

原生 CUDA 卷积/池化使用自写的原生库（Windows 为 `native_cublas.dll`，Linux 为 `native_cublas.so`）。库不存在或与源码不一致时，第一次用到原生算子会自动用 CMake 编译（需要 C++ 编译器，工具链见[开发环境](docs/development_environment.md)），编译失败会直接报错，不会改用 CuPy。也可以提前手动编译：

~~~powershell
.\.venv\Scripts\python.exe tools\build_native_cuda.py
~~~

~~~bash
.venv/bin/python tools/build_native_cuda.py
~~~

运行全部测试（同时收集 unittest 和函数式测试；`--scope` 可取 all、framework、apps）：

~~~powershell
.\.venv\Scripts\python.exe -m tools.run_tests --scope all
~~~

~~~bash
.venv/bin/python -m tools.run_tests --scope all
~~~

第一阶段已修复原先偶发失败的图像分类用例；2026-09-19 分布式阶段交付时统一回归为 204 项通过，2026-09-23 加入训练控制台与 GPU 采样后为 229 项通过。

## 当前有效入口

启动训练控制台（首次或前端改动后先构建，需要 Node.js 20.19+ 或 22.12+）：

~~~powershell
Set-Location apps\console\web; npm ci; npm run build; Set-Location ..\..\..
.\.venv\Scripts\python.exe -m apps.console.server
~~~

然后访问 http://127.0.0.1:8790 。控制台默认只监听本机；任务按提交顺序串行执行（单 GPU），产物写入 runs/console/。开发前端时可在 apps/console/web 下执行 `npm run dev`，Vite 会把 /api 代理到 8790 端口。

检查 DonkeyCar 数据：

~~~powershell
.\.venv\Scripts\python.exe -m tools.analyze_donkey_data --data mycar\data
~~~

运行 ResNet18 小规模训练：

~~~powershell
.\.venv\Scripts\python.exe -m apps.train.train_myflows_donkey --max-samples 200 --epochs 1 --device auto
~~~

用 2 个 Worker 做数据并行训练（`--distributed ring` 为 Ring AllReduce，`ps` 为 Parameter Server；`--batch` 是全局 batch，按 Worker 切分）：

~~~powershell
.\.venv\Scripts\python.exe -m apps.train.train_myflows_donkey --distributed ring --workers 2 --batch 8 --max-samples 1000 --val-size 100 --epochs 2 --export-onnx --device cuda
~~~

所有 Worker 共享本机一张 GPU，每个 Worker 的 CuPy 显存池默认上限为启动时空闲显存的 90% 除以 Worker 数（`--worker-memory-mb` 可改），超出时直接报显存不足。Socket JSON 每条消息有 64 MB 上限，只能用于 `--base-width 32` 及以下的模型。BN 前向用各 Worker 本地 batch 的统计量，所以多 Worker 结果与单进程不会逐位相同；1 个 Worker 时 PS、Ring 与单进程逐位一致（见 `tests/test_train_distributed.py`）。

评估现有 ONNX 模型：

~~~powershell
.\.venv\Scripts\python.exe -m apps.eval.eval_myflows_donkey_onnx --checkpoint mycar/models/myflow_resnet18_best.onnx --split-file mycar/logs/resnet18_split.json --split test --max-samples 200 --fixed-throttle 0.2 --force-fixed-throttle --device cuda
~~~

启动 DonkeyCar 模拟驾驶：

~~~powershell
Set-Location mycar
..\.venv\Scripts\python.exe manage.py drive --model=models/myflow_resnet18_best.onnx --type=myflows
~~~

Linux 下以上入口的写法：

~~~bash
# 训练控制台（首次或前端改动后先构建前端）
(cd apps/console/web && npm ci && npm run build)
.venv/bin/python -m apps.console.server
# 检查数据、小规模训练、ONNX 评估
.venv/bin/python -m tools.analyze_donkey_data --data mycar/data
.venv/bin/python -m apps.train.train_myflows_donkey --max-samples 200 --epochs 1 --device auto
.venv/bin/python -m apps.train.train_myflows_donkey --distributed ring --workers 2 --batch 8 --max-samples 1000 --val-size 100 --epochs 2 --export-onnx --device cuda
.venv/bin/python -m apps.eval.eval_myflows_donkey_onnx --checkpoint mycar/models/myflow_resnet18_best.onnx --split-file mycar/logs/resnet18_split.json --split test --max-samples 200 --fixed-throttle 0.2 --force-fixed-throttle --device cuda
# 模拟驾驶
cd mycar && ../.venv/bin/python manage.py drive --model=models/myflow_resnet18_best.onnx --type=myflows
~~~

模拟器按平台使用仓库根目录下的 DonkeySimWin、DonkeySimLinux 或 DonkeySimMac（从 gym-donkeycar 发布页下载解压），也可用环境变量 `DONKEY_SIM_PATH` 指定路径，或设为 `remote` 连接已经启动的模拟器。Linux 上目前只实测了原生库的构建和测试；ONNX 的 GPU 推理还需要配置 cuDNN，模拟驾驶尚未实测，详见[开发环境](docs/development_environment.md#linux-环境)。

## 文档

- [本学期开发 Spec](docs/semester_spec.md)：本学期范围、排期、进度和验收标准。
- [第一阶段开发 Spec](docs/stage1_cuda_ps_spec.md)：CUDA/CPU PS/Nsight 的接口、任务拆分与验证矩阵（已交付）。
- [分布式 GPU Spec](docs/stage1_distributed_gpu_spec.md)：同步 PS、Ring AllReduce、GPU Worker 与同条件评测（已交付）。
- [开发环境](docs/development_environment.md)：Python、CUDA、Nsight 与依赖锁定。
- [当前系统设计](docs/system_design.md)：只说明当前真实存在的系统。
- [当前模块说明](docs/module_design.md)：当前有效代码入口及职责。
- [文档索引](docs/README.md)：全部实验报告与阶段文档。
- [课程目标 PDF（本地课件）](<docs/深度学习框架-16 综合项目III.pdf>)：教师提供的原始目标，不纳入 Git。

旧实验截图、模型、数据和日志属于本地运行资产，不作为本学期功能已完成的证明。本学期的新结果必须由新代码重新运行并记录完整命令和环境。
