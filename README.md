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

已按 [第一阶段 Spec](docs/stage1_cuda_ps_spec.md) 实现 CUDA Conv/Pool 前反向、FP32 小 CNN、CPU PS 和 Nsight 入口；运行结果与阶段验收状态见 [第一阶段实验报告](docs/experiments/semester_2026_fall/stage1/README.md)。Agent 的进一步设计与实现安排到后续阶段。

## 当前基础

仓库目前已经具备：

- NumPy/CuPy 动态计算图和自动求导。
- Conv2D、MaxPool2D、BatchNorm、Dropout 等基础算子。
- 显式 FP32 CUDA C Conv/Pool 后端、三后端实验与 Systems/Compute 采集。
- CPU 同步 Parameter Server，小 MLP 的 1/2/4 worker 更新等价和故障清理。
- ResNet18 图像回归模型。
- DonkeyCar 数据读取、训练、验证和 ONNX 评估入口。
- TensorBoard 训练可视化、checkpoint、Grad-CAM、INT8 和推理服务基础。
- DonkeyCar Windows 模拟器、本地数据和模型运行环境。

以下内容仍是本学期的待开发项：

- 完整 ResNet 的 CUDA C 精度/训练接入，以及更多模型的 PS 集成。
- All-Reduce 训练。
- AutoPilot Agent。
- 完整 GPU 监控。
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
| benchmark/ | 性能测试脚本；本学期将按 Spec 重构 |
| mycar/ | DonkeyCar 工程和驾驶入口 |
| tools/ | 数据分析、模型导出、量化等工具 |
| docs/ | 当前文档、课程目标和本学期 Spec |

## 当前运行环境

项目使用仓库内的 Python 3.11 虚拟环境。不要直接使用系统默认的 python，因为当前系统默认 Python 3.14 没有项目依赖。

PowerShell 中使用：

~~~powershell
.\.venv\Scripts\python.exe --version
~~~

检查应用层测试：

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

检查框架层测试：

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s MyFlows\tests -v
~~~

当前框架测试中有一个图像分类用例存在偶发失败。在它修复前，不能把当前测试状态视为稳定基线。

## 当前有效入口

检查 DonkeyCar 数据：

~~~powershell
.\.venv\Scripts\python.exe -m tools.analyze_donkey_data --data mycar\data
~~~

运行 ResNet18 小规模训练：

~~~powershell
.\.venv\Scripts\python.exe -m apps.train.train_myflows_donkey --max-samples 200 --epochs 1 --device auto
~~~

评估现有 ONNX 模型：

~~~powershell
.\.venv\Scripts\python.exe -m apps.eval.eval_myflows_donkey_onnx --checkpoint mycar/models/myflow_resnet18_best.onnx --split-file mycar/logs/resnet18_split.json --split test --max-samples 200 --fixed-throttle 0.2 --force-fixed-throttle --device cuda
~~~

启动 DonkeyCar 模拟驾驶：

~~~powershell
Set-Location mycar
..\.venv\Scripts\python.exe manage.py drive --model=models/myflow_resnet18_best.onnx --type=myflows
~~~

## 文档

- [本学期开发 Spec](docs/semester_spec.md)：本学期范围、排期和验收标准，当前待审核。
- [第一阶段开发 Spec](docs/stage1_cuda_ps_spec.md)：当前 CUDA/PS/Nsight 的接口、任务拆分、验证矩阵与汇报目标。
- [当前系统设计](docs/system_design.md)：只说明当前真实存在的系统。
- [当前模块说明](docs/module_design.md)：当前有效代码入口及职责。
- [课程目标 PDF（本地课件）](<docs/深度学习框架-16 综合项目III.pdf>)：教师提供的原始目标，不纳入 Git。

旧实验截图、模型、数据和日志属于本地运行资产，不作为本学期功能已完成的证明。本学期的新结果必须由新代码重新运行并记录完整命令和环境。
