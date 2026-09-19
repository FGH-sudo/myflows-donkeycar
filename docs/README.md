# 项目文档

本目录只保留当前仍需维护的文档。

| 文档 | 作用 | 状态 |
| --- | --- | --- |
| [semester_spec.md](semester_spec.md) | 本学期范围、任务依赖与进度；后续计划继续审核 | v0.4 进度更新 |
| [stage1_cuda_ps_spec.md](stage1_cuda_ps_spec.md) | CUDA 卷积/池化、PS、Nsight 的实现约定与验收 | v0.2 实施记录 |
| [stage1_distributed_gpu_spec.md](stage1_distributed_gpu_spec.md) | 同步 PS、Ring AllReduce、GPU Worker 与同条件评测 | v0.3 实施中 |
| [PS / Ring 双任务实测报告（优化版主结果）](experiments/semester_2026_fall/distributed_gpu/20260919_optimized/README.md) | MNIST、ResNet18 道路任务；PS/Ring gRPC/protobuf 优化对照、课件表、逐 epoch 数据、Nsight 与验收边界；[第四周汇报 PPT（本地文件，不纳入 Git）](experiments/semester_2026_fall/distributed_gpu/20260919_optimized/%E7%AC%AC%E5%9B%9B%E5%91%A8%E6%B1%87%E6%8A%A5_PS_Ring_%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83.pptx) | 2026-09-19：56 个正式档案、42 次性能重复、204 项回归；保留 ResNet 严格轨迹超差说明 |
| [PS / Ring 双任务实测报告（原基线）](experiments/semester_2026_fall/distributed_gpu/20260917/README.md) | 优化前历史基线，供逐配置复核 | 2026-09-18：历史结果，保留 ResNet 严格轨迹超差说明 |
| [第一阶段报告](experiments/semester_2026_fall/stage1/README.md) | 实测结果、限制、第四次课演示、复现命令和实验包 | 当前阶段成果 |
| [第二阶段 CUDA im2col Spec](stage2_cuda_optimized_spec.md) | 同算法路径迁移与 CuPy/CUDA 控制变量对照 | v0.1 实施前方案 |
| [第二阶段首轮记录](experiments/semester_2026_fall/stage1/stage2_im2col_report.md) | CUDA im2col/col2im 首轮实现与组件计时 | 2026-09-06 首轮 |
| [原生池化对照报告](experiments/semester_2026_fall/stage1/native_pool_report.md) | 原生 C/C++ 调度最大池化与 CuPy 对照 | 2026-09-06 |
| [semester_spec_review.md](semester_spec_review.md) | 对照 PDF 与代码的审核发现、实测证据 | 2026-09-05 历史审核，对应 v0.2 |
| [system_design.md](system_design.md) | 当前系统结构，不包含尚未实现的能力 | 当前有效 |
| [module_design.md](module_design.md) | 当前有效模块和运行入口 | 当前有效 |
| [深度学习框架-16 综合项目III.pdf](<深度学习框架-16 综合项目III.pdf>) | 教师提供的本学期目标 | 本地课件，不纳入 Git |

## 使用原则

- PDF 说明课程希望完成什么。
- semester_spec.md 说明本组准备怎么做，审核通过后再作为开发依据。
- stage1_cuda_ps_spec.md 规定当前阶段的具体任务与验收；Agent 后续设计暂缓，阶段产物不代表学期全部目标完成。
- system_design.md 和 module_design.md 只记录当前代码事实，不能提前写成“已经实现”。
- 按用户确认，之前学期目标已完成；缺失的往期交付材料不列为本学期待办。本学期新增功能与改进结论使用对应代码重新验证。
- 新实验报告统一放到 docs/experiments/semester_2026_fall/，并记录代码版本、命令、环境、输入数据和原始结果。
