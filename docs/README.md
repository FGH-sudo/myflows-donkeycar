# 项目文档

本目录只保留当前仍需维护的文档。

| 文档 | 作用 | 状态 |
| --- | --- | --- |
| [semester_spec.md](semester_spec.md) | 本学期范围、任务依赖与进度；后续计划继续审核 | v0.4 进度更新 |
| [stage1_cuda_ps_spec.md](stage1_cuda_ps_spec.md) | CUDA 卷积/池化、PS、Nsight 的实现约定与验收 | v0.2 实施记录 |
| [第一阶段报告](experiments/semester_2026_fall/stage1/README.md) | 实测结果、限制、第四次课演示、复现命令和实验包 | 当前阶段成果 |
| [第二阶段 CUDA im2col Spec](stage2_cuda_optimized_spec.md) | 同算法路径迁移与 CuPy/CUDA 控制变量对照 | v0.1 实施前方案 |
| [第二阶段首轮记录](experiments/semester_2026_fall/stage1/stage2_im2col_report.md) | CUDA im2col/col2im 首轮实现与组件计时 | 2026-09-06 首轮 |
| [semester_spec_review.md](semester_spec_review.md) | 对照 PDF 与代码的审核发现、实测证据 | 2026-09-05 历史审核，对应 v0.2 |
| [system_design.md](system_design.md) | 当前系统结构，不包含尚未实现的能力 | 当前有效 |
| [module_design.md](module_design.md) | 当前有效模块和运行入口 | 当前有效 |
| [深度学习框架-16 综合项目III.pdf](<深度学习框架-16 综合项目III.pdf>) | 教师提供的本学期目标 | 原始资料 |

## 使用原则

- PDF 说明课程希望完成什么。
- semester_spec.md 说明本组准备怎么做，审核通过后再作为开发依据。
- stage1_cuda_ps_spec.md 规定当前阶段的具体任务与验收；Agent 后续设计暂缓，阶段产物不代表学期全部目标完成。
- system_design.md 和 module_design.md 只记录当前代码事实，不能提前写成“已经实现”。
- 按用户确认，之前学期目标已完成；缺失的往期交付材料不列为本学期待办。本学期新增功能与改进结论使用对应代码重新验证。
- 新实验报告统一放到 docs/experiments/semester_2026_fall/，并记录代码版本、命令、环境、输入数据和原始结果。
