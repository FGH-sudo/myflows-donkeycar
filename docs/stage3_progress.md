# 第三阶段开发记录

本文记录 `stage3_plan.md` 的实施事实、验收证据与待办；规划、学期规格、AutoPilot 规格和文档索引仍由用户审核维护。

## 基线与执行约定

- 开工日期：2026-11-05。
- 根仓库基线：`2c20a5a`；MyFlows 基线：`ce63df5`。两个工作区开工时均干净。
- 当前开发分支：两个仓库均为 `codex/stage3-development`。按工作包保存本地提交，先 MyFlows 后根仓库；公开推送留待具体交付完成并经用户同意。
- 使用仓库 `.venv`，不运行 `uv sync`；GPU 训练、测试、计时与模拟器串行。
- 全量训练和模拟器使用已获用户授权。原始运行放在 `runs/stage3/` 或规划指定的本地实验目录；只提交报告和源码。
- 每次中断前更新本文，继续时先读本文及最近的实验报告，不重复已完成验收。

## 执行清单

| 工作包 | 状态 | 下一项 / 证据 |
| --- | --- | --- |
| 开工回归 | 通过 | `runs/stage3/baseline-tests.log`，统一 `--scope all`，443 项，549.087 s |
| S1 监控交付 | 实验已完成，等待统一回归复测 | 9 次开销、两次真实 Nsight、完整三轮趋势、60 次冻结对照及五项自查见[交付报告](experiments/semester_2026_fall/monitoring/README.md) |
| S2 冻结反向截断 | 数值与性能已验证，等待统一回归复测 | 三架构四后端 40 组数值检查，2,012 组梯度与更新逐字节相同。冻结指纹保持，60 次交替计时见[冻结报告](experiments/semester_2026_fall/monitoring/freeze-001/report.md) |
| S3 迁移学习组件 | 待开始 | S2 后；含参数组、渐进解冻、缓存、增强、LoRA |
| S4 迁移实验 | 待开始 | S3 后；小规模后全量配对种子 |
| S5 跨框架比较 | 协议与 worker 已编写 | 待小规模校准、正式三种子质量与两档吞吐验证 |
| S6 闭环评估 | S6.1、S6.2 已实测 | 721 帧探测标定阈值，三模型短跑通过。三模型共 30 个正式 episode 已完成，S4.2 最佳迁移模型随后加入 |
| S7 CIFAR-10 系统测试 | 数据已备齐 | 二进制数据在 `runs/datasets/cifar10/` |
| S8 算子与显存优化 | 待开始 | S2 后串行，含所有已纳入拓展项 |
| S9 测试计划与 CI | 待开始 | 本地 / WSL 验证后准备公开 CI；推送前请求同意 |
| S10 AutoPilot 增量 | 待开始 | 失败归类、算子诊断、迁移先验 |
| S11 设计与最终报告 | S11.2 已归档 | formal-full-002 与既有 E2 / M6 报告已整理。设计文档持续同步，控制台与最终报告待完成 |

## 运行与决策记录

| 日期 | 项目 | 结论 / 证据 |
| --- | --- | --- |
| 2026-11-05 | 开工检查 | 启动文档指定的统一回归，实际收集数见下方基线记录。 |
| 2026-11-05 | S1 实现决策 | 保留旧字段并增加 schema 版本；TensorBoard 改用自身事件 API，避免训练运行时导入 torch。独立采样进程负责 NVML，训练仅增量读取 JSONL。 |
| 2026-11-05 | 基线回归 | 443 项通过，549.087 s；规划中的 446 项与当前实际收集数不同。 |
| 2026-11-05 | S1.1–S1.2 专项 | 19 项通过；覆盖假采样器、缺测、增量读取、PID 重用、正常停止、历史 formal-full-002 摘录读取、无 torch 的 TensorBoard 事件回读。`runs/stage3/s1-unit.log`。尚待真实训练集成及开销验收。 |
| 2026-11-05 | S1.1–S1.2 集成 | 63 项通过（54.463 s），另有 2 项进程生命周期检查通过；CPU/CUDA、PS/Ring、TrainingState 精确恢复与控制台/AutoPilot 回归见 `runs/stage3/s1-integration.log`。 |
| 2026-11-05 | S1.1–S1.2 真实训练 | `runs/stage3/monitor-smoke/`：80 条真实数据，64/16 train/val，w64、batch 2、1 epoch；33 行 metrics、28 行 resources 全部通过 v1 校验。TensorBoard 回读 11 条资源曲线及三段耗时；`tensorboard-check.json`。 |
| 2026-11-05 | S1.3 | `cb3d691` / `0a06459` 接入监控，`9cdb103` 引入交替测量。9 次独立 CUDA 训练均通过格式验证，0.5 s 采样的配对开销中位数 0.601%，重复间标准差 14.709 个百分点。详见 [开销报告](experiments/semester_2026_fall/monitoring/overhead-001/report.md)。 |
| 2026-11-05 | S1.4 | `3c16500` / `0e642cf`。真实 Nsight 捕获第 5–9 步，共 142.306 ms。792 个空闲片段均有源码定位，各阶段共享 step 分母，每步 494 个 kernel。冻结前三段后，首个卷积仍执行了 5 次反向。[时间线报告](experiments/semester_2026_fall/monitoring/timeline-001/report.md) |
| 2026-11-05 | S2 基线准备 | 单独检出 S1.4 双仓库并编译其原生 DLL。比较脚本使用固定输入和转换后的同一组权重。 |
| 2026-11-05 | S1.4 统一回归 | 463 项通过，583.505 s。日志见 `runs/stage3/s1-profile-regression.log`。 |
| 2026-11-05 | S1.5 / S2 | 根仓库 `89d6f7c`，MyFlows `e49d4e3`。定向测试 31 项通过。40 组独立进程比较、60 次交替计时均完成。在冻结 3 段的同一诊断区间，每步 kernel 数由 494 降至 373。 |
| 2026-11-05 | S1.6 | 2,000 帧三轮训练的 12 项趋势已生成。loader 0 / 2 的稳态读取占比为 16.86% / 2.08%，每轮数值轨迹一致。TensorBoard 阶段曲线支持与常规日志不同的采样步。 |
| 2026-11-05 | S6 | `b26dd0a`。真实 Unity 短跑三模型通过，10 个配对场景使用 30 s 上限、油门 0.2、连续 3 帧超过 2.0 的走廊出界规则。首次启动检测未识别已监听的端口，现同时读取 Unity 的就绪日志。 |
| 2026-11-05 | 统一回归复测 | 首次 473 项回归中一个既有驾驶 fallback mock 因新增参数枚举失败。已将冻结操作限定到按训练配置构建的模型路径，fallback 专项通过。完整复测日志为 `runs/stage3/s2-monitor-regression-002.log`。 |
| 2026-11-05 | S11.2 | [正式调优归档](experiments/semester_2026_fall/autopilot/formal-full-002/report.md)与[历史迁移归档](experiments/semester_2026_fall/e2_transfer/historical-001/report.md)包含原始文件位置与校验和。历史 ResNet18 分类产物实际冻结 2 段，另两种模型冻结 3 段。 |
| 2026-11-05 | S6.3 基础比较 | 三模型 × 10 次，30 s 全部存活，碰撞及出界为 0。候选离线 MAE 略低，闭环平均绝对 CTE 高 0.013。原始帧、配对差值与 checksum 见[闭环报告](experiments/semester_2026_fall/closed_loop/baseline-001/report.md)。 |

## 待用户审核的规划文档更新建议

- 完成各工作包后，将本记录的已验收条目同步到 `stage3_plan.md` 第 8 节与 `semester_spec.md` 第 7 节。
- 暂无需要新增依赖或改变验收标准的决定。
