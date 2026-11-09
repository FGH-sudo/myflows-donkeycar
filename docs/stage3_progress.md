# 第三阶段开发记录

本文记录 `stage3_plan.md` 的实施事实、验收证据与待办；规划、学期规格、AutoPilot 规格和文档索引仍由用户审核维护。

## 基线与执行约定

- 开工日期：2026-11-05。
- 根仓库基线：`2c20a5a`；MyFlows 基线：`ce63df5`。两个工作区开工时均干净。
- 开发分支：两个仓库均为 `codex/stage3-development`。按工作包保存本地提交，先 MyFlows 后根仓库。2026-11-09 经用户同意，两个仓库都快进合入 main 并推送到公开远端。
- 使用仓库 `.venv`，不运行 `uv sync`；GPU 训练、测试、计时与模拟器串行。
- 全量训练和模拟器使用已获用户授权。原始运行放在 `runs/stage3/` 或规划指定的本地实验目录；只提交报告和源码。
- 每次中断前更新本文，继续时先读本文及最近的实验报告，不重复已完成验收。

## 执行清单

| 工作包 | 状态 | 下一项 / 证据 |
| --- | --- | --- |
| 开工回归 | 通过 | `runs/stage3/baseline-tests.log`，统一 `--scope all`，443 项，549.087 s |
| S1 监控交付 | 已验收（476 项回归通过） | 9 次开销、两次真实 Nsight、完整三轮趋势、60 次冻结对照及五项自查见[交付报告](experiments/semester_2026_fall/monitoring/README.md) |
| S2 冻结反向截断 | 已验收（476 项回归通过） | 三架构四后端 40 组数值检查，2,012 组梯度与更新逐字节相同。冻结指纹保持，60 次交替计时见[冻结报告](experiments/semester_2026_fall/monitoring/freeze-001/report.md) |
| S3 迁移学习组件 | S3.1–S3.7 已实现并专项验证 | 两个入口各 166 个 checkpoint 数组保持一致。验证了缓存、解冻续训、[LoRA 六组对照与导出](experiments/semester_2026_fall/transfer/lora-001/report.md)和[冻结特征先验](experiments/semester_2026_fall/autopilot/diagnostics-001/report.md) |
| S4 迁移实验 | 两项正式实验完成 | [分类递进](experiments/semester_2026_fall/transfer/classify-001/report.md)有 24 个试验。[回归对照](experiments/semester_2026_fall/transfer/donkey-001/report.md)有 14 个小规模试验及 21 个全量试验，附逐轮曲线、首层特征和 Grad-CAM；另按各种子最佳验证轮补充到达时间与配对检验 |
| S5 跨框架比较 | 正式协议已完成，持续复测 | 三个共同种子、两种吞吐和独立质量实验均已完成。S8 各项与计划表修正后共复测 7 次，最近一次为 frameworks/resnet18-plans-001，汇总见最终报告第 4 节。[报告](experiments/semester_2026_fall/frameworks/resnet18-001/report.md) |
| S6 闭环评估 | 探测与两组正式实验完成 | 721 帧标定，三模型 30 次。加入 S4 候选后在同一仿真会话轮换四个模型，共 40 次、24,011 帧。[配对结果与会话差异](experiments/semester_2026_fall/closed_loop/transfer-001/report.md) |
| S7 CIFAR-10 系统测试 | 全量验收已测 | 32 样本过拟合门槛通过。45,000/5,000 训练与验证、同构 PyTorch 比较、锁定后官方 10,000 张测试均已完成。[报告](experiments/semester_2026_fall/classification/cifar10-001/report.md) |
| S8 算子与显存优化 | S8.1–S8.8 已实测 | 最大驻留 batch 为 37→46→93。多张量 Adam、BN 融合完成对拍。三种 sanitizer 28 项均 0 错误，见 [sanitizer-002](experiments/semester_2026_fall/operators/sanitizer-002/report.md)。自动调优测了 4,365 个组合及 17 个占用率样本。Ada 计划经锁频计数修正 3 项，另有 4 项改用 cp.async 流水 tile，结果逐位不变。见[逐形状结果](experiments/semester_2026_fall/operators/autotune-001/report.md)、[锁频对比](experiments/semester_2026_fall/operators/gemm-ncu-001/report.md)与[归约与流水](experiments/semester_2026_fall/operators/gemm-ncu-002/report.md)。[TF32/FP16/BF16](experiments/semester_2026_fall/operators/precision-001/report.md)与[Winograd](experiments/semester_2026_fall/operators/winograd-001/report.md)只作对照 |
| S9 测试计划与 CI | 已验收 | GEMM 门槛改用 Nsight Compute 锁频周期，逐形状门槛在 [gate-profile-004](experiments/semester_2026_fall/operators/gate-profile-004/report.md) 中全部通过。Windows 三轮统一回归与 WSL 独立 checkout 见运行记录。GitHub 托管 CI 首次运行 [42640093233](https://github.com/FGH-sudo/myflows-donkeycar/actions/runs/42640093233) 通过，作业 4 分 4 秒 |
| S10 AutoPilot 增量 | 已实现并专项验证 | M4、M6 和新增诊断规则 26 项通过。真实线性探测、配对差与算子指标见[报告示例](experiments/semester_2026_fall/autopilot/diagnostics-001/report.md) |
| S11 设计与最终报告 | S11.1–S11.4 已整合 | 设计文档按 2026-11-09 的代码更新。[最终报告](final_report.md)覆盖系统设计、测试、B2–B7、E1–E3、S4–S10 与限制 |

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
| 2026-11-06 | S1/S2、S5 | 统一回归 476 项通过（604.327 s）。S5 共 30 个独立 worker 成功，62 个可训练张量逐个比较。三 seed 验证角度 MAE 为 MyFlows 0.06397 ± 0.03676，PyTorch 0.08408 ± 0.04474。 |
| 2026-11-07 | S3.1–S3.5 | 组件提交 `f788e98` / `056a1e4`，分组与解冻 `b045741` / `4dc8838`。两种 BN 策略跨越解冻点逐字节续训，缓存开关下分类 123 步和回归 84 步的 loss 与最终状态一致。[缓存报告](experiments/semester_2026_fall/transfer/cache-001/report.md) |
| 2026-11-07 | S7 | 完成配置锁定与独立测试，官方测试准确率 60.13%，多数类 10%。 |
| 2026-11-07 | S4 准备 | 冻结源码训练流程可按已验证的 attempt 接续。首步前的真实验证对照、含增强的 CIFAR-10 小规模迁移、两个训练入口的缓存轨迹均已测。长任务日志位于 `runs/stage3/transfer-classify-study.log` 与 `runs/stage3/transfer-donkey-study.log`。 |
| 2026-11-08 | S3.6、S9.3 | LoRA 训练与合并导出（`d540ed9` / `e133baf`），秩 4 合并 ONNX 最大输出差 8.20×10⁻⁷。CPU 工作流 `c33329f` 在 WSL Ubuntu 22.04 独立 checkout 收集 495 项、56 项带原因跳过，213.739 s（`runs/stage3/wsl-ci-verified.log`）。 |
| 2026-11-08 | S8.1、S8.2 | 首次写入分配梯度（`832e3c3` / `f515d04`）：w16 最大驻留 batch 37→46，1,968 个梯度数组与改动前数值相等。重算 im2col（`cb4a62b` / `80da1d4`）：池峰值降 47%–51%、步时增 14%–24%、驻留 batch 46→93，默认关闭。两次 S5 复测见 frameworks/resnet18-lazy-001、resnet18-recompute-001。 |
| 2026-11-08 | S8.3–S8.5、S9.2、S10、S11.3 | 多张量 Adam（`9531e04` / `bc133a3`）CUDA 更新 3.435→2.337 ms，参数与两组矩逐位一致。BN/残差/ReLU 融合（`5196b69` / `29aa8c0`）17 处，步时降 2.7%–4.3%。sanitizer 三种工具 22 项 0 错误。首个 GPU 门槛（事件计时）64 个形状中 29 个超过 1.20 倍。诊断规则与控制台阶段页见 `00e38f0`、`49d1f63`。 |
| 2026-11-08 | S8.6 | `28e5572` / `cac6927`：逐形状实测 tile 与 split-K，4,365 个组合与 17 个占用率样本，见 operators/autotune-001。 |
| 2026-11-09 | S8.7、S8.8 | `34a8b89` / `e032648`：TF32、FP16/BF16 存储加动态 loss scaling、Winograd F(2×2,3×3)，六 batch FP64 对拍与 2,000 帧三种子训练见 operators/precision-001、winograd-001；S5 复测 frameworks/resnet18-experimental-001。 |
| 2026-11-09 | S9.2 计划实测 | `25de44a` / `0f19c48`：补充 12 种 tile 几何与窄输出并行归约，按 Ada 实测计划选择 tile。事件计时门槛 gate-profile-001 剩 4 个超限，复核替换后 gate-profile-002 变为 8 个，其中多数形状的计划未变，同一形状的轮间耗时相差约 2 倍。S6 补测部署模型离线分数（`616023a`）。 |
| 2026-11-09 | S9.2 锁频门槛 | 交接后复查。`benchmark/gemm_ncu.py` 在 Nsight Compute 锁定基频、清缓存的条件下按 NVTX 区间比较 GEMM 周期，同一调用 7 次的极差中位数 4.0%（事件计时 36.4%）。修正前 64 个形状 6 个超限、加权合计 0.982；事件门槛报出的 8 个中 7 个实际在门槛内。候选搜索后修改 3 项 Ada 计划（MyFlows `f9344b5`），其中 2 项输出逐位不变。分组读取的归约改动经测量更慢，已撤回；流水线指标显示归约受 FP64 流水线限制。见 [锁频对比](experiments/semester_2026_fall/operators/gemm-ncu-001/report.md)。完整门槛 [gate-profile-003](experiments/semester_2026_fall/operators/gate-profile-003/report.md)：步时比值 1.003、1.001、0.969，GEMM 加权合计 0.977，仍有 4 个小 dW GEMM 超过 1.20（1.204–1.466），不清缓存时比值不变，记为已知超限。 |
| 2026-11-09 | 计划表对拍与 S5 复测 | 计划表此前没有数值对拍。按门槛的 w16 batch 128 运行 6 个 batch 的 FP64 对照：只有 register 后端的 120/498 个数组改变，最大相对变化 3.51×10⁻⁶，误差统计不变；batch 8 的 1,992 个数组前后逐字节相同（计划表不覆盖）。S5 复测 frameworks/resnet18-plans-001：端到端 MyFlows 462.85 samples/s、PyTorch 确定性 301.61，验证 MAE 与此前逐位相同。见 [plans-numerics-001](experiments/semester_2026_fall/operators/plans-numerics-001/report.md)。 |
| 2026-11-09 | S9.4 阶段验收回归 | Windows `--scope all --repeat 3`（根仓库 `9db82b2`、MyFlows `f9344b5`）：每轮 530 项全部通过，无跳过，557.7 s、546.4 s、567.8 s。日志 `runs/stage3/s9-all-repeat3.log`。 |
| 2026-11-09 | S4 补充 | `8902daa`：donkey-001 按各种子最佳验证轮补充到达时间与配对检验。冻结 4 段 0.02327 ± 0.00108，人工基线 0.03436 ± 0.01525，配对差 −0.01109 的 95% 区间跨零；仍未检出负迁移。 |
| 2026-11-09 | S9.3 本地复验 | WSL Ubuntu 22.04 独立 checkout（根仓库 `8902daa`、MyFlows `f9344b5`）收集 530 项，463 项通过、67 项带原因跳过，209.3 s；此前失败的恢复测试已改为确定性场景并通过。日志 `runs/stage3/wsl-stage3-final.log`。 |
| 2026-11-09 | S9.2 剩余 dW GEMM | MyFlows `724d0dd`：split-K 归约结果仍与固定顺序 FP64 求和逐位相同，但 2 份切分用 FP32 计算，3–32 份用 64 位整数精确求和（指数跨度超限、Inf/NaN 或次正规结果回到 FP64 循环）；TN 布局新增 3 种 cp.async 多级流水 tile，4 项 Ada 计划只换 tile、切分数不变。整网对拍 before-after-001 中 4 种配置 × 3 个 batch 的 82 个梯度数组与 loss 前后逐位相同；`tools.run_tests --scope framework` 311 项通过。完整门槛 [gate-profile-004](experiments/semester_2026_fall/operators/gate-profile-004/report.md)：步时比值 1.007、0.994、1.001，64 个 GEMM 全部不超过 1.20（最高 1.173），加权合计 0.963；原超限 4 项为 1.112、1.153、0.967、1.136。见 [gemm-ncu-002](experiments/semester_2026_fall/operators/gemm-ncu-002/report.md)。 |
| 2026-11-09 | 合并前复核 | 根仓库 `0290d6c`、MyFlows `724d0dd`。复核了 split-K 归约的逐位证明与 cp.async 流水的同步顺序；原生 GEMM、cuBLAS、计划表与门槛测试 36 项通过。[sanitizer-002](experiments/semester_2026_fall/operators/sanitizer-002/report.md) 三种工具各 28 项，0 错误、0 竞争。before-after-001 只覆盖 w16 的 1 项计划变动，另在 4 个生产形状上直接比较新旧计划与旧引擎，16 组输入逐位相同。WSL 独立 checkout 收集 534 项，463 项通过、71 项带原因跳过，226.0 s（`runs/stage3/wsl-stage3-merge.log`）。Windows `--scope all --repeat 3` 每轮 534 项全部通过，无跳过，580.0 s、587.1 s、578.1 s（`runs/stage3/s9-merge-repeat3.log`）。两个仓库待推送内容的隐私扫描无命中。 |
| 2026-11-09 | 合入 main 与 S9.3 托管运行 | 按用户决定，两个仓库快进合入 main 并推送：MyFlows `ce63df5..724d0dd` 先推送，根仓库 `918e042..c8071f7` 后推送。推送触发的首次 GitHub Actions 运行 [42640093233](https://github.com/FGH-sudo/myflows-donkeycar/actions/runs/42640093233) 通过：ubuntu-22.04、Python 3.11，作业 4 分 4 秒，其中依赖安装 22 s、统一回归 3 分 32 s。测试日志 artifact 已上传，运行未使用密钥。GitHub 上的日志需登录才能查看，同一代码在 WSL 独立 checkout 中列出 71 项跳过及原因。 |

## 待用户审核的规划文档更新建议

- 完成各工作包后，将本记录的已验收条目同步到 `stage3_plan.md` 第 8 节与 `semester_spec.md` 第 7 节。
- S9.2 的逐形状 GEMM 比较已按规划 2.3 节改用 Nsight Compute 锁频周期；建议在规划 S9.2 中写明这一测量方法，以及 WDDM 上约 20 分钟的耗时。
- 逐形状 register/cuBLAS ≤ 1.20 已在 gate-profile-004 中全部通过（4 个 dW GEMM 经结果逐位不变的归约与 cp.async 流水修正）。建议把规划 S9.2 中的 4 个已知超限改为已通过，并注明余量最小的是 TN 512×256×640（1.153）。
- S9.3 的 GitHub 托管 CI 已通过（运行 42640093233），第三阶段已快进合入两个仓库的 main。建议在规划第 8 节记为完成。
- 暂无需要新增依赖的决定。
