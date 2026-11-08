# 学期项目测试计划

Windows 命令在仓库根目录执行，表中的 `python` 取 `.venv/Scripts/python.exe`。CPU CI 使用 Python 3.11 和 requirements-ci.txt。在一张 GPU 上串行执行训练、性能试验、动态检查与模拟器。阶段实验使用新的输出目录，保存原始命令、协议、源码版本和逐项数据。

## 执行节奏

| 类别 | 触发时机 | 命令或入口 | 通过标准与时间口径 |
| --- | --- | --- | --- |
| 日常 CPU | 每个功能提交 | 针对变动模块运行 `python -m unittest -v <测试模块>`，函数式测试使用 `python -m tools.run_tests --scope framework --pattern <文件名>` | 所有目标用例通过。多数规则、序列化、API 组在数秒到一分钟内结束 |
| 托管 CPU | main 推送、PR、手工 dispatch | `python -X utf8 -m tools.run_tests --scope all` | Ubuntu CPU 全部适用测试通过，skip 日志可追溯，目标约 15 分钟内完成。相同锁定依赖的 WSL 样本为 495 项、56 skip、214 s |
| 本机集成 | 每个工作包完成 | `python -X utf8 -m tools.run_tests --scope all` | CPU、实际 CUDA、可选本机资源的结果逐项列出。第三阶段已观测的完整运行约 9–10 分钟 |
| GPU 性能 | 性能变动合入前、阶段报告前 | `python -m benchmark.perf_gate --reference-tree <固定对照代码树> --out runs/stage3/perf-<批次>` | 同 GPU 与同会话交替，多配置步时比值 ≤ 1.10，逐形状 register/cuBLAS ≤ 1.20，同时记录卷积与池化前向、反向、合计耗时 |
| 阶段验收 | 最终报告前 | `python -X utf8 -m tools.run_tests --scope all --repeat 3`，再执行 GPU 性能门槛 | 三轮分别退出成功，记录每轮耗时和 skip 原因。性能超限项列出形状、实际比值与证据 |
| 数据与模拟器 | 协议或模型变化后 | 先短跑，再执行下表的阶段实验 | 模型选择只使用验证集。测试集评分与配对场景在协议锁定后运行 |

## 需求与证据

实验路径均相对 `docs/experiments/semester_2026_fall/`。本机原始数据和可提交的 report.md 同目录。GPU 测试缺少设备时按用例跳过，门槛工具和正式实验要求实际设备，完整结果应记录其测量数量。

| 条目 | 频率与复现入口 | 核验内容与证据 |
| --- | --- | --- |
| B1、S10 | 日常运行 `python -m unittest tests.system.test_autopilot_campaign tests.system.test_autopilot_m4 tests.system.test_autopilot_m6 tests.system.test_stage3_specialists` | 预算、重试、故障四类、NVML/NCU 诊断、迁移先验、配对负迁移和确认流程。正式证据在 autopilot/formal-full-002 和 autopilot/diagnostics-001 |
| B2 | GPU 运行 `python -m unittest MyFlows.tests.test_cuda_native_kernels MyFlows.tests.test_cuda_native_cublas`。阶段运行 `python -m benchmark.cuda_ops --suite correctness --backends numpy,cupy,cuda_native_register,cuda_native_cublas --out-dir runs/stage3/op-check` | 卷积及池化的输出与全部梯度通过数值门限，阶段报告在 stage1/correctness-final-002 与 cuda_lecture19 |
| B3、E1 | 日常及 GPU 集成运行 `python -m unittest MyFlows.tests.test_ps_training MyFlows.tests.test_ring_training tests.test_train_distributed` | 同步 PS 与 Ring 的参数、统计量、断点、退出与错误传播。多进程证据在 distributed_gpu/20260919_optimized 与 stage1/restored-ps-uneven-001 |
| B4、S5 | 每次重要算子优化运行 `python -m benchmark.compare_frameworks_resnet --out runs/stage3/frameworks-<批次>` | 同一 FP32 初值、划分、批次与 Adam，独立进程检查全部 62 个梯度、一次更新、固定张量与端到端吞吐、资源峰值和三个种子训练质量。证据在 frameworks |
| B4、S6 | 模拟器/部署候选改变后运行 `python -m apps.eval.sim_probe --out runs/stage3/probe-<批次>`，再按 closed_loop/transfer-001 报告运行 closed_loop | 真实 telemetry、走廊与去抖、固定油门、终止原因，逐帧核对同场景配对差。每模型十组、每组至多 30 秒。证据在 closed_loop |
| B5、S7 | 日常 `python -m unittest tests.test_cifar10`。阶段依 classification/cifar10-001 报告依次运行 train_classify、cifar10_reference 和 score-test | 平衡 32 张训练帧过拟合 ≥95%，两模型冒烟，通过同构 PyTorch 的验证参考。固定模型后评价官方测试集，记录 checkpoint SHA |
| B5、S9 | 每包运行统一回归，性能变动运行 perf_gate。发布前走上表的三轮验收 | .github/workflows/tests.yml 上传完整 CPU 日志。性能原始数据分别为 steps.json、operators.json、gemms.json，基线在 runs/perf-baselines 按 GPU 与代码哈希保存 |
| B6、S11 | 每个交付节点核对设计文档与报告。前端更新后运行 `npm --prefix apps/console/web run build` 与 `python -m unittest tests.test_console_backend` | 真实控制台验证旧/新格式、四段耗时占比、12 项指标、S5 对比加载。第三阶段的页面截图位于 runs/stage3/console-qa/output/playwright |
| B7、S1 | 日常 `python -m unittest tests.test_monitoring_format tests.test_monitor_process tests.test_timeline_gaps`。阶段运行 monitor_overhead、profile_train 与 timeline_gaps | 12 指标、缺测、时钟、双采样防护、TensorBoard 回读，默认监控开销 <5%，逐个空闲区间能定位源码。命令与原始采集见 monitoring/README.md |
| E2、S2 | 冻结实现改变后，按 monitoring/freeze-001 报告依次执行 `benchmark.freeze_before_after` 的 reference、verify、speed | 相同初值下未冻结梯度和更新保持一致，冻结参数及 BN 指纹保持，前后端与输入尺寸覆盖记录在报告中 |
| E2、S3 | 日常 `python -m unittest MyFlows.tests.test_transfer_components MyFlows.tests.test_lora tests.system.test_autopilot_m6`。阶段运行 `python -m benchmark.transfer_study lora --out runs/stage3/lora-<批次>` | 三种架构、参数组、冻结/解冻续训、缓存逐步轨迹、六组低秩比较与合并导出误差。证据在 transfer/cache-001、transfer/lora-001、autopilot/diagnostics-001 |
| E2、S4 | 数据或协议改变后分别运行 `python -m benchmark.transfer_study classification --out <新目录>` 与 `python -m benchmark.transfer_study donkey --out <新目录>` | 分类两种数据、六档方法、两个种子。回归先 2,000 帧，再沿用正式时间划分，3 seeds × 3 epochs。两种图像解释、全部试验和独立测试记录在 transfer/classify-001 和 transfer/donkey-001 |
| E3 | 日常 `python -m unittest tests.test_training_state tests.system.test_autopilot_campaign`，TPE 与分组续训另由统一回归覆盖 | 跨进程恢复参数、优化器、随机流、历史最佳和训练顺序。回滚与确认不重复计入噪声估计。证据在 autopilot/formal-full-002 |
| S8.1、S8.2 | 算子修改后运行 `benchmark.memory_study --old <对照树> --out <新目录>` 与 `benchmark.resnet_before_after --old <对照树> --out-dir <新目录> --skip speed --include-register`，均以 `python -m` 启动 | 6 种子、4 后端逐数组核对，FP64 误差单列 ReLU 异号。以 WDDM 进程非本地驻留变化、相邻整数批次确定容量并报告耗时。证据在 operators |
| S8.3、S8.4 | 融合前后各跑同种子数值对拍与交替 phase/step 计时 | Adam 参数、两个矩与分组步数逐位一致，BN/残差/激活的正反向与缓冲一致。无可测收益的 Adam 改动按实验结果处理 |
| S8.5 | 原生 kernel 变化后运行 `python -m tools.sanitize_native --out runs/stage3/sanitizer-<批次>` | memcheck、racecheck、synccheck 的适用测试实际运行，诊断错误与竞争告警均归零。原始输出和测试日志分开保存 |
| S8.6–S8.8 | 调优、精度或算法变动后进行逐形状和训练端到端对照，并执行数值与性能门槛 | tile/split-K 实测缓存，记录占用率与切分舍入。TF32、FP16/BF16、Winograd 各以相同输入对照 FP64，记录训练质量、速度、显存和适用条件 |
