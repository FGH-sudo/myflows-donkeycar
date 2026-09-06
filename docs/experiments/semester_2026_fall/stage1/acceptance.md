# 第一阶段完成核对

核对日期：2026-09-05。依据 [阶段 Spec](../../../stage1_cuda_ps_spec.md) 的原始 C0-C7 范围；本表补充 [报告](README.md) 的证据索引，不增加或缩减阶段目标。

| 约定 | 核对对象 | 结论 |
| --- | --- | --- |
| C0：原有函数式测试不能漏收集 | tools/run_tests.py 的 ProjectLoader；最终日志中的 10 个函数式名称；103 框架 + 24 应用/交付工具 | 通过 |
| C0：固定真实初始化，不能放宽断言 | make_initializer 显式共用 seed；test_seeded_layers.py；digits/tiny CNN 保留原指标；regression-final-003 三次 127 项、无 skip | 通过 |
| C0：NumPy/CuPy 基线、RawKernel 可用 | c0-baseline-001 的原始误差与编译小样例；后续正式矩阵实际 CUDA C 运行 | 通过 |
| C0/C4：小模型 FP32 全链路 | Dense/Conv 初始化、Graph 梯度种子、交叉熵 onehot、Adam 标量修正；assert_fp32 检查参数/中间值/梯度/累积状态 | 通过 |
| C1/C2：T0 可手算；T1-T4 参数矩阵 | test_cuda_convolution.py 与 correctness-final-002 的每张量输出/独立 FP64 参考 | 通过 |
| C2：固定非均匀 dy 与独立数值梯度 | fixture 保存随机 dy；directional difference 用 FP64 前向计算标量目标，不复用 CUDA backward | 通过 |
| C3：T5 三池化配置，T6 负数/tie/overlap | test_cuda_pooling.py 的首索引、重叠中心梯度为 4、非方形/非连续输入与 dy；T5 原始矩阵 | 通过 |
| T6：bias 有/无、全零、非连续 x/w/b/dy | test_no_bias_zero_and_strided_arrays；无 bias 返回 db=None | 通过 |
| T7：非法 shape/dtype/device/groups/dilation 拒绝 | wrapper/Op 参数检查；invalid 测试与 patch 验证 Conv 未发射；设备改变 backward 报错 | 通过 |
| 当前 stream、局部新梯度、+= | wrapper 只在当前 stream 分配/发射；nonblocking stream 测试；两个 Conv 共享 x/w/b 与 CuPy 比较 | 通过 |
| forward context 刷新、固定实际后端 | 多次改变 batch/空间 shape；Pool context 形状与梯度检查；显式后端与融合路径测试 | 通过 |
| C4：包含真实 Pool 的 CNN、100 步、同初值 | restored-cnn-001 与集成测试；两个 GPU 后端都达到 100% 小样本训练准确率，初始/早期梯度/更新对照 | 通过 |
| 原 NumPy/CuPy 与高级卷积无回归 | 最终完整三轮，包括原 test_convolution、ResNet/BN/graph_opt/序列化/应用测试 | 通过（限本次回归范围） |
| C5：P0/P1/P2 与 P3 六项全部三后端 | performance-final-004 的 27 项正确性和 108 组测量；原始数据已清理，结论保留在报告和汇总表 | 通过，无案例缩减/超时 |
| 10 warmup、50 次 × 3 组、预算与失败码 | 实际 config；cuda_ops 中估测/期限检查；每组原始计数；交付工具错误/覆盖测试 | 通过 |
| CPU/GPU/传输/编译计时口径 | CPU perf_counter；当前 stream Events+stop 同步；单独 transfer-inclusive 与 transfer_ms；compile-cold-001 空缓存/新进程 | 通过；kernel-only 按 Spec 允许标记未测 |
| 输出全部误差、均值/中位数/P95/N/加速比 | 已生成 results.json、timings.csv 和 analysis/tables.md；当前交付保留 analysis/tables.md，最终分母 1e-6，allclose 容差未改 | 通过 |
| 两种规模 Systems，含 CuPy 对照 | 四份最终 capture.nsys-rep、SQLite、kernel/API/NVTX CSV；期望 kernel 名称核验 | 通过 |
| 两种规模 Compute，实际计数器报告 | ncu-P0/P1-dx-window-002：退出码 0、非空 .ncu-rep、完整 metrics.csv、自写 dX 名称 | 通过，用户已解除计数器权限限制 |
| 有依据优化与前后普通计时 | P1 原 dX 最耗时；优化有效输出窗口；performance-001 / performance-dx-window-002 实际源码与固定输入；最终正确性回归 | 通过，不将 profile 时间当普通耗时 |
| C6：CPU PS/Worker/Launcher/Monitor | distributed/ 源码：spawn、有界消息、单调时间事件、唯一 server 更新、固定参数名/schema | 通过 |
| 1/2 worker 单步及 20 步等价 | restored-ps-1/2-001：初始及每步参数/mean loss/版本/样本数均比较 | 通过 |
| 不均匀 shard 权重与 4 worker 增强 | restored-ps-uneven-001 的 20+12；restored-ps-4-001 的 8+8+8+8；对应参数 NPZ | 通过 |
| 无重复平均、复制数组发送 | SmallMLP.update 拒绝累积缓存/acc_no 非空；clone 快照；按 n_i/global_batch 聚合 | 通过 |
| 重复/陈旧/错误形状或精度/非有限/空 shard | protocol 单元检查和最终单 worker 重复消息专项；失败非零退出 | 通过 |
| 默认 10 秒超时、5 秒内清理、无残留 | 四个 ps-fault run 默认 timeout=10；所有 cleanup<0.064 秒、alive_pids=[]、非零子进程/CLI 状态 | 通过，故障 run 保持 failed |
| worker/server 时间、数组字节、不伪造网络流量 | 运行时 events.jsonl/timings.csv 记录提交/确认/等待/compute/聚合/update/广播；当前仅保留报告摘要；payload=464 bytes | 通过，网络协议开销未测 |
| C7：配置、环境、实际双仓库源码/输入哈希 | 每个 run 曾生成 config/manifest/results/timings/stdout/report；manifest 保存 Git 状态和源码 hash；当前清理 fixture/输出快照，仅保留报告和最终 Nsight 原始证据 | 通过 |
| 交付不是 HEAD 或仅 Markdown | 当前源码、Git 历史、每次运行原始材料和逐文件校验信息 | 通过；不保存整项目压缩包 |
| 新环境与导出源码复现 | 隔离虚拟环境 include-system-site-packages=false、无 torch；三轮框架回归；独立无 Git 目录的正确性/CNN/1/2/4/非均匀 PS | 通过；旧可选 TensorBoard 测试单独记 skip |
| 第四次课可展示与后续边界 | 报告中的两张实际图、六步演示提纲、全部 CLI、原始 Nsight 文件；更新总体/阶段/系统/模块文档 | 通过 |

阶段未要求的完整 ResNet/BN、分类长训练、持续 GPU 资源监控、通用状态接口、PS 小 CNN、All-Reduce、Agent、跨框架/驾驶实验仍由总体 Spec 跟踪，不用本表宣告整个学期完成。有限输入是 CUDA 调用前提，kernel-only 和网络协议开销明确未测；这些边界均在报告中可见。
