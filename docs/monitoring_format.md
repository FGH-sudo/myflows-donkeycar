# 训练与资源记录格式 v1

适用入口：`python -m apps.train --run-dir <目录>`，包括其单进程与 PS/Ring Worker；资源记录器也供控制台、AutoPilot 与独立实验使用。v1 只增加字段，不改变既有字段含义。历史记录没有 `schema_version` 时按 v0 读取，不自动重写。新任务可以增加任务专有字段；读取方须忽略未知字段，保留缺测值。

验证命令：`.venv/Scripts/python.exe -m tools.check_run_format <run-dir>`。正在运行或仅有部分文件的历史档案用 `--allow-incomplete`。校验覆盖版本、类型、时钟、非有限数、缺测原因与关键时间关系，不能替代训练正确性或完整实验验收。

## 公共约定

- JSON 对象、JSONL 的每一行均含 `schema_version: 1`。每行 UTF-8 编码，以换行符结束；读取方缓冲尚未写完的行。`config.json` 与 `result.json` 是对象，后者原子替换。
- 时间点：`unix_s` 是 `time.time()` 的 Unix 秒；`monotonic_s` 是 `time.perf_counter()` 的秒数。`start_s`、`end_s` 也是 perf_counter 时间点。单调时钟只比较同一主机、同一次开机的进程，不与 Unix 秒直接相减。
- 对齐资源与 step 时优先用同机 `monotonic_s`。跨工具时间轴通过同一记录中的 `(monotonic_s, unix_s)` 锚点对齐，Nsight 内部则使用其纳秒时钟与 NVTX。系统校时可能使 Unix 时钟跳变；耗时均不依赖 Unix 差值。
- `*_ms` 为毫秒、`*_s` 为秒、`*_bytes` 为字节、`*_mib` 为 MiB（2²⁰ 字节）、`*_pct` 为百分比（0–100，进程多核 CPU 若另行提供可超过 100）。旧 `*_mb` 显存字段实际也是 MiB，不改变其口径。
- 数值必须有限。硬件不支持或无法读取时写 `null`，并在该层的 `missing` 字典写入 `字段名: 原因`，不能以 0 替代。未安排某次阶段测量时允许省略阶段字段，同时写 `phase_sampled: false` 和缺测原因。
- JSONL 记录的 `schema_version`、计时与资源字段可被旧读取方忽略；v0 的关键字段维持原名。控制台、AutoPilot evidence/verifier/reporter 均按字段读取，不要求新增字段存在。

## config.json

| 字段 | 类型 / 可空 | 含义 |
| --- | --- | --- |
| schema_version | int | 1 |
| kind, task, run_id | string | 运行类型、任务、运行标识 |
| args | object | 完整 CLI 有效参数；字段名为 `apps.train --help` 对应的下划线名称，类型由 parser 定义。路径只用于本地运行，报告中改写成相对路径 |
| device | string | 请求的 cpu/cuda/auto，实际设备见 result.backend |
| samples, val_samples | int | 训练/验证样本数 |
| input_h, input_w, steps_per_epoch | int | 输入尺寸、每轮步数 |
| log_file, tensorboard_dir, checkpoint, best, stop_file | string | 本地输出位置 |
| started_unix_s, unix_s, monotonic_s | number | 准备结束时的墙钟与时钟锚点 |
| pid | int | 训练入口进程 PID |
| parallel | object | mode（none/ps/ring）、transport（string/null）、workers（int）、shard_sizes（int[]） |
| seeds | object | init、shuffle、augment、mix、dropout 等实际种子；标量或每 Worker 的数组，未指定时可为 null |
| data_hash, val_hash | string/null | 训练/验证样本清单指纹 |
| split_file | string/null | 固定划分来源；无划分文件时为 null |
| monitoring | object | requested（auto/on/off）、enabled（bool）、external（bool）、interval_s（number）。external 表示控制台已有采样器，避免重复写入 |
| phase_timing_every | int | 每多少步采样三段设备耗时，0 关闭；在 args 中亦保存 |

## metrics.jsonl

每个完成的训练 step 一行 `kind: "step"`，每个完成的 epoch 一行 `kind: "epoch"`。只有 lead Worker 写聚合 metrics。TensorBoard 标量默认每 10 步写入（`--tb-log-interval`）；JSONL 不因此降采样。

| 字段 | 类型 / 可空 | 含义 |
| --- | --- | --- |
| schema_version, unix_s, monotonic_s | int / number | 公共版本与写出时间 |
| kind | string | step 或 epoch |
| epoch | int/null | 从 1 开始；通用 dashboard 未提供 epoch 时 step 行可为 null |
| step | int | 从 1 开始的全局 step，仅 step 行 |
| loss, running_loss | number / number|null | 当前 batch loss、当前 epoch 运行均值；epoch 的 loss 为整轮均值 |
| accuracy | number/null | 分类准确率 [0,1]，回归任务为 null |
| data_load_ms | number | loader 取出 batch 的主机墙钟时间 |
| train_step_ms | number | 前向+反向设备 span；CUDA Event，CPU 用 perf_counter，不含优化器 |
| step_time_ms | number | **旧口径**：batch 取出后到该步计算/同步完成的墙钟时间，不含 data_load_ms；含数据转换/增强、H2D、观察器与 Python 开销 |
| step_total_ms | number | data_load_ms + step_time_ms；不含本步末尾日志与 checkpoint，不能直接当完整 epoch 墙钟 |
| batch_size | int | 全局 batch 样本数；Worker 独立样本数在 rank steps 中 |
| samples_per_sec | number | **旧口径**：batch_size / (step_time_ms / 1000) |
| samples_per_sec_total | number | batch_size / (step_total_ms / 1000) |
| forward_ms, backward_ms, optimizer_ms | number，可省略 | 每 `--phase-timing-every` 步测得，默认 20；CUDA Event（同 stream、步末同步后读取），CPU 用 perf_counter |
| phase_sampled | bool | 本步是否采集上述三段 |
| missing | object，可省略 | 未采阶段或无 CUDA 显存池的原因 |
| gpu_mem_used_mb | number，可省略 | 兼容字段：上次写出以来在阶段边界采样的 CuPy used 峰值，单位 MiB |
| gpu_mem_pool_mb | number，可省略 | 兼容字段：CuPy pool.total_bytes 的当前值，单位 MiB |
| cupy_pool_used_mib, cupy_pool_total_mib | number/null | 同上述两个兼容字段；used 是区间采样峰值，不是 allocator 每次分配的精确高水位 |
| learning_rate, epoch_time_s | number | epoch 行的学习率和轮次墙钟耗时 |
| val_loss, val_accuracy | number/null | epoch 行验证结果；未验证或非分类时为 null |
| val_angle_mae, val_throttle_mae, best_loss | number/null，可省略 | DonkeyCar 验证与最佳分数；其它任务的 extra 指标只增加字段 |

CUDA 各阶段 event、主机 wall span 与异步 H2D 并非同一种测量，不能强行把三段设备时间加数据读取凑成完整 step。占比图需标明未覆盖的准备/同步/主机开销。NVTX 时间线上的阶段分区另以同一 step 的 CPU 时间为分母。

## resources.jsonl

默认每 0.5 s 采样，`--monitor-interval-s` 可调整；真实间隔以相邻 `monotonic_s` 为准。耗时超过间隔时不补采虚假数据。`auto` 在有 run-dir 且无外部采样器时开启；`off` 禁用训练自身采样。控制台提供采样时设置 `MYFLOWS_EXTERNAL_MONITOR=1`，即使训练请求 on 也只保留外部写入方。

独立命令：`python -m MyFlows.monitoring.record --pid <PID> --out <run-dir>/resources.jsonl --interval-s 0.5`。训练自动启动时还传递进程创建时间，防 PID 重用；正常退出发送停止文件并回收子进程，异常被杀时采样器检测父 PID 消失后退出。

| 字段 | 类型 / 可空 | 含义 |
| --- | --- | --- |
| schema_version, unix_s, monotonic_s | int / number | 版本、采样时间锚点 |
| interval_s, root_pid | number / int|null | 请求间隔、训练入口 PID（控制台全机视图可无 PID） |
| sampler_backend | string/null | nvml、nvidia-smi；均不可用为 null |
| sampling_status | string | ok/partial/unavailable/error |
| sampling_error | string/null，可省略 | 采样异常或 GPU 不存在的说明 |
| missing, gpu_missing | object | missing 为扁平字段原因；gpu_missing 保留旧的设备级原因字典 |
| cpu_utilization_pct | number/null | 整机 CPU 利用率，不是仅训练进程的 CPU 使用率 |
| ram_used_bytes, ram_total_bytes | number/null | 整机主机内存已用/总量 |
| process_rss_bytes | object/null | PID 字符串 → 当前 RSS 字节数，根进程与后代。进程退出或拒绝访问不伪造数据 |
| process_rss_total_bytes | number/null | 此次可读进程 RSS 之和；共享页面可重复计数，不是私有工作集 |
| gpu_utilization_pct | number/null | 选中设备的 GPU 利用率，含桌面与其它进程 |
| gpu_memory_mib, gpu_device_memory_mib | number/null | 两字段值一致：NVML / nvidia-smi 的**整卡**已用显存 |
| gpu_device_memory_peak_mib | number/null | 本记录器开始以来的整卡显存采样峰值，无法代表精确分配峰值 |
| gpu_temperature_c, gpu_power_w, gpu_power_limit_w | number/null | 摄氏度、当前功率 W、当时功耗上限 W |
| gpus | object[] | 每个可见设备的完整样本，采集不可用时空数组 |

`gpus[]` 包含 `index`（int）、`name`/`uuid`（string/null），以及 `utilization_pct`、`memory_used_mib`、`memory_total_mib`、`temperature_c`、`power_w`、`power_limit_w`（number/null）；`missing`（object）描述硬件缺测。`processes` 为 null 或 `{pid: int, used_memory_mib: number|null}[]`；WDDM 常不提供每进程显存，原因记录在 `missing.process_memory`，不以整卡显存冒充。

TensorBoard 的 `resources/*` 曲线增量读取这个文件，保持原始采样墙钟与采样序号，不再次请求 NVML。缺测标量跳过；采样文件保留原因。GPU 显存、CuPy 池与 RSS 使用独立 tag。

## rank-*/steps.jsonl

每个 Worker 完成一步写一行。`rank`、`epoch`、`step` 为从 0 开始的整数（历史约定）；`start_s`/`end_s` 为 perf_counter 时间点；`step_wall_s`、`data_s`、`gradient_sync_s`、`update_confirm_s` 为非负秒；`n_samples` 为本 Worker 样本数；`loss` 为聚合 loss。阶段 `*_ms`、显存字段、schema 与时钟同 metrics。

独立分布式 benchmark 还可输出：`sync_update_s`（反向结束至该步结束）、`digest_s`（状态指纹）、`input_h2d_wall_s`、`forward_wall_s`/`forward_gpu_s`、`backward_wall_s`/`backward_gpu_s`、`optimizer_wall_s`/`optimizer_gpu_s`、`gradient_d2h_s`、`gradient_to_device_wall_s`、`metrics_wall_s`、`rpc_observed_s`、`encode_s`、`decode_s`（均为秒）；`request_bytes`、`response_bytes`、`outgoing_gradient_bytes`、`gradient_payload_bytes` 为字节计数；`backends` 为实际后端映射。不同 span 可以相互重叠，不能将所有 `*_s` 直接求和。未启用相应测量时可省略字段。

## result.json

| 字段 | 类型 / 可空 | 含义 |
| --- | --- | --- |
| schema_version, written_unix_s | int / number | 版本、最终写入时间 |
| status, exit_code | string / int | succeeded/stopped/failed；退出码 0/1/3/4/5 分别成功/一般错误/非有限 loss/OOM/等 GPU 锁超时 |
| failure | object/null | kind、message；可附 step、epoch、loss（非有限 loss 用字符串） |
| task, config | string / object | 任务及影响训练的有效参数摘要，完整配置在 config.json |
| metrics | object | val_angle_mae（最佳 checkpoint 对应值）、val_angle_mae_final、best_val_angle_mae、val_loss、best_loss、last_train_loss、best_epoch；未产生时 null |
| epochs | object[] | 逐轮验证记录（epoch、val_loss、val_angle_mae、val_throttle_mae 等） |
| throughput | object | samples_per_s、epoch_train_wall_s、train_wall_s；秒和样本/秒，包含 epoch 内日志/保存，排除 epoch 结束后的验证/保存 |
| memory | object | source= cupy_memory_pool 或 null；peak_used_mb、peak_pool_mb、device_total_mb（MiB），另有明确命名的 cupy_pool_peak_used_mib、cupy_pool_peak_total_mib；missing 说明 CPU/失败时缺测 |
| steps, global_step, epochs_completed | int/null | 此次完成步数、含恢复前历史的总步数、完成轮数 |
| stopped_early, stopped_by_file | bool | 早停或用户停止文件 |
| backend | object | requested、actual（string/null）、device（string/null）、dtype（string） |
| timing | object | started_unix_s、finished_unix_s；lock_acquired_unix_s（null 表示未取得）；lock_wait_s、wall_s、run_s（秒/null，run_s 从取得锁起算） |
| samples | object，可省略 | train、val、trained、expected_trained（int/null）；失败在数据准备前可无此块 |
| data, seeds, parallel | object，可省略 | data_hash、val_hash、split_file；实际种子、并行配置同 config |
| paths | object，可省略 | checkpoint、best、log、states（路径/null），本地使用 |
| resume | object，可省略 | state、from_global_step、lr_before、lr，仅精确续训出现 |
| transfer | object/null | arch、freeze_stages、input_norm、load_report、freeze、frozen_before/after、frozen_unchanged；权重来源哈希、载入映射与冻结指纹由 pretrained 模块报告 |
| environment | object | 下面的环境块 |

environment 包含 `python`、`platform`（string），`packages`（包名 → 版本 string/null，查询 metadata，不导入 torch），`gpu`（nvidia-smi 的设备型号/驱动/总显存文本或 null），`cuda`（runtime_version、driver_version 的 CUDA 整数编码及 nvrtc_version 的 [major, minor]，未探测为 null），`native_source_sha256`（原生 C++ 与 CMake 源码摘要），`missing`（缺测路径 → 原因）。GPU/环境查询在训练计时之外，CPU 运行不初始化 CUDA。源码哈希只标识原生源码，完整运行还需保存两个仓库的 code_state。

## 12 项课程指标对应

| 指标 | 主字段 | 口径 |
| --- | --- | --- |
| GPU 利用率 | resources.gpu_utilization_pct | 整卡采样 |
| 当前显存 | resources.gpu_device_memory_mib / metrics.cupy_pool_total_mib | 硬件已用与进程池申请量分列 |
| 显存峰值 | resources.gpu_device_memory_peak_mib / result.memory.cupy_pool_peak_used_mib | 硬件采样峰值与阶段边界池采样峰值分列 |
| GPU 温度 | resources.gpu_temperature_c | 摄氏度 |
| GPU 功耗 | resources.gpu_power_w，gpu_power_limit_w | W，动态上限 |
| CPU 利用率 | resources.cpu_utilization_pct | 整机 |
| 单步时间 | metrics.step_total_ms | 含读取，不含步末日志/保存 |
| 前向 | metrics.forward_ms | CUDA Event / CPU perf_counter |
| 反向 | metrics.backward_ms | 同上 |
| 优化器 | metrics.optimizer_ms | 同上 |
| 数据读取 | metrics.data_load_ms | loader 等待 wall time |
| samples/sec | metrics.samples_per_sec_total | 与 step_total_ms 同分母；完整训练另看 result.throughput |

监控开销和频率是否可用于正式计时，需以 `monitoring/overhead-001/report.md` 的交替测量为准。格式本身不声称默认采样开销已达标。
