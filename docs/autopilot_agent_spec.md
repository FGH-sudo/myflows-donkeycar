# MyFlows AutoPilot Agent Spec

- 状态：v0.3 开发 Spec，未实现。第 16 节列出的待定项已于 2026-10-23 逐项确认。v0.1（2026-10-22）为初版设计稿，v0.2 加入了 dsh 最小验证的结论
- 日期：2026-10-23
- 依据：[深度学习框架-16 综合项目III.pdf](<深度学习框架-16 综合项目III.pdf>) 第 2、4、5、8、9、10、11 页；[semester_spec.md](semester_spec.md) 第 4.0、4.7、4.10、9 节（W7、E3）
- 代码基线：根仓库 80d02af，MyFlows ffb3378；DeepSeek Harness（下称 dsh）0.2.1-rc.2，上游提交 687f282e86
- 范围：细化 semester_spec 第 4.7 节；B1/B5 的验收口径不变

## 1. 定位与已确定的决策

AutoPilot Agent 是训练调优程序，不是 DonkeyCar 驾驶控制器（后者是 `mycar/myflows_pilot.py`）。按课件第 11 页，它围绕训练过程完成四件事：自动实验调度、性能诊断、配置搜索、生成优化建议。它必须形成一个可检验的闭环：

> 读取指标 → 判断瓶颈与配置有效性 → 选择下一组候选 → 运行 → 校验 → 记录 → 报告

各项决定见第 16 节，要点如下：

- **范围与进度**：
  - 不等 W1/W5，直接开工；
  - 开发目标全部完成，不按汇报节点裁剪；
  - 只在网页控制台里操作；
  - 终端里手动启动的训练也进同一个 GPU 队列。
- **实验口径**：
  - 人工基线为 lr 1e-3、batch 2、num_workers 0；
  - 开发和验证一律用小规模数据，跑全量之前先通知用户，等用户把机器空出来；
  - seed 联调用 2 个，正式用 3 个；
  - 正式 campaign 之前，补上按时间段划分和独立测试集。
- **LLM 层**：
  - 使用固定提交的 dsh 源码克隆，并保持上游原样；
  - 对话、权限模式、审批和 API Key 设置都用 AutoPilot 专用 dsh 自带的网页，嵌入控制台；
  - dsh 可以创建、修改、启动、暂停 campaign，可以提交和停止 trial，这些操作都受 dsh 权限模式控制，其中启动训练类的操作始终要人工确认；
  - 每个 campaign 最多调用 30 次。
- **不改动本机自用的 dsh**（`~/.dsh`、desktop profile），不做 `npm link`，也不做全局安装。

设计原则：

1. **证据驱动**。每条建议都要引用产生它的指标文件和具体数值（Finding）；每个结论都能回溯到某个 trial 和原始 JSONL。
2. **复用而不重写**。训练、分布式、任务队列、资源采样、参数校验、源码快照都已存在（第 4 节），AutoPilot 只负责编排和判断。
3. **规则内核，模型可插拔**。六组件和诊断都是确定性规则。TPE 和 LLM 只是 Planner 的策略或操作者，它们发起的写操作和规则建议一样要过 Verifier 和预算。
4. **结论诚实**。“人工基线未被超越”“分布式在单卡上不如单进程”都是合法结论；最佳配置必须独立重跑确认，小于噪声的提升不宣称。
5. **单 GPU 串行**。任何时刻最多一个训练占用 GPU，包括在终端里手动启动的训练。
6. **LLM 不在关键路径上**。LLM 不可用、超时或超出预算时，campaign 按规则策略继续。凡是由 LLM 发起、会启动训练的操作，都必须由人确认。

不做：自动修改 CUDA/算子源码、MCTS、多 GPU 并发 trial、让 LLM 获得 shell 或文件读写权限。

## 2. 课件要求与本设计的对应

| 课件 | 要求 | 本设计 | 层级 |
| --- | --- | --- | --- |
| 第 11 页 | Planner / Scheduler / Executor / Memory / Critic·Verifier / Reporter | 第 6 节，规则内核 | 基础 B1 |
| 第 11 页 | 专业子 Agent：GPU Monitor、Operator Optimization、PS Training、Transfer Learning、Benchmark | 第 7 节；Transfer Learning 在 E2 交付前登记为 unavailable | 基础 B1 |
| 第 11 页 | 能生成优化建议的智能系统 | 第 8 节：dsh 作为顾问和操作者，读取证据、提出建议、配置 campaign、解释结论 | B1 的增强，可关闭 |
| 第 10 页 | 监控数据作为 AutoPilot 的分析输入 | GPU Monitor 读取 `resources.jsonl` 和 `metrics.jsonl`；第 11 节补齐缺失字段 | B7 → B1 |
| 第 9 页 | NumPy / CuPy / CUDA C 的时间、加速比、误差、规模变化 | Operator Optimization 与 Benchmark 读取已有 benchmark 结果并生成对比表 | 基础 B2 |
| 第 8 页 | PS / Worker / Launcher / Monitor | PS Training Agent 比较 `apps.train --distributed` 各配置的吞吐与同步开销 | 基础 B3 |
| 第 4 页 | 系统测试的核心是 Agent 自动调优的测试 | 第 13 节 | 基础 B5 |
| 第 5 页 | TPE 多超参搜索 | `TPEStrategy`（Optuna，条件搜索空间） | 拓展 E3 |
| 第 5 页 | 实时异常检测与自动回滚 | Verifier 在线规则 + 基于完整 TrainingState 的精确回滚 | 基础 / E3 |
| 第 5 页 | MCTS | 不做 | — |

## 3. dsh 接入的最小验证（2026-10-23）

**目的**：在不影响本机自用 dsh 的前提下，确认 dsh 能作为 AutoPilot 的 LLM 层：模型只能用 MyFlows 开放的工具，工具能调到项目代码，结果能回到模型。

**做法**：把 dsh 上游（687f282e86）克隆到仓库根目录的同级目录 `../dsh-myflows`，用独立的 `DSH_HOME` 和仓库自带的 pnpm 构建，不做任何全局注册。在此基础上做了四项改动：

- **新 bundle `myflows-autopilot`**：以 `sdk-minimal` 为底的独立插件树，共 29 行，入口是一次性运行器（headless）。
  - 不挂 shell、终端、文件读写、工作目录工具，也不挂 `mcp-resources`。
  - MCP 客户端以 stdio 方式启动 Python 工具服务。
  - 系统提示词为中文顾问人设；sandbox 策略为 `read-only`。
- **白名单插件**：在 `agent/created` 时用 `tools.restrict()` 屏蔽名字不以 `mcp__autopilot__` 开头的全局工具，并注册全局 `tools.guard()`，拒绝对这些工具的调用。
- **Python MCP 工具服务**：只用标准库，提供 `list_runs`、`get_run_summary`、`get_search_space`、`propose_trial`。其中 `propose_trial` 先查搜索空间，再调用 `apps.console.presets.validate_single`；它只记录，不启动训练。
- **无 Key 验证**：用 dsh 仓库自带的 Messages 兼容 mock 服务代替 DeepSeek。模型适配器、agent 循环、MCP 客户端和工具服务都是真实的。

**结果**：

| 场景 | 结果 |
| --- | --- |
| 读取运行记录 | 通过：返回 `runs/console` 中 7 条运行的状态、参数和指标摘要 |
| 合法候选被接受 | 通过：返回对应的 apps.train 命令行参数 |
| 未知参数 `optimizer` | 通过：被 `validate_single` 拒绝 |
| `lr=0.01` | 通过：超出搜索空间，被拒绝 |
| 调用白名单外的工具 | 通过：被白名单插件拒绝 |

每个场景都检查以下各项：
- 退出码为 0，恰好发出两次模型请求；
- 模型看到的工具恰好是 4 个 `mcp__autopilot__*`；
- 工具结果回到了模型，最终回答已输出；
- 运行前后 `~/.dsh` 的文件数和最新修改时间不变。

一次完整运行约 5 秒，包括启动 dsh、拉起工具服务和两次模型请求；其中 Python 导入 MyFlows 校验代码约占 2.3 秒。

**实现要点**：

- **YAML 写法**：配置里含 `: ` 的 `!!js` 表达式必须写成块标量，否则 YAML 解析失败。
- **组合检查**：bundle 解析失败时只打印 “skipping profile bundle” 警告，启动不会失败，所以必须用组合检查兜底（`--dump-config` 或测试）。
- **资源工具**：只要配置了 MCP 服务器，`mcp-resources` 就会加入 3 个资源工具和一段系统提示词。MCP 客户端只在这个服务存在时才注册资源。
- **屏蔽列表的时机**：`tools.restrict()` 只能在 agent 作用域里调用，并且只能点名已注册的全局工具，所以屏蔽列表要在 `agent/created` 时计算。
- **stdout 与编码**：stdio 工具服务必须独占 stdout，MyFlows 导入时的打印要重定向到 stderr；在 Windows 上，JSON 要用 UTF-8 并做 ASCII 转义后再写出。

**代价**：从源码构建的克隆约 1.6 万个文件；安装依赖 1 分 40 秒，完整构建 137 秒；需要 Node.js ≥ 22.19 和 pnpm 11。

验证用的改动已经撤销，克隆恢复为上游原样并保持构建好的状态，作为第 8 节方案 B 的运行时继续使用。附录 A 保留了实现要点。

## 4. 可直接复用的现有能力

| 能力 | 现有入口 | AutoPilot 能拿到的东西 |
| --- | --- | --- |
| 正式训练（单进程与数据并行） | `apps/train/train_myflows_donkey.py` | CLI 参数，包括 `--seed`（权重初始化）、`--distributed ps\|ring`、`--workers`、`--transport`、`--worker-memory-mb`。<br>`--run-dir` 写出 `config.json` 和 `metrics.jsonl`：step 行有 loss、data_load_ms、train_step_ms、step_time_ms、samples_per_sec；epoch 行有 loss、val_loss、learning_rate、epoch_time_s。<br>`--stop-file` 可以优雅停止。断点原子写入并带训练元数据，可以从记录的 epoch/step 续训。 |
| 分布式实验框架 | `benchmark/distributed_experiment.py` | 冻结配置下的计时与通信量对比（`results.json`、`rank-*/steps.jsonl`），留给 PS Training Agent 做对照 |
| 任务队列 | `apps/console/jobs.py` 的 `JobManager` | 任务状态机；串行执行；`submit(..., run_dir=...)` 可以指定运行目录；按 PID 和创建时间重启后接管；`kill_tree` 清理进程树 |
| 参数校验 | `apps/console/presets.py` | `validate_single`（`SINGLE_ARGS` 已包含 seed、distributed、workers、transport、sync_timeout、base_width）、`validate_distributed`、`CONV_BACKENDS` |
| 资源采样 | `MyFlows/monitoring/gpu.py` | `GpuSampler`、`ResourceRecorder`；控制台按任务每 0.5 秒采样一次，写入 `resources.jsonl` |
| 运行发现与展示 | `apps/console/registry.py`、`readers.py`、运行详情页 | 增量读取 JSONL；曲线、耗时、资源和日志页面 |
| 原生库 | `MyFlows/ops/cuda_native/build.py` | 源码有变化时自动重新构建，不回退到 CuPy |
| 算子对比与 Nsight | `benchmark/cuda_ops.py`、`gemm_engines.py`、`matmul_compare.py`、`profile_cuda.py`；归档中的 `capture.sqlite` | 各后端耗时与误差，kernel 级耗时排行 |
| 溯源 | `benchmark/stage1_common.py` | 环境信息，两个仓库的 HEAD / diff 快照和哈希 |

## 5. 总体架构

~~~mermaid
flowchart TB
    subgraph browser[浏览器 · 控制台 AutoPilot 页面]
        pages[campaign 列表 / 新建 / 详情 / 待确认]
        embed[内嵌：AutoPilot 专用 dsh 网页<br/>对话 · 权限模式 · 审批 · Key 设置]
    end
    subgraph console[训练控制台进程 apps/console]
        api[/api/autopilot/*/]
        service[AutoPilotService<br/>六组件 + 子 Agent]
        jobs[JobManager<br/>手动任务与 trial 同一队列]
        host[DshHost<br/>管理 dsh 网页进程与后台运行]
        api --> service
        service --> jobs
        service --> host
    end
    pages -- REST / SSE --> api
    embed -- HTTP --> web
    subgraph dsh[dsh 源码克隆 · 固定提交 · DSH_HOME=runs/autopilot/dsh-home]
        web[网页 profile<br/>myflows-autopilot-web]
        bg[一次性 profile<br/>myflows-autopilot-headless]
    end
    host -- 启动 / 停止 --> web
    host -- 决策点调用 --> bg
    web -- HTTPS --> deepseek[DeepSeek API]
    bg -- HTTPS --> deepseek
    web -- MCP stdio --> tools[AutoPilot 工具服务]
    bg -- MCP stdio --> tools
    tools -- HTTP + 令牌 --> api
    jobs -- 子进程 --> train[apps.train]
~~~

主循环由 `AutoPilotService` 的后台线程驱动，LLM 只是参与者之一：

~~~python
memory.recover()                                  # 核验遗留任务与结果，只续排未完成的 trial
while not campaign.budget.exhausted(memory):
    evidence = memory.evidence()
    findings = [f for a in specialists if a.available() for f in a.diagnose(evidence)]
    proposals = planner.next(findings, memory, campaign)   # 规则 / LLM 顾问 / TPE
    for p in verifier.precheck(proposals, memory, campaign):
        scheduler.enqueue(p)                       # LLM 发起的启动训练操作先等待人工确认
    trial = scheduler.dispatch()                   # 交给 JobManager；没有候选就退出
    if trial is None:
        break
    outcome = executor.run(trial, monitor=verifier.online)
    memory.record(trial, outcome, verifier.postcheck(trial, outcome, memory))
planner.confirm(memory)                            # 最佳配置换 seed 独立重跑
reporter.write(memory, campaign)
~~~

| 阶段 | 内容 | 退出条件 |
| --- | --- | --- |
| baseline | 人工基线 × N 个 seed（联调 2 个，正式 3 个），得到基线与噪声水平 | 基线有效 |
| explore | 在 lr × batch 中抽样（≤ 6 组） | 候选用完，或预算用到 70% |
| exploit | 子 Agent 或 LLM 根据诊断提出的定向修改 | 没有新的、有证据支撑的建议 |
| confirm | 前 1～2 名换新 seed 独立重跑 | 完成，或预算耗尽（报告中标注“未确认”） |

## 6. 核心组件（规则内核）

### 6.1 Planner

负责阶段管理，并合并、去重各个来源的建议。具体的“下一组参数”由可替换的策略给出：

| 策略 | 用途 | 层级 |
| --- | --- | --- |
| `CandidateList` | 从离散空间按 seed 分层抽样，默认最多 6 个，不跑完整的笛卡尔积 | 基础 |
| `RuleRefine` | 把子 Agent 的 Proposal 变成配置差分 | 基础 |
| `LLMAdvisor` | 在决策点调用 dsh 一次性运行，模型通过 `request_trials` 提出带理由的建议（第 8 节） | 基础增强，可关闭 |
| `TPEStrategy` | Optuna 的 ask/tell，用历史有效 trial 预热，支持条件空间；正则项在这一阶段加入搜索 | E3 |

搜索空间：

| 参数 | 处理方式 |
| --- | --- |
| `lr` {1e-4, 3e-4, 1e-3} × `batch` {1, 2, 4} | 参与精度搜索，从 9 组中抽取 ≤ 6 组；batch 上限由经验库中的 OOM 记录收紧 |
| `num_workers`（数据读取进程数） | 不参与精度搜索，由 GPU Monitor 根据数据读取占比和吞吐量决定 |
| 卷积后端 | 固定为 `cuda_native_register`，由 Operator Optimization 按吞吐量复核 |
| 训练 worker 数（`--distributed` / `--workers`） | 不参与精度搜索，由 PS Training Agent 单独比较吞吐与同步开销 |
| weight_decay、数据增强、dropout | 在 TPE 阶段（E3）加入 |
| `freeze_stages` | E2 交付后开放 |

主指标：DonkeyCar 用固定验证集上的 `val_angle_mae`，越低越好；图像分类用 `val_accuracy`。次指标为 samples/sec、每 epoch 耗时和显存峰值。资源超出预算或数值无效的 trial 不参与排名。

### 6.2 Scheduler

- **排队**：所有 trial 都交给控制台的 `JobManager`，与手动提交的任务共用一个串行队列。训练入口启动前要先获取 GPU 锁（第 11 节 P8），所以 trial 和在终端里手动启动的训练会互相等待，不会同时占用 GPU。等锁的时间不计入 trial 的超时、卡死判断和总时长预算。
- **去重指纹**：`sha256(resolved_config, data_manifest_hash, split_hash, code_hash)`。控制台重启后，已成功的指纹不再运行。
- **预算**：`max_trials`、单 trial 超时、总时长（只计 trial 的实际运行时间，不计排队等待）、重试次数。调度前用历史 trial 的每步耗时估计本次时长，超出剩余预算就跳过，并记录原因。
- **重试**：只对基础设施失败重试 1 次。NaN、发散、OOM 是配置本身的问题，不重试同一配置。
- **优先级**：confirm > exploit > explore。LLM 发起、会启动训练的操作在人工确认之前处于 `pending_confirmation` 状态。

### 6.3 Executor

- **命令拼装**：单进程和数据并行训练都通过 `apps.train` 运行，命令由共用的拼装函数生成（第 11 节 P7），控制台手动提交和 AutoPilot 共用同一份参数白名单。`distributed_experiment` 只留给 PS Training Agent 做冻结配置下的对照实验。
- **提交**：调用 `JobManager.submit(..., run_dir=<campaign>/trials/<trial_id>/attempt-<n>)`。任务记录带 `campaign_id` 和 `trial_id`，现有的运行记录和运行详情页不用改就能显示。资源采样沿用控制台对每个任务的记录。
- **停止**：沿用控制台的做法——写停止文件，超时后结束整个进程树。
- **断点保留**：campaign 结束时，只保留基线和前 2 名的最佳模型，其余 trial 只保留日志和指标。

目录布局：

~~~text
runs/autopilot/
  dsh-home/                      # AutoPilot 专用 DSH_HOME：profile、会话日志、dsh 凭据
  knowledge.jsonl                # 经验库（不进 git）
  <campaign_id>/
    campaign.json  events.jsonl
    trials/<trial_id>/attempt-<n>/
      config.json  command.json  train.log  metrics.jsonl  resources.jsonl  result.json  verdict.json
      checkpoints/               # 只有基线和前 2 名会保留
    report/  report.md  leaderboard.csv  figures/*.png
~~~

正式 campaign 的 `report.md` 会复制到 `docs/experiments/semester_2026_fall/autopilot/<campaign_id>/`。

### 6.4 Memory

| 层 | 内容 | 存储 |
| --- | --- | --- |
| 工作记忆 | 当前阶段、预算用量、当前最佳、待确认列表、后台 dsh 运行的会话 id、LLM 调用次数 | `campaign.json` |
| trial 注册表 | 每次尝试的 `TrialResult`、Verdict、事件流（含 LLM 发起的操作） | `trials/*/attempt-*/result.json`、`events.jsonl` |
| 经验库 | 可复用的事实，带 task、model、关键配置、code_hash、`semantics_version`、`kind`（resource / convergence） | `runs/autopilot/knowledge.jsonl` |

所有写入都先写临时文件，再用 `os.replace` 替换。

经验库首次初始化时，从已有报告中导入资源类事实，例如：
- cifar stem、w64、batch 16 时显存吃紧，layer1 每个卷积的 im2col 缓冲约 708 MB；
- 两个 w64 worker 各用 batch 4 时超过 8 GB 显存，WDDM 会悄悄换页，看起来像卡死，需要 `--worker-memory-mb`；
- 模型宽度大于 32 时，Socket JSON 传输会被拒绝。

收敛类经验从空开始，因为此前的结果用的是旧版 Adam。经验库只用于预检和 TPE 预热，不参与排名。

### 6.5 Critic / Verifier

- **预检**：
  - 参数合法性（`validate_single`）；
  - 所需能力是否可用；
  - 经验库中已知必然 OOM 的配置；
  - 重复的指纹；
  - 剩余预算。

  LLM 发起的操作和规则建议走同一套预检。
- **在线检查**：每秒增量读取 `metrics.jsonl` 和 `resources.jsonl`。

| 异常 | 规则（阈值写在 CampaignSpec 里） | 基础版动作 | E3 动作 |
| --- | --- | --- | --- |
| 非有限 loss | loss 为 NaN 或 Inf | 立即停止，记为 `failed:nan` | 回滚到最近的健康断点，lr × 0.5 |
| 发散 | loss 的 EMA 连续 K 个窗口上升，且超过初始值的 3 倍 | 停止，记为 `failed:diverged` | 同上 |
| OOM | 日志匹配 OOM 错误，或 `result.json` 中有标记 | 停止，写入经验库 | 以 batch 减半派生新 trial |
| 卡死 | 获得 GPU 锁后，T 秒内没有新的 step 行 | 停止，记为 `failed:hang` | 同左 |
| GPU 空闲 | GPU 利用率低于阈值，但 step 仍在推进 | 不停止，记一条 Finding | 同左 |
| 超时 | 获得 GPU 锁后，超过单 trial 时长上限 | 停止，记为 `failed:timeout` | 同左 |

  判断卡死只看 step 是否推进。本机有后台进程，训练空闲时也占约 25% 的 GPU 利用率。
- **后检**：只有 Verdict 为 `valid` 的 trial 参与排名。后检要求：
  - 指标都是有限值；
  - 验证集指纹与本 campaign 一致；
  - 实际样本数与配置一致；
  - 实际后端与请求一致；
  - 资源没有超出预算。
- **确认**：最佳配置换新 seed 独立重跑，联调用 2 个 seed，正式用 3 个。提升小于基线各 seed 之间的标准差时，报告写“与基线无显著差别”。

### 6.6 Reporter

`report.md` 包含：
- 目标、搜索空间、预算与实际用量、代码版本与环境；
- 排行榜，失败和无效的 trial 也列出原因；
- 诊断链，其中至少一条由监控数据触发，并有前后对照；
- 最佳配置及选择理由、确认运行的 mean ± std、与人工基线的配对差值；
- 图表；
- LLM 部分：每个 LLM 发起的操作及其结果，模型写的解释（标注“LLM 生成”），调用次数和 token 用量。

## 7. 专业子 Agent

子 Agent 是领域规则的封装，不是独立的对话进程。统一接口为 `available()`、`diagnose(evidence) -> list[Finding]`、`propose(findings, space) -> list[Proposal]`。

| 子 Agent | 输入 | 主要规则与输出 | 首版 |
| --- | --- | --- | --- |
| GPU Monitor | `resources.jsonl`；`metrics.jsonl` 中的阶段耗时 | 数据读取占比 > 0.3 且 GPU 利用率中位数 < 50%：num_workers 0→2。<br>利用率 < 40% 且显存峰值 < 50%：batch 加倍，lr 按线性缩放成对提出。<br>显存峰值 > 90%：batch 减半。<br>主机同步占比 > 20%：标记。<br>温度或功耗受限：该 trial 的计时不做横向比较。 | M2 |
| Benchmark | 基线多 seed 的结果 | 给出噪声水平；固定比较协议（同一划分、同一预热、CUDA event 计时） | M2 |
| Operator Optimization | benchmark 结果、Nsight `capture.sqlite` | 生成课件第 9 页的对比表和 kernel 热点排行。kernel 之间比较用 `sm__cycles_elapsed.max`；按吞吐复核卷积后端。只提建议，不改源码。 | M4 |
| PS Training | `apps.train --distributed` 的短跑结果；`distributed_experiment` 的对照 | 在显存上限内比较 workers {1, 2, 4} × PS/Ring × 传输方式的吞吐与同步占比。同步等待占步时 > 50%：减少 worker 或换方案。各 rank 步时的变异系数 > 0.2：标记掉队者。吞吐都低于单进程：结论为保留单进程。 | M4 |
| Transfer Learning | — | E2 交付前 `available()` 返回 unavailable，搜索空间中不出现 `freeze_stages` | E2 之后 |

## 8. LLM 层：dsh 接入

### 8.1 运行时（方案 B：固定提交的源码克隆）与隔离

- **克隆**：使用仓库根目录的同级目录 `../dsh-myflows`，固定在上游 687f282e86，保持上游原样，不往克隆里提交任何改动。路径可以用 `MYFLOWS_DSH_ROOT` 修改。
- **构建**：依赖用克隆自带的 pnpm 安装，`--store-dir` 指向克隆内的 store。升级 dsh 时整体切换到新的提交，重新构建，并重跑第 13 节的第一、二层测试。
- **我们的代码**：bundle 和插件放在本仓库的 `apps/autopilot/dsh/`，作为外部插件装进 AutoPilot 专用 profile。具体安装方式在 M3 的 spike 中确定：`dsh plugin --profile <p>` 会把 `add file:<路径>` 这类参数交给 pnpm，在 profile 目录中执行；也可以直接链接进 profile 的 `node_modules`。
- **隔离**：
  - `DSH_HOME` 固定为 `runs/autopilot/dsh-home`，且必须非空；不读写 `~/.dsh`。
  - 不做 `npm link`、`npm install -g`、`corepack enable`。
  - dsh 网页改用独立端口（默认 8792），避开 dsh 网页的默认端口 3080，以免和自用 dsh 冲突。
  - 控制台只结束自己启动的 dsh 进程树，不按进程名批量结束进程。
- **环境要求**：Node.js ≥ 22.19（本机为 24.16）。

### 8.2 两个 profile

两个 profile 都以 `dsh-base` 为底，上面叠加各自的模式 bundle，最上层是我们的补丁层 `@myflows/dsh-autopilot`。

| profile | 组成 | 用途 |
| --- | --- | --- |
| `myflows-autopilot-web` | dsh-base + dsh-web-app + 补丁层 | 常驻网页，嵌入控制台，承担对话、权限切换、审批弹窗和 Key 设置 |
| `myflows-autopilot-headless` | dsh-base + dsh-headless + 补丁层 | 后台的一次性运行，承担 Planner 顾问和报告解释 |

补丁层包含以下内容：

- **禁用工具行**：禁用 dsh-base 中的 shell、终端、文件读写与搜索、网页搜索/抓取、子 agent、技能、工作流、计划模式、PTC 等工具行，并且不挂 `mcp-resources`。
- **保留的服务**：凭据（Key 存放在 AutoPilot 的 `DSH_HOME` 里，通过网页设置）、审批服务和权限预设。
- **MCP 客户端**：以 stdio 方式启动 AutoPilot 工具服务。
- **中文顾问人设**。
- **守卫插件 `myflows-autopilot-guard`**：用纯 JavaScript（ESM）编写，本仓库不需要 TypeScript 工具链。它承担三项职责：
  1. **白名单**：每个 agent 只看得到 `mcp__autopilot__*` 工具，对其他工具的调用一律拒绝（PoC 已验证）。
  2. **分 profile 的工具子集**：一次性 profile 只开放只读工具、`request_trials` 和 `write_note`。
  3. **权限判定**：在 `tools/pre-execute` 中按工具类别和当前权限模式决定允许、询问或拒绝（第 8.4 节）。

组合检查要覆盖两个 profile，作为测试的一部分（第 13 节）：断言不包含被禁用的工具行，且补丁层已经生效。

### 8.3 工具面

工具服务位于 `apps/autopilot/dsh/tools_server.py`，沿用 PoC 的 stdio 实现。所有数据都通过控制台的 `/api/autopilot/agent/*` 读写，所以 LLM 看到的数据和页面一致，写操作也和页面走同一套校验与预算。

| 类别 | 工具 | 作用 |
| --- | --- | --- |
| 只读 | `list_campaigns`、`get_campaign`、`list_trials`、`get_trial_summary`、`get_findings`、`get_search_space`、`list_runs` | 读取目标、阶段、预算、排行榜、指标摘要、诊断、搜索空间和普通运行；只返回摘要，不返回原始 JSONL |
| 配置 | `create_campaign`、`update_campaign`、`pause_campaign`、`stop_trial` | 写入 campaign 配置（任务、搜索空间、预算、seed）、暂停 campaign、停止运行中的 trial |
| 直接启动训练 | `start_campaign`、`submit_trials` | 启动或恢复 campaign、把候选配置放进队列 |
| 请求启动训练 | `request_campaign_start`、`request_trials` | 在控制台生成待确认项，不直接启动训练 |
| 说明 | `write_note` | 给报告补充文字说明，标注“LLM 生成” |

不提供读写文件、执行命令、删除 campaign 或 trial，也不提供修改预算以外的系统设置。

### 8.4 权限模式与审批

工具是否可用，由守卫插件根据 dsh 当前会话的权限模式决定。dsh 的权限预设把 sandbox 和审批策略打包在一起：工作区写入对应 `ask`，完全访问对应 `never`，Auto（dsh 中标为实验性）是由模型代为审核的 `ask`。需要注意，`never` 不是“直接放行”，而是“需要审批的请求一律拒绝”。所以守卫插件在完全访问模式下必须直接返回“允许”，而不是“询问”。

| 工具类别 | 工作区写入（默认） | 完全访问 | Auto | 一次性 profile（策略为 `never`） |
| --- | --- | --- | --- | --- |
| 只读、说明、请求启动训练 | 允许 | 允许 | 允许 | 允许（只读、`request_trials`、`write_note`） |
| 配置 | 询问：dsh 弹出审批框，由你确认 | 允许 | 询问：由 dsh 自动审查代为审批 | 不提供 |
| 直接启动训练 | 询问：dsh 弹出审批框，由你确认 | 拒绝，提示改用 `request_*` | 拒绝，提示改用 `request_*` | 不提供 |

这样，凡是 LLM 发起的启动训练操作，都要由人确认：
- 工作区写入模式下，在 dsh 的审批框里确认；
- 其他模式下，在控制台的“待确认”列表里确认。

规则策略在已启动的 campaign 内自行产生的 trial 不需要确认，因为自动调优本来就是 AutoPilot 的职责。

无论哪种模式，所有写操作都还要经过 Verifier 预检和预算检查。每个 campaign 的后台 LLM 调用最多 30 次；token 上限等第一次真实验收之后再定。网页对话的用量由 dsh 自带的 token 计量显示。

### 8.5 会话、事件与展示

- **网页对话**：会话保存在 dsh 自己的会话列表里，可以在内嵌网页中回看。
- **后台运行**：Planner 顾问在 baseline 完成后、每批 explore 结束后、出现 OOM 或 NaN 失败后，各调用一次 `myflows-autopilot-headless`；campaign 结束时再调用一次，生成报告解释。每个 campaign 使用一个后台会话，后续运行用 `--session-id` 续接，会话 id 记录在 `campaign.json` 中。`--json` 事件流会转写为 `events.jsonl` 里 `source: "llm"` 的事件。
- **控制台侧记录**：不论操作来自网页对话还是后台运行，每次调用 `/api/autopilot/agent/*` 都会记成一条事件，并推送到详情页的“LLM 操作记录”，所以控制台始终知道 dsh 做了什么。

### 8.6 失败处理与降级

| 情况 | 处理 |
| --- | --- |
| 克隆不存在、构建产物缺失、版本不符，或组合检查失败 | 关闭 LLM，在页面上说明原因；规则策略照常运行 |
| dsh 网页进程退出 | 内嵌区域显示状态，并提供“重新启动”按钮；后台运行不受影响 |
| 没有配置 Key | 内嵌区域引导你到 dsh 设置页配置；后台运行跳过 |
| 模型超时或报错 | 跳过本次调用并记录，下一个决策点重试；连续 3 次失败后，本 campaign 关闭 LLM |
| 超出调用上限 | 停止后台调用，并写入报告 |
| 写操作被预检或守卫插件拒绝 | 把原因作为工具结果返回给模型 |

### 8.7 安全

- **工具面三层限制**：
  1. 插件组合里本来就不包含 shell、文件、终端、网页、资源等工具；
  2. 白名单只开放 `mcp__autopilot__*` 工具；
  3. 写操作再按权限模式判定。
- **启动训练必须由人确认**：LLM 发起的启动训练操作，在任何模式下都要有人确认。
- **工具服务鉴权**：工具服务只调用 `/api/autopilot/agent/*`。这组接口要求一个令牌，令牌在控制台启动时随机生成，经由 dsh 的 MCP 客户端环境变量传给工具服务。控制台和 dsh 网页都只监听 127.0.0.1。
- **Key 保管**：Key 由你在内嵌的 dsh 设置页中填写，保存在 `runs/autopilot/dsh-home` 的凭据里（`runs/` 不进 git），不出现在本仓库代码、campaign 文件或日志中。
- **把模型输出当数据**：模型输出和工具结果都按数据处理，训练参数还要经过 `validate_single` 等白名单校验。

## 9. 控制台集成

**后端**：
- `create_app` 中创建 `AutoPilotService` 和 `DshHost`，它们与 `jobs` 一起在 lifespan 中启动和停止。
- 重启后从 `runs/autopilot/*/campaign.json` 恢复：已完成的 trial 不重跑，运行中的 trial 由 `JobManager` 重新接管。
- `DshHost` 在首次打开 AutoPilot 页面或控制台启动时，用 `myflows-autopilot-web` 启动 dsh 网页。它负责健康检查和重启，后台的一次性运行也由它统一调度和计数。

**新路由**（`apps/console/autopilot_api.py`）：

| 路由 | 用途 |
| --- | --- |
| `GET/POST /api/autopilot/campaigns`、`GET /api/autopilot/campaigns/{id}` | campaign 列表、新建、详情 |
| `POST .../{id}/start\|pause\|resume\|stop` | 控制 campaign |
| `GET .../{id}/trials`、`.../findings`、`.../pending` | 排行榜、诊断、待确认项 |
| `POST .../pending/{pid}/confirm\|reject` | 确认或拒绝 LLM 发起的启动训练请求 |
| `GET .../{id}/report`、`GET .../{id}/stream` | 报告；SSE 事件流 |
| `GET /api/autopilot/dsh/status`、`POST /api/autopilot/dsh/restart` | dsh 网页的状态（克隆、版本、组合检查、Key 是否已配置）与重启 |
| `/api/autopilot/agent/*` | 工具服务专用，需要令牌（第 8.3 节） |

**前端**：侧边栏新增“AutoPilot”，包括以下页面：
- campaign 列表；
- 新建页：任务、目标指标、搜索空间、预算、seed、策略、LLM 开关；
- 详情页：概览、排行榜、诊断链、待确认、LLM 操作记录、报告；
- AutoPilot 助手：嵌入 dsh 网页，在其中对话、切换权限、审批、设置 Key。从详情页打开时，自动带上当前 campaign 的上下文。

## 10. 数据契约

所有 JSON 都带 `schema_version`，字段名与 semester_spec 第 4.0 节的 `TrainConfig` / `TrialResult` 对齐。

`CampaignSpec`：

~~~json
{
  "schema_version": 3,
  "campaign_id": "donkey-r18-lr-batch-001",
  "task": "donkey_resnet18_single",
  "objective": {"metric": "val_angle_mae", "direction": "minimize"},
  "baseline": {"lr": 0.001, "batch": 2, "num_workers": 0, "seeds": [1, 2]},
  "fixed": {"max_samples": 1000, "epochs": 3, "split_file": "runs/autopilot/splits/donkey-1000.json",
            "dtype": "float32", "device": "cuda", "conv_backend": "cuda_native_register"},
  "space": {"lr": [0.0001, 0.0003, 0.001], "batch": [1, 2, 4]},
  "strategy": {"name": "candidate_list", "max_candidates": 6, "seed": 0},
  "budget": {"max_trials": 12, "trial_timeout_s": 1800, "wall_time_s": 14400, "max_retries": 1},
  "online": {"divergence_windows": 3, "divergence_factor": 3.0, "hang_timeout_s": 120},
  "specialists": ["gpu_monitor", "benchmark"],
  "llm": {"enabled": true, "max_calls": 30, "max_total_tokens": null},
  "created_by": "user"
}
~~~

`created_by` 的取值为 `user` 或 `llm`。由 LLM 创建的 campaign，启动时同样要经过第 8.4 节的确认。

`TrialResult` 的结构与 v0.1 相同：run_id、trial_id、attempt、phase、fingerprint、origin、resolved_config、command、status、verdict、metrics、resources、timing、paths、failure。其中 `origin.strategy` 可以是 `candidate_list`、`rule_refine`、`llm`、`tpe`。

待确认项（LLM 发起的启动训练请求）：

~~~json
{"id": "pa-007", "kind": "trials", "campaign_id": "donkey-r18-lr-batch-001", "source": "llm",
 "proposals": [{"delta": {"lr": 0.0003}, "rationale": "……", "expected": "val_angle_mae 下降，samples_per_sec 不变"}],
 "precheck": {"passed": true, "reasons": []}, "status": "pending_confirmation"}
~~~

`kind` 的取值为 `trials` 或 `campaign_start`；`status` 的取值为 `pending_confirmation`、`confirmed`、`rejected`、`expired`。

`events.jsonl` 的事件种类：`proposal`、`precheck`、`pending`、`confirmation`、`dispatch`、`online_stop`、`verdict`、`llm_call`（模型、token、延迟）、`llm_action`（工具名、参数摘要、结果摘要、来源是网页还是后台）、`llm_error`、`note`。

## 11. 开工前要补的接口

| # | 改动 | 位置 | 现状与原因 |
| --- | --- | --- | --- |
| P1 | 由 `--seed` 派生 dropout 和数据增强的种子，并写入 `config.json` | `apps/train/train_myflows_donkey.py` | 80d02af 已加入 `--seed`，用于权重初始化；数据增强目前固定用 `42 + rank` |
| P2 | 训练结束时写 `result.json`；NaN、OOM 和其他异常分别用不同的退出码 | 同上 | 目前没有；Executor 只能从退出码和日志推测结果 |
| P3 | 在验证集上计算 `val_angle_mae`，写入 epoch 行 | 同上，`apps/train/common/validation.py` | 主指标；目前只有 val_loss |
| P4 | 每 N 个 step 用 CUDA event 分别记录 forward / backward / optimizer 的耗时 | 同上 | 课件第 10 页的要求；单进程训练目前没有 |
| P5 | 记录显存峰值（CuPy 内存池的区间最大值） | 同上 / `MyFlows/monitoring` | 课件第 10 页的要求；WDDM 下 NVML 读不到单个进程的显存 |
| P6 | 训练中出现非有限 loss 时，用专用退出码结束 | 同上 | 在线规则的兜底 |
| P7 | 把 `submit_job` 中单进程任务的完整命令和提交参数提取为共用函数 | `apps/console/api.py`、`presets.py` | `single_cli_args` 已能把参数转成命令行；输出路径参数（`--run-dir`、日志、断点、`--stop-file`）和资源采样设置仍写在路由里 |
| P8 | 训练入口启动前获取 GPU 锁（持有者 PID + 创建时间，持有者死亡后锁失效），控制台和 AutoPilot 的任务同样经过这把锁；分布式训练由启动 worker 的父进程持有；开始等锁和获得锁时各写一条事件，作为 Verifier 计时的起点 | `apps/train`、`benchmark/distributed_experiment.py` | 已确认：终端里手动启动的训练也要排队 |
| P9（E3） | 完整的 `TrainingState`：随机数状态、epoch 内的数据游标、早停与调度器状态、配置指纹；事务式 checkpoint（临时写入、校验和、最后发布 manifest） | `MyFlows/utils/checkpoint.py`、`apps/train` | 80d02af 已支持原子写入和按 epoch/step 续训；已确认 E3 要做精确回滚 |

## 12. 代码布局

~~~text
apps/autopilot/                    # 不依赖网页的库，测试直接调用
  service.py  spec.py  planner.py  scheduler.py  executor.py  memory.py  verifier.py  reporter.py
  strategies/  candidate_list.py  rule_refine.py  llm_advisor.py  tpe.py
  agents/      gpu_monitor.py  operator_opt.py  ps_training.py  transfer.py  benchmark.py
  dsh/
    host.py          # DshHost：定位克隆、检查组合、管理网页进程与一次性运行
    tools_server.py  # MCP stdio 工具服务
    prompts.py
    bundle/          # @myflows/dsh-autopilot：package.json、cordis.patch.yml、guard.js、两个 profile 模板
  state.py           # E3：完整 TrainingState 的读写与校验
  rollback.py        # E3
apps/console/autopilot_api.py
apps/console/web/src/pages/autopilot/   CampaignList  NewCampaign  CampaignDetail  Assistant（内嵌 dsh）
tests/system/  fixtures/fake_train.py  fixtures/fake_dsh.py  test_autopilot_*.py
~~~

## 13. 测试与验收

**第一层：日常回归，不需要 GPU 和 LLM**

| 测试 | 检查 |
| --- | --- |
| 预算 | `max_trials`、总时长、单 trial 超时都生效；超时的 trial 被结束，进程树被清理 |
| 异常识别 | 用 `fake_train` 构造 NaN、发散、OOM、卡死、异常退出，确认分类正确且不进入排名 |
| 重试 | 基础设施失败重试 1 次；NaN 和 OOM 不重试同一配置 |
| 恢复 | 控制台在 campaign 中途重启后，已成功的 trial 不重跑，运行中的 trial 被接管 |
| GPU 锁 | 控制台任务、AutoPilot trial 和终端里启动的训练同时存在时，任意时刻最多一个在运行；持有者死亡后锁失效；等锁期间不触发卡死和超时判断 |
| 诊断闭环 | 构造数据读取占比高的场景：GPU Monitor 产生 `data_bound`，派生 trial 改变 num_workers，报告中的诊断链完整 |
| 待确认项 | LLM 发起的启动请求在确认前不运行；拒绝后不运行；确认后进入队列 |
| 工具服务 | stdio 协议（initialize / tools/list / tools/call）；缺少令牌时被拒绝；只读工具不产生写操作 |
| 守卫插件 | 用 `node --test` 测试 `guard.js` 的判定表（第 8.4 节）：在各权限模式和各 profile 下，各类工具的允许、询问、拒绝结果 |
| profile 组合 | 对两个 profile 做组合导出，断言不包含被禁用的工具行，补丁层已经生效 |
| DshHost | 用 `fake_dsh` 模拟网页进程的启动、退出与重启，以及一次性运行的事件转写和降级 |
| E3 | TrainingState 写入后再恢复，与不中断的轨迹逐位一致；TPE 提出的参数落在搜索空间内且不重复；回滚次数受限 |

**第二层：用本机克隆，不需要 Key**

用 mock 模型服务模拟以下场景：
- 第 3 节的 5 个场景；
- 工作区写入模式下，配置类工具触发审批；
- 完全访问模式下，配置类工具直接执行，直接启动训练的工具被拒绝；
- `request_trials` 在控制台生成待确认项；
- 内嵌页面能加载 dsh 网页。

**第三层：手动运行，会消耗 API 额度**

用真实的 DeepSeek 完成以下验收：
- 在内嵌网页中对话、配置并启动一个小 campaign；
- 跑一个开启 LLM 顾问的小 campaign。

记录 token 用量、延迟和费用，据此确定 token 上限。

**GPU 验收**：开发和验证阶段只用小规模数据；跑全量数据之前要先通知用户，等用户把机器空出来。
1. 冒烟：200 个样本，跑 3 个 trial，验证端到端流程。
2. 小规模 campaign：基线、≤ 6 个候选、确认重跑，与人工基线比较 `val_angle_mae` 和 samples/sec。
3. 至少一次由监控数据触发、有前后对照的诊断。
4. E3：用过大的 lr 演示一次异常检测和精确回滚。
5. 正式 campaign（全量数据、3 个 seed、按时间段划分并留有测试集）：经用户同意后运行。

## 14. 里程碑

所有开发目标都要完成，不按汇报节点裁剪。

| 里程碑 | 内容 | 退出检查 |
| --- | --- | --- |
| M0 | P1～P8 | 同一 seed 两次训练逐位一致；单进程和分布式训练都产出 `result.json`、阶段耗时和显存峰值；GPU 锁测试通过；现有回归测试通过 |
| M1 | `apps/autopilot` 规则内核，`fake_train` 系统测试 | 第一层中与 LLM 无关的测试全部通过 |
| M2 | 控制台集成（服务、接口、页面）、GPU Monitor、Benchmark、Reporter；第一个小规模真实 campaign（只用规则） | GPU 验收第 1～3 项 |
| M3 | dsh 接入：先做方案 B 的 spike（外部 bundle 能否装进 AutoPilot 专用 profile、两个 profile 的组合、网页能否嵌入、权限判定），再实现 DshHost、工具服务、守卫插件、助手页和待确认页 | 第一层、第二层全部通过；第三层跑通一次 |
| M4 | Operator Optimization、PS Training；完善报告 | 课件第 9 页的对比表由 Agent 生成；PS 结论自动得出 |
| M5（E3） | P9、TPE、精确回滚；正式 campaign 前补上按时间段划分和测试集 | E3 测试通过；回滚实机演示；正式 campaign 经用户同意后完成 |
| M6（E2 之后） | Transfer Learning | 冻结阶段的搜索能够运行 |

M3 的 spike 不依赖训练入口的改动，可以与 M0、M1 并行。

## 15. 风险

| 风险 | 应对 |
| --- | --- |
| dsh 是开发预览版，补丁层所依赖的 dsh-base 行 id 会随版本变化 | 固定提交；组合检查纳入测试；升级时整体切换提交并重跑测试 |
| dsh 网页可能不允许嵌入（同源策略、鉴权令牌） | 放在 M3 spike 第一项；不行就退回在新标签页打开，并在页面中说明 |
| 审批策略 `never` 会拒绝“询问” | 守卫插件按第 8.4 节的判定表，在完全访问模式下直接返回“允许”；判定表由单元测试覆盖 |
| dsh 网页进程和一次性运行共用一个 `DSH_HOME` | 两者使用各自的会话；在 spike 中验证并发读写凭据与设置是否安全 |
| 需要 Node.js 和构建好的克隆 | 由 `DshHost` 在启动时检查；缺失时关闭 LLM，规则策略照常运行 |
| 费用失控 | 每个 campaign 的后台调用上限 30 次；页面显示用量；第一次真实验收后再定 token 上限 |
| Key 泄露 | 只保存在 AutoPilot 专用 dsh 的凭据里，不进本仓库、campaign 文件或日志 |
| LLM 的配置或建议质量不稳定 | 必须过 Verifier 和预算检查；启动训练必须由人确认；报告对比规则与 LLM 的效果 |
| 课程验收要看自研内容 | 六组件、子 Agent、工具服务、守卫插件和 profile 都是本仓库的代码；dsh 只提供 agent 循环、界面和模型接入 |

## 16. 决定记录

以下为 2026-10-23 逐项确认的结果，原清单的第 4、5 项合并确认。

| # | 问题 | 决定 |
| --- | --- | --- |
| 1 | LLM 承担哪些角色 | 全部承担：Planner 顾问、报告解释、网页对话、配置 campaign（直接创建，不只是起草） |
| 2 | 审批方式 | 默认需要确认：使用 dsh 的权限模式（第 8.4 节）；启动训练类的操作在任何模式下都要由人确认 |
| 3 | API Key 与费用上限 | Key 在 AutoPilot 专用 dsh 的设置页中配置；每个 campaign 的后台调用最多 30 次；token 上限在第一次真实验收后确定 |
| 4、5 | dsh 运行时与对话方式 | 方案 B：使用固定提交的源码克隆；对话界面嵌入控制台，使用 dsh 网页（常驻）；后台角色走一次性运行 |
| — | dsh 可以执行的写操作 | 创建和修改 campaign、启动或暂停 campaign、提交 trial、停止运行中的 trial |
| 6 | 终端里手动启动的训练是否进同一个 GPU 队列 | 是，在训练入口加锁（P8） |
| 7 | AutoPilot 是否先于 W1/W5 开工 | 是 |
| 8 | 汇报节点 | 不考虑汇报，开发目标全部完成 |
| 9 | 人工基线 | lr 1e-3、batch 2、num_workers 0 |
| 10 | 数据规模与预算 | 开发和验证只用小规模数据；跑全量数据之前先通知用户，等用户把机器空出来 |
| 11 | seed 数 | 联调 2 个，正式 3 个 |
| 12 | 数据划分 | 先用现有的随机划分；正式 campaign 前补上按时间段划分和独立测试集 |
| 13 | 搜索空间 | 精度搜索只搜 lr × batch；num_workers、卷积后端、训练 worker 数按吞吐量决定；正则项在 TPE 阶段加入 |
| 14 | checkpoint 保留 | 只保留基线和前 2 名的最佳模型 |
| 15 | 经验库 | 导入资源类事实；收敛类从空开始；不进 git |
| 16 | E3 范围 | 先实现完整的 TrainingState，再做 TPE 和精确回滚 |

实现过程中需要确定、但不需要用户决定的事项：
- 外部 bundle 的安装方式；
- dsh 网页能否被嵌入；
- token 上限的具体数值。

以上事项都在 M3 的 spike 或第一次真实验收中确定。

## 附录 A：PoC 实现要点

PoC 使用的是基于 `sdk-minimal` 的独立插件树，共 29 行，分组如下：

| 类别 | 行 id |
| --- | --- |
| 入口 | `headless-startup`、`headless-runner` |
| 模型 | `llm`、`llm-deepseek`、`deepseek-llm-api-extensions`、`llm-retry`、`agent-default-model` |
| agent 与会话 | `agent`、`agent-loop`、`session`、`session-title`、`sessions`、`session-log-deepseek`、`session-projection`、`token-meter`、`compaction-basic` |
| 运行环境 | `timer`、`jobs`、`subprocess`、`sandbox`、`sandbox-policy`（read-only）、`filesystem`、`attachment-local`、`working-directory`、`plugin-package-inventory-deepseek` |
| 工具 | `tools`、`mcp-autopilot`、`tool-allowlist` |
| 提示词 | `system-prompt`（不带 Harness 身份和运行时上下文，用中文顾问人设） |

正式实现改为在 dsh-base 之上加补丁层（第 8.2 节）。这样可以保留网页、凭据、审批和权限预设，但需要显式禁用 dsh-base 中的工具行。

MCP 客户端行的写法如下（`!!js` 表达式写成块标量）：

~~~yaml
- id: mcp-autopilot
  name: '@deepseek-ai/dsh-mcp-client'
  config:
    serverName: autopilot
    transport: stdio
    command: !!js >-
      process.env.MYFLOWS_PYTHON
      ?? `${process.env.MYFLOWS_ROOT}/.venv/${process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python'}`
    args: !!js >-
      [process.env.MYFLOWS_AUTOPILOT_SERVER
      ?? (() => { throw new Error('set MYFLOWS_AUTOPILOT_SERVER') })()]
    cwd: !!js >-
      process.env.MYFLOWS_ROOT ?? (() => { throw new Error('set MYFLOWS_ROOT') })()
    env:
      PYTHONIOENCODING: utf-8
    failOnStartupError: true
~~~

白名单部分的核心逻辑如下；权限判定在此基础上增加 `tools/pre-execute` 监听：

~~~js
export const name = 'myflows-autopilot-guard'
export const inject = ['tools']

export function apply(ctx, config) {
  const allowed = toolName => toolName.startsWith(config.prefix)
  ctx.tools.guard(execution => allowed(execution.name)
    ? undefined
    : `myflows-autopilot only allows ${config.prefix}* tools; ${execution.name} is denied`)
  ctx.on('agent/created', ({ agent }) => {
    const denied = ctx.tools.schemas(agent).map(tool => tool.name).filter(n => !allowed(n))
    if (denied.length > 0) createScope(ctx, agent).ctx.tools.restrict({ deny: denied })
  })
}
~~~

`createScope` 来自 `@deepseek-ai/dsh-scope`。作为外部插件时，它如何从克隆的安装中解析，要在 spike 中确认。
