# MNIST 与 ResNet18 道路任务：PS / Ring 实测对比

根据用户恢复任务后的要求，正式比较覆盖 MNIST MLP 与现有 ResNet18 的 DonkeyCar 道路回归。PS 协议表包含 Socket/JSON 与 gRPC/protobuf；PS/Ring 架构表及加速比图统一使用 gRPC/protobuf。

汇报材料：第四周汇报 PPT 与汇报稿作为本地文件保存，不纳入 Git。仓库保留本报告、关键汇总数据和实验复现脚本。

本报告使用当前自研框架，在单机回环网络中运行独立 Worker，所有训练 Worker 共享一张 RTX 4060 Laptop GPU。
PS 使用独立 CPU 聚合进程，Worker 在 GPU 更新参数；Ring 使用真实相邻 gRPC 传递，完成 ScatterReduce 与 AllGather。
MNIST 使用 GPU 上的 CuPy 数组与矩阵乘法；ResNet18 的 Conv/Pool 使用现有原生 cuda_native_cublas。两任务沿用项目现有 Adam 更新规则。

**验收边界：ResNet 的独立整批训练轨迹与分片轨迹未通过原 Spec 的逐步梯度等价门槛。本文保留该失败，并分别报告已验证的梯度归约、CPU/GPU 更新一致性和完整训练质量，不将本报告称为全部 Spec 条款无条件通过。**

## 优化版与原基线对照

本轮优化版（`20260919_optimized`）与原基线（`20260917`）使用相同数据、初始参数、Worker 数、单 GPU、epoch 口径和三次性能重复；PS/Ring 架构比较全部使用 gRPC/protobuf。下面的下降率按 `(基线 epoch 时间 - 优化版 epoch 时间) / 基线 epoch 时间` 计算。

|任务|配置|原基线 epoch/s|优化版 epoch/s|耗时下降|优化版质量|
|---|---|---:|---:|---:|---:|
|MNIST|PS gRPC 2|34.329|12.207|64.45%|96.56%|
|MNIST|PS gRPC 4|62.145|15.399|75.22%|96.52%|
|MNIST|Ring 2|4.226|3.369|20.27%|96.56%|
|MNIST|Ring 4|6.396|5.684|11.12%|96.57%|
|ResNet18|PS gRPC 2|110.171|14.073|87.23%|angle MAE 0.037636|
|ResNet18|PS gRPC 4|210.157|16.936|91.95%|angle MAE 0.037397|
|ResNet18|Ring 2|13.276|13.187|0.67%|angle MAE 0.037636|
|ResNet18|Ring 4|16.488|15.422|6.46%|angle MAE 0.037379|

8 行 gRPC/protobuf 分布式配置全部比原基线快，整体中位耗时下降 42.36%；PS 为 81.22%。Ring 的中位下降为 8.79%，其中 MNIST 两个配置达到 10% 显著门槛，ResNet18 Ring 2/4 分别只下降 0.67%/6.46%，因此 Ring 的优化结论是部分改善，不能写成所有配置均显著。单进程控制组也只小幅变化（MNIST 1.418→1.385 s，ResNet18 9.121→9.052 s），不能把 Socket/JSON 的运行波动全部归因于本轮 gRPC 优化。

按“gRPC/protobuf 分布式 8 行全部不变慢、整体中位下降至少 10%、14/14 训练质量通过、最后一次全量回归通过”的规则，本优化版作为主结果，原基线永久保留作历史参照。逐配置数据见 [optimization_comparison.csv](optimization_comparison.csv) 和 [optimization_comparison.json](optimization_comparison.json)，交付检查摘要见 [delivery_summary.json](delivery_summary.json)。

## 课件要求与证据

|来源|要求|保留证据|
|---|---|---|
|PS 课件第 2、4 页|MNIST/MLP、道路 CNN；单机、2/4 Worker；Socket/JSON 与 gRPC|两任务各 7 个完整训练配置，*/convergence/*|
|PS 课件第 43 页|心跳、掉线超时处理|fault_checks.json；PS 独立 watchdog，超时中止同步训练|
|Ring 课件第 4、39 页|epoch 总耗时、计算、梯度同步、加速比、准确率|comparison.csv、每 run 的 epochs.csv、加速比与收敛图|
|Ring 课件第 4、41 页|完整 Ring、计算正确性、固定格式归档|numeric_checks.json、逐 rank/step 的参数/梯度/Adam 状态 NPZ、messages.jsonl|
|综合项目第 10 页|GPU/CPU 资源、forward/backward/update、加载、吞吐量|resources.jsonl、rank-*/steps.jsonl、epochs.csv|

数值与更新检查保留 36 组配置，覆盖 synthetic、MNIST、小型道路 CNN 和 ResNet18，MBGD/Adam、N=1/2/4 和非均匀分片。ResNet18 使用下文单列的归约/更新参考，不混称为独立整批轨迹全部等价。
故障/重复消息场景通过 17/17；训练质量门槛通过 14/14。

带非空 Adam 状态的恢复对照通过 9/9；N=1/2/4/5 的分块流向与非整除补零检查通过 12/12；实际梯度字节量通过 36/36。
数值/恢复矩阵中保留此前小型道路 CNN 的检查；它与本轮 ResNet18 正式结果单独存放。所有记录分别见 [数值](numeric_checks.json)、[恢复](state_checks.json)、[故障](fault_checks.json)、[分块](ring_chunk_flow.json)、[通信量](volume_checks.json)。

## 固定配置与测量口径

- MNIST：官方 60,000 训练部分划分为 50,000/10,000；官方 10,000 测试集仅在最终评估使用。784→64→10，global batch=128，Adam lr=0.001，10 epochs。
- 每个 epoch 先排列全部 50,000 个训练索引，再统一丢弃尾部 80 个，共 390 steps、49,920 samples；不同 Worker 数拆分同一个 global batch。
- 质量目标为验证 accuracy ≥95%，与单进程差 ≤1 个百分点。这是工作区 Spec 的项目门槛，不是课件给定的精确数值。
- 各配置使用相同 seed=0、FP32、数据顺序和初始参数。性能数据为从同一初始 checkpoint 出发的完整第 1 个 epoch，独立进程重复 3 次，轮换配置顺序；所有 GPU 实验串行。
- epoch 总时间排除进程启动、数据准备、首次编译预热、评估与保存；逐 Worker 的计算/同步累计后取最大值，不能把各列相加当作端到端耗时。
- CUDA Event 测量 forward/backward/update；另保留主机 wall time。梯度 D2H、H2D、摘要及更新确认分列。GPU/CPU 以约 1 秒周期采样，显存峰值为采样峰值，并非瞬时精确峰值。
- 利用率/显存/温度/功耗为整卡 GPU 观测，CPU 利用率为整机观测；没有扣除桌面与其他后台活动。进程树 RSS 另保留在 resources.jsonl。
- 所有主表 run 启用相同 CUDA Event 同步、逐步记录和资源监控；表中是带测量开销的时间。监控开关对照只衡量资源采样，不能代表全部仪表开销。
- Socket 的解码未单独计时，包含在 RPC 观测中；日志 decode_s=0 是占位值。编码、RPC、规约与等待均包含在梯度同步总时间，不称为纯网络传输时间。
- 消息量为实际 JSON/protobuf 正文长度；Socket 长度前缀单列。HTTP/2、TCP/IP 头、重传未抓包测量，因此不称为实际网卡总流量。
- 性能 run 与收敛 run 分别保存；表中的准确率/MAE来自完整收敛 run，耗时来自 3 次性能 run，具体 run_id 在 comparison.csv。

完整逐 epoch 数据见 [epoch_metrics.csv](epoch_metrics.csv)，包含 run/config、数据指纹、初值摘要、优化器、样本数、阶段时间、S/E 与质量。旧 MNIST 收敛 run 没有反向后至放行的独立时间戳，该字段留空并标记未测；三次正式性能重复均记录此字段。纯线路传输时间也留空，不用 0 冒充实测值。

## 通信量口径

令 N 为 Worker 数，M 为一个完整 FP32 梯度的字节数（PS 不计入 N）。PS 中心上行接收 NM、下行发送 NM；集群有向发送总量 2NM。
Ring 每个 rank 单向发送/接收均为 2(N−1)M/N，集群有向发送总量为 2(N−1)M。单 rank 的量趋近 2M，集群总量仍随 N 增长。
实际 Ring 尾部补零，用 M′=4N ceil(D/N) 代入。以上只计训练梯度；初始化、确认、心跳等控制消息由正文计数另行记录。

## ResNet18 配置与 BN 控制

复用 `MyFlows.layers.resnet.ResNet18`，8 个残差块，base_width=16，各 stage 为 [16, 32, 64, 128]，参数量 703,554。这是现有模型的宽度配置，不是默认宽度 64 的 11,182,338 参数版本。
图像 120×160 RGB；8,000/1,000/1,000 按来源编号分组划分；global batch=128，Adam lr=0.0003，5 epochs。每 epoch 先全量排列再丢弃尾部 64 张，使用 7936 张。
仅用 512 张训练图校准 BN running mean/variance，随后固定；gamma/beta 及全部卷积/FC 权重保持可训练。所有配置复用同一 BN 文件，checkpoint 与参数一致性摘要包含 BN buffers。没有使用本地 shard 的训练态 BN，也未声称实现 SyncBatchNorm。
预检仅访问训练/验证集，测试集在正式训练结束时评估。门槛：训练 loss 相对初始化下降至少 30%；验证 angle MAE 优于常数预测至少 10%；与单进程差 ≤max(0.001, 单进程 MAE 的 5%)。
配置与预检证据：[冻结配置](frozen_task_configs.json)、[选择记录](resnet_preflight/freeze_decision.json)。

### ResNet 数值判定范围

严格的“独立整批单进程轨迹 vs 分片轨迹，每一步梯度逐元素等价”在 ResNet 上出现超差，未将其标为通过。首个双 Worker MBGD 案例在第 4 步参数最大差约 4.3e-7，但梯度差约 1.4e-3；重算记录了 ReLU 掩码和 MaxPool argmax 的变化。某些相同权重下的不同 batch 形状也出现少量分支变化。详见原基线目录中的 [保留诊断](../20260917/resnet_gradient_diagnosis.json)。
原 synthetic/MNIST/小型 CNN 保持原连续 20 步独立轨迹对照。新增 ResNet 检查分为：共同初始点的整批梯度对照；每一步实际局部 FP32 梯度的独立 FP64 加权求和；使用已经核验的传输均值，从同一初始状态执行现有 CPU 优化器，比较全部 rank 的 GPU 参数、Adam v/s、版本和 BN buffers。各项仍使用 atol=1e-4、rtol=1e-3；同时保留独立整批轨迹的逐步误差，未放宽容差后把原失败改写为成功。
新增 ResNet 18 组逐步检查的归约最大绝对误差为 1.7e-07，CPU/GPU 参数更新最大绝对误差为 2.98e-08。每组原始局部/全局梯度、CPU 参考 checkpoint 及 resnet_step_audit.json 均留档。完整训练的质量差异另按固定验证门槛判断。

## MNIST MLP

### PS 与 Ring：统一 gRPC/protobuf

|配置|epoch 时间/s|计算/s|梯度同步/s|加速比|效率|验证准确率|
|---|---:|---:|---:|---:|---:|---:|
|单进程|1.385|0.605|0.000|1.000|1.000|96.57%|
|PS gRPC 2|12.207|0.810|5.850|0.113|0.057|96.56%|
|PS gRPC 4|15.399|0.962|6.885|0.090|0.022|96.52%|
|Ring gRPC 2|3.369|0.725|0.820|0.411|0.206|96.56%|
|Ring gRPC 4|5.684|1.069|2.184|0.244|0.061|96.57%|

课件表格图片：[PS 协议表](mnist_mlp-ps-table.png)、[统一 gRPC 的 PS/Ring 表](mnist_mlp-ring-table.png)。
计算列为前向加反向的主机耗时；CUDA Event 对应数据在阶段明细中另列。梯度同步包含聚合和等待。

### PS 传输协议对比（对应第一张课件表）

|配置|epoch 时间/s|计算/s|通信/s|消息正文总字节/epoch|验证准确率|
|---|---:|---:|---:|---:|---:|
|单进程|1.385|0.605|0.000|0|96.57%|
|PS JSON 2|33.217|0.993|28.490|1213291655|96.56%|
|PS gRPC 2|12.207|0.810|5.850|318286492|96.56%|
|PS JSON 4|52.964|1.873|42.595|2347842661|96.52%|
|PS gRPC 4|15.399|0.962|6.885|636577010|96.52%|

消息正文为请求与响应的有向发送总量，每条消息仅计一次；不包括 TCP/IP、HTTP/2 头。正文包括梯度及控制消息。

### 质量与阶段明细


|配置|梯度 D2H/s|均值送入设备/s|GPU 更新/s|摘要及更新确认/s|反向后至放行/s|
|---|---:|---:|---:|---:|---:|
|单进程|0.000|0.061|0.443|0.000|0.607|
|PS JSON 2|0.067|0.140|0.783|2.649|31.978|
|PS JSON 4|0.075|0.144|0.901|7.438|50.847|
|PS gRPC 2|0.061|0.130|0.678|5.331|11.088|
|PS gRPC 4|0.062|0.131|0.679|7.055|14.360|
|Ring gRPC 2|0.060|0.161|0.556|0.774|2.435|
|Ring gRPC 4|0.078|0.202|0.794|1.113|4.397|

反向后至放行为单独主机时间戳测量，包含指标处理、梯度拷贝/同步、更新及确认。均值送入设备一列在单进程中为 GPU 内部复制，分布式中含 H2D。摘要时间包含在更新确认列中。


|配置|验证准确率|官方测试准确率|与单进程验证差/百分点|
|---|---:|---:|---:|
|单进程|96.57%|96.79%|+0.000|
|PS JSON 2|96.56%|96.74%|-0.010|
|PS JSON 4|96.52%|96.73%|-0.050|
|PS gRPC 2|96.56%|96.74%|-0.010|
|PS gRPC 4|96.52%|96.73%|-0.050|
|Ring gRPC 2|96.56%|96.74%|-0.010|
|Ring gRPC 4|96.57%|96.79%|+0.000|

|配置|forward GPU/s|backward GPU/s|update GPU/s|数据准备/s|samples/s|单步中位/ms|单步P95/ms|
|---|---:|---:|---:|---:|---:|---:|---:|
|单进程|0.260|0.310|0.443|0.052|36041|2.897|6.901|
|PS JSON 2|0.433|0.414|0.783|0.073|1503|83.437|95.692|
|PS JSON 4|0.701|0.855|0.901|0.065|943|135.366|143.368|
|PS gRPC 2|0.360|0.377|0.678|0.072|4090|30.128|50.141|
|PS gRPC 4|0.395|0.457|0.679|0.070|3242|33.522|53.034|
|Ring gRPC 2|0.338|0.310|0.556|0.061|14816|8.400|11.504|
|Ring gRPC 4|0.452|0.485|0.794|0.063|8782|14.272|17.229|

|配置|GPU利用率均值/%|采样显存峰值/MiB|温度峰值/℃|功耗峰值/W|CPU利用率均值/%|有效采样点|
|---|---:|---:|---:|---:|---:|---:|
|单进程|34.7|1691.0|55.0|17.7|9.3|3|
|PS JSON 2|33.7|1798.0|54.0|30.3|8.9|92|
|PS JSON 4|35.3|2038.0|53.0|22.9|11.8|146|
|PS gRPC 2|27.2|1798.0|53.0|21.5|8.2|33|
|PS gRPC 4|32.5|2011.0|54.0|23.5|11.6|42|
|Ring gRPC 2|41.6|1798.0|56.0|31.9|18.5|9|
|Ring gRPC 4|42.5|2011.0|58.0|37.2|24.9|15|

资源均值为三次 run 各自有效样本均值的平均；峰值为三次 run 中的最大采样值。短 epoch 采样点少，不能据此精确估计持续利用率。

|配置|请求正文 MiB/epoch|响应正文 MiB/epoch|训练梯度上行 MiB/epoch|同步占 epoch 比例|
|---|---:|---:|---:|---:|
|单进程|0.00|0.00|0.00|0.0%|
|PS JSON 2|561.40|595.68|151.42|85.8%|
|PS JSON 4|1046.93|1192.14|302.84|80.4%|
|PS gRPC 2|151.85|151.69|151.42|47.9%|
|PS gRPC 4|303.70|303.39|302.84|44.7%|
|Ring gRPC 2|151.87|0.07|151.42|24.3%|
|Ring gRPC 4|456.53|0.33|454.28|38.4%|

正文为各 Worker 的实际请求/响应累计值再取三次中位数，包含控制消息；梯度上行为 PS Worker→Server 或 Ring 各有向链路发送的 FP32 数组字节，PS 返回梯度不计入该列。

![MNIST MLP 加速比](mnist_mlp-speedup.png)

误差线为 3 次独立进程测量的最小/最大耗时换算范围；N=1 为共同单进程基线。

![MNIST MLP 收敛](mnist_mlp-convergence.png)

本组分布式配置中耗时最短的是 Ring gRPC 2，加速比为 0.411。
单进程完整 epoch 中位数为 1.385 s；分布式同步包含主机梯度规约、编解码、回环 RPC 及等待，
且所有 Worker 争用同一张 GPU。Worker 数增加不增加 GPU 算力；消息轮次、Python 处理及更新确认会产生额外开销。

### 同 Worker 数的瓶颈对照

- 2 Worker：PS gRPC / Ring gRPC 的 epoch 耗时比为 3.62；各自梯度同步占 epoch 的 47.9% / 24.3%，更新确认累计为 5.331 / 0.774 s。PS JSON / PS gRPC 的耗时比为 2.72，消息正文总量比为 3.81。
- 4 Worker：PS gRPC / Ring gRPC 的 epoch 耗时比为 2.71；各自梯度同步占 epoch 的 44.7% / 38.4%，更新确认累计为 7.055 / 1.113 s。PS JSON / PS gRPC 的耗时比为 3.44，消息正文总量比为 3.69。

以上比例是当前实现的实测结果。PS 的通用消息转换、轮询和服务器处理，以及 Ring 的分块交换轮次均包含在同步耗时中；没有独立消融数据来把各项差距全部归因于某一种机制。

## DonkeyCar ResNet18

### PS 与 Ring：统一 gRPC/protobuf

|配置|epoch 时间/s|计算/s|梯度同步/s|加速比|效率|验证 angle MAE|
|---|---:|---:|---:|---:|---:|---:|
|单进程|9.052|6.285|0.000|1.000|1.000|0.037603|
|PS gRPC 2|14.073|6.257|3.491|0.643|0.322|0.037636|
|PS gRPC 4|16.936|6.312|5.068|0.535|0.134|0.037397|
|Ring gRPC 2|13.187|7.592|1.558|0.686|0.343|0.037636|
|Ring gRPC 4|15.422|7.726|2.066|0.587|0.147|0.037379|

课件表格图片：[PS 协议表](donkey_resnet18-ps-table.png)、[统一 gRPC 的 PS/Ring 表](donkey_resnet18-ring-table.png)。
计算列为前向加反向的主机耗时；CUDA Event 对应数据在阶段明细中另列。梯度同步包含聚合和等待。

### PS 传输协议对比（对应第一张课件表）

|配置|epoch 时间/s|计算/s|通信/s|消息正文总字节/epoch|验证 angle MAE|
|---|---:|---:|---:|---:|---:|
|单进程|9.052|6.285|0.000|0|0.037603|
|PS JSON 2|273.579|18.125|244.177|3872942449|0.037636|
|PS gRPC 2|14.073|6.257|3.491|698849712|0.037636|
|PS JSON 4|462.667|18.905|416.378|7737408025|0.037397|
|PS gRPC 4|16.936|6.312|5.068|1397704734|0.037397|

消息正文为请求与响应的有向发送总量，每条消息仅计一次；不包括 TCP/IP、HTTP/2 头。正文包括梯度及控制消息。

### 质量与阶段明细


|配置|梯度 D2H/s|均值送入设备/s|GPU 更新/s|摘要及更新确认/s|反向后至放行/s|
|---|---:|---:|---:|---:|---:|
|单进程|0.000|0.098|1.193|0.000|1.324|
|PS JSON 2|0.172|0.187|1.452|13.493|254.805|
|PS JSON 4|0.203|0.227|1.758|32.813|443.103|
|PS gRPC 2|0.157|0.222|1.579|1.796|7.137|
|PS gRPC 4|0.199|0.322|2.276|2.665|10.355|
|Ring gRPC 2|0.167|0.217|1.631|1.092|4.723|
|Ring gRPC 4|0.202|0.337|2.703|1.776|7.080|

反向后至放行为单独主机时间戳测量，包含指标处理、梯度拷贝/同步、更新及确认。均值送入设备一列在单进程中为 GPU 内部复制，分布式中含 H2D。摘要时间包含在更新确认列中。


|配置|验证 angle MAE|测试 angle MAE|测试 angle RMSE|测试 throttle MAE|
|---|---:|---:|---:|---:|
|单进程|0.037603|0.038653|0.052363|0.025922|
|PS JSON 2|0.037636|0.038533|0.052214|0.026046|
|PS JSON 4|0.037397|0.038372|0.052047|0.025867|
|PS gRPC 2|0.037636|0.038533|0.052214|0.026046|
|PS gRPC 4|0.037397|0.038372|0.052047|0.025867|
|Ring gRPC 2|0.037636|0.038533|0.052214|0.026046|
|Ring gRPC 4|0.037379|0.038456|0.052120|0.025847|

道路数据是 angle/throttle 连续值回归，课件的“准确率”列用有明确意义的 MAE/RMSE 替代。throttle 标签恒为 0.5，不代表具备变速决策能力。

|配置|forward GPU/s|backward GPU/s|update GPU/s|数据准备/s|samples/s|单步中位/ms|单步P95/ms|
|---|---:|---:|---:|---:|---:|---:|---:|
|单进程|1.404|4.874|1.193|1.120|877|145.945|153.030|
|PS JSON 2|6.820|11.203|1.452|0.621|29|4426.833|4506.295|
|PS JSON 4|6.916|11.784|1.758|0.382|17|7435.621|7877.358|
|PS gRPC 2|1.315|5.029|1.579|0.567|564|228.458|253.075|
|PS gRPC 4|1.300|4.991|2.276|0.371|469|273.825|307.490|
|Ring gRPC 2|1.777|5.781|1.631|0.625|602|210.597|227.161|
|Ring gRPC 4|1.743|5.995|2.703|0.438|515|248.392|278.367|

|配置|GPU利用率均值/%|采样显存峰值/MiB|温度峰值/℃|功耗峰值/W|CPU利用率均值/%|有效采样点|
|---|---:|---:|---:|---:|---:|---:|
|单进程|74.4|4713.0|64.0|60.7|9.1|27|
|PS JSON 2|25.1|4953.0|57.0|31.3|9.5|749|
|PS JSON 4|29.0|5277.0|57.0|109.4|12.7|1260|
|PS gRPC 2|60.4|4933.0|63.0|79.3|14.5|39|
|PS gRPC 4|64.5|5249.0|62.0|66.2|20.6|48|
|Ring gRPC 2|69.7|4928.0|63.0|73.3|14.0|35|
|Ring gRPC 4|72.5|5249.0|62.0|68.4|22.9|42|

资源均值为三次 run 各自有效样本均值的平均；峰值为三次 run 中的最大采样值。短 epoch 采样点少，不能据此精确估计持续利用率。

|配置|请求正文 MiB/epoch|响应正文 MiB/epoch|训练梯度上行 MiB/epoch|同步占 epoch 比例|
|---|---:|---:|---:|---:|
|单进程|0.00|0.00|0.00|0.0%|
|PS JSON 2|1842.49|1851.04|332.80|89.3%|
|PS JSON 4|3669.67|3709.30|665.59|90.0%|
|PS gRPC 2|333.25|333.22|332.80|24.8%|
|PS gRPC 4|666.51|666.44|665.59|29.9%|
|Ring gRPC 2|332.87|0.01|332.80|11.8%|
|Ring gRPC 4|998.75|0.05|998.39|13.4%|

正文为各 Worker 的实际请求/响应累计值再取三次中位数，包含控制消息；梯度上行为 PS Worker→Server 或 Ring 各有向链路发送的 FP32 数组字节，PS 返回梯度不计入该列。

![DonkeyCar ResNet18 加速比](donkey_resnet18-speedup.png)

误差线为 3 次独立进程测量的最小/最大耗时换算范围；N=1 为共同单进程基线。

![DonkeyCar ResNet18 收敛](donkey_resnet18-convergence.png)

本组分布式配置中耗时最短的是 Ring gRPC 2，加速比为 0.686。
单进程完整 epoch 中位数为 9.052 s；分布式同步包含主机梯度规约、编解码、回环 RPC 及等待，
且所有 Worker 争用同一张 GPU。Worker 数增加不增加 GPU 算力；消息轮次、Python 处理及更新确认会产生额外开销。

### 同 Worker 数的瓶颈对照

- 2 Worker：PS gRPC / Ring gRPC 的 epoch 耗时比为 1.07；各自梯度同步占 epoch 的 24.8% / 11.8%，更新确认累计为 1.796 / 1.092 s。PS JSON / PS gRPC 的耗时比为 19.44，消息正文总量比为 5.54。
- 4 Worker：PS gRPC / Ring gRPC 的 epoch 耗时比为 1.10；各自梯度同步占 epoch 的 29.9% / 13.4%，更新确认累计为 2.665 / 1.776 s。PS JSON / PS gRPC 的耗时比为 27.32，消息正文总量比为 5.54。

以上比例是当前实现的实测结果。PS 的通用消息转换、轮询和服务器处理，以及 Ring 的分块交换轮次均包含在同步耗时中；没有独立消融数据来把各项差距全部归因于某一种机制。

## Profiler、监控开销和归档核验

- nsys / ps：已捕获 im2col_forward, col2im_backward, maxpool2d_forward_native, maxpool2d_backward_native；[原始报告](profiles/donkey_resnet18-nsys-ps/capture.nsys-rep)。
- nsys / ring：已捕获 im2col_forward, col2im_backward, maxpool2d_forward_native, maxpool2d_backward_native；[原始报告](profiles/donkey_resnet18-nsys-ring/capture.nsys-rep)。
- ncu / ps：已捕获 im2col_forward；[原始报告](profiles/donkey_resnet18-ncu-ps/capture.ncu-rep)。
- ncu / ring：已捕获 im2col_forward；[原始报告](profiles/donkey_resnet18-ncu-ring/capture.ncu-rep)。

Profiler 使用独立的短训练进程，包含准备/预热，不计入性能表；Nsight Compute 抽取首个匹配 kernel 作代表性诊断。

|Nsight Systems / 模式|原生 kernel|调用次数|累计 GPU 时间/ms|kernel 时间占比|
|---|---|---:|---:|---:|
|ps|im2col_forward|240|4.721|7.6%|
|ps|col2im_backward|240|2.206|3.5%|
|ps|maxpool2d_backward_native|12|0.238|0.4%|
|ps|maxpool2d_forward_native|12|0.059|0.1%|
|ring|im2col_forward|240|4.300|7.7%|
|ring|col2im_backward|240|2.314|4.2%|
|ring|maxpool2d_backward_native|12|0.219|0.4%|
|ring|maxpool2d_forward_native|12|0.054|0.1%|

|Nsight Compute / 模式|采样 kernel|耗时/μs|SM 吞吐率/峰值|实际占用率|寄存器/线程|
|---|---|---:|---:|---:|---:|
|ps|im2col_forward|133.344|77.38%|89.02%|24|
|ring|im2col_forward|132.864|77.66%|89.36%|24|

两种模式均记录到完整的原生 Conv/Pool 前后向路径。短追踪的 kernel 累计时间及单个 im2col 的占用率仅用于确认执行路径和说明该次采样；不能用这些预热/小 batch 数据代替上文完整 epoch 的瓶颈占比。

|任务 / 配置|资源监控开/s|资源监控关/s|中位数差异|
|---|---:|---:|---:|
|mnist_mlp / 单进程|1.298|1.293|+0.4%|
|mnist_mlp / Ring gRPC 4|5.476|5.415|+1.1%|
|donkey_resnet18 / 单进程|9.021|8.955|+0.7%|
|donkey_resnet18 / Ring gRPC 4|15.132|15.990|-5.4%|

开/关各 3 次独立进程，交替顺序，均测完整 epoch；原始值见 [monitor_overhead.json](monitor_overhead.json)。短时间差异包含调度和设备状态波动，负差值不能解释成监控具有加速作用。

- mnist_mlp：28 个正式 run 通过数据 SHA256、无交叉划分、相同初值、步数/样本数、GPU 参数/Adam 状态、rank 最终摘要和性能源码一致性核验；[记录](mnist_mlp_artifact_audit.json)。
- donkey_resnet18：28 个正式 run 通过数据 SHA256、无交叉划分、相同初值、步数/样本数、GPU 参数/Adam 状态、rank 最终摘要和性能源码一致性核验；[记录](donkey_resnet18_artifact_audit.json)。
- 全量回归 204 项通过；[日志](suite_logs/run_tests-1.log)。
- 全量回归 204 项通过；[日志](suite_logs/run_tests-2.log)。
- 优化完成后的最后一次全量回归 204 项通过（53.919 s）；[日志](suite_logs/run_tests-final-optimized-20260919.log)。
- 两个任务共 56 个正式档案（14 个收敛、42 个性能重复）均已核验；逐配置优化对照见 [optimization_comparison.csv](optimization_comparison.csv)。

早期数值/恢复记录包含 Ring 控制消息仍用 Socket 的版本；新增 ResNet 复核和本轮正式 PS/Ring 矩阵均使用 gRPC/protobuf 控制消息，归档审计另行验证没有 Socket 帧。历史与新增结果保留各自源码指纹。

## 可复现性与限制

每个 run 保留 config.json、manifest.json（两仓库 Git 状态、源码哈希、依赖）、data_manifest.json、results.json、epochs.csv、events.jsonl、resources.jsonl、逐 rank 的 step/epoch 日志及完整 checkpoint。
MNIST 数组位于工作区 .codex/distributed-data-v1/mnist_mlp；数据 manifest 保存下载文件、数组和划分的 SHA256。
共享单 GPU 的结果不能用来推断多 GPU/多机器网络扩展性。道路数据为 generated-road，编号分组并非经过验证的真实采集 session，离线回归指标不代表真实车辆闭环表现。

实现层限制：PS 的 gRPC 适配器目前在通用消息与 protobuf 之间经过 Python list，Ring 梯度块直接使用 bytes。统一通信协议后仍存在这类实现开销差异，因此比较结论针对本项目当前实现，不能把全部时间差归因于网络拓扑的理论优势。

完整复现：项目 .venv 下先运行 `python -m benchmark.prepare_distributed_data`，再运行 `python -m benchmark.complete_course_experiments --root <新目录>`。该入口依次执行 ResNet 预检、冻结配置、数值/恢复/故障检查、完整训练、三次性能重复、Profiler、监控对照、归档核验及报告生成。相同 root 再次执行会保留历史尝试并复用配置匹配的成功实验；不要同时启动两组 GPU 实验。
