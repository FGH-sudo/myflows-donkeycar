# 本轮实验协议

本轮根据用户授权完成课件中的 PS/Ring 测试、数据记录和对比。三份课件作为要求来源，原有工作树和历史失败基线保留。

**最新执行范围（中断后用户更新）：MNIST 继续完成；道路任务使用现有 `MyFlows.layers.resnet.ResNet18`。PS 协议表保留 JSON/gRPC 的 2/4 Worker，PS/Ring 架构表统一 gRPC/protobuf。** 此前 MNIST-only 限制已由用户撤回。`execution_scope.json` 已同步更新。

原 PS JSON 四 Worker 的 MNIST 训练在第 10 轮中断，留档于 `mnist_mlp/convergence/ps-json-4-interrupted-20260917`；它只有 9 个完整 epoch 且无最终 checkpoint，不用作正式收敛验收，恢复后重新运行同一配置。

ResNet18 新增实验适配器复用现有模型和原生 Conv/Pool，不修改其 CUDA 实现。由于本机为 16 GB RAM、单张 8 GB GPU，候选配置为 `base_width=16`、ImageNet stem、8 个 BasicBlock、703,554 个参数、FP32；不是默认宽度 64 的 11,182,338 参数配置。先仅用训练集的 512 张图校准 BN 统计量，随后固定这些统计量，gamma/beta 和全部卷积/FC 权重仍参与训练。各 Worker 从相同 BN 文件/参数开始，checkpoint 和摘要包含 BN buffers。这样 N=1/2/4 拆分 batch 不改变 BN 数学语义。正式超参数在预检后写入 `frozen_task_configs.json`。

## 固定配置

|任务|模型|数据|global batch|优化器|预算|
|---|---|---|---:|---|---:|
|MNIST|784→64→10，无 BN/Dropout|50,000 train / 10,000 val / 10,000 official test|128|项目现有 Adam，lr=0.001|10 epochs|
|小型道路 CNN（此前预检留档）|3→8 Conv5/2 + Pool2/2 →16 Conv3/2 + Pool2/2 →1120→32→2|8,000 train / 1,000 val / 1,000 test|32|同上|20 epochs|
|ResNet18 道路任务（新增）|现有 ResNet18，base_width=16、ImageNet stem，固定校准 BN 统计量|同一 8,000 / 1,000 / 1,000 划分|128|项目 Adam，lr=0.0003|5 epochs|

ResNet18 已完成 GPU 预检并冻结：单进程预检训练 loss 从 0.095633 降至 0.001757，验证 angle MAE=0.037603，常数参考=0.135207；未访问测试集。完整 batch 每 epoch 使用 7,936 张（62 steps），随机排列后丢弃尾部 64 张，各配置完全相同。预检结束时 CuPy 分配器保留量为 2,884.21 MiB，属于分配器口径，不等同整卡瞬时显存峰值。

按用户统一协议的要求，Ring 的就绪/更新确认消息也已改用 gRPC/protobuf；中央控制屏障仅交换摘要和元数据，不汇总训练梯度。新增双 Worker 回归测试验证没有 Socket 帧。原有数值/恢复档案保留此前控制消息为 Socket 的源码指纹，正式 Ring 训练及新增 ResNet 检查采用更新后的协议路径。

所有配置 seed=0、FP32，CUDA 前反向和 GPU 更新。Adam 保留项目既有首次状态初始化规则，并非替换为其他框架的 Adam。CNN 全部 Conv/Pool 明确使用 `cuda_native_cublas`。PS 仅在 CPU 汇总平均梯度；Ring 在 CPU 对主机梯度执行相邻 gRPC 两阶段规约，每个 Worker 独立在 GPU 更新。

每个任务的矩阵为 single-1、PS JSON 2/4、PS gRPC 2/4、Ring gRPC 2/4。1 Worker 的三种分布式路径另在数值和恢复测试中覆盖。正式 GPU 任务串行运行。

MNIST 每 epoch 随机排列 50,000 个训练索引后取 49,920 个，统一丢弃尾部 80 个；ResNet18 每 epoch 使用 7,936 张道路图像，统一丢弃随机排列后的尾部 64 张；历史小型 CNN 的 batch=32 可整除 8,000 张。不同 Worker 数只是拆分同一个全局 batch。数据预先解码为 uint8 只读映射文件，各 Worker 只转换当前 shard，CPU 数据准备和输入 H2D 仍计入训练耗时。

道路图像根据文件名稳定数值编号，每 100 张为一组。重复解码像素涉及的组会合并后整体划分，避免重复图像跨集合。本数据未发现像素完全重复的图像。编号分组不等于经验证的真实采集 session；数据为 generated-road，不据此宣称真实道路驾驶泛化。throttle 全为 0.5，质量重点报告 angle MAE/RMSE。

## 质量判断

以下是工作区 Spec 的项目目标，不冒充 PDF 中给定的数值：

- MNIST 最终验证 accuracy ≥95%，分布式配置相对单进程准确率差 ≤1 个百分点。
- 道路训练集评估 loss 比初始化下降 ≥30%；验证 angle MAE 比训练集 angle 均值的常数预测低 ≥10%；与单进程差异 ≤max(0.001, 单进程 MAE 的 5%)。
- 超差时保留失败结果，诊断后重新验证，不降低标准。

预检使用训练和验证集，未访问测试集选择配置；MNIST 10 epochs 验证 accuracy 为 96.57%，道路 20 epochs 验证 angle MAE 为 0.036152，常数参考为 0.135207。随后冻结上述配置。正式矩阵分别在训练结束后访问测试集，不用测试结果调参。

## 时间与通信计数

- 启动、模型初始化、数据映射、前反向预热、初始评估、参数分发与训练时间分离。准备屏障仅交换文件就绪标记，不交换梯度或更新参数。
- epoch 从各 Worker 完成准备开始，到最后一步更新和全员状态确认完成；验证、最终保存不计入训练时间。
- `forward/backward/optimizer_gpu_s` 来自 CUDA Event；相应 `*_wall_s` 为主机观测。独立 CUDA 进程之间没有统一 GPU event 时间轴，因此全局 epoch 用本机单调时钟。
- 梯度同步包含编码、RPC、规约及等待，不包含梯度 D2H、平均梯度 H2D、GPU 更新和更新后确认。各阶段逐 Worker 累计，主表取最大值，阶段列不能直接相加。
- 正式性能重复另记录 `sync_update_s`：从反向完成到本步记录入口的独立主机时间戳差，覆盖指标处理、梯度拷贝/同步、更新和全员确认。早期收敛/数值档案未记录该字段时留空，不从不同阶段最大值相加伪造；该补充仅增加计时，不改变训练数学或固定配置。
- Socket 的 `rpc_observed_s` 包含接收与 JSON 解码，`decode_s=0` 是该路径未单独拆分计时的占位值，不表示解码免费；报告中不把它当作测得的独立解码时间。gRPC 的自动序列化/调度也包含在 RPC 观测中。纯线路传输时间未测。
- Socket 记录实际发送的 JSON 正文长度，4 字节长度前缀另计；gRPC 记录实际请求/响应的 protobuf `ByteSize()`。HTTP/2、TCP/IP 头及重传未抓包测量。
- 每个完整性能 epoch 独立启动进程，三次重复并轮换配置顺序，从相同单进程初始化 checkpoint 开始；性能 run 不替代完整收敛训练。性能表固定比较第 1 个 epoch。
- GPU/CPU 约每秒采样一次。GPU 显存为全卡观测，不能按 Worker 重复相加；采样峰值和 CuPy 分配器在结束时保留的内存量是不同口径。

## 正确性与故障

ResNet 追加验收说明：首轮 20 步独立整批轨迹对照在 MBGD / JSON / 双 Worker 第 4 步发生梯度超差；失败轨迹和 `resnet_gradient_diagnosis.json` 保留。重算显示约 4.3e-7 的参数差已伴随 ReLU/MaxPool 分支变化，导致约 1.4e-3 的梯度差；部分固定权重、不同 batch 形状也出现少量分支变化。因此原“独立整批轨迹每步梯度等价”不记为 ResNet 已通过项。

ResNet 另执行独立、可审计的分层检查：共同初始点的整批梯度；每步记录实际局部梯度，使用 FP64 按样本数加权求和检查 PS/Ring 的全局均值及各 rank 一致性；再使用核验后的传输均值，在同一初始/恢复状态上运行现有 CPU 优化器，逐步比较 GPU 参数、v/s、版本和 BN buffers。保留局部/全局梯度、CPU 参考 checkpoint、独立整批轨迹误差；仍使用 atol=1e-4、rtol=1e-3，不放宽后重写原失败。ResNet 非空状态恢复采用同样的归约/更新参考，原三个任务的检验口径不变。

数值验收使用 synthetic、MNIST、道路 CNN，MBGD/Adam，单进程与三种分布式实现 N=1/2/4，连续 20 步比较每个 Worker 的梯度、参数、Adam v/s 和步数。分片包括 20+12、11+9+7+5。FP32 容差 atol=1e-4、rtol=1e-3；保留每步 NPZ，不只比较最终 loss。

恢复验收从非空 Adam 状态和参数版本 2 的 checkpoint 恢复，比较继续更新到版本 4 与未中断参考，覆盖两个真实任务和全部 N=1/2/4 分布式组合。

PS 应用心跳由独立线程发送，服务端 watchdog 检查过期后中止整个同步训练，不悄悄缩减参与者。Ring 通过 RPC deadline、消息等待超时及 Launcher 子进程存活检查检测故障。新 request_id 的重复 PS 梯度被拒绝；同 request_id 的合法重试以及 Ring 同身份重试幂等确认。所有失败保留为失败，并验证五秒内清理，无自动续训声明。

Nsight 追踪另行运行，其耗时不进入性能对比。MNIST MLP 的 GPU 计算主要为 cuBLAS GEMM 与 CuPy 元素/规约 kernel；新增 ResNet18 在 PS/Ring Worker 中另做原生 Conv/Pool 追踪。Nsight Compute 抽取的第一个匹配 kernel 可能来自预热，作为代表 kernel 诊断；Nsight Systems 覆盖完整短训练工作负载。完整原始数据保留在本目录；首次故障测试中将“新 request_id 的重复梯度”误判为应成功的测试脚本错误，也保留在 `fault_checks_attempt-*.json`，修正预期后重跑。
