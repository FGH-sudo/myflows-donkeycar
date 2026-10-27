# CUDA 编程（第 19 讲）落地：数值精度与性能改进

更新时间：2026-10-22。

按《深度学习框架-19 CUDA 编程（上）》课件重新审查 MyFlows 的数值实现与 CUDA 路径。课件要求的几项（线程层次与网格配置、合并访存、共享内存 tile 与 bank 冲突、错误检查、cudaEvent 计时与单独计算传输、与 NumPy 对拍）全部落到代码和实验里，同时修正审查中发现的语义与精度问题。目标是在不牺牲速度的前提下，让结果尽量接近精确值（FP64 参考）。

## 1. 结论

| 项目 | 修改前 | 修改后 | 证据 |
|---|---|---|---|
| `apps/train` 默认配置每步耗时（ResNet18 w64、cifar stem、batch 2） | 958.8 ms（整网 FP64 + CuPy） | 52.0 ms（FP32 + 原生 CUDA），18.4× | [resnet-001](resnet-001/report.md) |
| ResNet18 w16、batch 128、训练模式 BN、原生后端 | 161.2 ms/步 | 54.8 ms/步，2.94× | resnet-001 |
| ResNet18 w64、batch 32、CuPy 后端 | 232.5 ms/步 | 110.6 ms/步，2.10× | resnet-001 |
| CPU（NumPy float32）路径 ResNet18 单步梯度，6 个 batch：出现 ReLU 异号的 batch / 无异号时的梯度中位误差 | 4/6（异号 batch 的梯度误差可达 1e-2）/ 1.1e-5–1.3e-5 | 1/6 / 3.0e-6–3.7e-6（与 GPU 相同） | resnet-001 |
| BN，均值是标准差的 1e5 倍：输出 / dγ 最大相对误差（GPU） | 1.9e-3 / 3.1e-3 | 8.8e-8 / 3.9e-8 | [ops-001](ops-001/report.md) |
| 交叉熵，logits 量级 1e4：loss 相对误差 | 99.9%（被 `log(p + 1e-10)` 截断） | 3.6e-8 | ops-001 |
| 原生卷积 bias 梯度最大相对误差 | 6.1e-7 | 3.7e-8 | ops-001 |
| Adam 与教科书公式的偏差（10 步后） | 5.7e-4（语义错误） | 1.7e-7（舍入级） | ops-001 |
| MNIST MLP 一个 epoch 的测试准确率（Adam 修正前 / 后） | 93.57% | 94.97% | [mlp-001](mlp-001/report.md) |
| 矩阵乘课件数据记录表，n = 1024 / 2048 / 4096 时对 NumPy 的加速比：CUDA C 共享内存版（课件写法） | — | 1.81× / 1.63× / 1.15× | [matmul-001](matmul-001/report.md) |
| 同上：CUDA C 寄存器分块版（课件之外的优化）；CuPy | — | 13.4× / 11.0× / 8.3×；14.2× / 8.7× / 6.8× | matmul-001 |
| 原生卷积的 GEMM 引擎（ResNet18） | cuBLAS | 寄存器分块，为默认（cuBLAS 可选）：每步耗时为 cuBLAS 的 0.96–1.02 倍，卷积 GEMM 合计 1.09–1.10 倍；单步梯度中位误差 3.2e-6–4.0e-6（cuBLAS 3.0e-6–3.9e-6） | [engine-001](engine-001/report.md) |
| 4096³ 矩阵乘的中位耗时与相对 RMS 误差：cuBLAS；寄存器分块 fp32；寄存器分块 kahan；寄存器分块 dot2 | — | 21.3 ms，1.1e-6；19.4 ms，1.1e-6；24.0 ms，1.1e-7；122.1 ms，2.5e-8 | matmul-001 |

GPU 上端到端梯度精度的下限来自卷积 GEMM 的 float32 累加。cuBLAS SGEMM 在 n ≥ 2048 的方阵上与教科书顺序 FMA 循环逐位相同，误差随 K 增长。

课件的三种写法每个线程只算一个输出，受访存限制，最快的也比 cuBLAS 慢 5–7 倍。改成寄存器分块后（每线程 8×8 个输出、float4 读取、共享内存双缓冲），自写 kernel 与 cuBLAS 一样快。它已推广为原生卷积与 `Dense` 的 GEMM 引擎（NN/NT/TN 三种布局、五种 tile、切分 K），并且是默认引擎，cuBLAS 保留为并列选项（5.5 节）。ResNet18 每步耗时与 cuBLAS 相同，梯度误差在同一量级（中位数高约 5%）。

补偿累加（kahan）版把误差降到 1e-7 量级，耗时为 cuBLAS 的 1.0–1.4 倍；按 2026-10-22 的决定，它与课件三版一样只作为实验内容。Triton 扩展不再开发（第 6 节）。

## 2. 修正的问题

| 问题 | 原因 | 修正 | 影响 |
|---|---|---|---|
| Adam 首步步长只有标准值的 0.32 倍 | 首步令 v = g、s = g²（缺少 1 − β 因子）后仍做偏差修正：v̂ = 10g、ŝ = 1000g²；之后 g₁² 在 s 中被放大 1000 倍，上千步才衰减 | 改为 Kingma & Ba 的标准形式；每个参数一次融合 kernel，`__f*_rn` 逐步显式舍入，CPU 回退同序，CPU/GPU 逐位一致 | MNIST 第 50 步 loss 0.904 → 0.451，准确率 93.57% → 94.97%。此前使用 Adam 的实验均基于旧实现，不能与新结果直接比较 |
| `apps/train` 整网 FP64 | `Dense`/`Conv2D`/`BatchNorm2d` 默认 dtype 为 float64，float32 输入与 float64 权重混算为 float64；原生后端只接受 float32，因而回落到 CuPy | 新增 `MyFlows/core/dtype.py`，默认 float32；训练脚本按 `--dtype` 设置；`--conv-backend auto` 在 CUDA + float32 时选原生后端 | 每步 959 → 52 ms，显存池 4925 → 1694 MiB。GeForce 卡的 FP64 吞吐只有 FP32 的 1/64 |
| BN 在 \|mean\| ≫ std 时丢失精度 | 均值以 float32 保存，`x − mean` 的舍入误差可达 \|mean\|/std 个 ulp；方差与 dγ 在 float32 中累加 | 统计量按 x[n=0, c] 平移后在 FP64 中单遍累加；均值存为 hi + lo 两个 float，`(x − hi) − lo` 对接近均值的 x 是精确的（Sterbenz）；反向由 x 重算 x̂，不再保存激活大小的 x̂ | 大均值用例的输出误差 1.9e-3 → 8.8e-8；BN 前向为一次统计 + 一次归一化，反向为一次规约 + 一次 dX |
| 交叉熵截断 | `-log(softmax + 1e-10)`：目标类概率下溢时 loss 停在约 23 | 每行一个 warp 的 FP64 log-sum-exp，softmax、log 与逐行损失在一个 kernel 内完成；越界标签使该行 loss 为 NaN（不在主机端检查，避免同步） | logits 量级 1e4 时 loss 误差 99.9% → 3.6e-8 |
| CPU 路径前向误差偏大，ReLU 处出现异号 | CPU 卷积输出是转置视图（内存按 NHWC 排列），NumPy 沿 (N, H, W) 跨步规约时不走成对求和：float32 方差的相对误差约 8e-6，连续数组只有 1.8e-7。旧 BN 的统计量因此不准，ReLU 输入误差 5.1e-6–6.4e-6，是 GPU 路径的 3–4 倍。离 0 足够近的 ReLU 输入因此与 FP64 异号（seed 1 的 batch 上有 2 个，都在最大值的 1.7e-6 以内），该处局部梯度在 0 与 dy 之间跳变，该 ReLU 之前各层的梯度偏差约 1% | BN 统计量在 FP64 中累加，与内存布局无关 | 6 个 batch 上：ReLU 输入误差 5.1e-6–6.4e-6 → 1.3e-6–1.7e-6，出现异号的 batch 4 → 1 个，无异号时梯度中位误差 1.1e-5–1.3e-5 → 3.0e-6–3.7e-6。前向误差变小只能降低异号的概率：离 0 足够近的输入在任何 float32 实现里都可能异号 |
| bias 梯度精度 | 卷积（原生 / CuPy / NumPy）与 Linear 的 bias 梯度在 float32 中求和 | FP64 累加：原生路径在 dY 转置 kernel 内用 warp shuffle 生成 FP64 部分和，再按固定顺序规约 | 原生 db 6.1e-7 → 3.7e-8；紧接 BN 的卷积 bias（真实梯度为 0）的舍入噪声在原生与 NumPy 路径约降为 1/5 |
| CuPy 卷积反向不可复现 | col2im 用 `xp.add.at`，CuPy 上是原子加，累加顺序随调度变化 | 改为每个输入像素按固定顺序收集窗口的 gather kernel，也省去全零缓冲与 padding 切片 | 结果逐位可复现；CuPy 后端 ResNet18 每步从 232.5 ms 降到 110.6 ms，改 col2im 之前的一次中间测量是 178 ms（该次运行目录未保留），差值主要来自这一项 |
| CUDA 头文件 / NVRTC 版本不匹配 | CuPy 导入时缓存 `CUDA_PATH`：先导入 CuPy 的进程会用系统 CUDA 13 头文件编译新 kernel；`import torch` 会先加载 torch 自带的同名 NVRTC 12.8，与 12.6 头文件配对后 CUB kernel 编译失败。CuPy 的磁盘缓存掩盖了这两个问题 | `device.py` 修正 CuPy 缓存的路径，用 `find_spec` 定位 `torch/lib` 而不导入 torch；原生 DLL 优先加载 `.venv` 内的 NVRTC 12.6 与 cuBLAS | 新增的 CuPy kernel 在任何导入顺序下都能编译 |
| 其他 | ReLU/LeakyReLU 反向用 `astype(float)` / `where(mask, 1.0, alpha)` 生成 float64 掩码，乘法与临时数组都按 FP64 计算（ConvTranspose2D_LeakyReLU 同样）；3 维输入的 `Linear` 反向公式错误；`GlobalAvgPool` 反向先物化广播副本；`MSELoss` 在前向调用 `float()` 同步 GPU；checkpoint 恢复不按模型 dtype 转换；`LogLoss` 用布尔索引（同步）且按行数而非元素数归一化 | 逐项修正，见各模块注释 | ReLU 的 dtype、3 维 `Linear`、`MSELoss` 与 checkpoint 有专门的单元测试，其余由现有测试覆盖 |

## 3. 与第 19 讲对应的实现

| 课件内容 | 落地位置 |
|---|---|
| grid / block / warp 配置 | 矩阵乘 basic 为 1-D grid（每线程一个输出），tiled/shared 为 TILE × TILE 的 2-D block，register 为 256 线程的 block（每线程 8×8 个输出，负责 128×128 的输出块），作为原生层的 GEMM 引擎另有 128×64、256×32、64×128、32×256 四种 tile，切分 K 时 grid 的 z 维是段数；卷积尾处理与 BN 统计按查询到的 SM 数（本机 24）设定 block 数，分别约每 SM 12 个和 8 个 block |
| 合并访存 | Nsight Compute 实测的 L1 全局 load 字节与按 warp 级 32 B sector 推出的解析值逐字节相等（5.1 节） |
| 共享内存 tile 与 bank 冲突 | shared 矩阵乘的访问模式是 `As[ty][k]` 广播与 `Bs[k][tx]` 连续，实测 bank 冲突为 0，不需要 +1 padding；需要 padding 的是按列读的转置，卷积尾处理的 NHWC → NCHW 转置用 32 × 33 tile。register 把 A 转置存放（行尾 +4 个 float），并把每个线程的输出拆成相隔 64 行/列的 4×4 小块，使 warp 读取连续的 float4。每次取 8 层 k 时读写都没有 bank 冲突；fp32 版每次取 16 层，转置写入有 2 路冲突，两种消除办法实测都更慢（5.1 节） |
| warp 级规约 | 卷积 bias 的 FP64 部分和、交叉熵逐行 log-sum-exp 都用 `__shfl_xor_sync` 树形规约，顺序固定 |
| 错误检查 | DLL 的 `MF_CU` / `MF_CUBLAS` 宏返回带源码行号、CUDA/cuBLAS 错误名与原因的信息；`MYFLOWS_NATIVE_SYNC=1` 或 `set_debug_sync(True)` 在每次 launch 后同步，异步错误能定位到 kernel；int32 下标溢出与 CUDA 上下文变化在 Python/C 两侧显式检查 |
| NVRTC 运行时编译 | NVRTC 支持当前架构时直接生成 SASS（cubin），否则生成 PTX；按源码、选项与 NVRTC 版本的哈希缓存（sm_89：加入 GEMM 引擎的 34 个 kernel 后冷编译约 7.5 s，只在首次运行时发生；命中约 3 ms）；`native.info()` 报告目标架构、产物类型、NVRTC 版本与缓存状态 |
| cudaEvent 计时、传输单独计 | `benchmark/matmul_compare.py` 分别测 kernel（event）与 H2D/D2H（可分页与锁页）；训练脚本的 `train_step_ms` 改用 event，每步只在读回 loss 时同步一次 |
| 与 NumPy 对拍 | 8 × 8 用例中全部实现与 NumPy 相差不超过 4.8e-7（满足课件的 1e-5）；更大规模改用相对 FP64 的误差（5.1 节说明原因） |
| 避免隐式同步 | `float()` / `bool()` 读取设备值都会同步：MSE loss 留在设备上，全局梯度范数只在记录时计算，交叉熵标签检查移进 kernel |

## 4. 实验环境与方法

- 硬件：Intel Core i7-14700HX，16 GB 内存；NVIDIA GeForce RTX 4060 Laptop GPU（8 GB，sm_89，24 SM，驱动 591.74）。
- 软件：Windows 11，Python 3.11.7，NumPy 2.2.6（OpenBLAS 0.3.29，24 线程），CuPy 14.0.1，NVRTC 12.6（生成 sm_89 cubin），cuBLAS 12.6.4.1（`CUBLAS_DEFAULT_MATH`，不使用 TF32）。
- 代码：修改前为根仓库 `b56408e` 与 MyFlows `6563dd0`，以 git worktree 检出并用其自身的 `tools/build_native_cuda.py` 编译 DLL；修改后为在这两个提交之上的工作区。各运行目录的 `manifest.json` 记录两个仓库的 HEAD、diff 与源码哈希（JSON/CSV/NPZ 不纳入 Git，提交的是各目录的 `report.md`）。
- 参考值：所有误差都以同一组 float32 输入在 FP64 中的计算结果为参考（Adam 以教科书公式的 FP64 结果为参考），不以 NumPy float32 结果为准——NumPy 自身也有舍入误差。
- 计时：每项单独进程运行；GPU 用 CUDA event 或同步后的 `perf_counter`。笔记本 GPU 的频率会浮动，同一配置两次运行相差可达 10%–15%，小于这个幅度的差异不作结论。

## 5. 结果

### 5.1 矩阵乘（matmul-001）

N(0,1) float32 方阵 C = AB。自写 kernel 为课件的 basic、tiled16/32、shared16/32 三种写法，加上课件之外的寄存器分块版 register；各配三种累加：fp32（顺序 FMA）、kahan（每 G 项 FMA 后用 TwoSum 合并，G 为 32 或 TILE）、dot2（Ogita–Rump–Oishi，相当于两倍工作精度累加后舍入一次）。

课件三版都是一个线程算一个输出，每做一次 FMA 都要从全局或共享内存读一到两个数，受 load/store 单元限制（见下文 Nsight Compute），只用到 GPU FP32 峰值的一小部分；NumPy 调用的则是 24 线程、AVX2 的 OpenBLAS，本机约 0.6–0.86 TFLOP/s。所以课件三版在大规模时与 NumPy 相当，甚至更慢。寄存器分块版按 cuBLAS 一类 GEMM 的做法改写（`native_cublas.cpp` 的 `gemm_register` 模板）：

- 线程块负责 128×128 的输出块，256 个线程各在寄存器里累加 8×8 个输出。每个 k 只从共享内存读 16 个 float（4 个 float4），做 64 次 FMA；课件 shared 版每次 FMA 要读 2 个 float。
- A、B 的 tile 用 float4 从全局内存读入。fp32 版每次取 16 层 k（128×16 与 16×128），即原生层 GEMM 引擎所用的 kernel；kahan/dot2 版取 8 层，因为 dot2 在 16 层时寄存器不够，慢 1.5–3 倍。
- A 转置存放，行尾加 4 个 float 的 padding。8 层时转置写入没有 bank 冲突，16 层时有 2 路冲突（见下文 Nsight Compute）。
- 每个线程的 8×8 输出拆成 2×2 个相隔 64 行/列的 4×4 小块，一个 warp 读到的是连续的 float4。
- 共享内存双缓冲：计算当前 tile 时，下一个 tile 的全局读取已经发出，每个 tile 只需一次 `__syncthreads`。
- 每个输出仍按 k = 0..K−1 的顺序累加，结果与课件三版逐位相同（kahan 每 32 项一组，与 basic、shared32 相同），单元测试逐位比对。K 或 N 不是 4 的倍数、或缓冲区未按 16 字节对齐时，改为逐元素读取。

课件数据记录表（按课件第 72–73 页的口径）：随机种子固定。每个规模测三轮，每轮依次测完全部实现，表中为三轮平均值的中位数。GPU 只计 kernel（CUDA event），传输时间见下文。分块版与共享内存版都用 16×16 的块；四个 CUDA C 版本都是 fp32 顺序累加。误差以 NumPy 的 float32 结果为基线：最大绝对误差为 max\|C − C_NumPy\|，相对误差为 ‖C − C_NumPy‖_F / ‖C_NumPy‖_F。加速比 = NumPy 时间 / 该实现时间。

| 规模 | 实现 | 平均执行时间 ms | 加速比 | 最大绝对误差 | 相对误差 |
|---|---|---|---|---|---|
| 1024 | NumPy（基线） | 3.664 | 1.00× | 0 | 0 |
| 1024 | CuPy | 0.258 | 14.21× | 7.63e-05 | 3.27e-07 |
| 1024 | CUDA C 基础版 | 3.212 | 1.14× | 2.67e-04 | 5.85e-07 |
| 1024 | CUDA C 分块版 | 2.697 | 1.36× | 2.67e-04 | 5.85e-07 |
| 1024 | CUDA C 共享内存版 | 2.021 | 1.81× | 2.67e-04 | 5.85e-07 |
| 1024 | CUDA C 寄存器分块版（优化） | 0.273 | 13.40× | 2.67e-04 | 5.85e-07 |
| 2048 | NumPy（基线） | 24.317 | 1.00× | 0 | 0 |
| 2048 | CuPy | 2.802 | 8.68× | 4.88e-04 | 8.44e-07 |
| 2048 | CUDA C 基础版 | 28.349 | 0.86× | 4.88e-04 | 8.44e-07 |
| 2048 | CUDA C 分块版 | 20.198 | 1.20× | 4.88e-04 | 8.44e-07 |
| 2048 | CUDA C 共享内存版 | 14.905 | 1.63× | 4.88e-04 | 8.44e-07 |
| 2048 | CUDA C 寄存器分块版（优化） | 2.210 | 11.00× | 4.88e-04 | 8.44e-07 |
| 4096 | NumPy（基线） | 162.285 | 1.00× | 0 | 0 |
| 4096 | CuPy | 23.860 | 6.80× | 1.01e-03 | 1.18e-06 |
| 4096 | CUDA C 基础版 | 328.848 | 0.49× | 1.01e-03 | 1.18e-06 |
| 4096 | CUDA C 分块版 | 185.067 | 0.88× | 1.01e-03 | 1.18e-06 |
| 4096 | CUDA C 共享内存版 | 140.989 | 1.15× | 1.01e-03 | 1.18e-06 |
| 4096 | CUDA C 寄存器分块版（优化） | 19.482 | 8.33× | 1.01e-03 | 1.18e-06 |

- 四个 CUDA C 版本都按 k 的顺序做 FMA，结果逐位相同，所以误差列一样；n ≥ 2048 时 CuPy（cuBLAS）也与它们逐位相同。这些误差有相当一部分来自 NumPy 自己：NumPy 对 FP64 的相对 RMS 误差约 3e-7（见下表）。
- 课件三版中最快的共享内存版也只比 NumPy 快 1.15–1.81 倍。寄存器分块版比 NumPy 快 8.3–13.4 倍，比课件共享内存版快 6.7–7.4 倍；n ≥ 2048 时比 CuPy 快约 20%，n = 1024 时慢约 6%。
- 加速比只能看量级。本机四次完整运行中，NumPy（OpenBLAS，24 线程）在 n = 1024 时的中位耗时为 6.7、3.7、3.7、3.7 ms。GPU 各实现在最近两次运行之间相差可达 20%，例如 cuBLAS 在 n = 4096 时为 25.9 与 21.3 ms。另外，测量时后台有别的程序在用 GPU，空闲时利用率约 25%。n = 1024 时课件三版的平均值比中位数高 2%–6%（单次 event 计时有长尾），寄存器分块版与 CuPy 只高 1%–2%。

补偿累加的对照（n = 4096，全部调用的中位耗时；误差相对 FP64）：

| n = 4096 | 耗时 ms | GFLOP/s | 相对 RMS 误差 | 正确舍入比例 | 与 NumPy 最大差 |
|---|---|---|---|---|---|
| NumPy（CPU，OpenBLAS） | 162.8 | 844 | 3.23e-7 | 8.45% | 0 |
| cuBLAS | 21.3 | 6447 | 1.14e-6 | 2.63% | 1.0e-3 |
| basic-fp32 | 294.3 | 467 | 1.14e-6 | 2.63% | 1.0e-3 |
| tiled32-fp32 | 170.4 | 807 | 1.14e-6 | 2.63% | 1.0e-3 |
| shared32-fp32 | 132.3 | 1039 | 1.14e-6 | 2.63% | 1.0e-3 |
| shared32-kahan | 133.3 | 1031 | 1.06e-7 | 24.0% | 1.5e-4 |
| shared16-kahan | 143.5 | 958 | 7.80e-8 | 31.1% | 1.5e-4 |
| shared32-dot2 | 187.8 | 732 | 2.54e-8 | 99.99% | 1.4e-4 |
| register-fp32 | 19.4 | 7074 | 1.14e-6 | 2.63% | 1.0e-3 |
| register-kahan | 24.0 | 5727 | 1.06e-7 | 24.0% | 1.5e-4 |
| register-dot2 | 122.1 | 1125 | 2.54e-8 | 99.99% | 1.4e-4 |

- cuBLAS 在 n = 2048 与 4096 时与自写的顺序 FMA kernel 逐位相同（另行核对），误差随 K 增长，在 K = 4096 时比 NumPy 大 3.5 倍；n = 1024 时 cuBLAS 换用了不同的规约顺序，误差 2.2e-7，优于顺序累加的 5.7e-7。
- kahan 的误差与 K 无关（n = 1024、2048、4096 均为 1.06e-7）。在访存/LSU 受限的 shared32 里，kahan 与 fp32 的耗时差在 −9%–+1% 之间，即测量噪声以内；dot2 达到舍入下限（正确舍入率 99.99%–100%），开销 25%–42%（n = 1024–4096）。
- 寄存器分块版不再受访存限制，补偿累加的额外运算就显现出来。
  - kahan：比 fp32 慢 24%–48%，耗时为 cuBLAS 的 1.0–1.4 倍，误差约为 cuBLAS 的 1/11（n = 4096）。fp32 版改为每次取 16 层 k 后，这个差距比 8 层时的 20%–30% 大。
  - dot2：每项约 10 次浮点运算，比 fp32 慢 6.3–7.4 倍，仍比 shared32-dot2 快 1.3–1.5 倍。
- 课件的 8 × 8 对拍：全部实现与 NumPy 相差 ≤ 4.8e-7。但“绝对误差 < 1e-5”不适用于大规模：n = 4096 时 \|C\| 可达约 300，float32 在该量级的 ulp 约 3e-5，连正确舍入的 dot2 结果与 FP64 的最大差也有 1.5e-5；dot2 与 NumPy 相差 1.4e-4，几乎全是 NumPy 自身的误差（其相对 FP64 最大误差 1.5e-4）。因此课件表按课件口径以 NumPy 为基线，其余各表报告相对 FP64 的误差与正确舍入比例。
- 速度：cuBLAS 峰值 8.3 TFLOP/s（n = 1024）；寄存器分块版 8.0–8.3 TFLOP/s（n = 1024–2048），n = 4096 时 7.1 TFLOP/s；课件 shared 版 0.97–1.15 TFLOP/s（n = 512–4096），tiled 0.74–0.89，basic 0.47–0.83。
- 传输：
  - 带宽（n ≥ 512）：H2D 可分页 6.3–9.0 GB/s、锁页 9.5–12.6 GB/s；D2H 可分页只有 2.9–4.7 GB/s（n = 4096 时 64 MB 需 16.7 ms），锁页 10.7–13.0 GB/s。
  - 计入传输后，cuBLAS 与寄存器分块版都从 n = 256 起快于 NumPy。n = 4096 时含传输分别为 57.2 与 55.3 ms，传输时间约为 kernel 的 1.7–1.9 倍。
  - 课件 shared32 含传输在 n = 256–512 时快于 NumPy；n ≥ 1024 时与 NumPy 相差不到 10%，互有快慢。

Nsight Compute（n = 2048，基础频率、每次 replay 前清缓存）：

| kernel | 时间 ms | L1 全局 load | 解析值 | L2 流量 | DRAM | SM 吞吐 | 占用率 | bank 冲突 |
|---|---|---|---|---|---|---|---|---|
| cuBLAS | 2.5 | 1.07 GB | — | 1.07 GB | 65 MB | 72% | 32% | 3.5 万 |
| basic-fp32 | 26.1 | 42.95 GB | 5n³ = 42.95 GB | 34.7 GB | 193 MB | 91% | 99% | 0 |
| tiled16-fp32 | 27.3 | 34.36 GB | 4n³ = 34.36 GB | 4.6 GB | 201 MB | 97% | 100% | 0 |
| tiled32-fp32 | 28.0 | 42.95 GB | 5n³ = 42.95 GB | 2.5 GB | 213 MB | 96% | 65% | 0 |
| shared16-fp32 | 21.5 | 4.30 GB | 8n³/16 = 4.30 GB | 4.6 GB | 185 MB | 96% | 100% | 0 |
| shared32-fp32 | 22.6 | 2.15 GB | 8n³/32 = 2.15 GB | 2.5 GB | 186 MB | 83% | 67% | 0 |
| register-fp32 | 2.3 | 0.54 GB | n³/16 = 0.54 GB | 0.55 GB | 47 MB | 75% | 32% | 2.3 万 |
| register-kahan | 3.2 | 0.54 GB | n³/16 = 0.54 GB | 0.56 GB | 45 MB | 68% | 17% | 0 |

- 解析值按 warp 计：basic 与 tiled32 的一个 warp 覆盖同一行的 32 个连续输出，每个 k 读 4 个 B sector 与 1 个广播的 A sector，共 160 B；tiled16 的 warp 是 2 行 × 16 列，B 的 2 个 sector 由两个半 warp 共用；shared 的每个操作数元素每个 block 只读一次；register 的每个 block 把 A 的 128 行、B 的 128 列各读一遍，float4 读取完全合并，共 4n³(1/128 + 1/128) = n³/16。实测与解析值逐字节相等，说明访存完全按预期合并。
- tile 的作用体现在 L2 流量：basic 的 L1 命中率约 19%，tiled32 约 94%；shared 把 L1 请求量降为 1/TILE，register 再降到 shared32 的 1/4。
- DRAM 流量：课件三版为 185–213 MB，register 只有 45–47 MB，与 50 MB 的必需流量（3n² × 4 B）相当。略低于必需值，可能是因为 kernel 结束时部分 C 还在 L2 中，没有计入 DRAM 写回。另测一次时，kahan 版只写回 8.9 MB，不到 C 的 16.8 MB。
- 课件三版的 DRAM 流量和 ncu 下的 kernel 时间在各次运行间变化较大，例如 tiled32-fp32 四次运行为 162 / 614 / 552 / 213 MB、24.0 / 37.3 / 34.1 / 28.0 ms。L1 全局 load 字节则逐字节不变。
- 课件三版的 SM 吞吐为 83%–97%，瓶颈是 load/store 单元：每次 FMA 都要读一到两次内存，没有寄存器分块，最快的 shared 版也比 cuBLAS 慢 5–7 倍（中位耗时，n = 1024–4096）。register 每 64 次 FMA 只读 16 个 float，ncu 下比 cuBLAS 还快（2.3 ms 对 2.5 ms）。kahan 版用满 255 个寄存器，每个 SM 只能驻留 1 个 block（占用率 17%），耗时 3.2 ms。
- TILE = 32 时 block 有 1024 个线程，sm_89 每个 SM 最多 1536 个线程，只能驻留 1 个 block，占用率 66.7%；TILE = 16 与 basic 接近 100%。register 的 block 只有 256 个线程、每线程 127 个寄存器，每个 SM 驻留 2 个 block，占用率约 32%；它靠每个线程内 64 个互不依赖的 FMA 掩盖延迟，不靠占用率。
- bank 冲突：表中只计共享内存读的冲突。fp32 版改为每次取 16 层 k 后，A 的转置写入出现 2 路冲突。
  - 原因：一个 warp 的 32 个 lane 写 8 行，每行 4 段 k（0、4、8、12）；共享 tile 每行比 32 的倍数长 4 个 bank，k 与 k + 8 落在同一 bank。
  - 实测：ncu 计得写冲突 229 万次（共 262 万条写指令）。读冲突的 2.3 万次也随写冲突出现，写冲突消除后为 0。每次取 8 层的 kahan 版，读写冲突都约为 0。
  - 试过两种消除办法，比较 6 个 GEMM 形状在 ncu 下的周期数：
    - 让两个 lane 合读一行（每行 32 字节）：冲突降为 0，但每个 warp 的全局读要跨两倍的缓存行。128×128 tile 的 NN/NT kernel 周期数增加 8%–19%。
    - 保持读法，让 k + 8 与 k + 12 的 lane 按另一顺序写：冲突降为原来的 0–10%，但每次写多了选择与地址运算。4 个形状的周期数增加 5%–25%，1 个减少 15%，TN 不受影响（没有转置写入）。
  - 结论：每个 k tile 每个线程只写 8 次共享内存，却要做 1024 次 FMA，写冲突的代价小于消除它的额外开销，所以保留原写法。消除 bank 冲突要以实测为准。

### 5.2 算子精度（ops-001）

同一组 float32 输入分别经过修改前和修改后的代码，误差为 max\|y − y64\| / max\|y64\|。

| 输出 | 修改前 | 修改后 |
|---|---|---|
| BN（大均值）y，GPU / CPU | 1.93e-3 / 1.93e-3 | 8.77e-8 / 6.82e-8 |
| BN（大均值）dγ，GPU / CPU | 3.06e-3 / 8.29e-3 | 3.88e-8 / 3.88e-8 |
| BN（大均值）dx，GPU / CPU | 1.61e-4 / 1.69e-4 | 9.30e-8 / 9.30e-8 |
| BN（N(0,1)）dγ / dβ，GPU | 9.00e-8 / 7.79e-8 | 3.02e-8 / 1.34e-8 |
| 交叉熵 loss（logits × 1e4） | 9.99e-1 | 3.59e-8 |
| 交叉熵 loss（N(0,1)），CPU | 6.43e-8 | 1.30e-8 |
| 卷积 db：原生 / CuPy / NumPy | 6.06e-7 / 1.02e-7 / 1.74e-7 | 3.65e-8（三者相同） |
| 卷积 y、dx、dW | 1.4e-7–3.6e-7 | 不变（GEMM 未改） |
| Adam 第 1 / 10 步（对教科书公式） | 2.22e-4 / 5.70e-4 | 3.86e-8 / 1.65e-7 |

CuPy 卷积的 dx 前后分别为 1.41e-7 与 1.50e-7：修改前的 col2im 用原子加，每次运行的累加顺序不同，这一项本来就在 1.4e-7–1.7e-7 之间波动；修改后结果固定。logits × 1e4 时 softmax 在 float32 中恰为 0/1，两版的梯度都与参考一致。

### 5.3 ResNet18 端到端（resnet-001）

本节的“原生”是 `cuda_native_cublas`：修改前的代码只有这一种 GEMM 引擎，前后对照时保持引擎不变。寄存器分块引擎与它的对照见 5.5 节。

单步梯度：ResNet18 w16、imagenet stem、8 × 3 × 120 × 160，训练模式 BN，相同 float32 参数，对 FP64 NumPy 参考；batch 由 6 个种子分别生成（`--seeds`，默认 1–6）。82 个参数中有 20 个是紧接 BN 的卷积 bias，真实梯度恰为 0（训练模式 BN 减去了通道均值），单独报告其舍入噪声（以其余参数 ‖g64‖ 的中位数为单位）；其余 62 个统计相对误差 ‖g − g64‖₂/‖g64‖₂。

离 0 足够近的 ReLU 输入在 float32 中可能落到 0 的另一侧（下称异号）。该元素的局部梯度于是在 0 与 dy 之间跳变，该 ReLU 之前（靠近输入一侧）各层的参数梯度偏差 1e-3–1e-2。这是 ReLU 的导数在 0 处不连续造成的，与累加精度无关，所以下表按有无异号分开统计，各项为 6 个 batch 的范围：

| 配置 | 有异号的 batch | 无异号 batch 的梯度中位误差 | 无异号 batch 的最大误差 | ReLU 输入误差 | 零梯度 bias 的噪声 |
|---|---|---|---|---|---|
| NumPy float32，修改前 | 4/6（共 6 个异号输入） | 1.1e-5–1.3e-5 | 4.6e-5 | 5.1e-6–6.4e-6 | 3.2e-6–6.2e-6 |
| NumPy float32，修改后 | 1/6（1 个） | 3.0e-6–3.7e-6 | 5.3e-6–9.5e-6 | 1.3e-6–1.7e-6 | 7.9e-7–1.2e-6 |
| CuPy，修改前 | 0/6 | 3.0e-6–3.7e-6 | 5.0e-6–7.9e-6 | 1.4e-6–1.8e-6 | 8.2e-7–1.5e-6 |
| CuPy，修改后 | 0/6 | 3.0e-6–3.9e-6 | 4.9e-6–7.4e-6 | 1.4e-6–1.7e-6 | 6.9e-7–1.2e-6 |
| 原生，修改前 | 0/6 | 3.0e-6–3.9e-6 | 4.9e-6–8.0e-6 | 1.5e-6–1.8e-6 | 3.1e-6–5.9e-6 |
| 原生，修改后 | 0/6 | 3.0e-6–3.9e-6 | 4.9e-6–7.2e-6 | 1.4e-6–1.7e-6 | 6.7e-7–1.1e-6 |

- 修改前的 CPU 路径：CPU 卷积输出是转置视图（内存按 NHWC 排列），NumPy 沿 (N, H, W) 跨步规约时不走成对求和，float32 方差的相对误差约 8e-6（连续数组为 1.8e-7）。BN 统计量因此不准，ReLU 输入误差是 GPU 路径的 3–4 倍：6 个 batch 中 4 个出现异号（seed 1 上 2 个，梯度中位误差 9.9e-3）；没有异号的 batch，梯度中位误差也有 1.1e-5–1.3e-5。
- 修改后的 CPU 路径：BN 统计量在 FP64 中累加，与内存布局无关，ReLU 输入误差与无异号时的梯度误差都与 GPU 相同。seed 2 仍有 1 个异号（梯度最大误差 2.7e-3，中位数不受影响）：前向误差变小只能降低异号的概率，离 0 足够近的输入在任何 float32 实现里都可能异号。
- GPU 两条路径：中位误差前后都在 3.0e-6–3.9e-6，来自卷积 GEMM（cuBLAS，K 为数百到数千）的 float32 舍入。单个 batch 的最大值前后互有高低（原生路径 seed 1 上 6.4e-6 → 7.2e-6，seed 4 上 8.0e-6 → 6.1e-6），没有系统性变化；只看 seed 1 会误以为修改后变差。
- 离 loss 最近的 fc 层，梯度误差 CPU 1.1e-6–1.7e-6 → 2.5e-7–3.9e-7，GPU 3.1e-7–5.7e-7 → 3.0e-7–4.3e-7。零梯度 bias 的噪声在 NumPy 与原生路径约降为 1/5（bias 梯度改为 FP64 累加，第 2 节），CuPy 路径修改前已在 1e-6 量级。

每步耗时（主机 batch 拷到 GPU、前向、反向、Adam、读回 loss；20 步中位数）：

| 配置 | 修改前 ms | 修改后 ms | 加速 | 显存池 MiB（前 → 后） |
|---|---|---|---|---|
| w16、batch 128、冻结 BN、原生 | 118.9 | 54.7 | 2.17× | 2805 → 2611 |
| w16、batch 128、训练 BN、原生 | 161.2 | 54.8 | 2.94× | 2805 → 2611 |
| w64、batch 32、训练 BN、原生 | 117.8 | 58.2 | 2.02× | 2392 → 2214 |
| w64、batch 32、训练 BN、CuPy | 232.5 | 110.6 | 2.10× | 2600 → 2459 |
| `apps/train` 默认：cifar stem、w64、batch 2 | 958.8（FP64） | 52.0 | 18.4× | 4925 → 1694 |

原生路径的提升来自：卷积尾处理与 dY 转置合并 bias、激活与 bias 梯度（去掉了单独的 bias 规约、ReLU 与转置 kernel）；输入不求梯度时跳过 stem 卷积的 dX；融合 BN（训练模式比冻结 BN 多出的 42 ms 降为 0.1 ms）；Adam 每参数一个 kernel；去掉 loss 读取与梯度范数造成的中途同步。

### 5.4 MNIST MLP（mlp-001）

MLP 784-256-10，batch 128，Adam lr 1e-3，相同初始化与 batch 顺序。

| 后端 | dW1 误差 | dW2 误差 | ms/步 | 测试准确率 | 与 FP64 轨迹的最大 loss 差 |
|---|---|---|---|---|---|
| NumPy float32 | 1.04e-7 | 1.63e-7 | 5.08 | 94.97% | 1.4e-7 |
| CuPy / 原生 cuBLAS | 7.78e-8 | 9.36e-8 | 1.67 / 1.80 | 94.97% | 1.1e-7 |
| 原生 kahan | 5.80e-8 | 7.06e-8 | 1.64 | 94.97% | 1.3e-7 |
| 原生 dot2 | 3.10e-8 | 3.05e-8 | 1.53 | 94.97% | 1.3e-7 |

- GEMM 精度能从梯度上看出来（dot2 已到舍入下限），但不改变一个 epoch 的训练结果：所有后端的逐步 loss 与 FP64 轨迹相差不超过 1.4e-7，epoch 结束时参数的相对距离都约 1.5e-6——这部分由 float32 参数更新的舍入决定，与 GEMM 精度无关。
- 这个规模的 GPU 训练受 kernel 发射开销限制（约 1.5–1.8 ms/步），补偿累加没有可测的代价。
- Adam 修正前后：第 10、50、100、390 步的 loss 分别为 2.24 → 1.82、0.904 → 0.451、0.460 → 0.308、0.247 → 0.182。

### 5.5 寄存器分块 GEMM 接入原生卷积（engine-001）

5.1 节的寄存器分块 kernel 推广为原生层的 GEMM 引擎，与 cuBLAS 并列，由后端名选择：

- `cuda_native_register`：GEMM 用寄存器分块 kernel。训练入口 `apps/train --conv-backend auto`（CUDA + float32 时）、分布式 donkey 任务（CUDA 上）和控制台都默认用它。
- `cuda_native_cublas`：GEMM 调用 cuBLAS。

两者的 im2col/col2im、尾处理、池化与 bias 梯度完全相同，只有 GEMM 不同。库层 `backend="auto"` 仍按设备选 NumPy/CuPy。

- 布局：卷积的三个 GEMM 是前向 rows = cols·Wᵀ（NT）、dW = dYᵀ·cols（TN）与 dX = dY·W（NN）。一个模板覆盖这三种布局：转置的操作数沿其连续维度用 float4 读入，再转置写进按 k 排列的共享内存 tile（行尾 +4 个 float）。不再需要先把操作数拷成连续数组。
- 每个 tile 取 16 层 k（BK = 16）。BK = 8 时，NT 前向每行每步只读 32 字节，受访存限制。
- 五种 tile，都是 256 个线程：
  - 128×128，每线程 8×8；
  - 128×64 与 256×32，用于窄输出（N ≤ 64、N ≤ 32）；
  - 64×128 与 32×256，用于矮输出（M ≤ 64、M ≤ 32）。
  
  M ≤ 128 时用 128×64。ResNet18 的 CO 只有 16–512，128×128 的 tile 会有一半以上落在矩阵外。
- 切分 K（split-K）：输出不到每 SM 3 个 tile 且 K ≥ M + N 时，把 K 切成不少于 64 的若干段（最多 32 段），使每 SM 约有 6 个 block。各段写出部分和，再由一个 kernel 按固定顺序在 FP64 中求和、舍入一次。实际用上它的是 dW：K 是 batch 的像素数，可达 61 万。前向 GEMM 的 K 只有数十到数千，切分后部分和的读写比省下的计算更费时，实测慢 1.6–2.2 倍，所以加了 K ≥ M + N 的条件。
- 可复现：不切分时每个输出按 k = 0..K−1 顺序累加，与课件 basic kernel 逐位相同。单元测试覆盖五种 tile × 三种布局 × 对齐/非对齐。切分时顺序也固定，重复运行逐位相同。
- 课件其余实现（basic/tiled/shared 与 kahan/dot2 累加）不接入训练入口与控制台，只作为 5.1、5.4 节的实验内容。`Dense` 的 `cuda_native_kahan` / `cuda_native_dot2` 保留给 mlp-001。

两个引擎在当前工作区交替运行（engine-001）。GEMM 一栏取 ResNet18（imagenet stem，3×120×160）一次训练步中原生卷积的全部 GEMM，同形状只测一次，按出现次数加权；每步耗时的配置同 5.3 节；梯度为 5.3 节同样的 6 个 batch。

| 项目 | cuBLAS | 寄存器分块 | 寄存器/cuBLAS |
|---|---|---|---|
| w16、batch 128：卷积 GEMM 前向 / dW / dX / 合计 ms | 5.35 / 5.97 / 3.19 / 14.51 | 5.37 / 7.20 / 3.33 / 15.90 | 1.00 / 1.21 / 1.04 / 1.10 |
| w64、batch 32：同上 | 7.51 / 6.27 / 6.87 / 20.66 | 8.84 / 7.32 / 6.45 / 22.60 | 1.18 / 1.17 / 0.94 / 1.09 |
| 每步耗时 ms：w16 冻结 BN / w16 训练 BN / w64 训练 BN / `apps/train` 默认 | 50.0 / 53.4 / 52.4 / 47.2 | 50.7 / 51.4 / 53.2 / 47.6 | 1.01 / 0.96 / 1.02 / 1.01 |
| 单步梯度中位误差（6 个 batch，均无 ReLU 异号） | 3.0e-6–3.9e-6 | 3.2e-6–4.0e-6 | 约 1.05 |
| 单步梯度最大误差 | 4.9e-6–7.2e-6 | 5.4e-6–9.3e-6 | |
| ReLU 输入误差 | 1.4e-6–1.7e-6 | 1.4e-6–1.8e-6 | |

- 速度：单看 GEMM，寄存器分块合计为 cuBLAS 的 1.09–1.10 倍。
  - dX（NN）持平或更快。
  - 前向 NT 在 w64 上慢 18%，dW 慢 17%–21%。
  - 差距最大的是 dW 的极小输出。例如 32×16×38400：最小的 tile 也有 8192 个输出，94% 落在矩阵外。

  GEMM 只占一步的一部分，每步耗时的比值为 0.96–1.02，在第 4 节所说的频率浮动以内，可以认为相同。显存池大小相同或略小（`apps/train` 默认配置 1610 对 1694 MiB）。
- 切分 K：64 个 GEMM 形状中有 21 个切分，其中 19 个是 dW，另 2 个是 K ≥ M + N 的 layer4 前向（640×512×4608 与 640×512×2304）。
- 精度：
  - 前向：22 个形状中 10 个与 cuBLAS 误差相同，此时 cuBLAS 也是顺序累加。另外 12 个比 cuBLAS 大，最多 6.3 倍（2560×128×576：1.6e-6 对 2.6e-7）：cuBLAS 对这些窄输出把 K 分段累加。
  - dW：22 个形状中 10 个比 cuBLAS 准，因为切分 K 后在 FP64 中求和，误差最低为 cuBLAS 的 0.38 倍；9 个更差。
  - dX：20 个形状中 15 个与 cuBLAS 相同。
  - 端到端：单步梯度中位误差比 cuBLAS 高约 5%，在同一量级。要在前向上追平 cuBLAS 的精度，需要在 block 内把 K 分段（sliced-K），或者用补偿累加；两者都要更多寄存器，或改变 kernel 结构（第 7 节）。

## 6. Triton：决定不做

2026-10-22 决定不再开发 Triton 扩展，GPU 部分以 CUDA C 的结果为准，理由如下：

- 原定的首个目标已用 CUDA C 完成：块级补偿 FP32 GEMM，即接近 cuBLAS 的速度加 kahan 级精度，就是寄存器分块版（5.1 节）。其 fp32 版本又推广成了原生卷积与 `Dense` 的 GEMM 引擎（5.5 节）。
- Windows 上只有社区维护的 `triton-windows`，版本须与 torch 2.11.0+cu128 绑定。接入还要处理两件事：torch 自带的 NVRTC 12.8 与 CuPy 所用 12.6 同名冲突（第 2 节），以及 torch 与 CuPy 两套显存分配器。维护成本与收益不相称。
- 其余候选不依赖 Triton。BN 归一化 + ReLU（+ 残差加）的融合可以用 CuPy `ElementwiseKernel` 完成。隐式 GEMM 卷积不物化 im2col 缓冲（cifar stem、batch 16 时 layer1 每个卷积的 cols 约 708 MB），可以在现有寄存器分块模板上改写 A 操作数的读取，留作后续工作。

## 7. 限制

- GPU 端到端梯度误差约 4e-6，由卷积 GEMM 的 float32 累加决定。补偿累加只用于 `Linear` 的实验后端与矩阵乘实验，没有接入卷积。
- 寄存器分块引擎在窄输出的前向 GEMM 上比 cuBLAS 误差大（最多 6.3 倍），因为每个输出顺序累加整个 K。要改善，需要 block 内分段累加 K（sliced-K）或补偿累加。它在极小输出的 dW 上也比 cuBLAS 慢（最小 tile 有 8192 个输出）。tile 与切分的选择规则是在本机（24 SM）的 ResNet18 卷积形状上定的，没有自动调优。
- 原生 DLL 的 kernel 源码加入 GEMM 引擎后，NVRTC 冷编译从约 0.8 s 增加到约 7.5 s。冷编译只在首次运行，或源码、NVRTC 版本变化时发生；之后命中磁盘缓存，约 3 ms。
- 计算图每步为所有节点分配并清零梯度再 `+=` 累加，比首次写入直接赋值多一次全张量读写。
- `apps/train` 在 `--num-workers 0` 时 batch 2 的数据读取每步 10–40 ms，与计算时间相当。
- ResNet18 每步耗时取 20 步的中位数，没有给出置信区间。本报告前后两次完整运行，各配置相差不超过 2.5%，但第 4 节所说的频率浮动仍可能造成更大的差异；CuPy 显存池的大小与分配顺序有关，两次运行最多相差 7%。
- ReLU 异号无法靠提高累加精度消除，只能降低其概率。用 FP64 参考评估 float32 梯度时，应像 resnet-001 那样在多个 batch 上统计并单列异号，否则单个 batch 的结果可能由一次异号决定。

## 8. 复现

```powershell
# 在仓库根目录执行。修改前代码：两个仓库的上一提交各检出一个 worktree，并编译其自身的 DLL
$Py = (Resolve-Path .\.venv\Scripts\python.exe).Path
git worktree add --detach <old> b56408e
git -C MyFlows worktree add --detach <old>\MyFlows 6563dd0
Push-Location <old>; & $Py tools\build_native_cuda.py; Pop-Location

$D = "docs\experiments\semester_2026_fall\cuda_lecture19"
& $Py -X utf8 -m benchmark.matmul_compare --out-dir $D\matmul-001 --ncu --ncu-size 2048
& $Py -X utf8 -m benchmark.ops_before_after --old <old> --out-dir $D\ops-001
& $Py -X utf8 -m benchmark.resnet_before_after --old <old> --out-dir $D\resnet-001
& $Py -X utf8 -m benchmark.mlp_backends --out-dir $D\mlp-001
& $Py -X utf8 -m benchmark.gemm_engines --out-dir $D\engine-001
& $Py -X utf8 -m tools.run_tests --scope all
```

`<old>` 的 DLL 须在该 worktree 根目录下编译，用当前 `.venv` 的 Python 即可。统一测试入口在修改后共 265 项，全部通过。
