# 无人值守采集与两个模拟器的闭环（2026-11-16–17）

本报告记录两件事：为 DonkeySim 和 Udacity 模拟器（term 1，即 [car-behavioral-cloning-paddle](https://github.com/jm12138/car-behavioral-cloning-paddle) 使用的环境）各写一个不需要人操作的数据采集脚本；在采集到的数据上跑通训练、导出、闭环评估、经推理服务驾驶和 AutoPilot campaign。设计说明见 [system_design.md 第 11 节](../../../system_design.md#11-第二个模拟器与无人值守采集2026-11-16)。

- 代码：根仓库 210e572 之后的工作树（随本报告提交），MyFlows 70113a0。
- 机器：Windows 11、RTX 4060 Laptop GPU（8 GB），用户的桌面程序同时占用约 30%–40% 的 GPU。
- 原始数据在本机：`runs/datasets/<id>`（采集）、`runs/pipelines/<id>`（划分、训练、驾驶），闭环评估归档在 `docs/experiments/semester_2026_fall/closed_loop/<id>`（控制台“运行对比 → 闭环评估”可查看）。仓库只提交各处的 report.md 与本文件。

## 1. 结论

两个模拟器的采集、数据集选择、训练、导出和闭环评估已接通。DonkeySim 自动采集的 4,000 帧能训练出在默认短场景中接近现有驾驶模型的模型；Udacity 的 16,000 帧训练出 ResNet18，验证转向 MAE 0.02560、独立测试 0.04671，实际导出的 ONNX 在同一测试集上复核为 0.04671。Udacity 本地 ONNX 与 FastAPI 各完成 4 个 60 秒场景，均未触发失败或出界规则。

较长的驾驶暴露了短场景没有覆盖的问题：12 mph 的 FastAPI 驾驶在 86.2 秒、约 436 米处卡住。因此这些短场景只证明该范围内的驾驶与部署链路可用，不能据此声称整圈稳定。长时间驾驶结果见第 3.6 节。新数据集上的真实 GPU AutoPilot 联调也已结束（5 个有效 trial，含确认与独立测试）；LLM 未启用，结果不支持候选配置优于基线的结论。

## 2. DonkeySim

### 2.1 采集

```
python -m apps.collect.donkey_auto --name donkey-auto-001 --frames 4000
```

专家是遥测 CTE 的 PD 控制器（kp 0.5、kd 0.35），固定油门 0.2；每段先用 `regen_road` 按种子换一条路，每 2–5 秒加一次 3–8 帧、幅度 0.3–0.8 的转向扰动，标签始终是专家输出。共 4 段、4000 帧，用时约 3.5 分钟，没有一段以失败结束。

| 段 | 种子 | 结束 | 秒 | 保存帧 | 平均 / 最大绝对 CTE |
| --- | --- | --- | --- | --- | --- |
| 0 | 11 | 到时 | 60.0 | 1190 | 0.261 / 0.860 |
| 1 | 12 | 到时 | 60.0 | 1191 | 0.235 / 0.759 |
| 2 | 13 | 到时 | 60.0 | 1191 | 0.214 / 0.953 |
| 3 | 14 | 帧数达标 | 21.9 | 428 | 0.222 / 0.704 |

6.9% 的帧带扰动。标签左转 32.1%、右转 42.4%、直行 25.5%，饱和 0.1%。

### 2.2 训练与闭环

时间块划分（验证 400、测试 400，块长 50、间隔 5，contract 2），ResNet18（cifar stem，120×160）batch 2、lr 1e-3、3 个 epoch，最佳验证转向 MAE 0.0722（第 1 个 epoch）。导出的 ONNX 与现有驾驶模型（`mycar/models/myflow_resnet18_best.onnx`，在 `mycar/data` 上训练）在同一协议下配对评估：4 个默认 reset 场景、每个 30 秒、固定油门 0.2。归档为 [closed_loop/donkey-auto-001](../closed_loop/donkey-auto-001/report.md)。

| 模型 | 碰撞 | 平均存活秒 | 平均距离 | 平均绝对 CTE | 最大绝对 CTE | 出界 |
| --- | --- | --- | --- | --- | --- | --- |
| current-pilot | 0 / 4 | 30.04 | 251.7 | 0.170 | 0.397 | 0 |
| auto-001 | 0 / 4 | 30.04 | 252.1 | 0.188 | 0.394 | 0 |

只用 4,000 帧自动采集的数据、训练 5 分钟，模型的闭环表现与人工数据训练的驾驶模型相当：配对差 Δ平均绝对 CTE +0.018 ± 0.001，两者都没有出界或碰撞。

## 3. Udacity 模拟器

### 3.1 接入

模拟器是 Socket.IO 1.x 客户端，python-socketio 5 无法与它握手，所以 `apps/sim/udacity.py` 直接用 websockets 实现 Engine.IO 3 服务端。实测事实：遥测字段都是字符串，`steering_angle` 单位是度（= 25 × 命令），图像为 320×160 JPEG；控制命令里的数值也必须是字符串；模拟器每收到一条命令回一帧，不加节拍时本机每秒约 270 帧，所以命令按 20 Hz 发送。自动模式只回传中心相机和车速，没有位置、CTE、碰撞和重置消息。

启动器（`apps/sim/udacity_launcher.py`）：Unity 配置对话框是普通 Windows 对话框，按控件 ID 设置分辨率（640×480）、画质（Fastest，没有路面阴影）和窗口模式后按 Play；命令行的 `-screen-*` 参数不会预选对话框。菜单按钮是 Unity UI，只投递点击消息无效，需要把光标移到按钮上（自动模式按钮在客户区的 (0.5, 0.63)）；投递 ESC 可回到菜单，即重启场景。窗口被最小化时，第一次真实采集在重启时因为点不到菜单而中断，现在启动器先还原窗口，采集脚本在模拟器无法重启时也会保留已存的帧。

### 3.2 视觉专家

专家只看图像（`apps/collect/vision.py`）。在真实画面上调试时确定的事实：

- 车道的消失行是第 73 行，不是天际线（约第 62 行）；车头只占底部约 10 行。
- 沥青的颜色（Lab 的 a/b）很稳定，但 JPEG 留下小于 8 的色度块噪声；左侧黄线太淡，检测不到，所以行驶线以右边缘为参照，间距取起步时的值。
- 过桥时右边缘从车道线跳到桥墙（斜率约 1.6 → 3.4 像素/行），出桥时跳回；跳变须连续两帧一致才接受，桥上行驶线保持不变，出桥后参照间距缓慢回到起步值。
- 模拟器把车放到路面上：每次进入场景的第一帧里，所有边缘斜率都只有落稳后的约 2/3（7 次会话的第一帧右边缘斜率为 1.61–1.64，第 10 帧为 2.19–2.44）。起步参照因此跳过头 10 帧，取之后 12 帧的中位数。

### 3.3 采集

```
python -m apps.collect.udacity_auto --name udacity-lake-001 --frames 16000 --episode-s 270 --speed 11 --seed 7
```

最终一次采集用时约 8 分钟：3 段、16000 帧（每帧另存左右镜像样本，转向取反），5.9% 的帧带扰动，四项自检全部通过。

| 段 | 结束 | 秒 | 保存帧 | 平均 / 最大横向位置估计 | 平均车速 mph |
| --- | --- | --- | --- | --- | --- |
| 0 | 丢路（桥口） | 88.8 | 3356 | 0.065 / 0.280 | 11.02 |
| 1 | 到时（约 1.2 圈） | 273.0 | 10528 | 0.065 / 0.301 | 10.99 |
| 2 | 帧数达标 | 57.0 | 2116 | 0.066 / 0.288 | 10.94 |

标签均值 0、标准差 0.087，左右转各 22.2%，没有饱和。第 0 段在进桥约 0.3 秒后连续 1 秒读不出路面，场景自动重启；第 1 段完整通过了桥。

此前的两次采集暴露了两个问题，均已修正，数据未使用：

| 尝试 | 起步参照 | 现象 |
| --- | --- | --- |
| 第一次 | 第一帧 | 参照来自落车前的画面，车整段比起步位置更靠右（右边缘斜率 1.63 对 2.40）；两段都跑完，但标签教的是贴右行驶 |
| 第二次 | 头 12 帧的中位数 | 落车的帧仍占一半，参照约 2.09；一次扰动恰在出桥时把车推向左侧，丢路；随后窗口被最小化，重启失败（修正前的启动器） |

之后扰动只在连续 40 帧稳定读路后施加，桥上与边缘切换时不加。

### 3.4 训练

16,000 条样本含镜像样本。时间块划分使用 contract 2（块长 50、间隔 5、seed 42）：训练 12,240、验证 1,600、测试 1,600，另剔除边界附近 560 条。ResNet18（cifar stem、宽度 64）把 160×320 相机图像缩到 80×160，batch 4、lr 1e-3、seed 1，原生寄存器分块 CUDA 后端，训练 3 个 epoch，约 11 分钟。测试集在训练和选取最佳 epoch 时不参与计算。

| epoch | 训练 loss | 验证 loss | 验证转向 MAE |
| --- | --- | --- | --- |
| 1 | 0.042880 | 0.004276 | 0.050997 |
| 2 | 0.005267 | 0.001594 | 0.038223 |
| 3（最佳） | 0.003213 | 0.000769 | 0.025597 |

最佳 checkpoint 在 1,600 条独立测试样本上的转向 MAE 为 0.046705。2026-11-17 又用实际导出的 `best.onnx` 经 ONNX Runtime CPU 复核相同测试集，MAE 仍为 0.046705（报告 `closed_loop/udacity-001/udacity-onnx-offline.json`，包含 ONNX 和划分文件的 SHA-256）。审查发现离线 ONNX 评估原来按原图尺寸构造输入，无法评估缩小输入尺寸的导出；现已按每个模型声明的空间尺寸与固定 batch 评估，末批补齐的样本不计入指标。

### 3.5 闭环评估：本地 ONNX 与 FastAPI 服务配对

归档：[closed_loop/udacity-001](../closed_loop/udacity-001/report.md)。4 个场景分别为 9、12、12、15 mph，各运行 60 秒，场景之间通过菜单重启，模型顺序轮换。服务使用同一个 `best.onnx`，输入为 80×160，GPU 推理；HTTP 客户端以 PNG 发送收到的相机帧。

| 模型 | 到时 / 失败 | 平均存活秒 | 平均距离（速度积分，m） | 平均绝对 CTE 估计 | 最差最大绝对 CTE 估计 | 出界 | 全帧推理 P50 / P95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| udacity-onnx | 4 / 0 | 60.02 | 320.7 | 0.05663 | 0.23781 | 0 | 5.19 / 10.99 |
| udacity-fastapi | 4 / 0 | 60.04 | 321.1 | 0.05778 | 0.23688 | 0 | 14.63 / 28.36 |

FastAPI 相对本地 ONNX 的逐场景配对差为 Δ平均绝对 CTE +0.00115 ± 0.01071，均存活到时。延迟包含客户端编码与网络调用，服务端自身的推理耗时另记在服务日志中。

此归档在 11 月 17 日审查前产生，包含起步标定期间按定义为零的横向位置。审查后的运行把标定帧作为缺测，排除在 CTE 统计之外，并拒绝缺失或非有限的车速。`--sim remote` 无法保证每次回到相同起点，因此 Udacity 配对评估拒绝这种连接方式；远程连接仍可用于采集与单次驾驶。

### 3.6 经推理服务的长片段驾驶

2026-11-17 在真实模拟器上追加了三次长片段。FastAPI 与 gRPC 都服务同一个最佳 ONNX，车辆按模型的转向行驶，油门由目标车速控制器给出。

| 服务 | 目标 mph | 计划时长 s | 结束 | 实际 s | 距离估计 m | 平均 / 最大绝对 CTE 估计 | 推理 P50 / P95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FastAPI | 12 | 240 | stuck | 86.16 | 436.37 | 0.08000 / 0.64382 | 12.43 / 29.13 |
| FastAPI | 9 | 320 | stuck | 112.73 | 436.16 | 0.08284 / 0.67276 | 11.11 / 27.14 |
| gRPC | 12 | 120 | stuck | 86.18 | 436.40 | 0.07983 / 0.64745 | 6.23 / 10.79 |

三次均在约 436 米处卡住，车速连续低于 0.5 mph 达 4 秒后结束，驾驶命令返回退出码 2。逐帧记录与 summary.json 分别保存在 `runs/pipelines/udacity-lake-001/drive-fastapi-lap`、`drive-fastapi-9mph`、`drive-grpc-12mph`。这些距离是车速积分，不能当作精确地图坐标。

两条部署链路都能连续接收相机帧、返回模型预测并记录终止原因，当前模型的整圈驾驶能力没有通过。降低车速仍在相近距离失败，后续应补充该路段及偏离后的恢复样本，重新训练后使用更长片段验收。当前默认 60 秒场景属于短片段评估。

## 4. AutoPilot campaign

真实 campaign 为 `runs/autopilot/udacity-dev-001`：使用 Udacity 数据集的前 4,000 条，时间块划分训练 3,065、验证 400、测试 400、剔除 135 条，80×160 输入，1 个 epoch。搜索 lr ∈ {0.0003, 0.001, 0.003}，batch 固定 4；筛选 seed 为 1，确认 seed 为 2，最多 6 个 trial，LLM 关闭，启用 GPU Monitor。

2026-11-17 通过控制台服务启动，最终为 `completed / done`，5 次训练均得到 valid verdict，经过 baseline → explore → exploit → confirm → done，报告和最终独立测试正常产生，无重试。

| trial | 用途 | lr | seed | 验证转向 MAE |
| --- | --- | --- | --- | --- |
| t001 | 基线 | 0.001 | 1 | 0.072353 |
| t002 | 探索 | 0.0003 | 1 | 0.160554 |
| t003 | 探索 | 0.003 | 1 | 0.055008 |
| t004 | 候选确认 | 0.003 | 2 | 0.049732 |
| t005 | 基线确认 | 0.001 | 2 | 0.033642 |

确认种子上候选比基线的 MAE 高 0.01609。只有一个筛选种子与一个确认种子，报告明确给出“无法判断（缺少基线噪声水平）”。最终独立测试的基线均值为 0.052419，候选为 0.060255；测试集只在最后评估，不参与排名。GPU Monitor 标出了部分运行受功耗或温度限制，计时不用于性能比较。

这是功能联调。运行期间审查改动涉及评估、控制台和测试文件，报告记录了 t003–t005 的代码哈希变化；训练计算路径没有改动，但本批结果仍不作为正式调参结论。原始证据在 `runs/autopilot/udacity-dev-001/report/`、`trials/` 与 `campaign.json`。

可复现配置如下；在控制台 API 创建后再通过 `/api/autopilot/campaigns/<id>/start` 显式启动：

```json
{
  "campaign_id": "udacity-dev-002",
  "mode": "dev",
  "baseline": {"lr": 0.001, "batch": 4, "num_workers": 0, "seeds": [1]},
  "fixed": {"data": "udacity-lake-001", "image_size": "80x160", "max_samples": 4000,
    "epochs": 1, "val_size": 400, "test_size": 400, "split_kind": "time_block", "block_size": 50, "split_seed": 42},
  "space": {"lr": [0.0003, 0.001, 0.003], "batch": [4]},
  "strategy": {"name": "candidate_list", "max_candidates": 3, "seed": 0},
  "budget": {"max_trials": 6, "trial_timeout_s": 900, "wall_time_s": 3600, "max_retries": 1},
  "confirm": {"top_k": 1, "seeds": [2]},
  "specialists": ["gpu_monitor"],
  "llm": {"enabled": false}
}
```

## 5. 限制

- Udacity 的横向位置与距离都不是模拟器真值：横向位置来自视觉专家对图像的估计（与专家共用同一把“尺子”），距离由车速积分。
- 视觉专家只在湖区赛道上调过；丛林赛道没有验证。专家偶尔在桥口读不出路面，这一段以丢路结束（采集时场景自动重启；闭环评估的丢路判定为连续 30 帧，即 1.5 秒）。
- 自动启动只支持 Windows，点击菜单时光标会移动约半秒。
- 每个闭环场景都从同一个起点出发，种子只用于记录；Udacity 场景之间的差别只有目标车速。
- 数据来自同一湖区赛道的少数片段及其镜像；按时间块留出的测试集不等于新赛道或独立采集会话的泛化评估。12 mph 的长时间驾驶已经出现短场景未捕获的失败。

## 6. 复现

在仓库根目录执行（Windows；Udacity 模拟器解压到 `UdacitySimWin/`，见 [development_environment.md](../../../development_environment.md#udacity-模拟器)）。没有模拟器时把采集与驾驶命令的 `--sim` 设为 `fake`，即可在替身上走通同一流程。

```
# 采集（会自行启动并关闭模拟器）
python -m apps.collect.donkey_auto --name donkey-auto-001 --frames 4000
python -m apps.collect.udacity_auto --name udacity-lake-001 --frames 16000 --episode-s 270 --speed 11 --seed 7

# 时间块划分（验证 1600、测试 1600）
python -c "from pathlib import Path; from apps.autopilot.data import ensure_split; print(ensure_split(Path('.'), Path('runs/pipelines/udacity-lake-001/splits'), max_samples=0, val_size=1600, test_size=1600, split_seed=42, kind='time_block', data='runs/datasets/udacity-lake-001')['split_file'])"

# 训练并导出 ONNX
python -m apps.train --data runs/datasets/udacity-lake-001 --image-size 80x160 --epochs 3 --batch 4 --lr 0.001 --seed 1 --device cuda --conv-backend cuda_native_register --split-file runs/pipelines/udacity-lake-001/splits/udacity-lake-001-all-val1600-test1600-tb50g5c2-s42.json --no-tensorboard --checkpoint-every 0 --export-onnx --run-dir runs/pipelines/udacity-lake-001/train --out runs/pipelines/udacity-lake-001/train/checkpoints/checkpoint --best-out runs/pipelines/udacity-lake-001/train/checkpoints/best

# 服务在单独的终端中运行；评估和驾驶命令在另一个终端执行
# models.json 中一项写 "path"，一项写 "endpoint": "http://127.0.0.1:8000"
python -m apps.serve.serve_fastapi --model runs/pipelines/udacity-lake-001/train/checkpoints/best.onnx --image-w 160 --image-h 80 --port 8000 --device cuda
python -m apps.eval.closed_loop --models runs/pipelines/udacity-lake-001/models.json --scenarios apps/eval/scenarios/udacity_lake.json --out docs/experiments/semester_2026_fall/closed_loop/udacity-001 --device cuda

# 长片段驾驶：当前模型会提前卡住，退出码 2，结果保存在输出目录
python -m apps.sim.udacity_drive --endpoint http://127.0.0.1:8000 --seconds 240 --speed 12 --out runs/pipelines/udacity-lake-001/drive-fastapi-lap

# gRPC 服务也在单独的终端启动
python -m apps.serve.serve_grpc --model runs/pipelines/udacity-lake-001/train/checkpoints/best.onnx --image-w 160 --image-h 80 --port 50051 --device cuda
python -m apps.sim.udacity_drive --endpoint grpc://127.0.0.1:50051 --seconds 120 --speed 12 --out runs/pipelines/udacity-lake-001/drive-grpc-12mph
```

AutoPilot campaign 在控制台“AutoPilot → 新建”中选数据集 `udacity-lake-001` 创建，或把第 4 节的 spec POST 到 `/api/autopilot/campaigns` 后启动。

## 7. 审查与验证（2026-11-17）

- Windows 最终统一回归：610 / 610 通过，0 跳过，559.0 秒（框架 330、应用 280）。接手时的版本另有一次 604 项通过；修复后新增的回归用例涵盖总时限、奇数镜像目标、空数据集、远程配对评估、标定帧、非有限车速和 ONNX 输入尺寸。
- WSL Ubuntu 22.04：610 项收集，539 通过、71 因缺少 CUDA、torch 等可选环境跳过，237.7 秒。使用包含来源清单的源码导出，复用已有 CPU 测试环境并补装锁定版本的 websockets；依赖检查通过。
- 前端类型检查与生产构建通过；lint 无错误，保留 8 条已有警告。新增与修改行的 Python F/E9 检查没有新问题，存量文件仍有 4 处既有诊断。
- Playwright 检查了新建训练、新建 campaign 的数据集选择和 80×160 自动填入；闭环页的模型表、配对差、逐场景车速、Udacity 缺少轨迹的说明，以及 DonkeySim 的俯视轨迹均有截图。390 px 视口的新建页无横向溢出，浏览器控制台无错误。截图与验证日志在 `runs/review-20261117/`。
- 临时控制台、推理服务与模拟器已退出，采集数据和所有实验产物保留。本轮改动尚未推送，托管 CI 未运行；GPU 算子的性能门槛未重跑。
