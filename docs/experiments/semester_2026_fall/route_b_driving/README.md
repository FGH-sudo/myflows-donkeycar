# Route B 实验报告：Udacity 自动驾驶闭环突破（2026-11-24）

本报告记录 Route B（端到端驾驶算法、训练配方、统一预处理链路与视觉定位鲁棒性改造）在已有 36,000 帧聚合数据集 `udacity-lake-aggregate-001` 与真实 Udacity 模拟器（Term 1 Lake 赛道，12 mph / 360 s 长时闭环）上的可控消融与闭环验收结果。

## 1. 环境与版本来源

- **代码提交**：父仓库 `42b3677`（含本报告新增提交），`MyFlows` 子模块 `8fe67e0`
- **硬件与环境**：Windows 11，RTX 4060 Laptop GPU（8 GB），Python 3.11（`.venv`），原生 CUDA 后端 `cuda_native_register`
- **数据集**：`runs/datasets/udacity-lake-aggregate-001`（36,000 帧，合并自 `udacity-lake-001`、`udacity-lake-recovery-001`、`udacity-lake-recovery-002`）
- **固定划分（保留原来源 val/test 归属与边界间隔）**：`runs/pipelines/udacity-spatial-002/splits/aggregate-preserved-holdouts.json`
  - SHA-256：`30866b8c02f0753081e53610781b1d051373c21608a7bbb9fc3f748177f6e2f4`
  - 训练 `27,483` / 验证 `3,600` / 独立测试 `3,595`（测试集仅在所有训练结束后用于最终离线评估，不参与训练和选模）
- **本地未入库运行目录**：`runs/pipelines/udacity-route-b-001/`

## 2. 核心改动摘要

1. **全链路统一 ROI 裁剪（消除 train/serve 偏移）**：
   - 训练使用 `--roi-crop 56,14 --image-size 64x160`（对 `160×320` 原始画面在 `resize` 前裁去顶部 56 行天空/远景和底部 14 行车头，保留道路视野区域并输出 `64×160` NCHW 张量）。
   - ROI 参数写入 ONNX `metadata_props["myflows.roi_crop"]`，由 `OnnxPredictor`、`apps.eval.eval_myflows_donkey_onnx`、`apps.serve.serve_fastapi`、`apps.sim.udacity_drive` 和 `mycar.myflows_pilot` 自动读取并在 `resize` 前应用。
2. **Route B 训练配方**：
   - **学习率调度**：`CosineWarmupScheduler`（`warmup_epochs=1.0`，`lr 1e-3 -> 1e-5`）
   - **驾驶控制加权 Huber 损失**：`WeightedControlLoss(base="huber", delta=0.15, angle_weight=1.0, throttle_weight=0.05, turn_weight=6.0)`，弱化油门噪声梯度并按 $|angle|$ 加权弯道样本
   - **选模判据**：`--best-metric val_angle_mae`，`weight_decay=1e-4`，轻量 dropout（`w16: 0.05` / `w24: 0.1`）
   - **驾驶专属增强**（`--augment-preset driving`）：`RandomHorizontalShift`（经 `VisionExpert` 实测转向灵敏度校准 `steer_per_frac=1.2`）+ `RandomShadow` + `ColorJitter`
3. **视觉定位防误报（`apps/collect/vision.py`）**：
   - 为 `RoadFinder` 增加纯几何双峰 2 点共识回退 `_consensus_fit`（配合 `curvature_prior=50x` 抑制自由曲率振荡）。
   - 在已存档真实长片段 `runs/pipelines/udacity-spatial-002/drive-probe-12mph/images`（4,205 帧）回放验证中，保持在路时段 `[0..2100]` 的 `found=2058/2100`，最大连续漏检从改造前的长段误报降为 `10` 帧，离开路面时段 `[4100..4205]` 保持 `0/105` 误检，`first_road_lost=2307`（与真实驶离路面帧一致）。

## 3. 离线可控消融与独立测试结果（3,595 条未见测试样本）

| 模型配置 | 输入尺寸 / ROI | Epoch | 验证 Angle MAE | 验证 Loss | 测试 Angle MAE | 测试 Angle MSE | 转向符号准确率 (近零) | 预测/真值平均 $\|angle\|$ | 训练墙钟时间 | 峰值显存 (CuPy pool) | ONNX SHA-256（前 12 位） |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 基线 `udacity-spatial-002`（`ResNet18 w16 spatial`） | `160×80` / 无 | 6 | 0.033218 | 0.001689 | 0.036455 | 0.002346 | 0.7274 (0.9231) | 0.0565 / 0.0623 | 572.4 s | 1222.7 MiB | — |
| `w16-recipe`（同宽度 `w16` + Route B 配方） | `160×64` / `56,14` | 10 | 0.015215 | 0.000441 | 0.018809 | 0.000687 | 0.8851 (1.0000) | 0.0562 / 0.0623 | 958.8 s | 1951.0 MiB | `610ddb5bc4a9` |
| `w24-recipe`（扩容 `w24` + Route B 配方） | `160×64` / `56,14` | 12 | **0.011928** | **0.000372** | **0.012645** | **0.000299** | **0.9344 (1.0000)** | **0.0580 / 0.0623** | 1502.9 s | 2915.6 MiB | `af67fc14eba5` |

- 相同 `base_width=16` 容量下，仅靠 ROI 裁剪与 Route B 配方，`test angle_mae` 下降 **48.4%**（`0.036455 -> 0.018809`），`test angle_mse` 下降 **70.7%**，`angle_sign_accuracy` 提升 **+15.8 个百分点**。
- `base_width=24`（`w24-recipe`）进一步将 `test angle_mae` 降至 **`0.012645`**（较基线降低 **65.3%**），`angle_sign_accuracy` 达到 **`93.44%`**（近零直行样本达到 `100.0%`），吞吐仍保持 `233.2 samples/s`、峰值显存仅 `2.92 GiB`。
- 两次离线 ONNX 评估均未手动传 `--roi-crop`，评估日志直接确认从 ONNX 元数据自动读取 `roi_crop=56,14`。

## 4. 真实 Udacity 模拟器闭环驾驶验证（Lake 赛道，12 mph）

验收协议与基线完全一致：标准 30 帧丢路规则、`excursions == 0`、标定后有效 CTE 覆盖率 `>= 90%`、跑满目标时长方返回通过（退出码 `0`）。

| 测试项 / 模型 | 视觉估计版本 | 目标时长 | 结果 (`passed`) | 终止原因 | 存活时间 | 速度积分距离 | 平均 / 最大绝对 CTE 估计 | 越界次数 | CTE 覆盖率 | 帧数 | 推理延迟 P50 / P95 |
| --- | --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 基线 `spatial-002` | 修复前 | 360 s | **失败** (`false`) | `road_lost` | 100.39 s | 535.87 m | 0.0826 / 0.4752 | 0 | 95.96% | 2,000 | 2.30 / 4.91 ms |
| 基线 `spatial-002` | 修复后（本版） | 360 s | **失败** (`false`) | `road_lost` | 116.14 s | 621.30 m | 0.0890 / 1.9220 | 1 | 97.61% | 2,319 | 1.76 / 3.14 ms |
| `w16-recipe` | 修复后（本版） | 360 s | **通过** (`true`) | `time_limit` | **360.03 s** | **1,929.55 m** | 0.2130 / 0.5204 | **0** | 98.76% | 7,196 | 1.45 / 3.89 ms |
| `w24-recipe` | 修复后（本版） | 360 s | **通过** (`true`) | `time_limit` | **360.04 s** | **1,929.49 m** | **0.1361** / 0.5938 | **0** | **99.53%** | 7,196 | 1.74 / 2.82 ms |
| `w24-recipe` + FastAPI 服务（`127.0.0.1:8012`） | 修复后（本版） | 60 s | **通过** (`true`) | `time_limit` | **60.03 s** | **320.36 m** | 0.0465 / 0.1626 | **0** | **99.91%** | 1,197 | 7.29 / 18.37 ms |

关键现象验证：
1. 在修复后的 `vision.py` 下重测基线 `spatial-002`，系统平稳通过此前在 `100.39 s / 535.9 m` 触发误报的桥出口路段，直到 `116.14 s / 621.30 m` 基线模型真实驶出路面（`excursions=1, max_abs_cte=1.922`）才判 `road_lost`，交叉印证了评价器修复准确且未放宽真偏离检测。
2. **`w16-recipe` 与 `w24-recipe` 均在标准协议下完整跑满 360 秒（约 1.93 km，覆盖 Lake 多圈）**，`0` 越界、`0` 卡死、`0` 丢路；`w24-recipe` 将平均绝对横向误差进一步从 `0.2130` 降至 `0.1361`（CTE 覆盖率 `99.53%`），且 FastAPI 实时 HTTP 推理（`/model_info` 自动识别 `"roi_crop": "56,14"`）同样无越界通过闭环验证。

## 5. 局限与说明

- 本报告中的 CTE 与行驶距离均由 `RoadFinder` 视觉测量及车速积分计算，Udacity Term 1 自动模式不回传模拟器真实坐标或官方圈数遥测。
- 验证集中在 Lake 赛道（标准 12 mph 协议），Jungle 赛道未在本轮训练与测试范围内。

## 6. 可复现命令

```powershell
# 1. 训练 w16-recipe（w24-recipe 仅调整 --base-width 24 --epochs 12 --dropout 0.1 及输出目录）
.\.venv\Scripts\python.exe -X utf8 -m apps.train.train_myflows_donkey `
  --data runs/datasets/udacity-lake-aggregate-001 `
  --split-file runs/pipelines/udacity-spatial-002/splits/aggregate-preserved-holdouts.json `
  --arch resnet18 --regression-head spatial --base-width 16 --stem cifar `
  --roi-crop 56,14 --image-size 64x160 `
  --batch 32 --epochs 10 --lr 0.001 --lr-schedule cosine --warmup-epochs 1.0 --min-lr 1e-5 `
  --weight-decay 1e-4 --dropout 0.05 `
  --loss huber --huber-delta 0.15 --angle-weight 1.0 --throttle-weight 0.05 --turn-weight 6.0 `
  --best-metric val_angle_mae --augment-preset driving `
  --seed 23 --device cuda --conv-backend cuda_native_register --num-workers 4 `
  --no-tensorboard --checkpoint-every 0 --no-compress-checkpoints --export-onnx `
  --run-dir runs/pipelines/udacity-route-b-001/w16-recipe/train `
  --log-dir runs/pipelines/udacity-route-b-001/w16-recipe/train/logs `
  --out runs/pipelines/udacity-route-b-001/w16-recipe/train/checkpoints/checkpoint `
  --best-out runs/pipelines/udacity-route-b-001/w16-recipe/train/checkpoints/best `
  --stop-file runs/pipelines/udacity-route-b-001/w16-recipe/train/STOP_TRAINING

# 2. 独立测试集 ONNX 评估（自动从 ONNX 元数据读取 roi_crop=56,14）
.\.venv\Scripts\python.exe -X utf8 -m apps.eval.eval_myflows_donkey_onnx `
  --data runs/datasets/udacity-lake-aggregate-001 `
  --checkpoint runs/pipelines/udacity-route-b-001/w24-recipe/train/checkpoints/best.onnx `
  --split-file runs/pipelines/udacity-spatial-002/splits/aggregate-preserved-holdouts.json `
  --split test --max-samples 0 --device cuda `
  --report-json runs/pipelines/udacity-route-b-001/w24-recipe/test_onnx_eval.json

# 3. 真实 Udacity 360s @ 12 mph 闭环驾驶
.\.venv\Scripts\python.exe -X utf8 -m apps.sim.udacity_drive `
  --model runs/pipelines/udacity-route-b-001/w24-recipe/train/checkpoints/best.onnx `
  --seconds 360 --speed 12 --camera-every 20 --device cuda `
  --out runs/pipelines/udacity-route-b-001/drive-w24-recipe-12mph
```
