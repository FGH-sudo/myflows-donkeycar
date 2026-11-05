# E2 权重、转向分档分类与 M6 归档

这组数据来自 2026 年 10 月的既有运行，保留原来使用的冻结等级、样本与预算。权重转换检查用 4 张相同道路图像，对齐 224×224 RGB、/255 和 ImageNet 归一化。

| 架构 | 载入张量 | 最大绝对误差 | Top-5 相同 | 转换权重 SHA-256 |
| --- | --- | --- | --- | --- |
| alexnet | 16/16 | 4.77e-06 | True | `d91ada8fd7eb7a75659e45345ffdf80f3b635db411dd392e43778f11fc3a2552` |
| mobilenet_v2 | 262/262 | 1.29e-05 | True | `a0d164d74a180f3bf00d3281d23945bfb85a659f42fcf63ae4b5c9efac8b0af0` |
| resnet18 | 102/102 | 8.58e-06 | True | `ed6abcb7256afa3dcf148797f35f91c801565530e4b75a87dd9f0a011a13d85c` |

分类任务在训练帧上确定转向分档边界，以各自固定的时间段划分做验证和测试。

| 架构 | 冻结段 | train / val / test | epoch | 初始→末轮验证准确率 | 测试准确率 | 冻结指纹相同 |
| --- | --- | --- | --- | --- | --- | --- |
| alexnet | 3 | 2085 / 400 / 400 | 2 | 0.198 → 0.980 | 0.9525 | True |
| mobilenet_v2 | 3 | 2085 / 400 / 400 | 2 | 0.302 → 0.960 | 0.9175 | True |
| resnet18 | 2 | 2085 / 400 / 400 | 2 | 0.265 → 0.935 | 0.9200 | True |

M6 小规模 campaign 在带 ImageNet 权重的 ResNet18 上搜索冻结等级。800 帧按 650/150 划分，每个 trial 为 1 轮，5 个 trial 共约 55 秒。人工基线两个 seed 的验证 MAE 均值为 0.20166，最优筛选候选 t003 冻结 3 段、MAE 为 0.21099，因此在筛选阶段结束。正式 DonkeyCar 迁移对照采用第三阶段的独立协议。

本地原始文件与 SHA-256：

- `runs/pretrained/converted/alexnet.report.json` `c93a62692581131c6ab3f9549f375ef610ba8a21662757d4edc435f81bc67fd0`
- `runs/pretrained/converted/mobilenet_v2.report.json` `053c62df61fc9a09d90acb12509a9b176544e915fd1eb056f2bb57afcf3a55d8`
- `runs/pretrained/converted/resnet18.report.json` `39c40b53ce46111bec2a6b96d3d3be974b6fad1c7b95c659cfacf2e2b1d72850`
- `runs/transfer/alexnet-f3-20261026-225302/result.json` `dcb32257ae558771ae851607f0cd0a94736b0335e9c7353063d1e6df2d56c41e`
- `runs/transfer/mobilenet_v2-f3-20261026-225110/result.json` `b03064297d3cf1e9643754ec0d257b8d03ffe6f2f5592126d80949efe17ee0c9`
- `runs/transfer/resnet18-f2-20261024-013155/result.json` `6e1dcb11cba936111b37a05814a749a1c3c590c54ce716349e216b82565fdf60`
- `runs/autopilot-m6demo/autopilot/m6-freeze-demo/report/report.md` `d52b91435e75bef981312c9a62b7f169b53844c3212ea4855f400692d7ad53ef`
