# formal-full-002 正式调优记录

全量 10,000 帧按时间段固定划分。训练 7,680 帧，验证和测试各 1,000 帧，边界隔离 320 帧。ResNet18 使用 cifar stem、w64、FP32、原生 register 后端，每次训练 3 轮。原报告与 trial 证据保存在 `runs/autopilot/formal-full-002/`。

人工基线为 lr=0.001、batch=2、loader workers=0。种子 1–3 的验证角度 MAE 为 0.03436 ± 0.01525。候选 t005 将 lr 调为 0.0001，在共同确认种子 2、3、4 上的配对差值为 +0.00033 ± 0.02665，验证结果支持保留人工基线。

| 确认 seed | 基线验证 MAE | 候选验证 MAE | 候选减基线 |
| --- | --- | --- | --- |
| 2 | 0.01971 | 0.04905 | +0.02935 |
| 3 | 0.05014 | 0.02709 | -0.02305 |
| 4 | 0.02366 | 0.01837 | -0.00529 |

独立测试集评估保留每个 checkpoint 的原始值。按共同确认种子比较，人工基线使用 t002、t003、t013，候选使用 t010、t011、t012。

| 配置 | 测试 MAE 各次 | mean ± sample std |
| --- | --- | --- |
| 人工基线 | 0.02211, 0.06732, 0.02874 | 0.03939 ± 0.02441 |
| 候选 | 0.05177, 0.02941, 0.02170 | 0.03429 ± 0.01562 |

若包含筛选 seed 1，四个 seed 的测试 MAE 均值为基线 0.03690、候选 0.03180，对应原报告的全种子口径。

本次 campaign 执行 13 个 trial，逐次异常、排队和资源数据见原始报告。结论依据验证集配对差值，测试数据单独列出。

可核对的原始文件：

- `runs/autopilot/formal-full-002/report/report.md` SHA-256 `c4bf606c76a15e1680b9eb493a9cb10b8ab3294e890953737cacb2e8ea65a381`
- `runs/autopilot/formal-full-002/report/confirmation.json` SHA-256 `3097531bcf5541cceff8a397996d064834f19c42bf7cffef3db43ef3c1f3171a`
- `runs/autopilot/formal-full-002/report/test-eval/evaluation.json` SHA-256 `c7b1521d93c132b2d64f2c4a6482522415506202b2ebbcdce702c5ce032e3d54`
