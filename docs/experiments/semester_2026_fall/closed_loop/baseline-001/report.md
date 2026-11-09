# DonkeyCar 配对闭环评估

每个场景使用相同的默认 reset、客户端种子、终止规则、时限和 0.2 油门。模型顺序按场景轮换。位置距离使用连续有效 3D 遥测点之间的欧氏距离。

| 模型 | 次数 | 碰撞次数 | 平均 / 最差存活秒 | 平均 / 最差距离 | 平均 / 最差绝对 CTE | 平均 / 最差最大绝对 CTE | 平均 / 最差出界数 | 全部帧 P50 / P95 推理 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current-pilot | 10 | 0 | 30.022 / 30.003 | 250.776 / 248.064 | 0.545 / 0.546 | 0.885 / 0.892 | 0.000 / 0.000 | 12.383 / 18.156 |
| manual-t002 | 10 | 0 | 30.037 / 30.000 | 249.968 / 247.969 | 0.554 / 0.555 | 0.925 / 0.927 | 0.000 / 0.000 | 16.869 / 20.267 |
| candidate-t012 | 10 | 0 | 30.020 / 30.000 | 250.881 / 250.498 | 0.567 / 0.569 | 0.892 / 0.899 | 0.000 / 0.000 | 16.317 / 20.074 |

配对差值采用模型值减去 manual-t002 在同一场景的值。

| 模型 | 离线角度 MAE | 平均差 Δ存活秒 | Δ距离 | Δ平均绝对 CTE | Δ出界数 |
| --- | --- | --- | --- | --- | --- |
| current-pilot | 0.03200 | -0.016 | 0.808 | -0.010 | 0.000 |
| manual-t002 | 0.02211 | 0.000 | 0.000 | 0.000 | 0.000 |
| candidate-t012 | 0.02170 | -0.017 | 0.912 | 0.013 | 0.000 |

manifest.json 记录规则、场景、模型来源和 SHA-256。frames.jsonl 保存逐帧遥测，episodes.jsonl 保存各次 reset 起点与结局，result.json 保存平均、最差和配对差异。Unity 场景随机种子和任意起点坐标无法通过当前接口指定。

按预设清单取 current-pilot 的前五次配对 reset，平均 / 最差存活时间为 30.015 / 30.003 s，距离为 250.406 / 248.064，平均绝对 CTE 为 0.545 / 0.546。五次均未越过走廊。表中 current-pilot 的离线值由相同 SHA-256 的 ONNX 在正式 1,000 帧清单上补测，原始记录见 transfer-001/current-pilot-offline.json。

人工配置的 t002 与候选配置的 t012 均为留存的最佳验证 checkpoint，训练 seed 分别为 2 和 4。两者的独立测试 MAE 分别为 0.02211 和 0.02170。候选模型的闭环平均绝对 CTE 高约 0.013，最大绝对 CTE 的均值低约 0.033。30 次运行均达到时限，碰撞数和走廊出界数均为 0。不同指标的排序并不一致，训练种子也不同。

启动时的评估入口提交为 `b26dd0a`。复现命令为 `python -m apps.eval.closed_loop --models apps/eval/scenarios/baseline_models.json --scenarios apps/eval/scenarios/generated_roads.json --baseline manual-t002 --out runs/stage3/closed-loop-repro`。原始目录与本报告同目录，校验和如下。

- `manifest.json` SHA-256 `f05783435e2a97fc97b6f0c5049b5bd78bc9790718f74dceaa6a4f9fa94baaef`
- `frames.jsonl` SHA-256 `cebf4a95005e43ba68ac1dd6974636e20676312d7229aa583e07742154dc722d`
- `episodes.jsonl` SHA-256 `b22a5c4de329a606d2553ddd21ef148d8fb5a89555ba6b7c7a8710c04a676324`
- `result.json` SHA-256 `95807b96c3ff4f35e00abac070f43f2cc5743a1ec650a675a670e62893929d71`
