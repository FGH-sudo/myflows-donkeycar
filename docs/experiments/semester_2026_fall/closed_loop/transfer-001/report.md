# DonkeyCar 配对闭环评估

每个场景使用相同的默认 reset、客户端种子、终止规则、时限和 0.2 油门。模型顺序按场景轮换。位置距离使用连续有效 3D 遥测点之间的欧氏距离。

| 模型 | 次数 | 碰撞次数 | 平均 / 最差存活秒 | 平均 / 最差距离 | 平均 / 最差绝对 CTE | 平均 / 最差最大绝对 CTE | 平均 / 最差出界数 | 全部帧 P50 / P95 推理 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current-pilot | 10 | 0 | 30.034 / 30.003 | 250.890 / 247.918 | 0.476 / 0.481 | 1.528 / 1.607 | 0.000 / 0.000 | 6.176 / 13.316 |
| manual-t002 | 10 | 0 | 30.025 / 30.003 | 247.564 / 247.041 | 29.823 / 31.395 | 101.412 / 106.283 | 1.000 / 1.000 | 11.494 / 18.986 |
| candidate-t012 | 10 | 0 | 30.025 / 30.000 | 250.523 / 250.100 | 0.584 / 0.636 | 3.082 / 3.617 | 1.000 / 1.000 | 11.533 / 19.033 |
| transfer-pretrained-k4 | 10 | 0 | 30.031 / 30.006 | 253.119 / 252.863 | 27.012 / 30.077 | 118.522 / 125.629 | 1.000 / 1.000 | 9.301 / 18.595 |

配对差值采用模型值减去 manual-t002 在同一场景的值。

| 模型 | 离线角度 MAE | 平均差 Δ存活秒 | Δ距离 | Δ平均绝对 CTE | Δ出界数 |
| --- | --- | --- | --- | --- | --- |
| current-pilot | 0.03200 | 0.010 | 3.325 | -29.346 | -1.000 |
| manual-t002 | 0.02211 | 0.000 | 0.000 | 0.000 | 0.000 |
| candidate-t012 | 0.02170 | -0.000 | 2.959 | -29.239 | 0.000 |
| transfer-pretrained-k4 | 0.02478 | 0.006 | 5.555 | -2.810 | 0.000 |

manifest.json 记录规则、场景、模型来源和 SHA-256。frames.jsonl 保存逐帧遥测，episodes.jsonl 保存各次 reset 起点与结局，result.json 保存平均、最差和配对差异。Unity 场景随机种子和任意起点坐标无法通过当前接口指定。

40 次运行共 24,011 帧，全部达到 30 秒时限，碰撞计数为零。迁移模型取全量迁移实验中按验证集选定的 pretrained-k4、seed 3 的最佳 checkpoint。其独立测试转向 MAE 为 0.02478，平均绝对 CTE 为 27.012，每次运行均触发一次走廊出界。candidate-t012 的平均绝对 CTE 为 0.584，每次有 23–41 帧的绝对 CTE 超过 2.0。current-pilot 的十次运行均在走廊内。

补测 current-pilot 使用与 manifest 相同的 ONNX 文件，并使用 formal-full-002 的同一份 1,000 帧测试清单，原始转向 MAE 为 0.03199813。本次未核实该旧模型原始训练集与这批帧是否重叠，其得分用于描述相同帧上的表现。离线排序首位是 candidate-t012，闭环平均绝对 CTE 最低的是 current-pilot。命令为 `python -m apps.eval.eval_myflows_donkey_onnx --checkpoint mycar/models/myflow_resnet18_best.onnx --split-file runs/autopilot/splits/donkey-all-val1000-test1000-tb50g5-s42.json --split test --max-samples 0 --device cuda --report-json runs/stage3/pilot-offline.json`。

与 baseline-001 对照，原有三个模型的文件哈希和评估协议均相同，但新会话中 manual-t002 的平均绝对 CTE 从 0.554 变为 29.823。生成道路在 Unity 进程之间无法固定。报告采用同一会话内的逐场景配对结果，十次 reset 也没有提供十条独立、已知的道路。离线 MAE 与此会话平均绝对 CTE 在 manual-t002 和迁移模型之间给出相反排序。

复现时将 transfer/donkey-001/formal/selected-model.json 追加到 apps/eval/scenarios/baseline_models.json 的模型列表，将得到的清单传给 `python -m apps.eval.closed_loop --models <模型清单> --scenarios apps/eval/scenarios/generated_roads.json --baseline manual-t002 --out runs/stage3/closed-loop-transfer-repro`。

| 原始文件 | SHA-256 |
| --- | --- |
| manifest.json | `640c083b28cc8ab5d41109544dc23c8929b208324ae275bcf56c242eca0f9095` |
| frames.jsonl | `b6080bd40717bb5a6a45413af5a1ae68130c685bc7b305e03fad5637666933c2` |
| episodes.jsonl | `1608aae67e94998d281a4bae8ee05faf8e6bd1a4665e3b9a265b07eb1417f085` |
| result.json | `99fc9a16503f107170c2bfc0951a10c50b533eb232c41654215e725d4ef6741b` |
