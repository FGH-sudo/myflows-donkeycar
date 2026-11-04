# 监控 v0 兼容夹具

摘自本地历史运行 `runs/autopilot/formal-full-002/trials/t003/attempt-1/`。
保留前两步、首轮 epoch、前两步附近的两个资源样本及最终指标。
仅删除输出路径与非必要配置，设备 UUID 和 PID 替换为固定占位标识。
数值、字段含义与缺测情况不变；这是读取兼容夹具，不是一份新的训练结果。

该版本的 config、metrics、resources 没有 schema_version；旧的 result 已有独立的
schema_version=1 封套，但尚无 environment。校验工具据 v0 config 识别这种历史组合。
