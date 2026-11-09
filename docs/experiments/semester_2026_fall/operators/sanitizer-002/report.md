# 原生 CUDA 动态检查

在同一 GPU 上串行执行原生卷积、GEMM、池化及 cuBLAS 路径测试。

| 工具 | 运行测试数 | 报告错误数 | racecheck 危险数 | 秒 | 通过 |
| --- | ---: | ---: | ---: | ---: | --- |
| memcheck | 28 | 0 | — | 29.1 | 是 |
| racecheck | 28 | — | 0 | 122.5 | 是 |
| synccheck | 28 | 0 | — | 19.8 | 是 |

工具版本、源码指纹、每次启动命令、测试日志及诊断原文均保存在本目录。复现命令为 `python -m tools.sanitize_native --out runs/stage3/native-sanitizer`。

本轮代码为根仓库 0290d6c、MyFlows 724d0dd（工作区干净），Compute Sanitizer 2026.2.1。与 [sanitizer-profile-001](../sanitizer-profile-001/report.md) 相比多出的 4 项是 `NativeSplitReduceTests`，覆盖 split-K 的 FP32 归约、整数精确归约与 3 种 cp.async 流水 tile。三种工具下这 4 项都没有错误或竞争告警。
