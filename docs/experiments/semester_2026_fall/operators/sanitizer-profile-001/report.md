# 原生 CUDA 动态检查

在同一 GPU 上串行执行原生卷积、GEMM、池化及 cuBLAS 路径测试。

| 工具 | 运行测试数 | 报告错误数 | racecheck 危险数 | 秒 | 通过 |
| --- | ---: | ---: | ---: | ---: | --- |
| memcheck | 24 | 0 | — | 22.8 | 是 |
| racecheck | 24 | — | 0 | 115.5 | 是 |
| synccheck | 24 | 0 | — | 15.8 | 是 |

工具版本、源码指纹、每次启动命令、测试日志及诊断原文均保存在本目录。复现命令为 `python -m tools.sanitize_native --out runs/stage3/native-sanitizer`。
