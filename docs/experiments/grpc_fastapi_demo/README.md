# gRPC / FastAPI 演示截图

本目录保存服务化部署模块中 gRPC 和 FastAPI 的演示截图。两种服务都加载 `mycar/models/myflow_resnet18_best.onnx`，并使用 CUDA/ONNX Runtime GPU provider 完成推理。

## gRPC 截图

| 图片 | 内容 | 答辩用途 |
|---|---|---|
| `grpc_server_startup.png` | gRPC 服务启动日志，包含端口、模型路径、CUDA provider 信息 | 证明 gRPC 推理服务已启动并加载 ResNet ONNX |
| `grpc_client_predict.png` | gRPC 客户端单张图片推理结果 | 展示请求输入和 `[angle, throttle]` 回归输出 |
| `grpc_bench_summary.png` | gRPC 压测命令和汇总结果 | 展示 gRPC 服务延迟、QPS、RSS 等性能指标 |
| `grpc_bench_requests_json.png` | gRPC 请求级 JSON 明细 | 展示压测采集到每次请求的状态、延迟和输出 |

## FastAPI 截图

| 图片 | 内容 | 答辩用途 |
|---|---|---|
| `fastapi_server_startup.png` | FastAPI/Uvicorn 服务启动日志，包含模型路径、端口、CUDA provider 信息 | 证明 HTTP 推理服务已启动并加载 ResNet ONNX |
| `fastapi_health_check.png` | `/health` 健康检查返回结果 | 展示服务存活和模型元信息接口 |
| `fastapi_client_predict.png` | FastAPI 客户端上传图片推理结果 | 展示 HTTP 文件上传推理和 `[angle, throttle]` 输出 |
| `fastapi_metrics_endpoint.png` | `/metrics` 接口返回的请求统计 | 展示服务端在线统计能力 |
| `fastapi_bench_summary.png` | FastAPI 压测命令和汇总结果 | 展示 HTTP 服务延迟、QPS、RSS 等性能指标 |
| `fastapi_bench_requests_json.png` | FastAPI 请求级 JSON 明细 | 展示压测采集到每次请求的状态、延迟和输出 |

## 建议展示顺序

1. 展示 `grpc_server_startup.png` 和 `grpc_client_predict.png`，说明 gRPC 适合服务间二进制调用。
2. 展示 `grpc_bench_summary.png`，说明压测统计口径。
3. 展示 `fastapi_server_startup.png`、`fastapi_health_check.png` 和 `fastapi_client_predict.png`，说明 FastAPI 适合 HTTP 演示和接口调试。
4. 展示 `fastapi_metrics_endpoint.png` 和 `fastapi_bench_summary.png`，说明 HTTP 服务的在线统计和压测结果。

请求级 JSON 明细图主要用于备查，PPT 中可按篇幅选择是否展示。
