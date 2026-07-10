# Docker 演示截图

本目录保存 Docker 封装推理服务的答辩截图材料。当前报告和 PPT 只展示 Docker Compose + Docker Desktop。

| 文件名 | 内容 | 推荐用途 |
|---|---|---|
| `docker_desktop_containers_running.png` | Docker Desktop `Containers` 页面，展示 gRPC/FastAPI 两个容器运行和端口映射 | Docker 封装服务主图 |
| `docker_fastapi_cuda_predict.png` | FastAPI health/model_info/predict 结果，包含 `CUDAExecutionProvider` 和 `device=cuda` | GPU 推理与 HTTP 服务证据 |
| `docker_grpc_client_predict.png` | gRPC 客户端请求结果，输出 DonkeyCar `[angle, throttle]` | gRPC 服务证据 |
| `docker_desktop_fastapi_logs_cuda.png` | Docker Desktop Logs 页面，包含 FastAPI 请求日志和 CUDA provider 信息 | 运行日志证据 |

建议 PPT 顺序：先放 `docker_desktop_containers_running.png`，再放 `docker_fastapi_cuda_predict.png` 或 `docker_grpc_client_predict.png`，最后用 `docker_desktop_fastapi_logs_cuda.png` 展示服务日志。
