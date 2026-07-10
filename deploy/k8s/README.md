# Kubernetes 部署（MyFlows 推理）

## 前置

1. 构建镜像（在 `testmyflow` 根目录）:

```bash
docker build -f deploy/docker/Dockerfile -t myflows-infer:latest .
```

2. 将 `deployment.yaml` 中 `hostPath` 改为本机 `mycar/models` 绝对路径。容器内模型路径默认为 `/models/myflow_resnet18_best.onnx`。

`deployment.yaml` 默认按 GPU 推理口径启动：`DEVICE=cuda`，并声明 `limits.nvidia.com/gpu: 1`。如果当前 Kubernetes 环境没有 NVIDIA device plugin 或没有 GPU 资源，Pod 会无法调度；此时可临时移除该 GPU limit 并把 `DEVICE` 改为 `auto` 或 `cpu` 做流程演示。

## 应用

```bash
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
```

gRPC 经 NodePort `30051` 访问（默认），本演示不切换 FastAPI。

## minikube 演示

```bash
minikube start
eval $(minikube docker-env)
docker build -f deploy/docker/Dockerfile -t myflows-infer:latest .
kubectl apply -f deploy/k8s/
minikube service myflows-infer --url
```

## Kubeflow Pipeline

见 `kubeflow_pipeline.py`（需安装 `kfp`）。本地无集群时可仅交付本 README + YAML 作为拓展(5) 演示材料。
