#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化 Kubeflow Pipeline：预处理 → 训练 → ONNX 导出 → 量化 → 部署说明。

需安装: pip install kfp

编译:
  python deploy/k8s/kubeflow_pipeline.py
"""
from __future__ import annotations

try:
  from kfp import dsl
  from kfp.compiler import Compiler
except ImportError:

  def main():
    print("请安装 kfp: pip install kfp")

  if __name__ == "__main__":
    main()
else:

  @dsl.component(base_image="python:3.11-slim")
  def preprocess_op(data_dir: str) -> str:
    print(f"预处理数据目录: {data_dir}")
    return data_dir

  @dsl.component(base_image="python:3.11-slim")
  def train_op(data_dir: str) -> str:
    print(f"训练 MyFlows ResNet（占位） data={data_dir}")
    return "mycar/models/myflow_resnet18_best.onnx"

  @dsl.component(base_image="python:3.11-slim")
  def export_onnx_op(checkpoint: str) -> str:
    print(f"导出 ONNX: {checkpoint}")
    return checkpoint

  @dsl.component(base_image="python:3.11-slim")
  def quantize_op(onnx_path: str) -> str:
    print(f"INT8 量化: {onnx_path}")
    return onnx_path.replace(".onnx", "_int8.onnx")

  @dsl.component(base_image="python:3.11-slim")
  def deploy_op(model_path: str) -> str:
    print(f"部署 gRPC/FastAPI 服务: {model_path}")
    return model_path

  @dsl.pipeline(name="myflows-donkey-pipeline", description="Donkey 训练到部署 DAG")
  def myflows_pipeline(data_dir: str = "mycar/data"):
    d = preprocess_op(data_dir)
    ckpt = train_op(d.output)
    onnx_p = export_onnx_op(ckpt.output)
    int8_p = quantize_op(onnx_p.output)
    deploy_op(int8_p.output)

  def main():
    Compiler().compile(myflows_pipeline, "deploy/k8s/myflows_pipeline.yaml")
    print("已生成: deploy/k8s/myflows_pipeline.yaml")

  if __name__ == "__main__":
    main()
