# MyFlows 项目文档索引

| 文档 | 说明 |
|------|------|
| [system_design.md](system_design.md) | 总体架构 |
| [module_design.md](module_design.md) | 模块划分 |
| [algorithm_design.md](algorithm_design.md) | 核心算法 |
| [detailed_design.md](detailed_design.md) | 接口与类设计 |
| [final_report.md](final_report.md) | 期末报告正文 |
| [project_structure.md](project_structure.md) | 目录结构与运行入口 |
| [task_verification_guide.md](task_verification_guide.md) | 按任务书逐项验证和演示系统功能 |
| [experiments/int8_report.md](experiments/int8_report.md) | FP32/INT8 对比 |
| [experiments/explainability/README.md](experiments/explainability/README.md) | Grad-CAM 可解释性输出 |

## 常用命令

```bash
# GPU 依赖（当前 Python 环境）
# pip install cupy-cuda12x onnxruntime-gpu

# 数据重建
python -m tools.convert_generated_road_to_tub_v2 --src mycar/generated-road-data --dst mycar/data --clear-dst

# 结构分层自检
python -m py_compile apps/common/donkey_data.py apps/common/image_preprocess.py
python -m py_compile MyFlows/utils/training_dashboard.py MyFlows/utils/checkpoint.py MyFlows/utils/onnx_exporter.py

# ResNet 回归从头训练（--max-samples 0 = 全量）
python -m apps.train.train_myflows_donkey --max-samples 0 --epochs 20 --batch 2 \
  --augment --device auto --checkpoint-every 500 --export-onnx
tensorboard --logdir mycar/logs/tensorboard

# 离线评估（先用 2000 条抽查）
python -m apps.eval.eval_myflows_donkey_onnx --checkpoint mycar/models/myflow_resnet18_best.onnx \
  --max-samples 2000 --device auto

# 仿真部署
cd mycar && python manage.py drive --model=models/myflow_resnet18_best.onnx --type=myflows

# VGG 角度分类
python -m apps.train.train_vgg_donkey_classify --max-samples 0 --epochs 10 --device auto

# ONNX INT8
python -m tools.quantize_onnx --input mycar/models/myflow_resnet18_best.onnx
python -m apps.eval.eval_myflows_donkey_onnx --checkpoint mycar/models/myflow_resnet18_best.onnx \
  --int8-model mycar/models/myflow_resnet18_best_int8.onnx --device auto

# gRPC / FastAPI
python -m apps.serve.serve_grpc --model mycar/models/myflow_resnet18_best.onnx --device auto
python -m apps.serve.grpc_client --image mycar/data/images/1042_0.0000.jpg --host 127.0.0.1 --port 50051
python -m apps.serve.serve_fastapi --model mycar/models/myflow_resnet18_best.onnx --port 8000
python -m apps.serve.fastapi_client --image mycar/data/images/1042_0.0000.jpg --url http://127.0.0.1:8000

# INT8 对比报告
python scripts/run_quantize_eval.py --fp32 mycar/models/myflow_resnet18_best.onnx

# Grad-CAM 解释可视化
python -m tools.explain_donkey_gradcam --model-type resnet --checkpoint mycar/models/myflow_resnet18_best --max-samples 8

# 跨框架 benchmark + 出图
python benchmark/compare_frameworks.py --epochs 2 --samples 64 --device auto
python benchmark/plot_compare.py

# 部署压测 / DataLoader 吞吐
python benchmark/serve_bench.py --mode local --model mycar/models/myflow_resnet18_best.onnx --out-json docs/experiments/serve_bench_local.json --out-md docs/experiments/serve_bench_local.md
python benchmark/dataloader_bench.py

# Docker
docker compose -f deploy/docker/docker-compose.yml up --build
```
