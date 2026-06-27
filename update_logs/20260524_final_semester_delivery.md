# 学期收官交付说明

- 时间：2026-05-24
- 对照任务书基本任务 (1)–(4) 与拓展 (3)(4)(5)

## 新增/更新（父仓库 testmyflow）

| 类别 | 路径 |
|------|------|
| FastAPI 部署 | `serve_fastapi.py`, `onnx_predictor.py` |
| INT8 闭环 | `scripts/calibration_reader.py`, `scripts/run_quantize_eval.py` |
| 压测 | `benchmark/serve_bench.py`, `benchmark/plot_compare.py`, `benchmark/dataloader_bench.py` |
| Docker | `deploy/docker/` |
| K8s | `deploy/k8s/` |
| 文档 | `docs/final_report.md`, 四份设计 md 充实 |

## 新增（MyFlows 子包）

| 类别 | 路径 |
|------|------|
| Conv+BN 折叠 | `core/graph_opt.py` `fold_bn_into_conv` |
| 生产者-消费者 | `data/pipeline.py` |
| 测试 | `tests/test_graph_opt_bn_fold.py` |

## 答辩 PPT 建议增补页

1. FastAPI + gRPC 双通道部署架构图  
2. FP32/INT8 对比表（`scripts/run_quantize_eval.py`）  
3. 计算图优化：Conv+BN 折叠前后节点数  
4. Docker compose 一键演示截图  

可将 `update_logs/MyFlows_中期答辩进度汇报.pptx` 复制为期末版后粘贴上述内容。
