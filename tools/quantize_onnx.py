#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 FP32 ONNX 量化为 INT8 动态量化模型。

示例:
  python -m tools.quantize_onnx --input mycar/models/myflow_resnet18_best.onnx
  python -m apps.eval.eval_myflows_donkey_onnx --checkpoint mycar/models/myflow_resnet18_best.onnx \\
      --int8-model mycar/models/myflow_resnet18_best_int8.onnx --device auto
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from MyFlows.utils.quantize import quantize_onnx_dynamic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="输入 .onnx")
    ap.add_argument("--output", type=str, default=None, help="输出路径，默认 *_int8.onnx")
    args = ap.parse_args()
    out = quantize_onnx_dynamic(args.input, args.output)
    print(f"已写入 INT8 模型: {out}")


if __name__ == "__main__":
    main()
