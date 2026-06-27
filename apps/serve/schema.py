# -*- coding: utf-8 -*-
"""Serving 请求/响应结构与输出解析。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping


def _to_float_list(outputs: Any) -> list[float]:
  if outputs is None:
    return []
  if hasattr(outputs, "reshape"):
    outputs = outputs.reshape(-1).tolist()
  return [float(v) for v in outputs]


def parse_prediction_outputs(outputs: Any, model_type: str = "regression") -> dict[str, Any]:
  values = _to_float_list(outputs)
  result: dict[str, Any] = {
      "outputs": values,
      "model_type": str(model_type or "regression"),
  }
  if result["model_type"] == "classification":
    if values:
      max_value = max(values)
      exp_values = [math.exp(v - max_value) for v in values]
      total = sum(exp_values) or 1.0
      class_id = int(max(range(len(exp_values)), key=exp_values.__getitem__))
      result.update({"class_id": class_id, "confidence": float(exp_values[class_id] / total)})
    return result
  if values:
    result["angle"] = float(values[0])
  if len(values) >= 2:
    result["throttle"] = float(values[1])
  return result


def format_prediction_dict(outputs: Any, model_type: str = "regression") -> dict[str, Any]:
  return parse_prediction_outputs(outputs, model_type=model_type)


@dataclass
class PredictionResult:
  outputs: list[float]
  model_type: str = "regression"
  request_id: str | None = None
  status: str | None = "ok"
  latency_ms: float | None = None
  input_shape: list[int] | None = None
  device: str | None = None
  angle: float | None = None
  throttle: float | None = None
  class_id: int | None = None
  confidence: float | None = None
  raw: dict[str, Any] | None = None

  @classmethod
  def from_outputs(cls, outputs: Any, model_type: str = "regression", **metadata: Any) -> "PredictionResult":
    parsed = parse_prediction_outputs(outputs, model_type=model_type)
    parsed.update({k: v for k, v in metadata.items() if v is not None})
    return cls(**_filter_result_fields(parsed))

  @classmethod
  def from_response(cls, data: Mapping[str, Any]) -> "PredictionResult":
    model_type = str(data.get("model_type", "regression"))
    parsed = parse_prediction_outputs(data.get("outputs", []), model_type=model_type)
    for key in _RESULT_FIELDS:
      if key in data and data[key] is not None:
        parsed[key] = data[key]
    parsed["raw"] = dict(data)
    return cls(**_filter_result_fields(parsed))

  def to_dict(self, include_none: bool = False, include_raw: bool = False) -> dict[str, Any]:
    data = {
        "request_id": self.request_id,
        "status": self.status,
        "model_type": self.model_type,
        "outputs": list(self.outputs),
        "angle": self.angle,
        "throttle": self.throttle,
        "class_id": self.class_id,
        "confidence": self.confidence,
        "latency_ms": self.latency_ms,
        "input_shape": self.input_shape,
        "device": self.device,
    }
    if include_raw:
      data["raw"] = self.raw
    if include_none:
      return data
    return {k: v for k, v in data.items() if v is not None}

  def to_json(self, include_raw: bool = False) -> str:
    return json.dumps(self.to_dict(include_raw=include_raw), ensure_ascii=False, indent=2)


_RESULT_FIELDS = {
    "outputs",
    "model_type",
    "request_id",
    "status",
    "latency_ms",
    "input_shape",
    "device",
    "angle",
    "throttle",
    "class_id",
    "confidence",
    "raw",
}


def _filter_result_fields(data: Mapping[str, Any]) -> dict[str, Any]:
  return {k: data[k] for k in _RESULT_FIELDS if k in data}
