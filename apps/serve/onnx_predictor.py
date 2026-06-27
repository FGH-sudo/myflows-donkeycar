# -*- coding: utf-8 -*-
"""ONNX Runtime 单图推理（gRPC / FastAPI 共用）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from apps.serve.schema import format_prediction_dict
from tools.device_runtime import create_ort_inference_session, print_ort_device


class OnnxPredictor:
  def __init__(
      self,
      model_path: Path,
      image_w: int,
      image_h: int,
      device: str = "auto",
      model_type: str = "regression",
  ):
    self.model_path = Path(model_path)
    self.device = str(device)
    self.session, providers = create_ort_inference_session(self.model_path, device)
    self.providers = list(providers)
    print_ort_device(providers, device)
    self.input_name = self.session.get_inputs()[0].name
    self.input_shape = self.session.get_inputs()[0].shape
    self.image_w = int(image_w)
    self.image_h = int(image_h)
    self.model_type = str(model_type)

  def predict(self, rgb: np.ndarray) -> np.ndarray:
    if rgb.shape[0] != self.image_h or rgb.shape[1] != self.image_w:
      rgb = cv2.resize(rgb, (self.image_w, self.image_h))
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[np.newaxis, ...]
    out = self.session.run(None, {self.input_name: x})[0]
    return np.asarray(out).reshape(-1)

  def format_prediction(self, outputs: np.ndarray) -> dict:
    return format_prediction_dict(np.asarray(outputs, dtype=np.float32), self.model_type)

  def info(self) -> dict:
    return {
        "model_path": str(self.model_path),
        "image_w": self.image_w,
        "image_h": self.image_h,
        "model_type": self.model_type,
        "input_name": self.input_name,
        "input_shape": list(self.input_shape),
        "providers": list(self.providers),
        "device": self.device,
    }
