# -*- coding: utf-8 -*-
"""ONNX Runtime 单图推理（gRPC / FastAPI 共用）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from apps.serve.schema import format_prediction_dict
from tools.device_runtime import create_ort_inference_session, print_ort_device


_INPUT_DTYPES = {"tensor(float)": np.float32, "tensor(double)": np.float64, "tensor(float16)": np.float16}


class OnnxPredictor:
  """Run one RGB frame on an ONNX model with a single NCHW image input."""

  def __init__(
      self,
      model_path: Path,
      image_w: int,
      image_h: int,
      device: str = "auto",
      model_type: str = "regression",
      *,
      sess_options=None,
  ):
    self.model_path = Path(model_path)
    self.device = str(device)
    self.session, providers = create_ort_inference_session(self.model_path, device, sess_options=sess_options)
    self.providers = list(providers)
    print_ort_device(providers, device)
    inputs = self.session.get_inputs()
    if len(inputs) != 1 or len(inputs[0].shape or ()) != 4:
      raise ValueError("the ONNX model must have one NCHW image input")
    image_input = inputs[0]
    self.input_name = image_input.name
    self.input_shape = image_input.shape
    self.input_dtype = _INPUT_DTYPES.get(image_input.type)
    if self.input_dtype is None:
      raise ValueError(f"unsupported ONNX image input type: {image_input.type}")
    batch, channels, height, width = self.input_shape
    if isinstance(channels, int) and channels != 3:
      raise ValueError(f"the ONNX image input must have 3 RGB channels (got {channels})")
    self.image_w = width if isinstance(width, int) else int(image_w)
    self.image_h = height if isinstance(height, int) else int(image_h)
    self.batch_size = batch if isinstance(batch, int) else 1
    if min(self.batch_size, self.image_w, self.image_h) < 1:
      raise ValueError("the ONNX image input must have positive batch and spatial sizes")
    self.model_type = str(model_type)

  def predict(self, rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
      raise ValueError("expected a non-empty HxWx3 RGB image")
    if rgb.shape[0] != self.image_h or rgb.shape[1] != self.image_w:
      rgb = cv2.resize(rgb, (self.image_w, self.image_h))
    x = rgb.astype(self.input_dtype) / 255.0
    x = np.transpose(x, (2, 0, 1))[np.newaxis, ...]
    if self.batch_size > 1:
      x = np.repeat(x, self.batch_size, axis=0)
    out = np.asarray(self.session.run(None, {self.input_name: x})[0])
    if self.batch_size > 1:
      if out.ndim == 0 or out.shape[0] != self.batch_size:
        raise RuntimeError("the ONNX output is missing its per-image batch dimension")
      out = out[0]
    prediction = out.reshape(-1)
    if not prediction.size or not np.isfinite(prediction).all():
      raise RuntimeError("the ONNX model returned an empty or non-finite prediction")
    return prediction

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
