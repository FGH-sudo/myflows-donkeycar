#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Donkey 图像目录 → ONNX Runtime 静态量化校准 DataReader。"""

from __future__ import annotations

import random
from pathlib import Path

from apps.common.image_preprocess import imread_nchw


class DonkeyCalibrationDataReader:
  """从 ``mycar/data/images`` 抽样 JPEG，输出 NCHW float32 batch=1。"""

  def __init__(
      self,
      data_dir: Path | str,
      *,
      image_w: int = 160,
      image_h: int = 120,
      max_samples: int = 200,
      seed: int = 42,
  ):
    self.data_dir = Path(data_dir)
    self.image_w = int(image_w)
    self.image_h = int(image_h)
    images_dir = self.data_dir / "images"
    paths = sorted(images_dir.glob("*.jpg"))
    if not paths:
      raise FileNotFoundError(f"未找到校准图片: {images_dir}")
    rng = random.Random(seed)
    if max_samples > 0 and len(paths) > max_samples:
      paths = rng.sample(paths, max_samples)
    self._paths = paths
    self._index = 0
    self.input_name = "input"

  def get_next(self) -> dict | None:
    if self._index >= len(self._paths):
      return None
    p = self._paths[self._index]
    self._index += 1
    try:
      x = imread_nchw(p, (self.image_w, self.image_h), dtype="float32")
    except FileNotFoundError:
      return self.get_next()
    return {self.input_name: x}

  def rewind(self) -> None:
    self._index = 0
