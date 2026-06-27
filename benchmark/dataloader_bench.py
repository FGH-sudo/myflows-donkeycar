#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 num_workers=0/2/4 时 Donkey 图像加载吞吐。"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from MyFlows.data.pipeline import MultiprocessDataLoader
from apps.common.donkey_data import load_donkey_index


def _load_bench_sample(sample):
  path_str, angle, image_w, image_h = sample
  img = cv2.imread(str(path_str), cv2.IMREAD_COLOR)
  img = cv2.resize(img, (int(image_w), int(image_h)))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  chw = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
  return chw, float(angle)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", type=str, default="mycar/data")
  ap.add_argument("--catalog", type=str, default="catalog_generated.catalog")
  ap.add_argument("--batch", type=int, default=8)
  ap.add_argument("--batches", type=int, default=50)
  ap.add_argument("--image-w", type=int, default=160)
  ap.add_argument("--image-h", type=int, default=120)
  args = ap.parse_args()

  data_dir = (ROOT / args.data).resolve()
  index = load_donkey_index(data_dir, fixed_throttle=0.5, angle_scale=1.0, catalog_name=args.catalog)
  dataset = [
      (str(data_dir / rel), angle, args.image_w, args.image_h)
      for rel, angle, _ in index[: args.batch * args.batches]
  ]
  if not dataset:
    raise SystemExit("无图片")

  for nw in (0, 2, 4):
    loader = MultiprocessDataLoader(
        dataset,
        args.batch,
        num_workers=nw,
        shuffle=True,
        seed=0,
        load_fn=_load_bench_sample,
    )
    t0 = time.perf_counter()
    n = 0
    for _ in loader:
      n += 1
      if n >= args.batches:
        break
    elapsed = time.perf_counter() - t0
    print(f"num_workers={nw}: {n} batches in {elapsed:.3f}s ({n * args.batch / elapsed:.1f} img/s)")


if __name__ == "__main__":
  main()
