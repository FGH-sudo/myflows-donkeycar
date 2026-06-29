#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 num_workers=0/2/4 时 Donkey 图像加载吞吐并生成报告产物。"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from MyFlows.data.pipeline import MultiprocessDataLoader
from apps.common.donkey_data import load_donkey_index


DEFAULT_CSV = "docs/experiments/dataloader_bench.csv"
DEFAULT_MD = "docs/experiments/dataloader_bench.md"
DEFAULT_PNG = "docs/experiments/dataloader_bench.png"
FIELDNAMES = [
  "num_workers",
  "batches",
  "batch_size",
  "total_images",
  "train_ms",
  "elapsed_s",
  "img_s",
  "ms_per_batch",
  "speedup",
]


def _load_bench_sample(sample):
  path_str, angle, image_w, image_h = sample
  img = cv2.imread(str(path_str), cv2.IMREAD_COLOR)
  img = cv2.resize(img, (int(image_w), int(image_h)))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  chw = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
  return chw, float(angle)


def parse_workers(value: str) -> list[int]:
  workers = []
  for item in str(value).split(","):
    item = item.strip()
    if not item:
      continue
    workers.append(max(0, int(item)))
  return workers or [0]


def enrich_results(rows: list[dict]) -> list[dict]:
  baseline = None
  for row in rows:
    if int(row["num_workers"]) == 0:
      baseline = float(row["img_s"])
      break
  if baseline is None and rows:
    baseline = float(rows[0]["img_s"])
  baseline = baseline or 0.0

  enriched = []
  for row in rows:
    item = dict(row)
    batches = max(1, int(item["batches"]))
    elapsed = float(item["elapsed_s"])
    img_s = float(item["img_s"])
    item["total_images"] = int(item.get("total_images", int(item["batch_size"]) * batches))
    item["train_ms"] = float(item.get("train_ms", 0.0))
    item["ms_per_batch"] = (elapsed / batches) * 1000.0
    item["speedup"] = (img_s / baseline) if baseline > 0 else 0.0
    enriched.append(item)
  return enriched


def write_csv(rows: list[dict], out_path: str | Path) -> None:
  path = Path(out_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def write_markdown(rows: list[dict], out_path: str | Path) -> None:
  path = Path(out_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  lines = [
    "# DataLoader Producer-Consumer Benchmark",
    "",
    "| num_workers | batches | batch_size | total_images | train_ms | elapsed_s | img_s | ms_per_batch | speedup |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in rows:
    lines.append(
      "| {num_workers} | {batches} | {batch_size} | {total_images} | {train_ms:.1f} | {elapsed_s:.4f} | {img_s:.2f} | {ms_per_batch:.2f} | {speedup:.2f} |".format(
        **row
      )
    )
  lines.extend(
    [
      "",
      "说明：`num_workers=0` 为主进程同步读取；`num_workers>0` 使用 MyFlows `MultiprocessDataLoader`，worker 进程作为生产者执行图片 IO/解码/预处理，主进程作为消费者按 batch 取数。",
    ]
  )
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(rows: list[dict], out_path: str | Path) -> None:
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("跳过绘图：需要 matplotlib")
    return

  names = [str(row["num_workers"]) for row in rows]
  img_s = [float(row["img_s"]) for row in rows]
  ms_batch = [float(row["ms_per_batch"]) for row in rows]

  fig, axes = plt.subplots(1, 2, figsize=(10, 4))
  axes[0].bar(names, img_s, color="#4C72B0")
  axes[0].set_title("DataLoader throughput")
  axes[0].set_xlabel("num_workers")
  axes[0].set_ylabel("images / second")
  axes[1].bar(names, ms_batch, color="#55A868")
  axes[1].set_title("Average batch load time")
  axes[1].set_xlabel("num_workers")
  axes[1].set_ylabel("ms / batch")
  fig.suptitle("Producer-Consumer Data Pipeline")
  plt.tight_layout()
  path = Path(out_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=140)
  print(f"已保存图表: {path}")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", type=str, default="mycar/data")
  ap.add_argument("--catalog", type=str, default="catalog_generated.catalog")
  ap.add_argument("--batch", type=int, default=8)
  ap.add_argument("--batches", type=int, default=50)
  ap.add_argument("--image-w", type=int, default=160)
  ap.add_argument("--image-h", type=int, default=120)
  ap.add_argument("--workers", type=str, default="0,2,4")
  ap.add_argument("--train-ms", type=float, default=0.0, help="每个 batch 模拟训练耗时，用于展示数据预取与训练并行")
  ap.add_argument("--out-csv", type=str, default=DEFAULT_CSV)
  ap.add_argument("--out-md", type=str, default=DEFAULT_MD)
  ap.add_argument("--out-png", type=str, default=DEFAULT_PNG)
  args = ap.parse_args()

  data_dir = (ROOT / args.data).resolve()
  index = load_donkey_index(data_dir, fixed_throttle=0.5, angle_scale=1.0, catalog_name=args.catalog)
  dataset = [
      (str(data_dir / rel), angle, args.image_w, args.image_h)
      for rel, angle, _ in index[: args.batch * args.batches]
  ]
  if not dataset:
    raise SystemExit("无图片")

  rows = []
  for nw in parse_workers(args.workers):
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
      if args.train_ms > 0:
        time.sleep(float(args.train_ms) / 1000.0)
      if n >= args.batches:
        break
    elapsed = time.perf_counter() - t0
    img_s = n * args.batch / elapsed
    row = {
      "num_workers": nw,
      "batches": n,
      "batch_size": args.batch,
      "total_images": n * args.batch,
      "train_ms": float(args.train_ms),
      "elapsed_s": elapsed,
      "img_s": img_s,
    }
    rows.append(row)
    print(f"num_workers={nw}: {n} batches in {elapsed:.3f}s ({img_s:.1f} img/s)")

  rows = enrich_results(rows)
  write_csv(rows, ROOT / args.out_csv)
  write_markdown(rows, ROOT / args.out_md)
  plot_results(rows, ROOT / args.out_png)
  print(f"已写入: {ROOT / args.out_csv}")
  print(f"已写入: {ROOT / args.out_md}")


if __name__ == "__main__":
  main()
