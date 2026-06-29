#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 benchmark/results.csv 渲染为对比柱状图。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METRIC_PANELS = [
  {
    "column": "time_s",
    "title": "DonkeyCar VGG Regression time",
    "ylabel": "seconds",
    "color": "#4C72B0",
    "scale": 1.0,
  },
  {
    "column": "peak_mb",
    "title": "DonkeyCar VGG Regression RSS",
    "ylabel": "MB",
    "color": "#55A868",
    "scale": 1.0,
  },
]


def _as_float(value):
  if value in (None, ""):
    return None
  try:
    return float(value)
  except ValueError:
    return None


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--csv", type=str, default="benchmark/results.csv")
  ap.add_argument("--out", type=str, default="docs/experiments/framework_compare.png")
  args = ap.parse_args()

  csv_path = (ROOT / args.csv).resolve()
  rows = []
  with csv_path.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
      if row.get("error"):
        continue
      rows.append(row)
  if not rows:
    raise SystemExit("无可绘制的行")

  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    raise SystemExit("需要 matplotlib: pip install matplotlib")

  fig, axes = plt.subplots(1, len(METRIC_PANELS), figsize=(5 * len(METRIC_PANELS), 4))
  if len(METRIC_PANELS) == 1:
    axes = [axes]
  for ax, panel in zip(axes, METRIC_PANELS):
    names = []
    values = []
    for row in rows:
      value = _as_float(row.get(panel["column"]))
      if value is None:
        continue
      names.append(row["framework"])
      values.append(value * panel["scale"])
    if not values:
      ax.set_visible(False)
      continue
    ax.bar(names, values, color=panel["color"])
    ax.set_title(panel["title"])
    ax.set_ylabel(panel["ylabel"])
    ax.tick_params(axis="x", rotation=20)
  plt.tight_layout()
  out = (ROOT / args.out).resolve()
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=120)
  print(f"已保存: {out}")


if __name__ == "__main__":
  main()
