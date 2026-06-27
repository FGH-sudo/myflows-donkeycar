#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 benchmark/results.csv 渲染为对比柱状图。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
      if row.get("framework", "").endswith("FLOPs_est"):
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

  names = [r["framework"] for r in rows]
  times = [float(r["time_s"]) for r in rows if r.get("time_s")]
  mems = [float(r["peak_mb"]) for r in rows if r.get("peak_mb")]

  fig, axes = plt.subplots(1, 2, figsize=(10, 4))
  axes[0].bar(names[: len(times)], times, color="#4C72B0")
  axes[0].set_title("Training time (s)")
  axes[0].set_ylabel("seconds")
  axes[1].bar(names[: len(mems)], mems, color="#55A868")
  axes[1].set_title("Peak RSS (MB)")
  axes[1].set_ylabel("MB")
  plt.tight_layout()
  out = (ROOT / args.out).resolve()
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=120)
  print(f"已保存: {out}")


if __name__ == "__main__":
  main()
