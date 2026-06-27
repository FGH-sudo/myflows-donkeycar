#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze DonkeyCar image/catalog labels before training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _normalize_angle_str(value: str) -> str:
    return value.strip().replace("−", "-").replace("–", "-").replace("—", "-").replace(",", ".")


def _parse_angle_from_filename(name: str) -> float | None:
    stem = Path(name).stem
    if "_" not in stem:
        return None
    try:
        return float(_normalize_angle_str(stem.split("_")[-1]))
    except ValueError:
        return None


def _load_rows(data_dir: Path, catalog_name: str) -> tuple[str, list[dict]]:
    catalog_path = data_dir / catalog_name
    rows: list[dict] = []
    if catalog_path.is_file():
        with catalog_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                image_name = rec.get("cam/image_array")
                if not image_name:
                    continue
                rel = Path(str(image_name))
                if not rel.parts or rel.parts[0] != "images":
                    rel = Path("images") / rel.name
                angle = rec.get("user/angle")
                if angle is None:
                    angle = _parse_angle_from_filename(rel.name)
                throttle = rec.get("user/throttle")
                rows.append(
                    {
                        "image": str(rel).replace("\\", "/"),
                        "angle": None if angle is None else float(angle),
                        "throttle": None if throttle is None else float(throttle),
                    }
                )
        return f"catalog:{catalog_name}", rows

    images_dir = data_dir / "images"
    for image_path in sorted(images_dir.glob("*.jpg")):
        angle = _parse_angle_from_filename(image_path.name)
        rows.append(
            {
                "image": str(Path("images") / image_path.name).replace("\\", "/"),
                "angle": angle,
                "throttle": None,
            }
        )
    return "images", rows


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    sorted_values = sorted(values)
    mid = n // 2
    median = sorted_values[mid] if n % 2 else (sorted_values[mid - 1] + sorted_values[mid]) / 2
    return {
        "count": n,
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "median": median,
        "std": math.sqrt(var),
    }


def _angle_buckets(angles: list[float]) -> dict[str, int]:
    buckets = {
        "large_left(angle<-0.30)": 0,
        "small_left(-0.30..-0.05)": 0,
        "straight(-0.05..0.05)": 0,
        "small_right(0.05..0.30)": 0,
        "large_right(angle>0.30)": 0,
    }
    for angle in angles:
        if angle < -0.30:
            buckets["large_left(angle<-0.30)"] += 1
        elif angle < -0.05:
            buckets["small_left(-0.30..-0.05)"] += 1
        elif angle <= 0.05:
            buckets["straight(-0.05..0.05)"] += 1
        elif angle <= 0.30:
            buckets["small_right(0.05..0.30)"] += 1
        else:
            buckets["large_right(angle>0.30)"] += 1
    return buckets


def _missing_images(data_dir: Path, rows: list[dict]) -> int:
    missing = 0
    for row in rows:
        image = row.get("image")
        if image and not (data_dir / image).is_file():
            missing += 1
    return missing


def _print_report(report: dict) -> None:
    print(f"source: {report['source']}")
    print(f"samples: {report['samples']}")
    print(f"missing_images: {report['missing_images']}")
    print("angle_stats:", _format_stats(report["angle_stats"]))
    print("angle_buckets:")
    total = max(int(report["angle_stats"].get("count", 0)), 1)
    for key, count in report["angle_buckets"].items():
        print(f"  {key}: {count} ({count / total:.1%})")
    if report["throttle_stats"].get("count", 0):
        print("throttle_stats:", _format_stats(report["throttle_stats"]))
    else:
        print("throttle_stats: no throttle labels")
    if report["warnings"]:
        print("warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    else:
        print("warnings: none")


def _format_stats(stats: dict) -> str:
    if not stats.get("count"):
        return "count=0"
    return (
        f"count={stats['count']} min={stats['min']:.4f} mean={stats['mean']:.4f} "
        f"median={stats['median']:.4f} max={stats['max']:.4f} std={stats['std']:.4f}"
    )


def _warnings(samples: int, buckets: dict[str, int], throttle_stats: dict) -> list[str]:
    warnings: list[str] = []
    if samples < 1000:
        warnings.append("样本量偏少；适合 smoke test，不适合证明最终模型效果。")
    straight = buckets["straight(-0.05..0.05)"] / max(samples, 1)
    left = buckets["large_left(angle<-0.30)"] + buckets["small_left(-0.30..-0.05)"]
    right = buckets["small_right(0.05..0.30)"] + buckets["large_right(angle>0.30)"]
    if straight > 0.65:
        warnings.append("直行样本占比过高，模型可能倾向预测接近 0 的转向。")
    if min(left, right) / max(max(left, right), 1) < 0.5:
        warnings.append("左右转样本明显不平衡，建议补采较少的一侧或做镜像增强。")
    if throttle_stats.get("count", 0) and float(throttle_stats.get("std", 0.0)) < 1e-6:
        warnings.append("throttle 几乎恒定；固定速度转向任务可忽略，若要研究速度变化需另采真实 throttle。")
    if not throttle_stats.get("count", 0):
        warnings.append("未发现 throttle 标签；固定速度转向任务可使用训练脚本的 --fixed-throttle。")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DonkeyCar data label distribution.")
    parser.add_argument("--data", type=str, default="mycar/data")
    parser.add_argument("--catalog", type=str, default="catalog_generated.catalog")
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    data_dir = (ROOT / args.data).resolve()
    source, rows = _load_rows(data_dir, args.catalog)
    angles = [float(row["angle"]) for row in rows if row.get("angle") is not None]
    throttles = [float(row["throttle"]) for row in rows if row.get("throttle") is not None]
    buckets = _angle_buckets(angles)
    angle_stats = _stats(angles)
    throttle_stats = _stats(throttles)
    report = {
        "data_dir": str(data_dir),
        "source": source,
        "samples": len(rows),
        "usable_angle_labels": len(angles),
        "usable_throttle_labels": len(throttles),
        "missing_images": _missing_images(data_dir, rows),
        "angle_stats": angle_stats,
        "angle_buckets": buckets,
        "throttle_stats": throttle_stats,
        "warnings": _warnings(len(angles), buckets, throttle_stats),
    }
    _print_report(report)

    if args.out_json:
        out_path = (ROOT / args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"written: {out_path}")


if __name__ == "__main__":
    main()
