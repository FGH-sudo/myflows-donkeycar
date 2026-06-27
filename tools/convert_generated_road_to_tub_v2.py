#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把老师的 generated-road-data 转成当前 apps.train.train_myflows_donkey 需要的 tub v2 格式：
需要：
  mycar/data/
    images/<image>.jpg
    catalog_*.catalog  （每行是一条 JSON，包含 cam/image_array、user/angle、user/throttle）

老师约定：
  - 油门 throttle 固定值：0.5
  - jpg 文件名中的浮点数：转向角 angle
    例如：0_-0.0889.jpg  => angle=-0.0889
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path


def _normalize_angle_str(s: str) -> str:
    # 兼容 unicode minus（例如 U+2212）和普通负号
    s = s.strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    # 兼容小数逗号
    s = s.replace(",", ".")
    return s


ANGLE_RE = re.compile(r"^.*_([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\.jpg$", re.IGNORECASE)


def _parse_angle_from_filename(name: str) -> float:
    m = ANGLE_RE.match(name)
    if not m:
        raise ValueError(f"无法从文件名解析 angle: {name}")
    return float(_normalize_angle_str(m.group(1)))


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hardlink (same-volume safe)
    except Exception:
        shutil.copy2(src, dst)


def _resolve_source_dir(src_dir: Path) -> Path:
    if any(src_dir.glob("*.jpg")):
        return src_dir
    nested = src_dir / "data"
    if nested.is_dir() and any(nested.glob("*.jpg")):
        return nested
    return src_dir


def _write_manifest(dst_dir: Path, catalog_name: str, count: int) -> None:
    manifest_path = dst_dir / "manifest.json"
    lines = [
        ["cam/image_array", "user/angle", "user/throttle", "user/mode"],
        ["image_array", "float", "float", "str"],
        {},
        {
            "created_at": time.time(),
            "sessions": {
                "all_full_ids": ["generated-road"],
                "last_id": 0,
                "last_full_id": "generated-road",
            },
        },
        {
            "paths": [catalog_name],
            "current_index": int(count),
            "max_len": int(count),
            "deleted_indexes": [],
        },
    ]
    with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert generated-road-data to tub v2 format.")
    ap.add_argument("--src", type=str, default="mycar/generated-road-data")
    ap.add_argument("--dst", type=str, default="mycar/data")
    ap.add_argument("--throttle", type=float, default=0.5)
    ap.add_argument("--clear-dst", action="store_true", help="清空 dst/images 和 dst/catalog_*.catalog")
    ap.add_argument("--catalog-stem", type=str, default="catalog_generated")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    src_dir = _resolve_source_dir((root / args.src).resolve())
    dst_dir = (root / args.dst).resolve()
    src_dir_str = str(src_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.is_dir():
        raise SystemExit(f"源目录不存在: {src_dir_str}")

    images_dst = dst_dir / "images"

    if args.clear_dst:
        # 仅清理训练会用到的内容（避免误删整个 mycar/data 其它文件）
        if images_dst.exists():
            shutil.rmtree(images_dst)
        for p in dst_dir.glob("catalog_*.catalog"):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        manifest_path = dst_dir / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()

    images_dst.mkdir(parents=True, exist_ok=True)

    # 收集图片
    jpgs = [p for p in src_dir.iterdir() if p.is_file() and p.name.lower().endswith(".jpg")]
    jpgs.sort(key=lambda p: p.name)

    if not jpgs:
        raise SystemExit(f"源目录未找到 jpg: {src_dir_str}")

    catalog_path = dst_dir / f"{args.catalog_stem}.catalog"
    if catalog_path.exists() and args.clear_dst:
        catalog_path.unlink()

    # 写 catalog：每行一条 JSON
    n = 0
    with catalog_path.open("w", encoding="utf-8", newline="\n") as f:
        for p in jpgs:
            angle = _parse_angle_from_filename(p.name)
            _hardlink_or_copy(p, images_dst / p.name)
            rec = {
                "cam/image_array": p.name,
                "user/angle": float(angle),
                "user/throttle": float(args.throttle),
                "user/mode": "user",
            }
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            n += 1

    # 简单校验
    cats = list(dst_dir.glob("catalog_*.catalog"))
    if not cats:
        raise SystemExit("转换后未生成任何 catalog_*.catalog")
    _write_manifest(dst_dir, catalog_path.name, n)
    print(f"转换完成：images={n}  catalog={catalog_path.name}  manifest=manifest.json")
    print("示例记录：")
    with catalog_path.open("r", encoding="utf-8") as f:
        for i in range(3):
            line = f.readline().strip()
            if not line:
                break
            print(line)


if __name__ == "__main__":
    main()

