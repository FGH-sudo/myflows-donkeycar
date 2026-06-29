# -*- coding: utf-8 -*-
"""DonkeyCar 数据索引与 catalog 读取工具。"""

from __future__ import annotations

import json
from pathlib import Path


def parse_angle_from_filename(name: str) -> float:
    """
    解析 generated-road 文件名里的 angle。

    典型样例：0_-0.0889.jpg / 1042_0.0000.jpg。规则是取扩展名前最后一个下划线之后的浮点数。
    """
    stem = Path(name).stem
    if "_" not in stem:
        raise ValueError(f"文件名不含 '_' 无法解析 angle: {name}")
    angle_str = stem.split("_")[-1].strip()
    angle_str = angle_str.replace("−", "-").replace("–", "-").replace("—", "-")
    angle_str = angle_str.replace(",", ".")
    return float(angle_str)


def image_rel_path(image_name: str) -> Path:
    rel = Path(str(image_name).replace("\\", "/"))
    if not rel.parts or rel.parts[0] != "images":
        rel = Path("images") / rel.name
    return rel


def load_generated_road_images_index(
    data_dir: Path,
    fixed_throttle: float,
    angle_scale: float,
) -> list[tuple[Path, float, float]]:
    """返回 (图片相对 data_dir 的路径, angle, throttle) 列表（不依赖 catalog）。"""
    images_dir = Path(data_dir) / "images"
    if not images_dir.is_dir():
        raise SystemExit(f"images 目录不存在: {images_dir}")

    jpgs = sorted(images_dir.glob("*.jpg"), key=lambda p: p.name)
    if not jpgs:
        raise SystemExit(f"images 目录下未找到 .jpg: {images_dir}")

    rows: list[tuple[Path, float, float]] = []
    bad = 0
    for path in jpgs:
        try:
            angle = parse_angle_from_filename(path.name) * float(angle_scale)
            rows.append((Path("images") / path.name, float(angle), float(fixed_throttle)))
        except Exception:
            bad += 1
            continue

    if bad:
        print(f"[data] 跳过无法解析 angle 的图片数: {bad} / {len(jpgs)}")
    return rows


def load_donkey_index(
    data_dir: Path,
    fixed_throttle: float,
    angle_scale: float,
    catalog_name: str | None = "catalog_generated.catalog",
    force_fixed_throttle: bool = False,
) -> list[tuple[Path, float, float]]:
    """返回 (图片相对 data_dir 的路径, angle, throttle)，优先读 catalog，失败则解析文件名。"""
    data_dir = Path(data_dir)
    if catalog_name:
        catalog_path = data_dir / catalog_name
        if catalog_path.is_file():
            rows: list[tuple[Path, float, float]] = []
            bad = 0
            missing = 0
            with catalog_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        image_name = rec.get("cam/image_array")
                        if not image_name:
                            bad += 1
                            continue
                        rel = image_rel_path(str(image_name))
                        if not (data_dir / rel).is_file():
                            missing += 1
                            continue
                        angle_value = rec.get("user/angle")
                        if angle_value is None:
                            angle_value = parse_angle_from_filename(rel.name)
                        throttle_value = rec.get("user/throttle", fixed_throttle)
                        angle = float(angle_value) * float(angle_scale)
                        if force_fixed_throttle:
                            throttle = float(fixed_throttle)
                        else:
                            throttle = float(fixed_throttle if throttle_value is None else throttle_value)
                    except Exception:
                        bad += 1
                        continue
                    rows.append((rel, angle, throttle))
            if missing:
                print(f"[data] catalog 中缺失图片数: {missing}")
            if bad:
                print(f"[data] catalog 中跳过无效记录数: {bad}")
            if rows:
                print(f"[data] source=catalog:{catalog_name} samples={len(rows)}")
                return rows
            print(f"[data] catalog 无可用样本，回退到 images 文件名解析: {catalog_path}")

    rows = load_generated_road_images_index(data_dir, fixed_throttle, angle_scale)
    print(f"[data] source=images samples={len(rows)}")
    return rows
