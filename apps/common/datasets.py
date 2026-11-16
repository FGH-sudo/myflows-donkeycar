"""Data sets the training entry, AutoPilot and the console may use, and the tub layout collectors write.

A data set is a directory with ``images/<frame>_<angle>.jpg`` and a line-JSON catalog, the layout
``apps.common.donkey_data.load_donkey_index`` reads. ``mycar/data`` is the hand-made DonkeyCar set every earlier
run used; collectors (``apps.collect``) write new ones under ``runs/datasets/<id>`` together with a
``dataset.json`` that says which simulator and which labelling rule produced them.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np

DEFAULT_DATA = "mycar/data"
COLLECTED_ROOT = "runs/datasets"
CATALOG = "catalog_generated.catalog"
DATASET_ID = re.compile(r"[a-z0-9][a-z0-9_-]{2,63}")
SCHEMA_VERSION = 1
# Catalog fields DonkeyCar's own tools know; collectors add their own after these.
TUB_FIELDS = (("cam/image_array", "image_array"), ("user/angle", "float"), ("user/throttle", "float"),
              ("user/mode", "str"))


class DatasetError(ValueError):
    pass


def normalize(data: str | None) -> str:
    """The repository-relative path of a known data set. Accepts ``None`` (the default set), that path, or the
    bare id of a collected set; anything else — absolute paths, ``..``, other directories — is refused, because
    the value reaches this function from web forms and from the LLM."""
    if data is None or data == "":
        return DEFAULT_DATA
    if not isinstance(data, str):
        raise DatasetError("数据集必须是字符串")
    text = data.strip().replace("\\", "/").rstrip("/")
    if text == DEFAULT_DATA:
        return DEFAULT_DATA
    name = text[len(COLLECTED_ROOT) + 1:] if text.startswith(COLLECTED_ROOT + "/") else text
    if DATASET_ID.fullmatch(name):
        return f"{COLLECTED_ROOT}/{name}"
    raise DatasetError(f"未知的数据集：{data}（可用 {DEFAULT_DATA} 或 {COLLECTED_ROOT}/<id>，id 由小写字母、数字、- 和 _ 组成）")


def dataset_key(data: str | None) -> str:
    """Short name used in split file names: ``donkey`` for the default set (the name older files carry)."""
    path = normalize(data)
    return "donkey" if path == DEFAULT_DATA else path.rsplit("/", 1)[1]


def dataset_dir(repo_root: str | Path, data: str | None = None, *, must_exist: bool = True) -> Path:
    path = Path(repo_root) / normalize(data)
    if must_exist and not path.is_dir():
        raise DatasetError(f"数据目录 {normalize(data)} 不存在")
    return path


def read_meta(directory: str | Path) -> dict:
    try:
        meta = json.loads((Path(directory) / "dataset.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def describe(repo_root: str | Path, data: str | None = None) -> dict:
    """What a form needs to offer the set: where it is, how many frames, which simulator, which image size."""
    path = normalize(data)
    directory = Path(repo_root) / path
    meta = read_meta(directory)
    catalog = directory / CATALOG
    frames = meta.get("frames")
    if not isinstance(frames, int):
        if catalog.is_file():
            with catalog.open("rb") as handle:
                frames = sum(1 for line in handle if line.strip())
        else:
            frames = sum(1 for _ in (directory / "images").glob("*.jpg")) if (directory / "images").is_dir() else 0
    default = path == DEFAULT_DATA
    return {"data": path, "id": dataset_key(path), "default": default, "frames": int(frames),
            "ready": (directory / "images").is_dir() and frames > 0,
            "environment": meta.get("environment") or ("donkey-generated-roads-v0" if default else None),
            "simulator": meta.get("simulator") or ("donkey" if default else None),
            "image": meta.get("image") or ({"width": 160, "height": 120} if default else None),
            "image_size": meta.get("image_size"), "source": meta.get("source") or ("人工驾驶记录" if default else None),
            "created_unix_s": meta.get("created_unix_s"), "summary": meta.get("summary")}


def list_datasets(repo_root: str | Path) -> list[dict]:
    """The default set first, then every collected set, newest first."""
    out = [describe(repo_root)]
    root = Path(repo_root) / COLLECTED_ROOT
    found = []
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and DATASET_ID.fullmatch(child.name) and (child / "images").is_dir():
                found.append(describe(repo_root, f"{COLLECTED_ROOT}/{child.name}"))
    found.sort(key=lambda row: (-(row["created_unix_s"] or 0), row["id"]))
    return out + found


def label_summary(angles, *, straight: float = 0.05) -> dict:
    """Shape of a steering label list: enough to see a set that only drives straight or only turns one way."""
    values = np.asarray(list(angles), dtype=np.float64)
    if not values.size:
        return {"frames": 0}
    edges = np.linspace(-1., 1., 21)
    counts, _ = np.histogram(np.clip(values, -1., 1.), bins=edges)
    return {"frames": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
            "mean_abs": float(np.abs(values).mean()), "min": float(values.min()), "max": float(values.max()),
            "straight_share": float((np.abs(values) < straight).mean()),
            "left_share": float((values <= -straight).mean()), "right_share": float((values >= straight).mean()),
            "saturated_share": float((np.abs(values) >= .999).mean()),
            "histogram": {"edges": [round(float(e), 2) for e in edges], "counts": [int(c) for c in counts]}}


class TubWriter:
    """Append frames to a data set directory, one image and one catalog line per frame.

    The catalog line is written after its image, so a collector that dies leaves a readable set. Frame
    numbers keep counting across episodes; ``close`` writes the manifests and ``dataset.json``.
    """

    def __init__(self, directory: str | Path, *, meta: dict, extra_fields: dict | None = None, jpeg_quality: int = 95):
        self.directory = Path(directory)
        self.images = self.directory / "images"
        if (self.directory / CATALOG).exists() or (self.images.is_dir() and any(self.images.iterdir())):
            raise DatasetError(f"数据集目录已有数据：{self.directory}")
        self.images.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta)
        self.extra_fields = dict(extra_fields or {})
        self.quality = int(jpeg_quality)
        self.count = 0
        self.angles: list[float] = []
        self.size = None
        self.created = time.time()
        self._catalog = (self.directory / CATALOG).open("w", encoding="utf-8", newline="\n")

    def add(self, rgb, angle: float, throttle: float, *, mode: str = "expert", **extra) -> str:
        import cv2

        angle, throttle = float(angle), float(throttle)
        if not (math.isfinite(angle) and math.isfinite(throttle)):
            raise DatasetError("标签必须是有限数值")
        frame = np.ascontiguousarray(rgb)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise DatasetError(f"图像必须是 HxWx3 的 uint8：{frame.shape} {frame.dtype}")
        size = (int(frame.shape[1]), int(frame.shape[0]))
        if self.size is None:
            self.size = size
        elif size != self.size:
            raise DatasetError(f"图像尺寸变化：{size} != {self.size}")
        unknown = set(extra) - set(self.extra_fields)
        if unknown:
            raise DatasetError(f"未声明的附加字段：{', '.join(sorted(unknown))}")
        name = f"{self.count}_{angle:.4f}.jpg"
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise DatasetError("JPEG 编码失败")
        (self.images / name).write_bytes(encoded.tobytes())
        record = {"cam/image_array": name, "user/angle": round(angle, 6), "user/throttle": round(throttle, 6),
                  "user/mode": mode, **extra}
        self._catalog.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._catalog.flush()
        self.count += 1
        self.angles.append(angle)
        return name

    def close(self, *, summary: dict | None = None) -> dict:
        """Write the manifests and ``dataset.json``; returns the metadata that was written."""
        if not self._catalog.closed:
            self._catalog.close()
        names = [name for name, _ in TUB_FIELDS] + list(self.extra_fields)
        types = [kind for _, kind in TUB_FIELDS] + list(self.extra_fields.values())
        session = str(self.meta.get("id") or self.directory.name)
        lines = [names, types, {},
                 {"created_at": self.created, "sessions": {"all_full_ids": [session], "last_id": 0, "last_full_id": session}},
                 {"paths": [CATALOG], "current_index": self.count, "max_len": self.count, "deleted_indexes": []}]
        with (self.directory / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        with (self.directory / f"{CATALOG}_manifest").open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"created_at": self.created, "line_lengths": [], "path": f"{CATALOG}_manifest",
                                     "start_index": self.count}) + "\n")
        meta = {"schema_version": SCHEMA_VERSION, **self.meta, "created_unix_s": self.created, "frames": self.count,
                "image": {"width": self.size[0], "height": self.size[1]} if self.size else None,
                "catalog": CATALOG, "labels": label_summary(self.angles), "summary": summary or {}}
        tmp = self.directory / "dataset.json.tmp"
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(self.directory / "dataset.json")
        return meta

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self._catalog.closed:
            self._catalog.close()
