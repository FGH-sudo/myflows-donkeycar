"""Data sets the training entry, AutoPilot and the console may use, and the tub layout collectors write.

A data set is a directory with ``images/<frame>_<angle>.jpg`` and a line-JSON catalog, the layout
``apps.common.donkey_data.load_donkey_index`` reads. ``mycar/data`` is the hand-made DonkeyCar set every earlier
run used; collectors (``apps.collect``) write new ones under ``runs/datasets/<id>`` together with a
``dataset.json`` that says which simulator and which labelling rule produced them.
"""

from __future__ import annotations

import json
import math
import numbers
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
    if not path.resolve().is_relative_to(Path(repo_root).resolve()):
        raise DatasetError("数据集路径离开了项目目录")
    if must_exist and not path.is_dir():
        raise DatasetError(f"数据目录 {normalize(data)} 不存在")
    meta = read_meta(path)
    if meta.get("status") == "collecting":
        raise DatasetError("数据集正在采集，请在采集完成或续采恢复后再训练")
    if meta.get("status") == "interrupted" or (isinstance(meta.get("summary"), dict)
                                               and meta["summary"].get("accepted") is False):
        raise DatasetError("数据集已中断或未通过采集自检，请先续采并通过自检")
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
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 0:
        if catalog.is_file():
            with catalog.open("rb") as handle:
                frames = sum(1 for line in handle if line.strip())
        else:
            frames = sum(1 for _ in (directory / "images").glob("*.jpg")) if (directory / "images").is_dir() else 0
    default = path == DEFAULT_DATA
    created = meta.get("created_unix_s")
    created = created if isinstance(created, (int, float)) and math.isfinite(created) else None
    summary = meta.get("summary") if isinstance(meta.get("summary"), dict) else {}
    status = meta.get("status") if isinstance(meta.get("status"), str) else None
    return {"data": path, "id": dataset_key(path), "default": default, "frames": int(frames),
            "ready": (directory / "images").is_dir() and frames > 0 and status not in ("collecting", "interrupted")
                     and summary.get("accepted") is not False,
            "status": status, "accepted": summary.get("accepted") if isinstance(summary.get("accepted"), bool) else None,
            "environment": meta.get("environment") or ("donkey-generated-roads-v0" if default else None),
            "simulator": meta.get("simulator") or ("donkey" if default else None),
            "image": meta.get("image") or ({"width": 160, "height": 120} if default else None),
            "image_size": meta.get("image_size"), "source": meta.get("source") or ("人工驾驶记录" if default else None),
            "created_unix_s": created, "summary": summary}


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

    def __init__(self, directory: str | Path, *, meta: dict, extra_fields: dict | None = None, jpeg_quality: int = 95,
                 resume: bool = False, checkpoint_every: int = 100):
        from apps.common.gpu_lock import GpuLock

        self.meta = dict(meta)
        self.extra_fields = dict(extra_fields or {})
        if set(self.extra_fields.values()) - {"float", "int", "boolean", "str"}:
            raise DatasetError("不支持的 catalog 附加字段类型")
        self.quality = int(jpeg_quality)
        self.checkpoint_every = int(checkpoint_every)
        if self.checkpoint_every < 1 or not 1 <= self.quality <= 100:
            raise DatasetError("checkpoint_every 必须为正数，JPEG 质量必须在 1..100")
        self.directory = Path(directory)
        self.images = self.directory / "images"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = GpuLock(self.directory / ".writer.lock", label="dataset writer", say=lambda _: None)
        try:
            self._lock.acquire(timeout_s=0.)
        except TimeoutError as exc:
            raise DatasetError("数据集正在被另一个采集进程写入") from exc
        self._catalog = None
        self._final_meta = None
        self.count = 0
        self.angles: list[float] = []
        self.line_lengths: list[int] = []
        self.records: list[dict] = []  # existing records on resume, used to recover frame/episode logs
        self.size = None
        self.created = time.time()
        try:
            if resume:
                self._recover()
            elif (self.directory / CATALOG).exists() or (self.images.is_dir() and any(self.images.iterdir())):
                raise DatasetError(f"数据集目录已有数据：{self.directory}")
            self.images.mkdir(parents=True, exist_ok=True)
            self._catalog = (self.directory / CATALOG).open("a" if resume else "w", encoding="utf-8", newline="\n")
            self.snapshot()
        except BaseException:
            if self._catalog is not None:
                self._catalog.close()
            self._lock.release()
            raise

    def _recover(self) -> None:
        """Keep complete catalog records, quarantine an incomplete tail and orphan images, then append."""
        from PIL import Image

        previous = read_meta(self.directory)
        if not previous or not (self.directory / CATALOG).is_file():
            raise DatasetError("续采需要已有的 dataset.json 和 catalog")
        if previous.get("schema_version") != SCHEMA_VERSION:
            raise DatasetError("不支持该数据集版本的续采")
        for key in ("id", "simulator", "environment", "image_size", "expert", "learner"):
            if previous.get(key) != self.meta.get(key):
                raise DatasetError(f"续采配置与原数据集不同：{key}")
        # A change to the image calibration changes the expert labels even if its gains stay the same.
        old_collection, new_collection = previous.get("collection") or {}, self.meta.get("collection") or {}
        for key in ("horizon", "hood_rows", "chroma_tolerance"):
            if old_collection.get(key) != new_collection.get(key):
                raise DatasetError(f"续采配置与原数据集不同：{key}")
        self.created = float(previous.get("created_unix_s") or self.created)
        if not math.isfinite(self.created):
            raise DatasetError("数据集创建时间无效")
        history = list(previous.get("collection_runs") or [])
        if not history:
            history.append({"started_unix_s": self.created, "collection": previous.get("collection"),
                            "code": previous.get("code")})
        history.append({"resumed_unix_s": time.time(), "previous_frames": previous.get("frames"),
                        "collection": self.meta.get("collection"), "code": self.meta.get("code")})
        self.meta = {**previous, **self.meta, "collection_runs": history}
        raw = (self.directory / CATALOG).read_bytes()
        position = 0
        tail = b""
        names = set()
        missing_newline = False
        for line in raw.splitlines(keepends=True):
            complete = line.endswith(b"\n")
            try:
                record = json.loads(line)
            except ValueError as exc:
                if complete:
                    raise DatasetError("catalog 中存在损坏的完整记录，不能自动续采") from exc
                tail = raw[position:]
                break
            try:
                name = record["cam/image_array"]
                if not isinstance(name, str) or Path(name).name != name or name in names:
                    raise ValueError("invalid or duplicate image name")
                if not re.fullmatch(rf"{len(self.records)}_[^/\\]+\.jpg", name):
                    raise ValueError("image numbers are not consecutive")
                image = self.images / name
                if image.resolve().parent != self.images.resolve():
                    raise ValueError("image path escapes the dataset")
                angle, throttle = float(record["user/angle"]), float(record["user/throttle"])
                if not math.isfinite(angle) or not math.isfinite(throttle):
                    raise ValueError("nonfinite labels")
                with Image.open(image) as header:
                    size = header.size
                if self.size is not None and size != self.size:
                    raise ValueError("image dimensions differ")
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise DatasetError(f"catalog 记录无法恢复：{exc}") from exc
            self.size = size
            names.add(name)
            self.records.append(record)
            self.angles.append(angle)
            self.line_lengths.append(len(line) + (0 if complete else 1))
            position += len(line)
            missing_newline = not complete
        self.count = len(self.records)
        orphans = [p for p in self.images.glob("*.jpg*") if p.name not in names and p.is_file()]
        if tail or orphans:
            quarantine = self.directory / "recovery" / str(time.time_ns())
            quarantine.mkdir(parents=True)
            if not quarantine.resolve().is_relative_to(self.directory.resolve()):
                raise DatasetError("恢复目录离开了数据集")
            if tail:
                (quarantine / "catalog-tail.bin").write_bytes(tail)
            for orphan in orphans:
                orphan.replace(quarantine / orphan.name)
        if tail or missing_newline:
            with (self.directory / CATALOG).open("r+b") as handle:
                handle.truncate(position)
                if missing_newline:
                    handle.seek(0, 2)
                    handle.write(b"\n")

    def add(self, rgb, angle: float, throttle: float, *, mode: str = "expert", **extra) -> str:
        import cv2

        if self._catalog is None or self._catalog.closed:
            raise DatasetError("数据集写入器已关闭")
        frame = np.ascontiguousarray(rgb)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or min(frame.shape[:2]) < 1:
            raise DatasetError(f"图像必须是 HxWx3 的 uint8：{frame.shape} {frame.dtype}")
        size = (int(frame.shape[1]), int(frame.shape[0]))
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise DatasetError("JPEG 编码失败")
        return self._append(encoded.tobytes(), size, angle, throttle, mode, extra)

    def add_jpeg(self, payload: bytes, angle: float, throttle: float, *, mode: str = "expert", **extra) -> str:
        """Copy an existing JPEG without re-encoding it, for aggregating collection rounds."""
        import io
        from PIL import Image

        try:
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != "JPEG" or image.width * image.height > 4096 * 4096 or not payload.endswith(b"\xff\xd9"):
                    raise DatasetError("合并需要完整的 JPEG，且不超过 4096×4096 像素")
                size = image.size
        except (OSError, Image.DecompressionBombError) as exc:
            raise DatasetError("JPEG 文件头无效") from exc
        return self._append(payload, size, angle, throttle, mode, extra)

    def _append(self, payload, size, angle, throttle, mode, extra) -> str:
        if self._catalog is None or self._catalog.closed:
            raise DatasetError("数据集写入器已关闭")
        angle, throttle = float(angle), float(throttle)
        if not (math.isfinite(angle) and math.isfinite(throttle)):
            raise DatasetError("标签必须是有限数值")
        if self.size is not None and size != self.size:
            raise DatasetError(f"图像尺寸变化：{size} != {self.size}")
        unknown = set(extra) - set(self.extra_fields)
        if unknown:
            raise DatasetError(f"未声明的附加字段：{', '.join(sorted(unknown))}")
        if not isinstance(mode, str):
            raise DatasetError("user/mode 必须是字符串")
        extra = dict(extra)
        for key, value in extra.items():
            if value is None:
                continue
            kind = self.extra_fields[key]
            valid = ((kind == "float" and isinstance(value, numbers.Real) and not isinstance(value, bool)
                      and math.isfinite(value))
                     or (kind == "int" and isinstance(value, numbers.Integral) and not isinstance(value, bool))
                     or (kind == "boolean" and isinstance(value, (bool, np.bool_)))
                     or (kind == "str" and isinstance(value, str)))
            if not valid:
                raise DatasetError(f"catalog 字段类型无效：{key} 应为 {kind}")
            extra[key] = float(value) if kind == "float" else int(value) if kind == "int" else bool(value) if kind == "boolean" else value
        name = f"{self.count}_{angle:.4f}.jpg"
        record = {"cam/image_array": name, "user/angle": round(angle, 6), "user/throttle": round(throttle, 6),
                  "user/mode": mode, **extra}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        if (self.images / name).exists():
            raise DatasetError(f"拒绝覆盖已有图像：{name}")
        temporary = self.images / (name + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(self.images / name)
        self._catalog.write(line)
        self._catalog.flush()
        self.count += 1
        self.size = size
        self.angles.append(angle)
        self.line_lengths.append(len(line.encode("utf-8")))
        if self.count % self.checkpoint_every == 0:
            self.snapshot()
        return name

    @staticmethod
    def _atomic(path: Path, text: str) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(path)

    def snapshot(self, *, summary: dict | None = None, status: str = "collecting") -> dict:
        """Publish a recoverable checkpoint; metadata is replaced after the catalog and manifests."""
        if self._catalog is None or self._catalog.closed:
            raise DatasetError("数据集写入器已关闭")
        if self._catalog is not None and not self._catalog.closed:
            self._catalog.flush()
        names = [name for name, _ in TUB_FIELDS] + list(self.extra_fields)
        types = [kind for _, kind in TUB_FIELDS] + list(self.extra_fields.values())
        session = str(self.meta.get("id") or self.directory.name)
        lines = [names, types, {},
                 {"created_at": self.created, "sessions": {"all_full_ids": [session], "last_id": 0, "last_full_id": session}},
                 {"paths": [CATALOG], "current_index": self.count, "max_len": self.count, "deleted_indexes": []}]
        self._atomic(self.directory / "manifest.json", "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines))
        self._atomic(self.directory / f"{CATALOG}_manifest",
                     json.dumps({"created_at": self.created, "line_lengths": self.line_lengths,
                                 "path": f"{CATALOG}_manifest", "start_index": 0}) + "\n")
        meta = {"schema_version": SCHEMA_VERSION, **self.meta, "created_unix_s": self.created, "frames": self.count,
                "image": {"width": self.size[0], "height": self.size[1]} if self.size else None,
                "catalog": CATALOG, "status": status, "labels": label_summary(self.angles), "summary": summary or {}}
        self._atomic(self.directory / "dataset.json", json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False))
        return meta

    def close(self, *, summary: dict | None = None, status: str = "complete") -> dict:
        if self._final_meta is not None:
            return self._final_meta
        try:
            self._final_meta = self.snapshot(summary=summary, status=status)
            return self._final_meta
        finally:
            if self._catalog is not None:
                self._catalog.close()
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._catalog is not None and not self._catalog.closed:
            self.close(status="interrupted", summary={"accepted": False, "ended_early": "collection interrupted"})
