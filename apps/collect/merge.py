"""Aggregate completed collection rounds without changing image bytes or losing their provenance."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

from apps.common.datasets import (CATALOG, COLLECTED_ROOT, DATASET_ID, DatasetError, TubWriter, dataset_dir,
                                  describe, label_summary, normalize, read_meta)
from apps.common.donkey_data import load_donkey_index
from apps.common.gpu_lock import GpuLock
from apps.eval.simulator import ROOT
from .recorder import EXTRA_FIELDS


def merge(root: Path, name: str, sources: list[str]) -> dict:
    paths = [normalize(source) for source in sources]
    if not DATASET_ID.fullmatch(name) or len(paths) < 2 or len(set(paths)) != len(paths):
        raise DatasetError("需要新的数据集 id 和至少两个不同的来源数据集")
    out = Path(root) / COLLECTED_ROOT / name
    if out.exists():
        raise DatasetError("合并输出目录已存在")
    rows, origins = [], []
    with ExitStack() as stack:
        for source in sorted(paths):
            directory = dataset_dir(root, source)
            lock = GpuLock(directory / ".writer.lock", say=lambda _: None, label="dataset merge")
            try:
                lock.acquire(timeout_s=0.)
            except TimeoutError as exc:
                raise DatasetError("来源数据集正在写入") from exc
            stack.callback(lock.release)
            meta = read_meta(directory)
            if meta.get("status") == "collecting":
                raise DatasetError(f"来源数据集需要先完成或恢复采集：{source}")
            if (meta.get("summary") or {}).get("accepted") is False:
                raise DatasetError(f"来源未通过采集自检：{source}")
            catalog = directory / CATALOG
            if not catalog.is_file():
                raise DatasetError(f"来源缺少 catalog：{source}")
            records = [json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines() if line.strip()]
            index = load_donkey_index(directory, .5, 1.)
            if not records or len(records) != len(index) or len({str(r[0]) for r in index}) != len(index):
                raise DatasetError(f"来源含无效或缺失记录：{source}")
            description = describe(root, source)
            catalog_digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
            content = hashlib.sha256(bytes.fromhex(catalog_digest))
            images_root = (directory / "images").resolve()
            for rel, _, _ in index:
                image = directory / rel
                if not image.resolve().is_relative_to(images_root):
                    raise DatasetError("来源图像离开了 images 目录")
                content.update(hashlib.sha256(image.read_bytes()).digest())
            origins.append({"data": source, "frames": len(index), "environment": description["environment"],
                            "simulator": description["simulator"], "image_size": description["image_size"],
                            "catalog_sha256": catalog_digest, "content_sha256": content.hexdigest(), "expert": meta.get("expert")})
            rows.append((source, directory, records, index))
        if len({r["environment"] for r in origins}) != 1 or len({r["image_size"] for r in origins}) != 1:
            raise DatasetError("不能合并不同模拟环境或不同建议输入尺寸的数据")
        if len({r["content_sha256"] for r in origins}) != len(origins):
            raise DatasetError("来源包含相同 catalog 和图像的重复副本")
        fields = {**EXTRA_FIELDS, "merge/source": "str", "merge/source_image": "str", "merge/source_episode": "int"}
        meta = {"id": name, "simulator": origins[0]["simulator"], "environment": origins[0]["environment"],
                "image_size": origins[0]["image_size"], "source": f"合并采集数据（{len(origins)} 组）", "sources": origins,
                "expert": {"kind": "aggregate", "catalogs": [r["catalog_sha256"] for r in origins]}}
        with TubWriter(out, meta=meta, extra_fields=fields) as writer:
            episodes = {}
            for source, directory, records, index in rows:
                images_root = (directory / "images").resolve()
                for record, (rel, angle, throttle) in zip(records, index):
                    image = directory / rel
                    if not image.resolve().is_relative_to(images_root):
                        raise DatasetError("来源图像离开了 images 目录")
                    old_episode = record.get("collect/episode", 0)
                    if not isinstance(old_episode, int) or old_episode < 0:
                        raise DatasetError("来源 episode 无效")
                    episode = episodes.setdefault((source, old_episode), len(episodes))
                    extra = {key: record.get(key) for key in EXTRA_FIELDS}
                    extra.update({"collect/episode": episode, "merge/source": source,
                                  "merge/source_image": record["cam/image_array"], "merge/source_episode": old_episode})
                    writer.add_jpeg(image.read_bytes(), angle, throttle, mode=record.get("user/mode", "expert"), **extra)
            return writer.close(summary={"accepted": True, "frames_stored": writer.count,
                                         "episodes": len(episodes), "labels": label_summary(writer.angles)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    args = parser.parse_args(argv)
    try:
        result = merge(ROOT, args.name, args.sources)
    except (DatasetError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{result['frames']} frames in {COLLECTED_ROOT}/{args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
