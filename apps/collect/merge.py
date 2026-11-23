"""Aggregate completed collection rounds without changing image bytes or losing their provenance."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
from pathlib import Path
import tempfile

from apps.common.datasets import (CATALOG, COLLECTED_ROOT, DATASET_ID, DatasetError, TubWriter, dataset_dir,
                                  describe, label_summary, normalize, require_usable_dataset)
from apps.common.donkey_data import read_donkey_catalog
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
            meta = require_usable_dataset(directory)
            catalog = directory / CATALOG
            if not catalog.is_file():
                raise DatasetError(f"来源缺少 catalog：{source}")
            records = list(read_donkey_catalog(directory, catalog, .5, 1., strict=True))
            if not records:
                raise DatasetError(f"来源 catalog 不含可用记录：{source}")
            description = describe(root, source)
            catalog_digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
            origins.append({"data": source, "frames": len(records), "environment": description["environment"],
                            "simulator": description["simulator"], "image_size": description["image_size"],
                            "catalog_sha256": catalog_digest, "expert": meta.get("expert")})
            rows.append((directory, records))
        if len({r["environment"] for r in origins}) != 1 or len({r["image_size"] for r in origins}) != 1:
            raise DatasetError("不能合并不同模拟环境或不同建议输入尺寸的数据")
        fields = {**EXTRA_FIELDS, "merge/source": "str", "merge/source_image": "str", "merge/source_episode": "int"}
        meta = {"id": name, "simulator": origins[0]["simulator"], "environment": origins[0]["environment"],
                "image_size": origins[0]["image_size"], "source": f"合并采集数据（{len(origins)} 组）",
                "expert": {"kind": "aggregate", "catalogs": [r["catalog_sha256"] for r in origins]}}
        out.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(stack.enter_context(tempfile.TemporaryDirectory(dir=out.parent, prefix=f".merge-{name}-"))) / name
        fingerprints = set()
        with TubWriter(staging, meta=meta, extra_fields=fields) as writer:
            episodes = {}
            for origin, (directory, records) in zip(origins, rows, strict=True):
                source = origin["data"]
                content = hashlib.sha256(bytes.fromhex(origin["catalog_sha256"]))
                for record, (rel, angle, throttle) in records:
                    old_episode = record.get("collect/episode", 0)
                    if isinstance(old_episode, bool) or not isinstance(old_episode, int) or old_episode < 0:
                        raise DatasetError("来源 episode 无效")
                    episode = episodes.setdefault((source, old_episode), len(episodes))
                    extra = {key: record.get(key) for key in EXTRA_FIELDS}
                    extra.update({"collect/episode": episode, "merge/source": source,
                                  "merge/source_image": record["cam/image_array"], "merge/source_episode": old_episode})
                    jpeg = (directory / rel).read_bytes()
                    writer.add_jpeg(jpeg, angle, throttle, mode=record.get("user/mode", "expert"), **extra)
                    content.update(hashlib.sha256(jpeg).digest())
                fingerprint = content.hexdigest()
                if fingerprint in fingerprints:
                    raise DatasetError("来源包含相同 catalog 和图像的重复副本")
                fingerprints.add(fingerprint)
                origin["content_sha256"] = fingerprint
            writer.meta["sources"] = origins
            result = writer.close(summary={"accepted": True, "frames_stored": writer.count,
                                           "episodes": len(episodes), "labels": label_summary(writer.angles)})
        staging.rename(out)
        return result


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
