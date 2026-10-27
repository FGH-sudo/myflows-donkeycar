"""Stand-in for apps.train.evaluate_donkey: same command line and output, deterministic made-up scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from apps.common.gpu_lock import GpuLock


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-dir")
    args, _ = ap.parse_known_args()
    spec = json.loads(Path(args.items).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir or Path(args.out).parent)
    with GpuLock(label="fake_evaluate", run_dir=run_dir, say=print):
        results = [{"key": item["key"], "split": spec.get("split", "test"), "samples": 5, "loss": 0.01,
                    "angle_mae": 0.1 + int(hashlib.sha256(item["key"].encode()).hexdigest(), 16) % 100 / 1000.0,
                    "checkpoint": item["checkpoint"]} for item in spec["items"]]
    Path(args.out).write_text(json.dumps({"schema_version": 1, "split": spec.get("split", "test"),
                                          "results": results, "failures": []}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
