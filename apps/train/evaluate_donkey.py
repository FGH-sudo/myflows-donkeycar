"""Score saved ResNet-18 DonkeyCar models on one split of a split file (AutoPilot's independent test set).

    python -m apps.train.evaluate_donkey --items items.json --out evaluation.json [--gpu-lock-timeout 0]

``items.json``::

    {"split": "test", "items": [{"key": "t001", "checkpoint": ".../checkpoints/best", "args": {...}}]}

``args`` are the apps.train arguments the model was trained with (from the trial); they select the same rows
(``--data``, ``--max-samples``, ``--split-file``) and rebuild the same model (width, stem, dtype, backend).
Each item reports ``loss`` (the training loss, MSE/2 over angle and throttle) and ``angle_mae`` over the
split; models are evaluated in inference mode (BN running statistics, no dropout). The GPU lock is taken
once for all items. Exit code 0 when every item was scored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from apps.common.gpu_lock import GpuLock
from apps.common.splits import select_split
from apps.console.presets import single_cli_args
from apps.train import train_myflows_donkey as train
from apps.train.common.outcome import write_result

ROOT = Path(__file__).resolve().parents[2]


def evaluate(item: dict, split: str) -> dict:
    import MyFlows as ms
    from MyFlows.core.device import asnumpy, set_device
    from MyFlows.data.pipeline import MultiprocessDataLoader

    args = train.build_parser().parse_args(single_cli_args(item["args"]))
    device = train.resolve_myflows_device(args.device)
    set_device(device)
    dtype = np.float32 if args.dtype == "float32" else np.float64
    ms.set_default_dtype(dtype)
    conv_backend = train._resolve_conv_backend(args.conv_backend, device, dtype)
    data_dir = (ROOT / args.data).resolve()
    index, splits, _path = train.load_rows(args, data_dir)
    if splits is None:
        raise SystemExit("评估需要划分文件（--split-file）")
    rows = select_split(index, splits, split)
    if not rows:
        raise SystemExit(f"划分中没有 {split} 样本")
    first = train.cv2.imread(str(data_dir / rows[0][0]))
    h, w = train.input_size(args, first)
    batch = int(args.batch)
    x_var, y_var, model, logits, _loss, graph = train._build_model(args, batch, h, w, dtype, conv_backend,
                                                                   load_pretrained=False)
    ms.load_checkpoint([model], None, str(item["checkpoint"]))
    model.eval()
    samples = [(str(data_dir / rel), angle, throttle, w, h, args.dtype) for rel, angle, throttle in rows]
    squared = angle_abs = 0.0
    count = 0
    for x_items, y_items in MultiprocessDataLoader(samples, batch, num_workers=0, shuffle=False,
                                                   load_fn=train._load_training_sample):
        real = len(x_items)
        x_batch = np.asarray(x_items, dtype=dtype)
        y_batch = np.asarray(y_items, dtype=dtype).reshape(-1, 2)
        x_batch, y_batch = train._pad_fixed_batch(x_batch, y_batch, batch)
        x_var.value, y_var.value = x_batch, y_batch
        graph.forward()
        pred = np.asarray(asnumpy(logits.value), dtype=np.float64)[:real]
        target = y_batch[:real].astype(np.float64)
        squared += float(np.sum((pred - target) ** 2)) / 2.0
        angle_abs += float(np.sum(np.abs(pred[:, 0] - target[:, 0])))
        count += real
    return {"key": item["key"], "split": split, "samples": count, "loss": squared / count,
            "angle_mae": angle_abs / count, "checkpoint": str(item["checkpoint"]), "device": device,
            "conv_backend": conv_backend}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-dir", default=None, help="控制台任务目录（GPU 锁事件与 result.json）")
    ap.add_argument("--gpu-lock-timeout", type=float, default=0.0)
    args, _unknown = ap.parse_known_args(argv)
    spec = json.loads(Path(args.items).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.out).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    lock = GpuLock(label="evaluate_donkey", run_dir=run_dir, say=print)
    lock.acquire(timeout_s=args.gpu_lock_timeout or None)
    results, failures = [], []
    try:
        for item in spec["items"]:
            try:
                results.append(evaluate(item, spec.get("split", "test")))
                print(f"[evaluate] {item['key']}: {results[-1]['angle_mae']:.6f}")
            except (Exception, SystemExit) as exc:
                failures.append({"key": item["key"], "error": f"{type(exc).__name__}: {exc}"})
                print(f"[evaluate] {item['key']} 失败：{exc}", file=sys.stderr)
    finally:
        lock.release()
    payload = {"schema_version": 1, "split": spec.get("split", "test"), "results": results, "failures": failures,
               "elapsed_s": time.time() - started}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    write_result(run_dir, {"status": "failed" if failures else "succeeded", "exit_code": 1 if failures else 0,
                           "task": "evaluate_donkey", "metrics": {}, "evaluation": payload})
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
