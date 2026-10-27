"""E2 evidence per pretrained model: the full load report (mapping list, success rate, missing / unexpected,
ignored buffers, source and converted checksums) and the output comparison with the source model on the same
real images and preprocessing (semester_spec §4.6: FP32 atol=1e-4, rtol=1e-3).

    .venv/Scripts/python.exe -m tools.pretrained_report [--arch resnet18 ...] [--device cpu|cuda]

Writes ``runs/pretrained/converted/<arch>.report.json`` and prints one line per model. The source model is the
torch.nn re-implementation in ``MyFlows/tests/torch_reference.py`` loaded with the official state_dict
(strict=True); the converted .npz it reads was produced by ``tools/convert_torchvision_weights.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "runs" / "pretrained" / "converted"
ATOL, RTOL = 1e-4, 1e-3


def frames(count: int = 4, size: int = 224) -> tuple[np.ndarray, list[str]]:
    from apps.common.image_preprocess import imread_nchw

    paths = sorted((ROOT / "mycar" / "data" / "images").glob("*.jpg"))[:count]
    if len(paths) < count:
        raise SystemExit("需要 mycar/data/images 下的真实图片")
    return np.concatenate([imread_nchw(p, (size, size), np.float32) for p in paths]), [p.name for p in paths]


def report(arch: str, device: str) -> dict:
    import torch

    from MyFlows.core.device import asnumpy, set_device, xp
    from MyFlows.core.graph import Graph
    from MyFlows.core.node import Variable
    from MyFlows.ops.pooling import IMAGENET_MEAN, IMAGENET_STD
    from MyFlows.tests.torch_reference import TORCH_MODELS
    from MyFlows.utils import pretrained

    path = CONVERTED / f"{arch}.npz"
    x, names = frames()
    mean = np.asarray(IMAGENET_MEAN, np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(IMAGENET_STD, np.float32).reshape(1, 3, 1, 1)
    reference = TORCH_MODELS[arch]()
    with np.load(path) as data:
        reference.load_state_dict({k: torch.from_numpy(data[k]) for k in data.files}, strict=True)
    reference.eval()
    with torch.no_grad():
        expected = reference(torch.tensor((x - mean) / std)).numpy().astype(np.float64)

    set_device(device)
    model = pretrained.build_model(arch, 1000, input_norm="imagenet",
                                   conv_backend="cuda_native_register" if device == "cuda" else "auto")
    load = pretrained.load_torchvision_weights(model, path)
    model.eval()
    x_var = Variable(xp.asarray(x), name="X", requires_grad=False)
    out = model(x_var)
    Graph(out).forward()
    actual = np.asarray(asnumpy(out.value), dtype=np.float64)
    diff = np.abs(actual - expected)
    bound = ATOL + RTOL * np.abs(expected)
    top5 = lambda z: np.argsort(-z, axis=1)[:, :5]  # noqa: E731
    return {
        "arch": arch, "device": device, "load_report": load,
        "comparison": {"images": names, "input": "224x224 RGB /255, ImageNet mean/std", "atol": ATOL, "rtol": RTOL,
                       "max_abs_diff": float(diff.max()), "max_rel_diff": float((diff / np.maximum(np.abs(expected), 1e-12)).max()),
                       "within_tolerance": bool((diff <= bound).all()), "logit_abs_max": float(np.abs(expected).max()),
                       "top1_equal": bool(np.array_equal(actual.argmax(1), expected.argmax(1))),
                       "top5_equal": bool(np.array_equal(top5(actual), top5(expected))),
                       "top1": expected.argmax(1).tolist()},
        "source_reference": "MyFlows/tests/torch_reference.py (torchvision definitions, strict state_dict load)",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", nargs="*", default=["resnet18", "mobilenet_v2", "alexnet"])
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = ap.parse_args(argv)
    ok = True
    for arch in args.arch:
        result = report(arch, args.device)
        out = CONVERTED / f"{arch}.report.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        load, cmp_ = result["load_report"], result["comparison"]
        ok &= cmp_["within_tolerance"] and load["success_rate"] == 1.0
        print(f"{arch}: loaded {load['loaded']}/{load['expected']} (ignored {load['ignored']['keys']} "
              f"num_batches_tracked), max |diff| {cmp_['max_abs_diff']:.2e} (logits up to {cmp_['logit_abs_max']:.1f}), "
              f"within atol/rtol={cmp_['within_tolerance']}, top-1 equal={cmp_['top1_equal']}, "
              f"top-5 equal={cmp_['top5_equal']} -> {out.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
