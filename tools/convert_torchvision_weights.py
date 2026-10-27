"""把 PyTorch 官方 torchvision ImageNet 权重（.pth）转换成 MyFlows 读取的 .npz + .json manifest。

下载的 .pth 是不受信任的数据：本脚本应以隔离模式运行（``python -I``，不把当前目录和脚本目录加入 sys.path），
用 ``torch.load(weights_only=True)`` 只反序列化张量，不导入本项目的任何模块。输出写到单独的目录。

    .venv/Scripts/python.exe -I tools/convert_torchvision_weights.py --arch resnet18 \\
        --pth runs/pretrained/imagenet/resnet18/resnet18-f37072fd.pth --out runs/pretrained/converted/resnet18.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

URLS = {
    "resnet18": ("resnet18-f37072fd.pth", "ResNet18_Weights.IMAGENET1K_V1"),
    "mobilenet_v2": ("mobilenet_v2-b0353104.pth", "MobileNet_V2_Weights.IMAGENET1K_V1"),
    "alexnet": ("alexnet-owt-7be5be79.pth", "AlexNet_Weights.IMAGENET1K_V1"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(arch: str, pth: Path, out: Path) -> dict:
    import numpy as np
    import torch

    expected_name, weights = URLS[arch]
    if pth.name != expected_name:
        raise SystemExit(f"{pth.name} is not the official {arch} file ({expected_name})")
    source_sha = sha256(pth)
    prefix = re.search(r"-([0-9a-f]{8})\.pth$", pth.name).group(1)
    if not source_sha.startswith(prefix):
        raise SystemExit(f"{pth} sha256 {source_sha[:16]}... does not start with {prefix}")

    state = torch.load(pth, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise SystemExit(f"{pth} is not a plain state_dict of tensors")
    arrays = {key: value.detach().cpu().numpy() for key, value in state.items()}

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    manifest = {
        "arch": arch,
        "weights": weights,
        "source_file": pth.name,
        "source_url": f"https://download.pytorch.org/models/{pth.name}",
        "source_sha256": source_sha,
        "source_bytes": pth.stat().st_size,
        "torch_version": torch.__version__,
        "converted_unix_s": time.time(),
        "npz_sha256": sha256(out),
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
        "parameter_values": int(sum(value.size for key, value in arrays.items() if value.dtype.kind == "f")),
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", required=True, choices=sorted(URLS))
    ap.add_argument("--pth", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    manifest = convert(args.arch, args.pth.resolve(), args.out.resolve())
    print(f"{args.arch}: {len(manifest['tensors'])} tensors, {manifest['parameter_values']:,} values, "
          f"source sha256 {manifest['source_sha256'][:16]}..., npz sha256 {manifest['npz_sha256'][:16]}... -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
