"""Compare a trained, unmerged LoRA checkpoint with its merged ONNX on fixed validation images."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def verify(run, tree):
    sys.path.insert(0, str(tree.resolve()))
    import onnxruntime as ort
    import MyFlows as ms
    from apps.common.image_preprocess import imread_nchw
    from apps.train import train_myflows_donkey as train

    ms.set_device("cpu")
    ms.set_default_dtype(np.float32)
    config = json.loads((run/"config.json").read_text(encoding="utf-8"))
    args = train.build_parser().parse_args([])
    for name, value in config["args"].items():
        if hasattr(args, name):
            setattr(args, name, value)
    h, w = config["input_h"], config["input_w"]
    x, _, model, logits, _, _ = train._build_model(args, 1, h, w, np.float32, "auto", load_pretrained=False)
    ms.load_checkpoint([model], None, str(run/"best"))
    model.eval()
    graph = ms.Graph(logits)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(run/"best.onnx"), options, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1:
        raise ValueError("deployment model must accept only the camera input")
    rows, splits, _ = train.load_rows(args, Path(args.data))
    selected = np.linspace(0, len(splits["val"])-1, min(8, len(splits["val"])), dtype=int)
    evidence = []
    for position in selected:
        rel = rows[splits["val"][position]][0]
        image = imread_nchw(Path(args.data)/rel, (w, h), np.float32)
        x.value = image
        graph.forward()
        output, = session.run(None, {session.get_inputs()[0].name: image})
        np.testing.assert_allclose(output, logits.value, rtol=3e-5, atol=1e-5)
        evidence.append({"frame": str(rel), "native": logits.value.tolist(), "onnx": output.tolist(),
                         "max_abs": float(np.max(np.abs(output-logits.value)))})
    with (run/"best.onnx").open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"sha256": digest, "samples": evidence, "max_abs": max(v["max_abs"] for v in evidence)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tree", type=Path, default=ROOT)
    options = parser.parse_args()
    evidence = verify(options.run.resolve(), options.tree)
    (options.run/"onnx-parity.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(evidence["max_abs"])
