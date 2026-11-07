"""Train the isomorphic PyTorch small CNN on a completed CIFAR-10 run's exact NPZ and batch order."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from apps.common.cifar10 import load

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    run = (ROOT / args.run).resolve()
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))["args"]
    if config["arch"] != "small":
        parser.error("the paired PyTorch protocol uses the small CNN")
    import torch
    from torch import nn
    import torch.nn.functional as F
    from apps.common.gpu_lock import GpuLock

    width = config["width"]
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, width, 3, padding=1)
            self.conv2 = nn.Conv2d(width, 2*width, 3, padding=1)
            self.hidden = nn.Linear(2*width*4*4, 4*width)
            self.output = nn.Linear(4*width, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = F.adaptive_avg_pool2d(x, 4).flatten(1)
            return self.output(F.relu(self.hidden(x)))

    device = config["device"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    images, labels = load(ROOT / config["data"], "train")
    with np.load(run / "split-and-order.npz", allow_pickle=False) as data:
        training, validation = data["batches"], data["val"]
    with GpuLock(label="CIFAR-10 PyTorch comparison", run_dir=run, enabled=device=="cuda"):
        model = Model().to(device)
        with np.load(run / "initial.npz", allow_pickle=False) as data:
            model.load_state_dict({k: torch.from_numpy(data[k]) for k in data.files}, strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(.9, .999), eps=1e-8,
                                     weight_decay=0., foreach=False, fused=False)
        def batch(ids):
            return torch.from_numpy(images[ids].astype(np.float32)/255.).to(device), torch.from_numpy(labels[ids]).to(device)
        history = []
        for epoch, batches in enumerate(training, 1):
            start = time.perf_counter()
            model.train()
            train_loss = 0.
            for ids in batches:
                x, y = batch(ids)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            correct = val_loss = 0.
            model.eval()
            with torch.no_grad():
                for first in range(0, len(validation), config["batch"]):
                    ids = validation[first:first+config["batch"]]
                    x, y = batch(ids)
                    logits = model(x)
                    correct += (logits.argmax(1) == y).sum().item()
                    val_loss += F.cross_entropy(logits, y, reduction="sum").item()
            history.append({"epoch": epoch, "train_loss": train_loss/len(batches), "val_loss": val_loss/len(validation),
                            "val_accuracy": correct/len(validation), "epoch_s": time.perf_counter()-start})
            print(f"PyTorch epoch {epoch} val_accuracy={correct/len(validation):.4f}", flush=True)
    own = json.loads((run / "result.json").read_text(encoding="utf-8"))["epochs"][-1]["val_accuracy"]
    difference = 100 * (own - history[-1]["val_accuracy"])
    report = {"schema_version": 1, "torch": torch.__version__, "device": device, "epochs": history,
              "myflows_final_val_accuracy": own, "difference_points": difference,
              "within_three_points": abs(difference) <= 3.0 + 1e-10, "actual_training_samples": int(training.size)}
    (run / "pytorch-validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["within_three_points"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
