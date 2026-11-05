"""Isolated MyFlows/PyTorch workers for the paired ResNet18 protocol."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import runpy
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class MyFlowsRun:
    def __init__(self, initial, batch):
        import MyFlows as ms
        from MyFlows.core.device import _lazy_cupy, asnumpy, set_device
        from MyFlows.utils.pretrained import build_model, load_torchvision_weights

        set_device("cuda")
        ms.set_default_dtype(np.float32)
        self.cp, self.host = _lazy_cupy(), asnumpy
        self.pool = self.cp.get_default_memory_pool()
        self.model = build_model("resnet18", 2, dropout=0., conv_backend="cuda_native_register")
        load_torchvision_weights(self.model, initial, strict=True)
        self.x = ms.Variable(np.zeros((batch, 3, 120, 160), np.float32), requires_grad=False)
        self.y = ms.Variable(np.zeros((batch, 2), np.float32), requires_grad=False)
        self.logits = self.model(self.x)
        self.loss = ms.MSELoss(self.logits, self.y)
        self.graph = ms.Graph(self.loss)
        self.opt = ms.Adam(self.graph, learning_rate=.001)
        self.peak_used = self.peak_total = 0
        self.info = {"framework": "MyFlows", "conv_backend": "cuda_native_register", "cupy": self.cp.__version__,
                     "tf32": False, "dtype": "float32"}

    def device_batch(self, x, y):
        return self.cp.asarray(x), self.cp.asarray(y)

    def event(self):
        return self.cp.cuda.Event()

    def elapsed(self, a, b):
        return float(self.cp.cuda.get_elapsed_time(a, b))

    def sync(self):
        self.cp.cuda.get_current_stream().synchronize()

    def sample_memory(self):
        self.peak_used = max(self.peak_used, self.pool.used_bytes())
        self.peak_total = max(self.peak_total, self.pool.total_bytes())

    def memory(self):
        return {"allocator": "CuPy pool, sampled at phase boundaries", "peak_used_mib": self.peak_used / 2**20,
                "peak_reserved_mib": self.peak_total / 2**20}

    def tensors(self, *, gradients=False):
        result = {}
        for name, target, kind in self.model.torchvision_tensors():
            if kind == "buffer":
                continue
            value = self.opt.get_gradient(target) if gradients else target.value
            array = self.host(value)
            result[name] = np.ascontiguousarray(array.T if kind == "param_t" else array)
        return result

    def train(self, x, y, phase, capture):
        self.x.value, self.y.value = x, y
        self.opt.one_step(phase)
        gradients = self.tensors(gradients=True) if capture else None
        with phase("optimizer"):
            self.opt.update()
        return self.loss.value, self.logits.value, gradients

    def predict(self, x, y):
        self.x.value, self.y.value = x, y
        self.graph.forward()
        return self.host(self.logits.value)


class TorchRun:
    def __init__(self, initial, batch, *, autotuned):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = not autotuned
        torch.backends.cudnn.benchmark = autotuned
        cls = runpy.run_path(str(ROOT / "MyFlows/tests/torch_reference.py"))["TorchResNet18"]
        self.model = cls(num_classes=2)
        with np.load(initial, allow_pickle=False) as arrays:
            self.model.load_state_dict({k: torch.from_numpy(arrays[k]) for k in arrays.files}, strict=True)
        self.model.cuda()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8,
                                    weight_decay=0., foreach=False, fused=False)
        torch.cuda.reset_peak_memory_stats()
        self.info = {"framework": "PyTorch", "version": torch.__version__, "cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version(), "cudnn_deterministic": torch.backends.cudnn.deterministic,
                     "cudnn_benchmark": torch.backends.cudnn.benchmark, "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                     "cudnn_tf32": torch.backends.cudnn.allow_tf32, "dtype": "float32"}

    def device_batch(self, x, y):
        return self.torch.from_numpy(x).cuda(), self.torch.from_numpy(y).cuda()

    def host(self, value):
        return value.detach().cpu().numpy()

    def event(self):
        return self.torch.cuda.Event(enable_timing=True)

    def elapsed(self, a, b):
        return a.elapsed_time(b)

    def sync(self):
        self.torch.cuda.synchronize()

    def sample_memory(self):
        pass  # PyTorch records allocation peaks continuously.

    def memory(self):
        return {"allocator": "torch.cuda.max_memory_allocated", "peak_used_mib": self.torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": self.torch.cuda.max_memory_reserved() / 2**20}

    def tensors(self, *, gradients=False):
        return {name: self.host(p.grad if gradients else p).copy() for name, p in self.model.named_parameters()}

    def train(self, x, y, phase, capture):
        self.opt.zero_grad(set_to_none=True)
        with phase("forward"):
            logits = self.model(x)
            # Match MyFlows' FP32 squared differences, FP64 mean, one FP32 rounding.
            loss = (logits - y).square().mean(dtype=self.torch.float64).float()
        with phase("backward"):
            loss.backward()
        grads = self.tensors(gradients=True) if capture else None
        with phase("optimizer"):
            self.opt.step()
        return loss, logits, grads

    def predict(self, x, y):
        with self.torch.no_grad():
            return self.host(self.model(x))


class Timer:
    def __init__(self, runner):
        self.runner = runner
        self.events = {k: (runner.event(), runner.event()) for k in ("forward", "backward", "optimizer")}

    @contextmanager
    def __call__(self, phase):
        a, b = self.events[phase]
        a.record()
        try:
            yield
        finally:
            b.record()
            self.runner.sample_memory()

    def fields(self):
        return {f"{name}_ms": self.runner.elapsed(a, b) for name, (a, b) in self.events.items()} | {
            "train_step_ms": self.runner.elapsed(self.events["forward"][0], self.events["backward"][1])}


def evaluate(runner, images, labels, indices, batch):
    runner.model.eval()
    squared = angle = 0.
    try:
        for first in range(0, len(indices), batch):
            real = indices[first:first + batch]
            padded = np.pad(real, (0, batch - len(real)), mode="edge")
            prediction = runner.predict(*runner.device_batch(images[padded], labels[padded]))[:len(real)]
            diff = prediction.astype(np.float64) - labels[real].astype(np.float64)
            squared += float(np.square(diff).sum())
            angle += float(np.abs(diff[:, 0]).sum())
    finally:
        runner.model.train()
    return {"val_loss": squared / (2 * len(indices)), "val_angle_mae": angle / len(indices), "val_samples": len(indices)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--framework", choices=("myflows", "torch", "torch-autotuned"), required=True)
    parser.add_argument("--mode", choices=("align", "fixed", "e2e", "quality"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    fixture, run, mode, framework = args.fixture, args.run, args.mode, args.framework
    config = json.loads((fixture / "protocol.json").read_text(encoding="utf-8"))
    batch = config["batch"]
    images = np.load(fixture / "images.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(fixture / "targets.npy", mmap_mode="r", allow_pickle=False)
    seed_dir = fixture / f"seed-{args.seed}"
    initial = seed_dir / "initial.npz"
    with np.load(seed_dir / "batches.npz", allow_pickle=False) as saved:
        training, val, timing = saved["training"], saved["val"], saved["timing"]
    def build():
        return MyFlowsRun(initial, batch) if framework == "myflows" else TorchRun(initial, batch, autotuned=framework == "torch-autotuned")
    init_start = time.perf_counter()
    runner = build()
    timer = Timer(runner)
    runner.sync()
    init_s = time.perf_counter() - init_start
    warmup_s = 0.
    if mode in ("fixed", "e2e"):
        warm = time.perf_counter()
        for ids in timing[:config["warmup"]]:
            runner.train(*runner.device_batch(images[ids], labels[ids]), timer, False)
        runner.sync()
        del runner, timer
        import gc
        gc.collect()
        runner = build()  # restore the common initial weights, BN and Adam state; start at the first batch again
        timer = Timer(runner)
        runner.sync()
        warmup_s = time.perf_counter() - warm
    if mode == "quality":
        epochs = training
    else:
        epochs = [timing[:1] if mode == "align" else timing[:config["timed_steps"]]]
    fixed = runner.device_batch(images[timing[0]], labels[timing[0]]) if mode == "fixed" else None
    runner.sync()
    clock_start = time.perf_counter()
    durations, epoch_rows = [], []
    step_number = 0
    with (run / "metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for epoch, batches in enumerate(epochs, 1):
            train_losses = []
            ep_start = time.perf_counter()
            for ids in batches:
                runner.sync()
                start = time.perf_counter()
                if fixed is None:
                    x, y = runner.device_batch(images[ids], labels[ids])
                    runner.sync()
                    after_load = time.perf_counter()
                else:
                    x, y = fixed
                    after_load = start
                loss, logits, grads = runner.train(x, y, timer, mode == "align")
                runner.sync()
                end = time.perf_counter()
                data_ms, compute_ms, total_ms = (after_load-start)*1000, (end-after_load)*1000, (end-start)*1000
                step_number += 1
                scalar = float(runner.host(loss))
                train_losses.append(scalar)
                durations.append(total_ms)
                row = {"schema_version": 1, "kind": "step", "epoch": epoch, "step": step_number,
                       "unix_s": time.time(), "monotonic_s": end, "loss": scalar, "batch_size": batch,
                       "data_load_ms": data_ms, "step_time_ms": compute_ms, "step_total_ms": total_ms,
                       "samples_per_sec": batch * 1000 / compute_ms, "samples_per_sec_total": batch * 1000 / total_ms,
                       **timer.fields()}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if mode == "align":
                    np.savez(run / "alignment.npz", logits=runner.host(logits), loss=scalar,
                             **{f"grad/{key}": value for key, value in grads.items()},
                             **{f"updated/{key}": value for key, value in runner.tensors().items()})
            if mode == "quality":
                train_s = time.perf_counter() - ep_start
                validation = evaluate(runner, images, labels, val, batch)
                row = {"schema_version": 1, "kind": "epoch", "epoch": epoch, "loss": float(np.mean(train_losses)),
                       "epoch_time_s": time.perf_counter() - ep_start, "train_wall_s": train_s, "learning_rate": .001,
                       "unix_s": time.time(), "monotonic_s": time.perf_counter(), **validation}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                epoch_rows.append(row)
    end = time.perf_counter()
    result = {"schema_version": 1, "status": "succeeded", "exit_code": 0, "mode": mode, "framework": framework,
              "seed": args.seed, "runtime": runner.info, "allocator": runner.memory(), "init_s": init_s, "warmup_s": warmup_s,
              "measurement_start_s": clock_start, "measurement_end_s": end, "wall_s": end-clock_start,
              "steps": step_number, "actual_samples": batch * step_number, "median_step_ms": float(np.median(durations)),
              "samples_per_s": batch * step_number * 1000 / sum(durations), "epochs": epoch_rows, "last_loss": scalar}
    (run / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if framework == "myflows" and "torch" in sys.modules:
        raise RuntimeError("MyFlows worker loaded PyTorch")
    if framework != "myflows" and "cupy" in sys.modules:
        raise RuntimeError("PyTorch worker loaded CuPy")


if __name__ == "__main__":
    main()
