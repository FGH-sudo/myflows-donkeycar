"""MNIST MLP on every Linear backend: gradients against FP64, training speed, and the Adam fix.

Three measurements on the prepared MNIST split (``.codex/distributed-data-v1/mnist_mlp``):

* gradient accuracy: the same float32 parameters and batches on every backend, compared with an
  FP64 NumPy evaluation of the same (exactly representable) values;
* training: one epoch per backend from the same initialization and batch order, timed per step,
  with the parameter distance to an FP64 NumPy trajectory and the test accuracy at the end;
* optimizer: the corrected Adam against the previous implementation (first moments seeded with
  g and g^2, then bias-corrected), same backend and data.

  python -m benchmark.mlp_backends --out-dir docs/experiments/semester_2026_fall/cuda_lecture19/mlp-001
"""

import argparse
import time

import numpy as np

import MyFlows as ms
from benchmark.stage1_common import ROOT, RunArtifacts, array_hash
from MyFlows.core.device import asnumpy, set_device, xp
from MyFlows.core.node import Variable
from MyFlows.distributed.tasks.prepared import load_prepared
from MyFlows.layers.layer import Dense
from MyFlows.ops.activation import ReLU
from MyFlows.ops.loss import CrossEntropy
from MyFlows.train.opt import Adam

DATA = ROOT / ".codex" / "distributed-data-v1" / "mnist_mlp"
BACKENDS = ("numpy", "cupy", "cuda_native_cublas", "cuda_native_kahan", "cuda_native_dot2")


class LegacyAdam(Adam):
    """The Adam implementation before this change, kept only to measure the fix."""

    def _update(self):
        self.t += 1
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                grad = self.get_gradient(node)
                if grad is None:
                    continue
                if node not in self.s:
                    self.v[node] = grad
                    self.s[node] = xp.power(grad, 2)
                else:
                    self.v[node] = self.beta_1 * self.v[node] + (1 - self.beta_1) * grad
                    self.s[node] = self.beta_2 * self.s[node] + (1 - self.beta_2) * xp.power(grad, 2)
                v_correct = self.v[node] / (1 - self.beta_1 ** self.t)
                s_correct = self.s[node] / (1 - self.beta_2 ** self.t)
                node.value -= self.learning_rate * v_correct / (xp.sqrt(s_correct) + self.eps)


def initial_parameters(hidden, seed):
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, np.sqrt(2.0 / 784), (784, hidden)).astype(np.float32)
    w2 = rng.normal(0.0, np.sqrt(2.0 / hidden), (hidden, 10)).astype(np.float32)
    return [w1, np.zeros((1, hidden), np.float32), w2, np.zeros((1, 10), np.float32)]


def build(backend, init):
    """MLP 784-hidden-10 with ReLU and softmax cross-entropy; ``fp64`` is the NumPy FP64 reference."""
    dtype = np.float64 if backend == "fp64" else np.float32
    set_device("cpu" if backend in ("numpy", "fp64") else "cuda")
    previous = ms.set_default_dtype(dtype)
    try:
        layer_backend = backend if backend.startswith("cuda_native") else "auto"
        x = Variable(xp.zeros((1, 784), dtype), name="input", requires_grad=False)
        y = Variable(xp.zeros((1, 1), np.int64), name="target", requires_grad=False)
        fc1 = Dense(784, init[0].shape[1], activation=ReLU, backend=layer_backend, name="fc1")
        fc2 = Dense(init[0].shape[1], 10, backend=layer_backend, name="fc2")
        logits = fc2(fc1(x))
        loss = CrossEntropy(logits, y)
        graph = ms.Graph(loss)
    finally:
        ms.set_default_dtype(previous)
    params = [fc1.W, fc1.b, fc2.W, fc2.b]
    for node, value in zip(params, init):
        node.value = xp.asarray(value.astype(dtype))
    return {"x": x, "y": y, "logits": logits, "loss": loss, "graph": graph, "params": params, "dtype": dtype,
            "eval_graph": ms.Graph(logits)}


def batches(split, batch, epoch, seed):
    images, labels = split["train"]
    order = np.random.default_rng(seed + 17 + epoch).permutation(len(images))
    for start in range(0, (len(order) // batch) * batch, batch):
        index = np.sort(order[start:start + batch])
        yield np.asarray(images[index], dtype=np.float32), np.asarray(labels[index], dtype=np.int64).reshape(-1, 1)


def gradient_accuracy(split, args, init, run):
    rows = []
    sample = [b for _, b in zip(range(args.grad_batches), batches(split, args.batch, 99, args.seed))]
    reference = build("fp64", init)
    expected = []
    for xb, yb in sample:
        reference["x"].value, reference["y"].value = xb.astype(np.float64), yb
        reference["graph"].forward()
        reference["graph"].backward()
        expected.append((float(reference["loss"].value), [p.grad.copy() for p in reference["params"]]))
    for backend in args.backends:
        model = build(backend, init)
        errors = {name: [] for name in ("loss", "W1", "b1", "W2", "b2")}
        for (xb, yb), (loss64, grads64) in zip(sample, expected):
            model["x"].value, model["y"].value = xp.asarray(xb), xp.asarray(yb)
            model["graph"].forward()
            model["graph"].backward()
            errors["loss"].append(abs(float(model["loss"].value) - loss64) / abs(loss64))
            for name, node, g64 in zip(("W1", "b1", "W2", "b2"), model["params"], grads64):
                g = asnumpy(node.grad).astype(np.float64)
                errors[name].append(float(np.linalg.norm(g - g64) / max(np.linalg.norm(g64), 1e-300)))
        row = {"backend": backend, **{f"{k}_median": float(np.median(v)) for k, v in errors.items()},
               **{f"{k}_max": float(np.max(v)) for k, v in errors.items()}}
        rows.append(row)
        print("grad", backend, {k: f"{v:.2e}" for k, v in row.items() if k.endswith("_median")}, flush=True)
    run.results["gradient_accuracy"] = rows
    table = [[r["backend"]] + [f"{r[f'{k}_median']:.2e} ({r[f'{k}_max']:.2e})" for k in ("loss", "W1", "b1", "W2", "b2")]
             for r in rows]
    run.notes.append(f"## 梯度相对 FP64 的误差（{args.grad_batches} 个 batch 的中位数，括号内最大值）\n\n"
                     "相对误差 = ‖g − g64‖₂ / ‖g64‖₂；loss 为相对误差。参数、输入均为同一组 float32 值。\n\n"
                     + _table(["后端", "loss", "dW1", "db1", "dW2", "db2"], table))


def train(backend, split, args, init, optimizer=Adam, steps=None):
    model = build(backend, init)
    opt = optimizer(model["graph"], learning_rate=args.lr)
    losses = []
    synchronize = (lambda: xp.cuda.Device().synchronize()) if backend not in ("numpy", "fp64") else (lambda: None)
    data = list(batches(split, args.batch, 0, args.seed))[:steps]
    synchronize()
    start = time.perf_counter()
    for xb, yb in data:
        model["x"].value = xb.astype(model["dtype"])
        model["y"].value = yb
        opt.one_step()
        opt.update()
        losses.append(model["loss"].value)  # read back after the epoch: no per-step synchronization
    synchronize()
    elapsed = time.perf_counter() - start
    losses = [float(v) for v in losses]
    test_x, test_y = split["test"]
    correct = 0
    for start_index in range(0, len(test_x), 1000):
        model["x"].value = xp.asarray(np.asarray(test_x[start_index:start_index + 1000], dtype=model["dtype"]))
        model["eval_graph"].forward()
        pred = asnumpy(model["logits"].value).argmax(axis=1)
        correct += int((pred == np.asarray(test_y[start_index:start_index + 1000]).reshape(-1)).sum())
    return {"ms_per_step": elapsed * 1000.0 / len(data), "steps": len(data), "losses": losses,
            "test_accuracy": correct / len(test_x), "params": [asnumpy(p.value).astype(np.float64) for p in model["params"]]}


def training(split, args, init, run):
    reference = train("fp64", split, args, init)
    rows = []
    for backend in args.backends:
        train(backend, split, args, init, steps=20)  # warm-up: kernel compilation, cuBLAS heuristics
        result = train(backend, split, args, init)
        distance = [float(np.linalg.norm(p - q) / np.linalg.norm(q)) for p, q in zip(result["params"], reference["params"])]
        loss_gap = float(np.max(np.abs(np.asarray(result["losses"]) - np.asarray(reference["losses"]))))
        first_gap = next((i for i, (a, b) in enumerate(zip(result["losses"], reference["losses"])) if abs(a - b) > 1e-4),
                         None)
        rows.append({"backend": backend, "ms_per_step": result["ms_per_step"], "steps": result["steps"],
                     "final_loss": float(np.mean(result["losses"][-20:])), "test_accuracy": result["test_accuracy"],
                     "param_rel_distance": distance, "max_loss_gap": loss_gap, "first_step_gap_1e-4": first_gap})
        print("train", backend, f"{result['ms_per_step']:.3f} ms/step acc={result['test_accuracy']:.4f} "
              f"W1 dist={distance[0]:.2e}", flush=True)
    run.results["training"] = {"fp64": {"test_accuracy": reference["test_accuracy"],
                                        "final_loss": float(np.mean(reference["losses"][-20:])),
                                        "ms_per_step": reference["ms_per_step"]}, "rows": rows}
    table = [["fp64（NumPy 参考）", f"{reference['ms_per_step']:.3f}", f"{np.mean(reference['losses'][-20:]):.4f}",
              f"{100 * reference['test_accuracy']:.2f}", "0", "0", "-"]]
    table += [[r["backend"], f"{r['ms_per_step']:.3f}", f"{r['final_loss']:.4f}", f"{100 * r['test_accuracy']:.2f}",
               f"{r['param_rel_distance'][0]:.2e}", f"{r['max_loss_gap']:.1e}",
               "未超过" if r["first_step_gap_1e-4"] is None else r["first_step_gap_1e-4"]] for r in rows]
    run.notes.append(f"## 一个 epoch 训练（{rows[0]['steps']} 步，batch={args.batch}，Adam lr={args.lr}）\n\n"
                     "相同初始化与 batch 顺序；W1 距离为 epoch 结束时 ‖W1 − W1_fp64‖ / ‖W1_fp64‖，"
                     "loss 差为逐步 loss 与 FP64 轨迹的最大绝对差。GPU 每步不同步，epoch 结束后统一读回 loss。\n\n"
                     + _table(["后端", "ms/步", "末 20 步 loss", "测试准确率 %", "W1 相对距离", "最大 loss 差",
                               "loss 差首次 >1e-4 的步"], table))


def optimizer_fix(split, args, init, run):
    rows = []
    for name, optimizer in (("Adam（修正后）", Adam), ("Adam（修正前）", LegacyAdam)):
        result = train("cupy", split, args, init, optimizer=optimizer)
        losses = result["losses"]
        rows.append([name] + [f"{np.mean(losses[max(0, s - 10):s]):.4f}" for s in (10, 50, 100, 200, len(losses))]
                    + [f"{100 * result['test_accuracy']:.2f}"])
        run.results.setdefault("optimizer", []).append({"optimizer": name, "losses": losses,
                                                        "test_accuracy": result["test_accuracy"]})
        print("optimizer", name, rows[-1][1:], flush=True)
    steps = len(run.results["optimizer"][0]["losses"])
    run.notes.append("## Adam 修正前后（cupy 后端，同初始化与数据）\n\n每格为截至该步前 10 步的平均 loss。\n\n"
                     + _table(["优化器", "第 10 步", "第 50 步", "第 100 步", "第 200 步", f"第 {steps} 步", "测试准确率 %"], rows))


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grad-batches", type=int, default=10)
    args = parser.parse_args(argv)
    args.backends = [b for b in args.backends.split(",") if b]
    unknown = sorted(set(args.backends) - set(BACKENDS))
    if unknown:
        parser.error(f"unknown backends: {unknown}")
    with RunArtifacts(args.out_dir, vars(args)) as run:
        split = load_prepared(args.data)
        run.manifest["dataset"] = {k: v for k, v in split["meta"].items() if k != "files"}
        init = initial_parameters(args.hidden, args.seed)
        run.manifest["init_sha256"] = array_hash(*init)
        gradient_accuracy(split, args, init, run)
        training(split, args, init, run)
        optimizer_fix(split, args, init, run)
        run.notes.append(f"MLP 784-{args.hidden}-10（ReLU，softmax 交叉熵），He 初始化（固定种子），输入为 [0,1] 像素。"
                         "fp64 参考在 CPU 上用 NumPy FP64 计算同一组 float32 值。")


if __name__ == "__main__":
    main()
