"""Operator accuracy before/after: the current tree against another one, both against FP64.

Fixed float32 inputs go through Conv2D (NumPy, CuPy, native), BatchNorm2d (including a batch whose
mean is 1e5 times its standard deviation), softmax cross-entropy (including logits of scale 1e4)
and ten Adam steps, in a fresh worker process per code tree. The driver computes FP64 references
of the same float32 values (textbook Adam for the optimizer) and reports the error of each tree.

  python -m benchmark.ops_before_after --old <worktree of the previous commit> \
      --out-dir docs/experiments/semester_2026_fall/cuda_lecture19/ops-001
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONV = {"stride": (1, 1), "padding": (1, 1)}
ADAM = {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "steps": 10}


def fixture(seed=0):
    rng = np.random.default_rng(seed)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    data = {
        "conv_x": f32(rng.standard_normal((4, 16, 20, 24))), "conv_w": f32(0.1 * rng.standard_normal((32, 16, 3, 3))),
        "conv_b": f32(rng.standard_normal(32)), "conv_dy": f32(rng.standard_normal((4, 32, 20, 24))),
        "bn_gamma": f32(rng.uniform(0.5, 1.5, 16)), "bn_beta": f32(rng.standard_normal(16)),
        "bn_dy": f32(rng.standard_normal((8, 16, 12, 12))),
        "bn_x_unit": f32(rng.standard_normal((8, 16, 12, 12))),
        "bn_x_offset": f32(1000.0 + 0.01 * rng.standard_normal((8, 16, 12, 12))),
        "ce_target": np.eye(10, dtype=np.float32)[rng.integers(0, 10, 64)],
        "adam_param": f32(rng.standard_normal(1000)),
    }
    logits = rng.standard_normal((64, 10))
    data["ce_logits_unit"], data["ce_logits_large"] = f32(logits), f32(1e4 * logits)
    data["adam_grads"] = f32(rng.standard_normal((ADAM["steps"], 1000)) * 10.0 ** rng.uniform(-6, 1, (ADAM["steps"], 1)))
    return data


# ------------------------------------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------------------------------------
def worker(args):
    sys.path.insert(0, str(Path(args.tree).resolve()))
    import MyFlows

    if not Path(MyFlows.__file__).resolve().is_relative_to(Path(args.tree).resolve()):
        raise RuntimeError(f"imported {MyFlows.__file__} instead of {args.tree}")
    from MyFlows.core.device import asnumpy, cuda_available, set_device, xp
    from MyFlows.core.graph import Graph
    from MyFlows.core.node import Variable
    from MyFlows.ops.batchnorm import BatchNorm2d_Op
    from MyFlows.ops.convolution import Conv2D_Op
    from MyFlows.ops.loss import CrossEntropy
    from MyFlows.train.opt import Adam

    f = dict(np.load(args.fixture))
    out = {}
    # xp.array copies: on the CPU xp.asarray would hand the fixture itself to in-place updates
    # (Adam's param -= ...), and the CUDA run would then start from already-updated values.
    devices = ("cpu", "cuda") if cuda_available() else ("cpu",)
    for device in devices:
        set_device(device)
        for backend in (("numpy",) if device == "cpu" else ("cupy", "cuda_native_cublas")):
            x, w, b, dy = (xp.array(f[k]) for k in ("conv_x", "conv_w", "conv_b", "conv_dy"))
            nodes = [Variable(x), Variable(w, trainable=True), Variable(b, trainable=True)]
            op = Conv2D_Op(nodes[0], nodes[1], bias=nodes[2], backend=backend, **CONV)
            op.forward(x, w, b)
            for node in nodes:
                node.clear_grad()
            op.grad = dy
            op.backward()
            for key, value in zip(("y", "dx", "dw", "db"), (op.value, *(n.grad for n in nodes))):
                out[f"conv/{backend}/{key}"] = asnumpy(value)
        for case in ("unit", "offset"):
            nodes = [Variable(xp.array(f[f"bn_x_{case}"])), Variable(xp.array(f["bn_gamma"]), trainable=True),
                     Variable(xp.array(f["bn_beta"]), trainable=True)]
            running = xp.zeros(16, xp.float32), xp.ones(16, xp.float32)
            op = BatchNorm2d_Op(*nodes, *running, momentum=0.1, eps=1e-5, training=True)
            op.forward(*(n.value for n in nodes))
            for node in nodes:
                node.clear_grad()
            op.grad = xp.array(f["bn_dy"])
            op.backward()
            for key, value in zip(("y", "dx", "dgamma", "dbeta"), (op.value, *(n.grad for n in nodes))):
                out[f"bn/{device}/{case}/{key}"] = asnumpy(value)
        for case in ("unit", "large"):
            logits, target = Variable(xp.array(f[f"ce_logits_{case}"])), Variable(xp.array(f["ce_target"]))
            loss = CrossEntropy(logits, target)
            graph = Graph(loss)
            graph.forward()
            graph.backward()
            out[f"ce/{device}/{case}/loss"] = np.asarray(float(loss.value))
            out[f"ce/{device}/{case}/grad"] = asnumpy(logits.grad)
        param = Variable(xp.array(f["adam_param"]), trainable=True)

        class _Graph:
            nodes = (param,)

        opt = Adam(_Graph(), learning_rate=ADAM["lr"], beta_1=ADAM["beta1"], beta_2=ADAM["beta2"], eps=ADAM["eps"])
        for step, grad in enumerate(f["adam_grads"], 1):
            opt.acc_gradient, opt.acc_no = {param: xp.asarray(grad)}, 1
            opt._update()
            if step in (1, ADAM["steps"]):
                out[f"adam/{device}/step{step}"] = asnumpy(param.value).copy()
    np.savez(args.output, **out)


# ------------------------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------------------
def references(f):
    from MyFlows.tests.test_convolution import naive_conv_backward, naive_conv_forward

    d = {k: np.asarray(v, dtype=np.float64) for k, v in f.items()}
    ref = {}
    y = naive_conv_forward(d["conv_x"], d["conv_w"], CONV["stride"], CONV["padding"]) + d["conv_b"][None, :, None, None]
    dx, dw = naive_conv_backward(d["conv_x"], d["conv_w"], d["conv_dy"], CONV["stride"], CONV["padding"])
    ref.update({"conv/y": y, "conv/dx": dx, "conv/dw": dw, "conv/db": d["conv_dy"].sum(axis=(0, 2, 3))})
    ch = lambda v: v.reshape(1, -1, 1, 1)
    for case in ("unit", "offset"):
        x, dy, gamma, beta = d[f"bn_x_{case}"], d["bn_dy"], d["bn_gamma"], d["bn_beta"]
        mean, var = x.mean(axis=(0, 2, 3)), x.var(axis=(0, 2, 3))
        inv = 1.0 / np.sqrt(var + 1e-5)
        xhat = (x - ch(mean)) * ch(inv)
        m = x.shape[0] * x.shape[2] * x.shape[3]
        dbeta, dgamma = dy.sum(axis=(0, 2, 3)), (dy * xhat).sum(axis=(0, 2, 3))
        ref.update({f"bn/{case}/y": xhat * ch(gamma) + ch(beta), f"bn/{case}/dgamma": dgamma, f"bn/{case}/dbeta": dbeta,
                    f"bn/{case}/dx": ch(gamma * inv) * (dy - ch(dbeta / m) - xhat * ch(dgamma / m))})
    for case in ("unit", "large"):
        x, target = d[f"ce_logits_{case}"], d["ce_target"]
        logp = x - x.max(axis=1, keepdims=True)
        logp -= np.log(np.exp(logp).sum(axis=1, keepdims=True))
        ref[f"ce/{case}/loss"] = np.asarray(-(target * logp).sum(axis=1).mean())
        ref[f"ce/{case}/grad"] = (np.exp(logp) - target) / x.shape[0]
    theta, m1, m2 = d["adam_param"].copy(), 0.0, 0.0
    for step, g in enumerate(d["adam_grads"], 1):
        m1 = ADAM["beta1"] * m1 + (1 - ADAM["beta1"]) * g
        m2 = ADAM["beta2"] * m2 + (1 - ADAM["beta2"]) * g * g
        theta = theta - ADAM["lr"] * (m1 / (1 - ADAM["beta1"] ** step)) / (
            np.sqrt(m2 / (1 - ADAM["beta2"] ** step)) + ADAM["eps"])
        if step in (1, ADAM["steps"]):
            ref[f"adam/step{step}"] = theta.copy()
    return ref


def _reference_key(key):
    parts = key.split("/")
    if parts[0] == "conv":
        return f"conv/{parts[2]}"
    if parts[0] == "adam":
        return f"adam/{parts[2]}"
    return f"{parts[0]}/{parts[2]}/{parts[3]}"


def _error(actual, expected):
    """Max error relative to max |reference|; Adam is measured as a distance from the textbook step."""
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.abs(actual - expected).max() / max(float(np.abs(expected).max()), 1e-300))


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tree", help=argparse.SUPPRESS)
    parser.add_argument("--fixture", help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    parser.add_argument("--old")
    parser.add_argument("--out-dir")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args)
    if not args.old or not args.out_dir:
        parser.error("--old and --out-dir are required")
    from benchmark.stage1_common import RunArtifacts, array_hash

    with RunArtifacts(args.out_dir, vars(args)) as run:
        data = fixture(args.seed)
        path = run.path / "ops-fixture.npz"
        np.savez(path, **data)
        run.manifest["fixture_sha256"] = array_hash(*data.values())
        expected = references(data)
        results = {}
        for name, tree in (("before", Path(args.old).resolve(strict=True)), ("after", ROOT)):
            output = run.path / f"ops-{name}.npz"
            command = [sys.executable, "-X", "utf8", "-m", "benchmark.ops_before_after", "--worker",
                       "--tree", str(tree), "--fixture", str(path), "--output", str(output)]
            done = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  encoding="utf-8", errors="replace", timeout=1800, check=False)
            if done.returncode:
                raise RuntimeError(f"worker failed for {tree}: {done.stdout[-3000:]}")
            results[name] = dict(np.load(output))
        rows = []
        for key in sorted(results["after"]):
            reference = expected[_reference_key(key)]
            before = _error(results["before"][key], reference) if key in results["before"] else None
            after = _error(results["after"][key], reference)
            rows.append({"output": key, "before": before, "after": after})
            print(f"{key:36s} before={before if before is None else f'{before:.2e}'} after={after:.2e}", flush=True)
        run.results["errors"] = rows
        run.notes.append(
            "## 算子输出相对 FP64 的最大误差（max|y − y64| / max|y64|）\n\n"
            "conv：4×16×20×24 输入、32×16×3×3 卷积核、stride 1、padding 1；bn：8×16×12×12，unit 为 N(0,1)，"
            "offset 为 1000 + 0.01·N(0,1)（均值是标准差的 1e5 倍）；ce：64×10 logits，unit 为 N(0,1)，large 为 1e4 倍，"
            "one-hot 标签；adam：1000 个参数、10 步、梯度量级 1e-6–10，参考为教科书 Adam（零初始化矩估计 + 偏差修正）的 FP64 结果，"
            "修改前一栏反映的是旧实现的语义偏差而不只是舍入误差。\n\n"
            + _table(["输出", "修改前", "修改后", "倍数"],
                     [[r["output"], "-" if r["before"] is None else f"{r['before']:.2e}", f"{r['after']:.2e}",
                       "-" if not r["before"] or not r["after"] else f"{r['before'] / r['after']:.1f}×"] for r in rows]))


if __name__ == "__main__":
    main()
