"""Stage-one correctness, performance, CNN training and short Nsight workloads."""

import argparse
import time

import numpy as np

from benchmark.stage1_common import RunArtifacts, array_hash, error_metrics
from MyFlows.core.device import set_device, xp, asnumpy
from MyFlows.core.node import Variable
from MyFlows.examples.stage1_cnn import build, dataset, assert_fp32
from MyFlows.ops.convolution import Conv2D_Op, MaxPool2d_Op
from MyFlows.ops.cuda.loader import compilation_metadata, get_kernel
from MyFlows.tests.cuda_fixtures import CONV_CASES, conv_fixture
from MyFlows.tests.test_convolution import (
    naive_conv_forward, naive_conv_backward, naive_maxpool_forward, naive_maxpool_backward,
)


PERFORMANCE_CONV = (
    ("P0", (1, 3, 16, 16), (8, 3, 3, 3), (1, 1), (1, 1)),
    ("P1", (4, 16, 32, 32), (32, 16, 3, 3), (1, 1), (1, 1)),
    ("P2", (1, 3, 120, 160), (8, 3, 7, 7), (2, 2), (3, 3)),
)


def cases(suite, seed):
    for case in CONV_CASES if suite == "correctness" else PERFORMANCE_CONV:
        yield {"id": case[0], "kind": "conv", "stride": case[3], "padding": case[4],
               "arrays": conv_fixture(case, seed)}
    shapes = [(2, 3, 7, 9)] if suite == "correctness" else [(4, 16, 32, 32), (1, 3, 120, 160)]
    rng = np.random.default_rng(seed)
    for si, shape in enumerate(shapes):
        for ki, (kernel, stride) in enumerate(((2, 2), (3, 2), (3, 1))):
            x = rng.normal(size=shape).astype(np.float32)
            out_shape = (*shape[:2], (shape[2] - kernel) // stride + 1, (shape[3] - kernel) // stride + 1)
            dy = rng.normal(size=out_shape).astype(np.float32)
            yield {"id": f"{'T5' if suite == 'correctness' else 'P3'}-{si}-{ki}",
                   "kind": "pool", "kernel": (kernel, kernel), "stride": (stride, stride), "arrays": [x, dy]}


def oracle(case):
    arrays = [a.astype(np.float64) for a in case["arrays"]]
    if case["kind"] == "conv":
        x, w, b, dy = arrays
        y = naive_conv_forward(x, w, case["stride"], case["padding"]) + b[None, :, None, None]
        dx, dw = naive_conv_backward(x, w, dy, case["stride"], case["padding"])
        return dict(y=y, dx=dx, dw=dw, db=dy.sum(axis=(0, 2, 3)))
    x, dy = arrays
    return dict(y=naive_maxpool_forward(x, case["kernel"], case["stride"]),
                dx=naive_maxpool_backward(x, dy, case["kernel"], case["stride"]))


class Operator:
    def __init__(self, case, backend):
        self.backend = backend
        set_device("cpu" if backend == "numpy" else "cuda")
        started = time.perf_counter()
        arrays = [xp.asarray(a).copy() for a in case["arrays"]]
        self.synchronize()
        self.transfer_ms = (time.perf_counter() - started) * 1000
        self.nodes = [Variable(a) for a in arrays[:-1]]
        self.dy = arrays[-1]
        if case["kind"] == "conv":
            x, w, b = self.nodes
            self.op = Conv2D_Op(x, w, stride=case["stride"], padding=case["padding"], bias=b, backend=backend)
        else:
            self.op = MaxPool2d_Op(self.nodes[0], case["kernel"], case["stride"], backend=backend)

    def synchronize(self):
        if self.backend != "numpy":
            xp.cuda.get_current_stream().synchronize()

    def forward(self):
        self.op.forward(*(node.value for node in self.nodes))

    def backward(self):
        self.op.grad = self.dy
        for node in self.nodes:
            node.grad = None
        self.op.backward()

    def combined(self):
        self.forward()
        self.backward()

    def outputs(self):
        self.combined()
        return {key: asnumpy(value).copy() for key, value in
                zip(("y", "dx", "dw", "db"), (self.op.value, *(node.grad for node in self.nodes)))}

    def timed(self, phase, case=None):
        if phase == "transfer_inclusive":
            start = time.perf_counter()
            Operator(case, self.backend).outputs()
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, elapsed
        call = {"forward": self.forward, "backward": self.backward, "combined": self.combined}[phase]
        if self.backend == "numpy":
            start = time.perf_counter()
            call()
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, elapsed
        start_event, stop_event = xp.cuda.Event(), xp.cuda.Event()
        start = time.perf_counter()
        start_event.record()
        call()
        stop_event.record()
        stop_event.synchronize()
        return float(xp.cuda.get_elapsed_time(start_event, stop_event)), (time.perf_counter() - start) * 1000


def compile_cuda(run):
    start = time.perf_counter()
    for filename, names in {
        "conv2d.cu": ("conv2d_forward_direct", "conv2d_backward_input", "conv2d_backward_weight", "conv2d_backward_bias"),
        "maxpool2d.cu": ("maxpool2d_forward_direct", "maxpool2d_backward_gather"),
        "im2col.cu": ("conv2d_im2col_forward", "conv2d_col2im_backward"),
        "gemm.cu": ("gemm_nt", "gemm_nn", "gemm_tn"),
    }.items():
        for name in names:
            get_kernel(filename, name)
    run.results["compile_or_cache_load_wall_ms"] = (time.perf_counter() - start) * 1000
    run.manifest["cuda_compilation"] = compilation_metadata()


def run_matrix(args, run):
    rows, measurements, hashes = [], [], {}
    run.results.update(correctness=rows, measurements=measurements)
    run.manifest["fixture_hashes"] = hashes
    run.manifest["cases"] = []
    run.manifest["timing_scope"] = {
        "operator_call": "Op.forward/backward; allocation, contiguous copies, launch, and backward gradient reset/merge included",
        "kernel_only": "not measured", "transfer_inclusive": "perf_counter: private input copies, H2D, node construction, forward/backward, D2H output copies; CPU uses corresponding private copies",
        "transfer_ms": "initial host-to-device copy or CPU copy, measured separately",
        "gpu_ms": "current-stream CUDA events synchronized at stop", "wall_ms": "perf_counter including stop synchronization"}
    suite_start = time.perf_counter()
    for case in cases(args.suite, args.seed):
        if args.case and case["id"] != args.case:
            continue
        case_start = time.perf_counter()
        run.manifest["cases"].append({**{k: v for k, v in case.items() if k != "arrays"},
                                      "arrays": [{"shape": a.shape, "dtype": str(a.dtype),
                                                  "contiguous": bool(a.flags.c_contiguous)} for a in case["arrays"]]})
        hashes[case["id"]] = array_hash(*case["arrays"])
        np.savez(run.path / f"fixture-{case['id']}.npz", **{f"array_{i}": a for i, a in enumerate(case["arrays"])})
        expected = oracle(case)
        for backend in args.backends:
            op = Operator(case, backend)
            if backend in ("cuda_c", "cuda_im2col", "cuda_im2col_gemm") and "cuda_compilation" not in run.manifest:
                compile_cuda(run)
            actual = op.outputs()
            metrics = {key: error_metrics(actual[key], reference) for key, reference in expected.items()}
            rows.append({"case": case["id"], "backend": backend, "errors": metrics,
                         "transfer_ms": op.transfer_ms, "actual_backend": op.op.actual_backend})
            np.savez(run.path / f"outputs-{case['id']}-{backend}.npz", **actual)
            if not all(m["allclose"] for m in metrics.values()):
                raise AssertionError(f"correctness failed: {case['id']} {backend}: {metrics}")
            print(f"{case['id']} {backend} correctness passed", flush=True)
            if args.suite == "correctness":
                continue
            for phase in ("forward", "backward", "combined", "transfer_inclusive"):
                if time.perf_counter() - case_start > args.case_timeout or time.perf_counter() - suite_start > args.suite_timeout:
                    raise TimeoutError(f"budget exceeded before {case['id']} {backend} {phase}")
                # A measured dry run estimates the cost before the formal repetitions.
                estimated, estimated_wall = op.timed(phase, case)
                if estimated_wall / 1000 * (args.warmup + args.measure * args.repeats) > args.case_timeout - (time.perf_counter() - case_start):
                    raise TimeoutError(f"estimated budget exceeds {case['id']} remaining time")
                for _ in range(args.warmup):
                    op.timed(phase, case)
                samples = []
                for repeat in range(args.repeats):
                    for iteration in range(args.measure):
                        ms, wall_ms = op.timed(phase, case)
                        row = {"case": case["id"], "backend": backend, "phase": phase,
                               "scope": "transfer_inclusive" if phase == "transfer_inclusive" else "operator_call",
                               "clock": "perf_counter" if backend == "numpy" or phase == "transfer_inclusive" else "cuda_event",
                               "shape": str(case["arrays"][0].shape), "dtype": "float32",
                               "repeat": repeat, "iteration": iteration, "ms": ms, "wall_ms": wall_ms,
                               "status": "passed", "max_abs": max(m["max_abs"] for m in metrics.values()),
                               "max_rel": max(m["max_rel"] for m in metrics.values())}
                        run.timings.append(row)
                        samples.append(ms)
                        if time.perf_counter() - case_start > args.case_timeout or time.perf_counter() - suite_start > args.suite_timeout:
                            raise TimeoutError(f"budget exceeded during {case['id']} {backend} {phase}")
                measurements.append({"case": case["id"], "backend": backend, "phase": phase,
                                     "mean_ms": float(np.mean(samples)), "median_ms": float(np.median(samples)),
                                     "p95_ms": float(np.percentile(samples, 95)), "n": len(samples),
                                     "dry_run_ms": estimated})
    if not rows:
        raise ValueError("no case matched --case")
    for row in measurements:
        if row["backend"] == "cuda_c":
            for ref in ("numpy", "cupy"):
                match = next((m for m in measurements if (m["case"], m["phase"], m["backend"]) == (row["case"], row["phase"], ref)), None)
                if match:
                    row[f"{ref}_speedup"] = match["mean_ms"] / row["mean_ms"]
    run.notes.append("All operands are identical FP32 fixtures; errors use independent FP64 loop references. Max relative error uses denominator max(abs(reference), 1e-6).")
    run.notes.append("GPU event intervals cover the whole operator call and may include GPU idle time while Python submits work. These are not kernel-only measurements. Profiler timings are excluded.")


def run_training(args, run):
    x, y = dataset(args.seed)
    run.manifest["fixture_sha256"] = array_hash(x, y)
    np.savez(run.path / "fixture.npz", x=x, y=y)
    run.results["training"] = results = []
    checkpoints = {}
    for backend in args.backends:
        set_device("cpu" if backend == "numpy" else "cuda")
        model = build(backend, args.seed)
        initial = [asnumpy(p.value).copy() for p in model["params"]]
        initial_hash = array_hash(*initial)
        snapshots = []
        for step in range(100):
            opt = model["optimizer"]
            opt.one_step()
            assert_fp32(model)
            loss = float(model["loss"].value)
            if step < 3:
                snapshots.extend(asnumpy(p.grad).copy() for p in model["params"])
            opt.update()
            assert_fp32(model)
            if step < 3:
                snapshots.extend(asnumpy(p.value).copy() for p in model["params"])
            run.timings.append({"backend": backend, "step": step, "loss_before_update": loss, "status": "passed"})
        model["graph"].forward()
        accuracy = float(np.mean(asnumpy(model["logits"].value).argmax(1) == y.ravel()))
        final_loss = float(model["loss"].value)
        np.savez(run.path / f"training-{backend}.npz", **{f"early_{i}": a for i, a in enumerate(snapshots)},
                 **{f"initial_{i}": a for i, a in enumerate(initial)},
                 **{f"final_{i}": asnumpy(p.value) for i, p in enumerate(model["params"])})
        results.append({"backend": backend, "initial_sha256": initial_hash, "accuracy": accuracy,
                        "loss": final_loss, "updates": 100, "dtype_checks": "passed"})
        checkpoints[backend] = snapshots
        if accuracy < 0.95:
            raise AssertionError(f"{backend} failed to overfit: {accuracy}")
        print(results[-1], flush=True)
    if len({r["initial_sha256"] for r in results}) != 1:
        raise AssertionError("initial parameters differ")
    reference = args.backends[0]
    for backend in args.backends[1:]:
        for a, b in zip(checkpoints[backend], checkpoints[reference]):
            np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-3)
        np.testing.assert_allclose(next(r["loss"] for r in results if r["backend"] == backend), results[0]["loss"], atol=1e-4, rtol=1e-3)
    run.notes.append("Locked fixture: seed=0 by default, 32 alternating vertical/horizontal stripe samples with Gaussian noise std=0.05; Adam lr=0.01, 100 full-batch updates, no BN/dropout/fusion/graph optimization.")


def run_profile(args, run):
    if len(args.backends) != 1 or args.backends[0] == "numpy" or args.case not in ("P0", "P1"):
        raise ValueError("profile requires one GPU backend and --case P0 or P1")
    case = next(c for c in cases("performance", args.seed) if c["id"] == args.case)
    op = Operator(case, args.backends[0])
    expected, actual = oracle(case), op.outputs()
    if not all(error_metrics(actual[k], v)["allclose"] for k, v in expected.items()):
        raise AssertionError("profile input failed correctness")
    if op.backend == "cuda_c":
        compile_cuda(run)
    run.manifest["fixture_sha256"] = array_hash(*case["arrays"])
    for _ in range(args.warmup):
        op.combined()
    op.synchronize()
    xp.cuda.profiler.start()
    try:
        for _ in range(args.measure):
            for label, call in (("forward", op.forward), ("backward", op.backward)):
                xp.cuda.nvtx.RangePush(f"{args.case}/{op.backend}/{label}")
                try:
                    call()
                finally:
                    xp.cuda.nvtx.RangePop()
        op.synchronize()
    finally:
        xp.cuda.profiler.stop()
    run.notes.append("Short NVTX-labelled profiling workload. No profiler-derived durations are regular benchmark evidence.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("correctness", "performance", "train", "profile"), required=True)
    parser.add_argument("--backends", default="numpy,cupy,cuda_c")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--case")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case-timeout", type=float, default=120)
    parser.add_argument("--suite-timeout", type=float, default=1800)
    args = parser.parse_args()
    args.backends = args.backends.split(",")
    allowed_backends = ("numpy", "cupy", "cuda_c", "cuda_im2col", "cuda_im2col_gemm")
    if not args.backends or len(args.backends) != len(set(args.backends)) or any(b not in allowed_backends for b in args.backends):
        parser.error("--backends must be a unique comma-separated subset of numpy,cupy,cuda_c,cuda_im2col,cuda_im2col_gemm")
    if min(args.measure, args.repeats, args.case_timeout, args.suite_timeout) <= 0 or args.warmup < 0:
        parser.error("budgets must be positive, warmup must be nonnegative")
    with RunArtifacts(args.out_dir, vars(args)) as run:
        {"train": run_training, "profile": run_profile}.get(args.suite, run_matrix)(args, run)
    set_device("cpu")


if __name__ == "__main__":
    main()
