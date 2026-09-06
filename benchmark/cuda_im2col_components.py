"""Compare im2col/GEMM/col2im components on the same GPU fixtures."""

import argparse
import time

import numpy as np

from benchmark.cuda_ops import PERFORMANCE_CONV
from benchmark.stage1_common import RunArtifacts, array_hash
from MyFlows.core.device import set_device
from MyFlows.ops.convolution import col2im, im2col
from MyFlows.ops.cuda.kernels import (
    col2im_cuda, im2col_cuda, gemm_forward_cuda, gemm_input_cuda, gemm_weight_cuda,
)
from MyFlows.tests.cuda_fixtures import conv_fixture


def event_time(call, cp):
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    call()
    stop.record()
    stop.synchronize()
    return float(cp.cuda.get_elapsed_time(start, stop))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    with RunArtifacts(args.out_dir, vars(args)) as run:
        import cupy as cp
        set_device("cuda")
        run.results["measurements"] = []
        run.manifest["fixtures"] = {}
        components = ("im2col", "gemm_forward", "gemm_weight", "gemm_input", "col2im")
        for case_id, x_shape, w_shape, stride, padding in PERFORMANCE_CONV:
            case = (case_id, x_shape, w_shape, stride, padding)
            x, w, b, dy = [cp.asarray(value) for value in conv_fixture(case, args.seed)]
            run.manifest["fixtures"][case_id] = {"hash": array_hash(*[cp.asnumpy(v) for v in (x, w, b, dy)]),
                                                  "x": list(x.shape), "w": list(w.shape), "dy": list(dy.shape)}
            out_shape = (x.shape[0], w.shape[0],
                         (x.shape[2] + 2 * padding[0] - w.shape[2]) // stride[0] + 1,
                         (x.shape[3] + 2 * padding[1] - w.shape[3]) // stride[1] + 1)
            dy_rows = dy.transpose(0, 2, 3, 1).reshape(-1, w.shape[0])
            weight_rows = w.reshape(w.shape[0], -1)
            contexts = {"cupy": None, "cuda_im2col": None, "cuda_im2col_gemm": None}
            columns = {}
            for backend in ("cupy", "cuda_im2col", "cuda_im2col_gemm"):
                def make_cols(backend=backend):
                    if backend == "cupy":
                        columns[backend], contexts[backend], _ = im2col(
                            x, w.shape[2:], stride=stride, padding=padding, context=contexts[backend])
                    else:
                        columns[backend], contexts[backend] = im2col_cuda(
                            x, w.shape[2:], stride=stride, padding=padding)
                make_cols()
                for component in components:
                    if component == "im2col":
                        call = make_cols
                    elif component == "gemm_forward":
                        if backend == "cuda_im2col_gemm":
                            call = lambda backend=backend: gemm_forward_cuda(columns[backend], weight_rows)
                        else:
                            call = lambda backend=backend: columns[backend] @ weight_rows.T
                    elif component == "gemm_weight":
                        if backend == "cuda_im2col_gemm":
                            call = lambda backend=backend: gemm_weight_cuda(dy_rows, columns[backend])
                        else:
                            call = lambda backend=backend: dy_rows.T @ columns[backend]
                    elif component == "gemm_input":
                        if backend == "cuda_im2col_gemm":
                            call = lambda: gemm_input_cuda(dy_rows, weight_rows)
                        else:
                            call = lambda: dy_rows @ weight_rows
                    else:
                        grad_cols = (gemm_input_cuda(dy_rows, weight_rows)
                                     if backend == "cuda_im2col_gemm" else dy_rows @ weight_rows)
                        def make_cols_back(backend=backend, grad_cols=grad_cols):
                            if backend == "cupy":
                                col2im(grad_cols, x.shape, w.shape[2:], stride=stride,
                                       padding=padding, context=contexts[backend])
                            elif backend == "cuda_im2col":
                                col2im_cuda(grad_cols, x.shape, w.shape[2:], stride=stride, padding=padding)
                            else:
                                col2im_cuda(grad_cols, x.shape, w.shape[2:], stride=stride, padding=padding)
                        call = make_cols_back
                    for _ in range(args.warmup):
                        event_time(call, cp)
                    samples = []
                    for repeat in range(args.repeats):
                        for iteration in range(args.measure):
                            elapsed = event_time(call, cp)
                            samples.append(elapsed)
                            run.timings.append({"case": case_id, "backend": backend, "component": component,
                                                "repeat": repeat, "iteration": iteration, "ms": elapsed,
                                                "status": "passed"})
                    run.results["measurements"].append({"case": case_id, "backend": backend,
                        "component": component, "mean_ms": float(np.mean(samples)),
                        "median_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
                        "n": len(samples)})
                    print(case_id, backend, component, run.results["measurements"][-1], flush=True)
        run.notes.append("所有路径使用相同 FP32 fixture；CuPy 路径使用 CuPy GEMM，cuda_im2col_gemm 使用自写 CUDA GEMM。组件计时不含 H2D/D2H 和分析器开销；CuPy 的 im2col 上下文在稳态测量中复用。")
        run.manifest["timing_scope"] = "CUDA events on current stream; output allocation is included where the component allocates it. GEMM is deliberately shared."
    set_device("cpu")


if __name__ == "__main__":
    main()
