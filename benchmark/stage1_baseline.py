"""Record the C0 NumPy/CuPy baseline and independently compile a CUDA probe."""

import argparse
import os
from pathlib import Path
import shutil
import time

import numpy as np

from benchmark.nsight import ENVIRONMENT
from benchmark.stage1_common import RunArtifacts, array_hash, command, error_metrics
from MyFlows.core.device import set_device, xp, asnumpy
from MyFlows.core.node import Variable
from MyFlows.ops.convolution import Conv2D_Op


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    with RunArtifacts(args.out_dir, vars(args)) as run:
        rng = np.random.default_rng(0)
        x, w, b, dy = [rng.normal(size=shape).astype(np.float32) for shape in
                       ((2, 3, 7, 9), (4, 3, 3, 3), (4,), (2, 4, 4, 5))]
        np.savez(run.path / "fixture.npz", x=x, w=w, b=b, dy=dy)
        run.manifest["fixture_sha256"] = array_hash(x, w, b, dy)
        outputs = {}
        for backend, device in (("numpy", "cpu"), ("cupy", "cuda")):
            set_device(device)
            nodes = [Variable(xp.asarray(a)) for a in (x, w, b)]
            op = Conv2D_Op(nodes[0], nodes[1], stride=2, padding=1, bias=nodes[2])
            op.forward(*(node.value for node in nodes))
            op.grad = xp.asarray(dy)
            op.backward()
            outputs[backend] = [asnumpy(a).copy() for a in [op.value, *(n.grad for n in nodes)]]
        run.results["numpy_cupy"] = [error_metrics(a, b) for a, b in zip(outputs["cupy"], outputs["numpy"])]
        assert all(row["allclose"] for row in run.results["numpy_cupy"])
        import cupy as cp
        source = 'extern "C" __global__ void stage1_probe(float* x) { x[threadIdx.x] += 1.0f; }'
        start = time.perf_counter()
        kernel = cp.RawKernel(source, "stage1_probe", options=("--std=c++11",))
        kernel.compile()
        run.results["rawkernel_compile_wall_ms"] = (time.perf_counter() - start) * 1000
        values = cp.zeros(32, dtype=cp.float32)
        kernel((1,), (32,), (values,))
        np.testing.assert_array_equal(cp.asnumpy(values), np.ones(32))
        run.results["rawkernel_probe"] = "passed"
        run.manifest["cuda"] = {"runtime": cp.cuda.runtime.runtimeGetVersion(),
                                "driver": cp.cuda.runtime.driverGetVersion(),
                                "probe_source": source}
        if os.name == "nt":
            suffix = ".exe"
            roots = [Path("C:/Program Files/NVIDIA Corporation"),
                     Path("C:/Program Files/NVIDIA GPU Computing Toolkit")]
        else:
            suffix = ""
            roots = [Path("/opt/nvidia"), *sorted(Path("/usr/local").glob("cuda*"))]
        # Tools installed elsewhere are named by the environment (CUDA_PATH; MF_NSYS / MF_NCU as in
        # benchmark/nsight.py), not by machine-specific search roots.
        if os.environ.get("CUDA_PATH"):
            roots.append(Path(os.environ["CUDA_PATH"]))
        found = {}
        for name in ("nvcc", "nsys", "ncu"):
            configured = os.environ.get(ENVIRONMENT[name]) if name in ENVIRONMENT else None
            paths = [shutil.which(name), configured if configured and Path(configured).is_file() else None]
            for root in roots:
                if root.exists():
                    paths.extend(str(p) for p in root.rglob(name + suffix) if p.is_file())
            found[name] = sorted(set(filter(None, paths)))
        run.results["tool_paths"] = found
        run.results["tool_versions"] = {p: command([p, "--version"])
                                        for paths in found.values() for p in paths}
        run.notes.append("C0 proves baseline and NVRTC compilation only. Missing profiler tools do not satisfy G-PROFILE.")
        print(run.results)
    set_device("cpu")


if __name__ == "__main__":
    main()
