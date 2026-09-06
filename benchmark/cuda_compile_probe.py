"""Measure first CUDA-source compilation with a new empty CuPy disk cache."""

import argparse
import os
import time

from benchmark.stage1_common import RunArtifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    with RunArtifacts(args.out_dir, vars(args)) as run:
        cache = run.path / "cupy-cache"
        cache.mkdir()
        os.environ["CUPY_CACHE_DIR"] = str(cache)
        from MyFlows.core.device import set_device
        from benchmark.cuda_ops import compile_cuda
        set_device("cuda")
        start = time.perf_counter()
        compile_cuda(run)
        run.results["cold_compile_wall_ms"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        compile_cuda(run)
        run.results["warm_lookup_wall_ms"] = (time.perf_counter() - start) * 1000
        run.manifest["cache_directory"] = str(cache)
        run.notes.append("Fresh cache directory, fresh Python process; first lookup compiles both source modules and resolves all six kernels. Includes host compiler/library overhead. Warm lookup reuses in-process modules.")
        print(run.results, flush=True)


if __name__ == "__main__":
    main()
