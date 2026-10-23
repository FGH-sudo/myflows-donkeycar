"""Build the native CUDA/cuBLAS scheduler library with CMake.

The build itself lives in MyFlows/ops/cuda_native/build.py, which native.py also runs on first use
when the library is missing or older than its sources. This script always rebuilds and streams
CMake's output; --if-needed skips the build when the library is current.
"""

import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Load the build module by path: it only needs the standard library, unlike the MyFlows package.
_spec = importlib.util.spec_from_file_location(
    "myflows_native_build", ROOT / "MyFlows" / "ops" / "cuda_native" / "build.py")
native_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native_build)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--if-needed", action="store_true", help="build only when the library is missing or stale")
    args = parser.parse_args(argv)
    try:
        if args.if_needed:
            native_build.ensure_built()
            print(native_build.OUTPUT)
        else:
            print(native_build.build())
    except native_build.BuildError as exc:
        raise SystemExit(f"native build failed: {exc}")


if __name__ == "__main__":
    main()
