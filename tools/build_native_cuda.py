"""Build the user-space native CUDA/cuBLAS scheduler DLL with CMake/MinGW."""

from pathlib import Path
import subprocess
import sys
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "MyFlows" / "ops" / "cuda_native" / "native_cublas.cpp"
PROJECT = SOURCE.parent
OUTPUT = ROOT / ".codex" / "native-cuda" / "native_cublas.dll"
BUILD = ROOT / ".codex" / "native-cuda" / "build"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cmake = shutil.which("cmake") or r"C:\Program Files\CMake\bin\cmake.exe"
    configure = [cmake, "-S", str(PROJECT), "-B", str(BUILD), "-G", "MinGW Makefiles",
                 "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={OUTPUT.parent}"]
    build = [cmake, "--build", str(BUILD), "--config", "Release", "--parallel"]
    print("$", " ".join(configure), flush=True)
    subprocess.run(configure, cwd=ROOT, check=True)
    print("$", " ".join(build), flush=True)
    subprocess.run(build, cwd=ROOT, check=True)
    if not OUTPUT.exists():
        raise FileNotFoundError(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
