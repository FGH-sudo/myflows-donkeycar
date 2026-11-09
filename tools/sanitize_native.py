"""Run the native CUDA operator suite under Compute Sanitizer and verify every tool's summary."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SUITE = ("MyFlows.tests.test_cuda_native_kernels", "MyFlows.tests.test_cuda_native_cublas",
         "MyFlows.tests.test_gemm_autotune.RegisterPlanTest.test_all_compiled_tiles_cover_aligned_and_ragged_operands",
         "MyFlows.tests.test_gemm_autotune.RegisterPlanTest.test_narrow_tile_96_splits_and_parallel_reduction_are_deterministic")


def find_sanitizer(explicit=None):
    name = "compute-sanitizer.exe" if os.name == "nt" else "compute-sanitizer"
    candidates = [explicit, shutil.which(name)]
    cuda_root = os.environ.get("CUDA_PATH")
    if cuda_root:
        candidates += [Path(cuda_root)/folder/name for folder in ("compute-sanitizer", "bin")]
    if os.name == "nt":
        base = Path(os.environ.get("ProgramFiles", "C:/Program Files"))/"NVIDIA GPU Computing Toolkit/CUDA"
        versions = sorted(base.glob("v*"), key=lambda p: tuple(int(x) for x in re.findall(r"\d+", p.name)), reverse=True)
        candidates += [p/"compute-sanitizer"/name for p in versions]
    else:
        candidates += [Path("/usr/local/cuda")/"bin"/name]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise FileNotFoundError("Compute Sanitizer was not found in the CUDA Toolkit")


def parse_result(tool, output, diagnostics, exit_code):
    count = re.search(r"Ran (\d+) tests? in ", output)
    errors = [int(n) for n in re.findall(r"ERROR SUMMARY:\s*(\d+) errors?", diagnostics)]
    hazards = [int(n) for n in re.findall(r"RACECHECK SUMMARY:\s*(\d+) hazards?", diagnostics)]
    checks = hazards if tool == "racecheck" else errors
    passed = bool(count and int(count[1]) and re.search(r"^OK\s*$", output, re.M)
                  and exit_code == 0 and checks and not any(errors+hazards))
    return {"tests": int(count[1]) if count else 0, "errors": errors, "hazards": hazards,
            "exit_code": exit_code, "passed": passed}


def main(argv=None):
    from apps.common.gpu_lock import GpuLock
    from apps.autopilot.provenance import code_state
    from apps.autopilot.files import write_json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/experiments/semester_2026_fall/operators/sanitizer-001"))
    ap.add_argument("--tools", nargs="+", choices=("memcheck", "racecheck", "synccheck", "initcheck"),
                    default=["memcheck", "racecheck", "synccheck"])
    args = ap.parse_args(argv)
    exe = find_sanitizer(args.exe)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    version = subprocess.check_output([str(exe), "--version"], encoding="utf-8", errors="replace")
    write_json(out/"manifest.json", {"source": code_state(ROOT), "version": version, "suite": SUITE, "tools": args.tools})
    results = {}
    with GpuLock(label="native CUDA sanitizer", run_dir=out):
        for tool in args.tools:
            log, diagnostics = out/f"{tool}-tests.log", out/f"{tool}-sanitizer.log"
            cmd = [str(exe), "--tool", tool, "--error-exitcode", "66", "--target-processes", "all",
                   "--launch-timeout", "180", "--log-file", str(diagnostics),
                   sys.executable, "-X", "utf8", "-m", "unittest", "-v", *SUITE]
            write_json(out/f"{tool}-command.json", cmd)
            print(f"starting {tool}", flush=True)
            start = time.monotonic()
            with log.open("wb") as handle:
                result = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            diagnostic_text = diagnostics.read_text(encoding="utf-8", errors="replace") if diagnostics.exists() else ""
            row = parse_result(tool, log.read_text(encoding="utf-8", errors="replace"), diagnostic_text, result.returncode)
            results[tool] = {**row, "seconds": time.monotonic()-start}
            write_json(out/"results.json", results)
            print(tool, results[tool], flush=True)
    lines = ["# 原生 CUDA 动态检查", "", "在同一 GPU 上串行执行原生卷积、GEMM、池化及 cuBLAS 路径测试。",
             "", "| 工具 | 运行测试数 | 报告错误数 | racecheck 危险数 | 秒 | 通过 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for tool, row in results.items():
        errors = sum(row["errors"]) if row["errors"] else "—"
        hazards = sum(row["hazards"]) if row["hazards"] else "—"
        lines.append(f"| {tool} | {row['tests']} | {errors} | {hazards} | {row['seconds']:.1f} | {'是' if row['passed'] else '否'} |")
    lines += ["", "工具版本、源码指纹、每次启动命令、测试日志及诊断原文均保存在本目录。"
              "复现命令为 `python -m tools.sanitize_native --out runs/stage3/native-sanitizer`。"]
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return int(not all(r["passed"] for r in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
