"""Collect an actual Nsight report; workload success alone is insufficient."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys

from benchmark.stage1_common import ROOT, RunArtifacts


def execute(command, log, timeout=180):
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace", timeout=timeout)
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"profiler exited {result.returncode}; see {log.name}: {result.stdout[-1500:]}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=("nsys", "ncu"), required=True)
    parser.add_argument("--tool-path", required=True)
    parser.add_argument("--case", choices=("P0", "P1"), required=True)
    parser.add_argument("--backend", choices=("cupy", "cuda_native_cublas"), default="cuda_native_cublas")
    parser.add_argument("--kernel", default="im2col_forward")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    with RunArtifacts(args.out_dir, vars(args)) as run:
        tool = str(Path(args.tool_path).resolve(strict=True))
        run.manifest["profiler_version"] = execute([tool, "--version"], run.path / "version.txt")
        prefix = run.path / "capture"
        workload = [sys.executable, "-X", "utf8", "-m", "benchmark.cuda_ops", "--suite", "profile",
                    "--case", args.case, "--backends", args.backend, "--warmup", "5", "--measure", "3",
                    "--out-dir", str(run.path / "workload")]
        if args.tool == "nsys":
            command = [tool, "profile", "--trace=cuda,nvtx,cublas", "--sample=none", "--cpuctxsw=none",
                       "--capture-range=cudaProfilerApi", "--capture-range-end=stop", f"--output={prefix}", *workload]
            report = prefix.with_suffix(".nsys-rep")
        else:
            command = [tool, "--config-file", "off", "--set", "full", "--kernel-name", args.kernel,
                       "--launch-count", "1", "--profile-from-start", "off", "--export", str(prefix), *workload]
            report = prefix.with_suffix(".ncu-rep")
        run.manifest["profiler_command"] = command
        execute(command, run.path / "profiler.log")
        if not report.exists() or report.stat().st_size == 0:
            raise RuntimeError("profiler did not produce a nonempty report")
        if args.tool == "nsys":
            stats_command = [tool, "stats", "--report", "cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum", "--format", "csv",
                             "--output", str(run.path / "summary"), str(report)]
            execute(stats_command, run.path / "stats.log")
            with (run.path / "summary_cuda_gpu_kern_sum.csv").open(encoding="utf-8-sig", newline="") as stream:
                kernels = list(csv.DictReader(stream))
            expected_kernel = {"cuda_native_cublas": "im2col_forward"}.get(args.backend)
            if not kernels or (expected_kernel and not any(expected_kernel in row["Name"] for row in kernels)):
                raise RuntimeError("trace does not contain the expected GPU kernels")
            run.results["kernel_summary"] = kernels
        else:
            execute([tool, "--import", str(report), "--page", "raw", "--csv"], run.path / "metrics.csv")
            if args.kernel not in (run.path / "metrics.csv").read_text(encoding="utf-8"):
                raise RuntimeError("Compute report does not contain the requested kernel")
        run.results["profile_report"] = report.name
        run.notes.append("已验证实际 profiler 报告并提取 kernel 数据；普通性能实验单独记录。")
        print(f"{args.tool} {args.case} {args.backend}: {report.name}", flush=True)


if __name__ == "__main__":
    main()
