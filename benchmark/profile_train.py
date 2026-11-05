"""Capture bounded production training steps in Nsight Systems and analyze the exported SQLite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from benchmark.nsight import find_tool
from benchmark.stage1_common import source_state
from MyFlows.monitoring.nvtx import parse_steps
from tools.check_run_format import check_run
from tools.timeline_gaps import analyze, write_outputs

ROOT = Path(__file__).resolve().parents[1]


def run(command, log):
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name} exited {result.returncode}, see {log}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/semester_2026_fall/monitoring/timeline-001"))
    parser.add_argument("--nsys")
    parser.add_argument("--profile-steps", default="5:9")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--pretrained", default="imagenet", help="converted NPZ, imagenet, or none")
    parser.add_argument("--freeze-stages", type=int, default=3)
    parser.add_argument("--threshold-ms", type=float, default=.1)
    args = parser.parse_args(argv)
    first, last = parse_steps(args.profile_steps)
    if args.batch <= 0 or args.max_samples <= 0 or last > (args.max_samples + args.batch - 1) // args.batch:
        parser.error("capture must fit within the available training batches")
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT):
        parser.error("output must be in the workspace")
    out.mkdir(parents=True, exist_ok=False)
    workload = out / "workload"
    rel = workload.relative_to(ROOT).as_posix()
    command = [sys.executable, "-X", "utf8", "-m", "apps.train", "--device", "cuda", "--seed", "1",
               "--epochs", "1", "--stem", "imagenet", "--base-width", "64", "--batch", str(args.batch),
               "--max-samples", str(args.max_samples), "--no-tensorboard", "--checkpoint-every", "0",
               "--phase-timing-every", "1", "--nvtx-nodes", "--profile-steps", args.profile_steps,
               "--monitor", "off", "--run-dir", rel, "--log-dir", rel, "--out", f"{rel}/checkpoint",
               "--best-out", f"{rel}/best", "--stop-file", f"{rel}/STOP"]
    if args.pretrained != "none":
        command.extend(["--pretrained", args.pretrained, "--freeze-stages", str(args.freeze_stages)])
    elif args.freeze_stages:
        parser.error("freeze-stages requires pretrained weights")
    nsys = str(find_tool("nsys", args.nsys))
    profile = [nsys, "profile", "--trace=cuda,nvtx,cublas", "--sample=none", "--cpuctxsw=none",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--wait=all",
               f"--output={out / 'capture'}", *command]
    database = out / "capture.sqlite"
    export = [nsys, "export", "--type=sqlite", "--output", str(database), str(out / "capture.nsys-rep")]
    manifest = {"schema_version": 1, "source": source_state(), "commands": [profile, export], "capture_steps": [first, last]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    run(profile, out / "profile.log")
    checked = check_run(workload)
    if not checked["valid"] or json.loads((workload / "result.json").read_text(encoding="utf-8"))["status"] != "succeeded":
        raise RuntimeError(f"profiled training failed, see profile.log and workload/result.json: {checked['errors']}")
    run(export, out / "export.log")
    mapping = json.loads((workload / "nvtx_map.json").read_text(encoding="utf-8"))
    report = analyze(database, mapping, threshold_ms=args.threshold_ms)
    write_outputs(report, out)
    captured = [row["step"] for row in report["steps"]]
    if captured != list(range(first, last + 1)) or not report["top_kernels"] or report["unmapped_gaps"]:
        raise RuntimeError(f"capture incomplete: steps {captured}, unmapped gaps {report['unmapped_gaps']}")
    if abs(sum(row["wall_ms"] for row in report["phases"]) - report["step_wall_ms"]) > 1e-6:
        raise RuntimeError("phase partition does not cover complete step time")
    checksums = {}
    for path in (database, workload / "nvtx_map.json", out / "timeline.json", out / "timeline.png"):
        with path.open("rb") as handle:
            checksums[path.relative_to(out).as_posix()] = hashlib.file_digest(handle, "sha256").hexdigest()
    (out / "checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    with (out / "report.md").open("a", encoding="utf-8") as handle:
        handle.write(f"\n原始采集位于 `{out.relative_to(ROOT).as_posix()}/capture.sqlite`。"
                     "manifest.json 保存双仓库源码状态、采集和导出命令。checksums.json 可用于核对 SQLite、来源映射、分析数据和图像。\n\n"
                     f"本次使用 batch {args.batch}，样本数 {args.max_samples}，冻结 {args.freeze_stages} 个阶段，"
                     f"采集全局步 {first} 至 {last}。\n\n"
                     f"复现采集可运行 `python -m benchmark.profile_train --out runs/stage3/profile-repro "
                     f"--profile-steps {args.profile_steps} --max-samples {args.max_samples} --batch {args.batch} "
                     f"--pretrained {args.pretrained} --freeze-stages {args.freeze_stages}`。\n")
    print(f"{captured}: {len(report['gaps'])} mapped idle fragments, {len(report['top_kernels'])} kernel groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
