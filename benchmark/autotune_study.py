"""Calibrate the five native GEMM tiles, remeasure the winners, and capture occupancy counters."""

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import numpy as np

from benchmark.gemm_engines import GEMM_MODELS, _event_ms, conv_gemms
from benchmark.nsight import find_tool
from benchmark.resnet_before_after import ROOT

METRICS = (
    "sm__cycles_elapsed.max",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__block_size",
    "launch__grid_size",
)


def occupancy_child():
    from MyFlows.core.device import set_device, xp
    from MyFlows.ops.cuda_native import native

    set_device("cuda")
    a = xp.asarray(np.random.default_rng(1).normal(size=(512, 512)).astype(np.float32))
    b = a.copy()
    native.gemm(a, b, plan=("128x128", 1))
    xp.cuda.Device().synchronize()
    xp.cuda.profiler.start()
    try:
        for tile in (16, 32):
            native.matmul(a, b, variant="shared", tile=tile, accumulate="fp32")
        for tile in native.GEMM_TILES:
            for splits in (1, 4, 16):
                native.gemm(a, b, plan=(tile, splits))
        xp.cuda.Device().synchronize()
    finally:
        xp.cuda.profiler.stop()


def occupancy(out):
    tool = str(find_tool("ncu"))
    prefix = out / "occupancy"
    cmd = [
        tool,
        "--config-file",
        "off",
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--kernel-name",
        "regex:matmul_shared|gemm_register_[shtwx]_",
        "--metrics",
        ",".join(METRICS),
        "--export",
        str(prefix),
        "--force-overwrite",
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "benchmark.autotune_study",
        "--profile-child",
    ]
    with (out / "ncu.log").open("wb") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    raw = subprocess.check_output(
        [
            tool,
            "--import",
            str(prefix.with_suffix(".ncu-rep")),
            "--page",
            "raw",
            "--csv",
            "--print-units",
            "base",
        ],
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    (out / "occupancy.csv").write_text(raw, encoding="utf-8")
    rows = list(csv.DictReader(line for line in raw.splitlines() if line.startswith('"')))
    results = [
        {"kernel": r["Kernel Name"], **{m: float(r[m].replace(",", "")) for m in METRICS}}
        for r in rows
        if r.get("Kernel Name") and r.get(METRICS[0], "").replace(",", "").replace(".", "").isdigit()
    ]
    if len(results) != 17:
        raise RuntimeError(f"expected 2 shared-memory and 15 register launches; captured {len(results)}")
    return results


def main(argv=None):
    from apps.autopilot.files import write_json
    from apps.autopilot.provenance import code_state
    from apps.common.gpu_lock import GpuLock
    from MyFlows.core.device import set_device, xp
    from MyFlows.ops.cuda_native import autotune, native

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--profile-child", action="store_true")
    args = ap.parse_args(argv)
    if args.profile_child:
        return occupancy_child()
    if not args.out:
        ap.error("--out is required")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates").mkdir(exist_ok=True)
    set_device("cuda")
    states = code_state(ROOT)
    manifest = out / "manifest.json"
    if (
        manifest.exists()
        and json.loads(manifest.read_text(encoding="utf-8"))["source"]["repos"]["MyFlows"]
        != states["repos"]["MyFlows"]
    ):
        raise ValueError("framework source differs from the archived calibration")
    write_json(manifest, {"schema_version": 1, "source": states, "metrics": METRICS})
    old = os.environ.get("MYFLOWS_GEMM_AUTOTUNE")
    with GpuLock(label="native GEMM tile and split calibration", run_dir=out):
        try:
            os.environ["MYFLOWS_GEMM_AUTOTUNE"] = "0"
            cases = []
            for model, stem, width, batch in GEMM_MODELS:
                cases.extend(
                    (model, part, layout, m, n, k)
                    for part, layout, m, n, k in sorted(conv_gemms(stem, width, batch))
                )
                gc.collect()
                xp.get_default_memory_pool().free_all_blocks()
            summary = out / "shapes.json"
            rows = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else []
            done = {(r["model"], r["part"], r["layout"], *r["shape"]) for r in rows}
            for case in cases:
                if case in done:
                    continue
                model, part, layout, m, n, k = case
                autotune.selected_plan(layout, m, n, k)
                path = autotune.cache_file(layout, m, n, k)
                saved = json.loads(path.read_text(encoding="utf-8"))
                shutil.copyfile(path, out / "candidates" / path.name)
                rng = xp.random.RandomState(99)
                ta, tb = layout == "TN", layout == "NT"
                a = rng.standard_normal((k, m) if ta else (m, k), dtype=xp.float32)
                b = rng.standard_normal((n, k) if tb else (k, n), dtype=xp.float32)
                dest = xp.empty((m, n), dtype=xp.float32)
                winner = saved["selected"]
                plans = {"manual": None, "tuned": (winner["tile"], winner["requested_slices"])}
                times = {"manual": [], "tuned": [], "cublas": []}
                for i in range(5):
                    for label in times if i % 2 == 0 else reversed(times):
                        kw = {"engine": "cublas"} if label == "cublas" else {"plan": plans[label]}
                        call = lambda: native.gemm(a, b, out=dest, trans_a=ta, trans_b=tb, **kw)
                        call()
                        times[label].append(float(_event_ms(xp, call, 8)))
                rows.append(
                    {
                        "model": model,
                        "part": part,
                        "layout": layout,
                        "shape": [m, n, k],
                        "manual": saved["manual"],
                        "tuned": {"tile": winner["tile"], "slices": winner["slices"]},
                        "times_ms": times,
                        "measured": sum(not r["skip"] for r in saved["candidates"]),
                        "skipped": sum(bool(r["skip"]) for r in saved["candidates"]),
                        "sample_error": winner["sample_relative_vs_fp64"],
                        "cache": path.name,
                    }
                )
                write_json(summary, rows)
                print(
                    len(rows),
                    len(cases),
                    model,
                    part,
                    layout,
                    (m, n, k),
                    winner["tile"],
                    winner["slices"],
                    flush=True,
                )
                del a, b, dest
                xp.get_default_memory_pool().free_all_blocks()
            counters = occupancy(out)
        finally:
            if old is None:
                os.environ.pop("MYFLOWS_GEMM_AUTOTUNE", None)
            else:
                os.environ["MYFLOWS_GEMM_AUTOTUNE"] = old
    write_json(out / "occupancy.json", counters)
    lines = [
        "# 寄存器 GEMM 自动调优",
        "",
        "固定布局与 M、N、K 后，实测五种 tile 和 1–32 个请求分片中各个不同的有效切分。"
        "缓存按 GPU、驱动、运行库、原生源码哈希和布局区分。另做五轮交替 CUDA Event 复测，表中为整次 GEMM 的中位耗时，含 split-K 归约。",
        "",
        "| 网络 | 部分 | 布局 | M×N×K | 手选 tile / 切分 | 实测 tile / 切分 | 候选数 | 手选 ms | 实测 ms | cuBLAS ms |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        t = {k: statistics.median(v) for k, v in row["times_ms"].items()}
        oldplan, newplan = row["manual"], row["tuned"]
        lines.append(
            f"| {row['model']} | {row['part']} | {row['layout']} | {'×'.join(map(str, row['shape']))} | "
            f"{oldplan['tile']} / {oldplan['slices']} | {newplan['tile']} / {newplan['slices']} | {row['measured']} | "
            f"{t['manual']:.4f} | {t['tuned']:.4f} | {t['cublas']:.4f} |"
        )
    lines += [
        "",
        "## 占用率与周期",
        "",
        "相同 512×512 输入，前两行改变共享内存算法的线程块尺寸。"
        "后续各 tile 分别以 1、4、16 个 K 分片运行。占用率取活跃周期中实际活跃 warp 比例。",
        "",
        "| kernel | 线程/块 | 总块数 | 占用率 % | SM 周期 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in counters:
        lines.append(
            f"| {r['kernel']} | {r[METRICS[2]]:.0f} | {r[METRICS[3]]:.0f} | {r[METRICS[1]]:.2f} | {r[METRICS[0]]:.0f} |"
        )
    lines += [
        "",
        "candidates 保存每个候选的耗时、FP64 抽样误差与相对不切分的最大输出差。"
        "shapes.json、occupancy.ncu-rep 和 occupancy.csv 保存独立复测原始数据。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
