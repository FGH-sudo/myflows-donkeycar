"""Rebuild stage-one tables and figures from immutable run artifacts."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    root, out = Path(args.runs_dir).resolve(), Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "generate_evidence.py")
    tables = []
    data = {}

    def result(run):
        return json.loads((root / run / "results.json").read_text(encoding="utf-8"))

    def rows(run, filename):
        with (root / run / filename).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def table(title, headers, entries):
        tables.extend([f"## {title}", "", "| " + " | ".join(headers) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |"])
        tables.extend("| " + " | ".join(str(v) for v in entry) + " |" for entry in entries)
        tables.append("")

    measurements = result("performance-final-004")["measurements"]
    data["performance_run"] = "performance-final-004"
    data["measurements"] = measurements
    table("Performance: operator-call and transfer-inclusive (ms)",
          ["Case", "Backend", "Phase", "Mean", "Median", "P95", "N", "NumPy/CUDA", "CuPy/CUDA"],
          [[r["case"], r["backend"], r["phase"], f'{r["mean_ms"]:.6f}', f'{r["median_ms"]:.6f}',
            f'{r["p95_ms"]:.6f}', r["n"], f'{r["numpy_speedup"]:.3f}' if "numpy_speedup" in r else "",
            f'{r["cupy_speedup"]:.3f}' if "cupy_speedup" in r else ""] for r in measurements])
    errors = []
    for run in ("correctness-final-002", "performance-final-004"):
        for row in result(run)["correctness"]:
            for tensor, error in row["errors"].items():
                errors.append({"run": run, "case": row["case"], "backend": row["backend"], "tensor": tensor, **error})
    data["errors"] = errors
    table("Errors against independent FP64 reference", ["Run", "Case", "Backend", "Tensor", "Max abs", "Max rel", "Allclose"],
          [[r["run"], r["case"], r["backend"], r["tensor"], f'{r["max_abs"]:.8g}', f'{r["max_rel"]:.8g}', r["allclose"]] for r in errors])
    before, after = (result(n)["measurements"] for n in ("performance-001", "performance-dx-window-002"))
    optimization = []
    for old in before:
        if old["backend"] != "cuda_c" or old["case"] not in ("P0", "P1", "P2"):
            continue
        new = next(r for r in after if (r["case"], r["backend"], r["phase"]) == (old["case"], old["backend"], old["phase"]))
        optimization.append({"case": old["case"], "phase": old["phase"], "before_ms": old["mean_ms"],
                             "after_ms": new["mean_ms"], "speedup": old["mean_ms"] / new["mean_ms"]})
    data["optimization"] = optimization
    table("dX optimization: ordinary operator-call timings", ["Case", "Phase", "Before ms", "After ms", "Speedup"],
          [[r["case"], r["phase"], f'{r["before_ms"]:.6f}', f'{r["after_ms"]:.6f}', f'{r["speedup"]:.3f}'] for r in optimization])
    system_summary = []
    for case in ("P0", "P1"):
        for backend in ("cupy", "cuda_c"):
            run = f"nsys-{case}-cupy-002" if backend == "cupy" else f"nsys-{case}-dx-window-003"
            kernel = rows(run, "summary_cuda_gpu_kern_sum.csv")
            nvtx = rows(run, "summary_nvtx_sum.csv")
            system_summary.append({"run": run, "kernel_instances": sum(int(r["Instances"]) for r in kernel),
                                   "kernel_total_us": sum(float(r["Total Time (ns)"]) for r in kernel) / 1000,
                                   "nvtx_total_us": sum(float(r["Total Time (ns)"]) for r in nvtx) / 1000})
    data["systems_summary"] = system_summary
    table("Systems: three forward/backward pairs per capture", ["Run", "Kernel calls", "Kernel total us", "NVTX total us"],
          [[r["run"], r["kernel_instances"], f'{r["kernel_total_us"]:.3f}', f'{r["nvtx_total_us"]:.3f}'] for r in system_summary])
    metrics = ["gpu__time_duration.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
               "sm__warps_active.avg.pct_of_peak_sustained_active", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
               "l1tex__throughput.avg.pct_of_peak_sustained_active", "launch__registers_per_thread",
               "launch__block_size", "launch__occupancy_limit_registers", "launch__waves_per_multiprocessor"]
    ncu = {case: rows(f"ncu-{case}-dx-window-002", "metrics.csv") for case in ("P0", "P1")}
    data["compute"] = {case: {key: content[1][key] for key in metrics} for case, content in ncu.items()}
    table("Compute: conv2d_backward_input", ["Metric", "Unit", "P0", "P1"],
          [[key, ncu["P0"][0][key], ncu["P0"][1][key], ncu["P1"][1][key]] for key in metrics])
    ps = []
    phase_stats = []
    for label in ("1", "2", "4", "uneven"):
        run = f"restored-ps-{label}-001"
        r = result(run)
        ps.append({"run": run, "workers": r["ps"]["config"]["workers"], "shards": r["ps"]["config"]["shard_sizes"],
                   "max_abs": r["equivalence_max_abs"], "ps_wall_s": r["ps"]["elapsed_s"],
                   "single_wall_s": r["single_process_wall_s"], "cleanup_s": r["ps"]["cleanup_s"]})
        events = [json.loads(line) for line in (root / run / "events.jsonl").read_text().splitlines()]
        for key in ("compute_s", "upload_submit_s", "upload_ack_wait_s", "parameter_wait_receive_s",
                    "collect_wait_s", "aggregate_s", "update_s", "broadcast_submit_s", "step_s"):
            vals = [e[key] * 1000 for e in events if key in e]
            phase_stats.append({"run": run, "phase": key, "n": len(vals), "mean_ms": float(np.mean(vals)), "p95_ms": float(np.percentile(vals, 95))})
    data["ps"] = ps
    data["ps_phases"] = phase_stats
    table("PS equivalence and one-run wall time", ["Run", "Shards", "Max abs", "PS wall s", "Single wall s", "Cleanup s"],
          [[r["run"], r["shards"], f'{r["max_abs"]:.8g}', f'{r["ps_wall_s"]:.6f}', f'{r["single_wall_s"]:.6f}', f'{r["cleanup_s"]:.6f}'] for r in ps])
    table("PS phase timings (includes first-step startup wait)", ["Run", "Phase", "N", "Mean ms", "P95 ms"],
          [[r["run"], r["phase"], r["n"], f'{r["mean_ms"]:.6f}', f'{r["p95_ms"]:.6f}'] for r in phase_stats])

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    colors = {"numpy": "#687582", "cupy": "#2e8b79", "cuda_c": "#df7438"}
    for axis, phase, title in zip(axes, ("combined", "transfer_inclusive"), ("Forward + backward: operator call", "Transfer-inclusive: end to end")):
        for index, backend in enumerate(colors):
            selected = [next(r for r in measurements if (r["case"], r["backend"], r["phase"]) == (case, backend, phase)) for case in ("P0", "P1", "P2")]
            x = np.arange(3) + (index - 1) * 0.24
            axis.bar(x, [r["mean_ms"] for r in selected], width=0.22, color=colors[backend], label=backend)
            axis.scatter(x, [r["p95_ms"] for r in selected], marker="_", s=85, color="#20242a", zorder=3)
        axis.set(xticks=range(3), xticklabels=("P0 small", "P1 medium", "P2 image-sized"), yscale="log", ylabel="Milliseconds (log scale)", title=title)
        axis.grid(axis="y", alpha=0.15)
    axes[0].legend(frameon=False)
    fig.suptitle("Three backends | bars: mean; ticks: P95 | 150 samples per group\nperformance-final-004", fontsize=12)
    fig.savefig(out / "performance.png", dpi=170)
    plt.close(fig)

    run = "nsys-P1-dx-window-003"
    with sqlite3.connect(root / run / "capture.sqlite") as connection:
        nvtx = connection.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'P1/cuda_c/%' ORDER BY start").fetchall()
        kernels = connection.execute("SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id ORDER BY k.start").fetchall()
    origin = min(r[0] for r in nvtx)
    palette = {"forward": "#2e8b79", "backward": "#bec8ce", "dX": "#df7438", "dW": "#46789c", "db": "#b89036", "add": "#87949b"}
    fig, axis = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
    for start, end, name in nvtx:
        label = name.rsplit("/", 1)[-1]
        axis.broken_barh([((start - origin) / 1e6, (end - start) / 1e6)], (1.1, .5), facecolors=palette[label])
        axis.text(((start + end) / 2 - origin) / 1e6, 1.35, label, ha="center", va="center", fontsize=8)
    for start, end, name in kernels:
        label = "dX" if "backward_input" in name else "dW" if "backward_weight" in name else "db" if "backward_bias" in name else "forward" if "forward_direct" in name else "add"
        axis.broken_barh([((start - origin) / 1e6, (end - start) / 1e6)], (.25, .5), facecolors=palette[label])
    axis.set(yticks=(.5, 1.35), yticklabels=("GPU kernels", "CPU NVTX ranges"), ylim=(0, 2.2), xlabel="Time since first NVTX range (ms)", title=f"Actual captured timeline: {run}\nGPU lane shows kernels; transfers/memsets are not drawn")
    axis.legend(handles=[Patch(color=palette[k], label=k) for k in ("forward", "dX", "dW", "db", "add")], loc="upper right", ncol=5, frameon=False)
    axis.grid(axis="x", alpha=.15)
    fig.savefig(out / "systems-timeline.png", dpi=170)
    plt.close(fig)
    (out / "tables.md").write_text("# Stage-one derived evidence\n\nAll values derive from named run artifacts.\n\n" + "\n".join(tables), encoding="utf-8")
    (out / "tables.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote tables and figures to {out}")


if __name__ == "__main__":
    main()
