"""Lecture 19 matrix multiplication: hand-written CUDA kernels against NumPy, CuPy and cuBLAS.

The kernels live in MyFlows/ops/cuda_native/native_cublas.cpp: basic (one thread per output on a
1-D grid), tiled (TILE x TILE thread blocks reading global memory), shared (TILE x TILE tiles
staged in shared memory) and register (128 x 128 block tiles, 8 x 8 outputs per thread held in
registers, float4 loads, double-buffered shared tiles 16 deep in k for fp32, which is the GEMM engine
of the native layers, and 8 deep for kahan / dot2), each with fp32 / kahan / dot2
accumulation, plus cuBLAS SGEMM called through the same DLL. Every implementation multiplies the same float32 operands; errors are
measured against the FP64 product of those operands and against NumPy's float32 product (the
"NumPy baseline" of the lecture). Host-device copies are timed separately from the kernels.

The report opens with the lecture's data table (pages 72-73): NumPy (baseline), CuPy and the
basic / tiled / shared CUDA C kernels at n = 1024, 2048 and 4096, with the mean time (each size
measured in three rounds, the median of the round means), the speedup over NumPy and the maximum
absolute and relative errors against NumPy.

  python -m benchmark.matmul_compare --out-dir docs/experiments/semester_2026_fall/cuda_lecture19/matmul-001
  python -m benchmark.matmul_compare --out-dir <dir> --ncu      # adds Nsight Compute metrics at --ncu-size
"""

import argparse
import csv
import subprocess
import sys
import time

import numpy as np

from benchmark.nsight import find_tool
from benchmark.stage1_common import ROOT, RunArtifacts, array_hash
from MyFlows.core.device import set_device, xp
from MyFlows.ops.cuda_native import native

U32 = float(np.finfo(np.float32).eps) / 2  # unit roundoff of float32
NCU_METRICS = (
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
)
KERNELS = tuple(
    [("cublas", "cublas", 32, "fp32")]
    + [(f"basic-{acc}", "basic", 32, acc) for acc in native.MATMUL_ACCUMULATION]
    + [(f"{variant}{tile}-{acc}", variant, tile, acc)
       for variant in ("tiled", "shared") for tile in (16, 32) for acc in native.MATMUL_ACCUMULATION]
    + [(f"register-{acc}", "register", 32, acc) for acc in native.MATMUL_ACCUMULATION])
IMPLEMENTATIONS = ("numpy", "cupy") + tuple(name for name, *_ in KERNELS)
# The lecture's data table: its five implementations (the tiled version uses 16 x 16 output blocks)
# at its three sizes.
LECTURE_SIZES = (1024, 2048, 4096)
LECTURE_ROWS = (("numpy", "NumPy（基线）"), ("cupy", "CuPy"), ("basic-fp32", "CUDA C 基础版"),
                ("tiled16-fp32", "CUDA C 分块版"), ("shared16-fp32", "CUDA C 共享内存版"),
                ("register-fp32", "CUDA C 寄存器分块版（优化）"))


def operands(n, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n), dtype=np.float32), rng.standard_normal((n, n), dtype=np.float32)


def accuracy(c, exact, magnitude, baseline):
    """Forward error of ``c`` against the FP64 product, plus its distance to NumPy's float32 product."""
    c64 = c.astype(np.float64)
    error = np.abs(c64 - exact)
    return {
        "max_abs": float(error.max()),
        "rel_rms": float(np.sqrt(np.mean(error ** 2) / np.mean(exact ** 2))),
        # In units of u * (|A| |B|)_ij: the classical bound for a sequential dot product is ~K.
        "max_scaled": float((error / np.maximum(magnitude * U32, 1e-300)).max()),
        "correctly_rounded": float(np.mean(c == exact.astype(np.float32))),
        "max_abs_vs_numpy": float(np.abs(c64 - baseline).max()),
        "rel_vs_numpy": float(np.linalg.norm(c64 - baseline) / np.linalg.norm(baseline)),
    }


def traffic(name, n):
    """Analytic L1 global-load bytes (32-byte sectors of warp-wide loads) and compulsory DRAM bytes.

    basic and tiled32 run a warp over 32 consecutive columns of one output row: per k it reads
    four B sectors and one broadcast A sector, 160 bytes for 32 outputs, i.e. 5 n^3 bytes in all.
    tiled16 runs a warp over 2 rows x 16 columns: two shared B sectors plus two A sectors, 4 n^3.
    shared stages each operand element once per thread block: 8 n^3 / TILE. register does the same
    with 128 x 128 block tiles read as float4: 4 n^3 (1/128 + 1/128) = n^3 / 16.
    """
    ideal = 3 * n * n * 4
    if name in ("numpy", "cupy", "cublas"):
        return None, ideal
    if name.startswith("register"):
        return n ** 3 // 16, ideal
    if name.startswith("shared"):
        return 8 * n ** 3 // int(name[len("shared"):name.index("-")]), ideal
    return (4 if name.startswith("tiled16") else 5) * n ** 3, ideal


class Timer:
    def __init__(self, budget_s, max_samples):
        self.budget_ms = budget_s * 1000.0
        self.max_samples = max_samples

    def _cuda_once(self, call):
        start, stop = xp.cuda.Event(), xp.cuda.Event()
        wall = time.perf_counter()
        start.record()
        call()
        stop.record()
        stop.synchronize()
        return float(xp.cuda.get_elapsed_time(start, stop)), (time.perf_counter() - wall) * 1000.0

    @staticmethod
    def _cpu_once(call):
        wall = time.perf_counter()
        call()
        elapsed = (time.perf_counter() - wall) * 1000.0
        return elapsed, elapsed

    def samples(self, call, on_gpu):
        once = (lambda: self._cuda_once(call)) if on_gpu else (lambda: self._cpu_once(call))
        estimate = max(once()[1], 1e-3)  # dry run: also pays one-time costs (cuBLAS heuristics, ...)
        for _ in range(int(np.clip(200.0 / estimate, 1, 10))):
            once()
        count = int(np.clip(self.budget_ms / estimate, 3, self.max_samples))
        return [once() for _ in range(count)]


def transfers(a, b, repeats=5):
    """Median H2D (A and B) and D2H (C) times, from pageable and from pinned host memory."""
    import cupyx

    device = xp.cuda.Device()
    pinned_a, pinned_b = cupyx.empty_pinned(a.shape, a.dtype), cupyx.empty_pinned(b.shape, b.dtype)
    pinned_a[...], pinned_b[...] = a, b
    pinned_c = cupyx.empty_pinned(a.shape, np.float32)
    rows = {"h2d_ms": [], "d2h_ms": [], "h2d_pinned_ms": [], "d2h_pinned_ms": []}
    for _ in range(repeats):
        for key, host_a, host_b, host_c in (("", a, b, None), ("_pinned", pinned_a, pinned_b, pinned_c)):
            device.synchronize()
            start = time.perf_counter()
            da, db = xp.asarray(host_a), xp.asarray(host_b)
            device.synchronize()
            rows[f"h2d{key}_ms"].append((time.perf_counter() - start) * 1000.0)
            start = time.perf_counter()
            if host_c is None:
                da.get()
            else:
                da.get(out=host_c)
            device.synchronize()
            rows[f"d2h{key}_ms"].append((time.perf_counter() - start) * 1000.0)
            del da, db
    return {key: float(np.median(values)) for key, values in rows.items()}


def make_call(name, da, db, dc, a, b):
    if name == "numpy":
        return lambda: np.matmul(a, b)
    if name == "cupy":
        return lambda: xp.matmul(da, db, out=dc)
    _, variant, tile, accumulate = next(k for k in KERNELS if k[0] == name)
    return lambda: native.matmul(da, db, variant=variant, tile=tile, accumulate=accumulate, out=dc)


def run_size(n, args, run):
    a, b = operands(n, args.seed)
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    exact, magnitude = a64 @ b64, np.abs(a64) @ np.abs(b64)
    baseline = (a @ b).astype(np.float64)
    del a64, b64
    run.manifest.setdefault("fixture_sha256", {})[str(n)] = array_hash(a, b)
    copy = transfers(a, b)
    da, db = xp.asarray(a), xp.asarray(b)
    dc = xp.empty((n, n), dtype=xp.float32)
    timer = Timer(args.budget, args.max_samples)
    flops = 2.0 * n ** 3
    calls, metrics_by_name = {}, {}
    for name in args.implementations:
        if name.startswith("basic") and n > args.basic_limit:
            continue
        calls[name] = make_call(name, da, db, dc, a, b)
        result = calls[name]()
        metrics_by_name[name] = accuracy(xp.asnumpy(dc) if name != "numpy" else result, exact, magnitude, baseline)
    # Each round measures every implementation once, so a slow drift of the laptop's CPU/GPU clocks
    # spreads over all of them instead of biasing the one that happened to run during it.
    rounds_by_name = {name: [] for name in calls}
    for _ in range(args.rounds):
        for name, call in calls.items():
            rounds_by_name[name].append(timer.samples(call, name != "numpy"))
    rows = []
    for name, rounds in rounds_by_name.items():
        on_gpu = name != "numpy"
        metrics = metrics_by_name[name]
        for round_index, samples in enumerate(rounds):
            for index, (sample_ms, wall_ms) in enumerate(samples):
                run.timings.append({"size": n, "implementation": name, "round": round_index, "iteration": index,
                                    "ms": sample_ms, "wall_ms": wall_ms,
                                    "clock": "cuda_event" if on_gpu else "perf_counter"})
        samples = [s for r in rounds for s in r]
        ms = [s[0] for s in samples]
        round_means = [float(np.mean([s[0] for s in r])) for r in rounds]
        median = float(np.median(ms))
        requests, ideal = traffic(name, n)
        row = {"size": n, "implementation": name, **metrics, "round_mean_ms": round_means,
               "stable_mean_ms": float(np.median(round_means)),
               "median_ms": median, "mean_ms": float(np.mean(ms)), "p95_ms": float(np.percentile(ms, 95)),
               "min_ms": float(np.min(ms)), "samples": len(ms), "wall_median_ms": float(np.median([s[1] for s in samples])),
               "gflops": flops / (median * 1e6), "load_request_bytes": requests, "compulsory_bytes": ideal}
        if on_gpu:
            row["with_transfers_ms"] = copy["h2d_ms"] + median + copy["d2h_ms"]
            row["with_pinned_transfers_ms"] = copy["h2d_pinned_ms"] + median + copy["d2h_pinned_ms"]
        rows.append(row)
        print(f"n={n:5d} {name:14s} {median:10.4f} ms {row['gflops']:9.1f} GFLOP/s "
              f"rel_rms={metrics['rel_rms']:.2e} rounded={metrics['correctly_rounded']:.4f} "
              f"vs_numpy={metrics['max_abs_vs_numpy']:.2e}", flush=True)
    return rows, {"size": n, **copy, "bytes_in": 2 * a.nbytes, "bytes_out": a.nbytes}


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _error(value):
    return "0" if value == 0 else f"{value:.2e}"


def report(run, rows, copies, sizes, rounds):
    by = {(r["implementation"], r["size"]): r for r in rows}
    lecture = []
    for n in (n for n in LECTURE_SIZES if n in sizes and ("numpy", n) in by):
        base = by[("numpy", n)]["stable_mean_ms"]
        lecture += [[n, label, f"{by[(name, n)]['stable_mean_ms']:.3f}", f"{base / by[(name, n)]['stable_mean_ms']:.2f}×",
                     _error(by[(name, n)]["max_abs_vs_numpy"]), _error(by[(name, n)]["rel_vs_numpy"])]
                    for name, label in LECTURE_ROWS if (name, n) in by]
    if lecture:
        run.notes.append(
            "## 课件数据记录表（NumPy 为基线）\n\n"
            f"按课件第 72–73 页的口径：随机种子固定，每个规模测 {rounds} 轮（每轮依次测完全部实现再开始下一轮），"
            f"每轮取多次调用的平均，表中为 {rounds} 轮的中位数；下一节的中位耗时取全部调用的中位数。"
            "GPU 只计 kernel（CUDA event），传输时间单独见“传输”一节；"
            "加速比 = NumPy 时间 / 该实现时间。分块版的线程块负责 16×16 的输出子块，共享内存版用 16×16 的共享内存 tile。"
            "寄存器分块版是课件三版之外的优化：线程块负责 128×128 的输出块，每个线程在寄存器里算 8×8 个输出，"
            "float4 读取，共享内存双缓冲（fp32 每次取 16 层 k，即原生卷积默认的 GEMM 引擎）。"
            "四版都是 fp32 顺序累加，结果逐位相同。"
            "误差以 NumPy 的 float32 结果为基线：最大绝对误差为 max|C − C_NumPy|，"
            "相对误差为 ‖C − C_NumPy‖_F / ‖C_NumPy‖_F。\n\n"
            + _table(["规模", "实现", "平均执行时间 ms", "加速比", "最大绝对误差", "相对误差"], lecture))
    names = [name for name in IMPLEMENTATIONS if any(r["implementation"] == name for r in rows)]
    timing = [[name] + [f"{by[(name, n)]['median_ms']:.3f} ({by[(name, n)]['gflops']:.0f})"
                        if (name, n) in by else "-" for n in sizes] for name in names]
    run.notes.append("## 中位耗时 ms（括号内 GFLOP/s）\n\nGPU 为 CUDA event 计时的单次 kernel（不含传输），"
                     "NumPy 为 perf_counter。\n\n" + _table(["实现"] + [str(n) for n in sizes], timing))
    largest = max(sizes)
    focus = [n for n in (8, 1024, largest) if n in sizes]
    accuracy_rows = []
    for name in names:
        cells = [name]
        for n in focus:
            r = by.get((name, n))
            cells.append("-" if r is None else f"{r['rel_rms']:.2e} / {100 * r['correctly_rounded']:.2f}% / "
                                                f"{r['max_scaled']:.1f} / {r['max_abs_vs_numpy']:.1e}")
        accuracy_rows.append(cells)
    run.notes.append(
        "## 精度（相对 FP64 乘积）\n\n每格：相对 RMS 误差 / 正确舍入比例 / 最大误差（以 u·(|A||B|)ij 为单位，"
        "顺序累加的经典上界约为 K）/ 与 NumPy float32 结果的最大绝对差。\n\n"
        + _table(["实现"] + [f"n={n}" for n in focus], accuracy_rows))
    copy_rows = [[c["size"], f"{c['h2d_ms']:.3f}", f"{c['d2h_ms']:.3f}", f"{c['h2d_pinned_ms']:.3f}",
                  f"{c['d2h_pinned_ms']:.3f}", f"{c['bytes_in'] / max(c['h2d_ms'], 1e-9) / 1e6:.1f}",
                  f"{c['bytes_in'] / max(c['h2d_pinned_ms'], 1e-9) / 1e6:.1f}"] for c in copies]
    run.notes.append("## 传输（中位数，ms）\n\n" + _table(
        ["n", "H2D A+B", "D2H C", "H2D 锁页", "D2H 锁页", "H2D GB/s", "锁页 H2D GB/s"], copy_rows))
    cross = []
    for name in ("cublas", "shared32-fp32", "shared32-kahan", "register-fp32", "register-kahan"):
        winner = next((n for n in sizes if (name, n) in by and ("numpy", n) in by
                       and by[(name, n)]["with_transfers_ms"] < by[("numpy", n)]["median_ms"]), None)
        cross.append([name, winner if winner is not None else "未出现"] +
                     [f"{by[(name, n)]['with_transfers_ms']:.3f} vs {by[('numpy', n)]['median_ms']:.3f}"
                      for n in sizes if (name, n) in by and ("numpy", n) in by])
    run.notes.append("## 含传输时 GPU 首次快于 NumPy 的规模\n\n每格：GPU（H2D + kernel + D2H）对 NumPy 的 ms。\n\n"
                     + _table(["实现", "交叉点 n"] + [str(n) for n in sizes], cross))


def run_ncu(args, run):
    tool = find_tool("ncu", args.ncu_exe)
    prefix = run.path / f"ncu-{args.ncu_size}"
    child = [sys.executable, "-X", "utf8", "-m", "benchmark.matmul_compare", "--profile-child",
             "--ncu-size", str(args.ncu_size), "--seed", str(args.seed)]
    command = [str(tool), "--config-file", "off", "--target-processes", "all",
               "--kernel-name", "regex:matmul_|gemm|xmma|cutlass", "--metrics", ",".join(NCU_METRICS),
               "--export", str(prefix), "--force-overwrite", *child]
    run.manifest["ncu_command"] = command
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace", timeout=1800, check=False)
    (run.path / "ncu.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"ncu exited {result.returncode}: {result.stdout[-1500:]}")
    dump = subprocess.run([str(tool), "--import", str(prefix.with_suffix(".ncu-rep")), "--page", "raw", "--csv",
                           "--print-units", "base"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          encoding="utf-8", errors="replace", timeout=600, check=False)
    (run.path / "ncu-metrics.csv").write_text(dump.stdout, encoding="utf-8")
    lines = [line for line in dump.stdout.splitlines() if line.startswith('"')]
    table = list(csv.reader(lines))
    header, rows = table[0], table[2:]  # second row holds the units
    order = [label for label, *_ in KERNELS]
    kernels = []
    for index, row in enumerate(rows):
        record = dict(zip(header, row))
        value = {m: float(record[m].replace(",", "")) for m in NCU_METRICS if record.get(m, "") not in ("", "n/a")}
        label = order[index] if index < len(order) else record.get("Kernel Name", "?")
        kernels.append({"implementation": label, "kernel": record.get("Kernel Name"), **value})
    run.results["ncu"] = kernels
    n = args.ncu_size
    rows_md = []
    for k in kernels:
        requests, ideal = traffic(k["implementation"], n)
        dram = k.get("dram__bytes_read.sum", 0.0) + k.get("dram__bytes_write.sum", 0.0)
        rows_md.append([k["implementation"], f"{k.get('gpu__time_duration.sum', 0) / 1e6:.3f}",
                        f"{dram / 1e6:.1f}", f"{ideal / 1e6:.1f}",
                        f"{k.get('l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum', 0) / 1e6:.0f}",
                        "-" if requests is None else f"{requests / 1e6:.0f}",
                        f"{k.get('lts__t_bytes.sum', 0) / 1e6:.0f}",
                        f"{k.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', 0):.1f}",
                        f"{k.get('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 0):.1f}",
                        f"{k.get('sm__warps_active.avg.pct_of_peak_sustained_active', 0):.1f}",
                        f"{k.get('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum', 0):.0f}"])
    run.notes.append(
        f"## Nsight Compute（n={n}，ncu 默认锁基础频率并在每次 replay 前清缓存）\n\n"
        "全局 load 为 L1 收到的全局 load 字节（l1tex__t_bytes_pipe_lsu_mem_global_op_ld）；解析值按 warp 级 32 B "
        "sector 计：basic/tiled32 为 5n³（每个 k 读 4 个 B sector、1 个广播的 A sector，服务 32 个输出），"
        "tiled16 为 4n³，shared 为 8n³/TILE，register（128×128 块、float4 读取）为 n³/16；必需 DRAM 为 3n²·4。\n\n" + _table(
            ["实现", "kernel ms", "DRAM MB", "必需 DRAM MB", "全局 load MB", "解析 load MB", "L2 MB",
             "SM 吞吐 %", "访存吞吐 %", "占用率 %", "共享内存 bank 冲突"], rows_md))


def profile_child(args):
    set_device("cuda")
    a, b = operands(args.ncu_size, args.seed)
    da, db = xp.asarray(a), xp.asarray(b)
    dc = xp.empty_like(da)
    native.info()  # loads the DLL and its kernels before the profiled launches
    for name, *_ in KERNELS:  # one launch each, in KERNELS order (cuBLAS first)
        make_call(name, da, db, dc, a, b)()
    xp.cuda.Device().synchronize()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir")
    parser.add_argument("--sizes", default="8,64,128,256,512,1024,2048,4096")
    parser.add_argument("--implementations", default=",".join(IMPLEMENTATIONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, default=0.6,
                        help="measurement seconds per implementation, size and round")
    parser.add_argument("--rounds", type=int, default=3, help="timing rounds per implementation and size")
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--basic-limit", type=int, default=4096, help="skip basic kernels above this size")
    parser.add_argument("--ncu", action="store_true", help="also collect Nsight Compute metrics")
    parser.add_argument("--ncu-size", type=int, default=2048)
    parser.add_argument("--ncu-exe", help="ncu executable; default: MF_NCU, PATH, then the newest install")
    parser.add_argument("--profile-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.profile_child:
        return profile_child(args)
    if not args.out_dir:
        parser.error("--out-dir is required")
    args.sizes = [int(s) for s in args.sizes.split(",") if s]
    args.implementations = [s for s in args.implementations.split(",") if s]
    unknown = sorted(set(args.implementations) - set(IMPLEMENTATIONS))
    if unknown:
        parser.error(f"unknown implementations: {unknown}")
    set_device("cuda")
    with RunArtifacts(args.out_dir, vars(args)) as run:
        run.manifest["native"] = native.info()
        try:
            import threadpoolctl

            run.manifest["cpu_blas"] = threadpoolctl.threadpool_info()
        except ImportError:
            pass
        rows, copies = [], []
        for n in args.sizes:
            size_rows, copy = run_size(n, args, run)
            rows += size_rows
            copies.append(copy)
        run.results.update(rows=rows, transfers=copies)
        report(run, rows, copies, args.sizes, args.rounds)
        if args.ncu:
            run_ncu(args, run)
        run.notes.append(
            "输入为 N(0,1) float32 方阵（固定种子），所有实现共用同一份数据；FP64 参考为同一 float32 输入的 FP64 乘积。"
            "GPU 计时为单次调用的 CUDA event 区间（包含 Python 发射开销对 GPU 的空闲影响），"
            f"中位耗时取 {args.rounds} 轮的全部调用；"
            "含传输时间 = 中位 H2D(A,B) + 中位 kernel + 中位 D2H(C)，均为单独测量后相加。")


if __name__ == "__main__":
    main()
