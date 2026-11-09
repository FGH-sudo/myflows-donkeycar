"""Run the Stage 3 transfer experiments sequentially from an identified training checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from apps.autopilot.provenance import code_state
from apps.common.donkey_data import load_donkey_index
from apps.common.splits import build_time_block_split, frame_numbers, save_split
from tools.check_run_format import check_run

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/experiments/semester_2026_fall/transfer"
WEIGHTS = ROOT / "runs/pretrained/converted/resnet18.npz"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def hash_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def flags(params):
    result = []
    for key, value in params.items():
        if value is None or value is False:
            continue
        result.append("--" + key.replace("_", "-"))
        if value is not True:
            result.append(str(value))
    return result


def execute(tree, command, log):
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(command, cwd=tree, stdout=handle, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=7200)
        except BaseException:
            from apps.console.jobs import kill_tree
            kill_tree(proc.pid)
            raise
    if code:
        raise RuntimeError(f"process exited {code}, see {log}")


def attempt(tree, source, folder, params, *, classification=False):
    recipe = {"source": source["code_hash"], "params": params, "classification": classification}
    folder.mkdir(parents=True, exist_ok=True)
    for previous in sorted(folder.glob("attempt-*")):
        if (previous / "recipe.json").exists() and read(previous / "recipe.json") != recipe:
            raise ValueError(f"study inputs changed for {folder}")
        if (previous / "verified.json").exists() and read(previous / "verified.json").get("valid"):
            return previous, read(previous / "result.json")
    run = folder / f"attempt-{len(list(folder.glob('attempt-*'))) + 1:02d}"
    run.mkdir()
    write(run / "recipe.json", recipe)
    if classification:
        command = [sys.executable, "-X", "utf8", "-m", "apps.train.transfer_classify", *flags(params),
                   "--no-require-loss-decrease", "--out-dir", str(run)]
    else:
        command = [sys.executable, "-X", "utf8", "-m", "apps.train", *flags(params), "--run-dir", str(run),
                   "--log-dir", str(run), "--out", str(run / "checkpoint"), "--best-out", str(run / "best"),
                   "--stop-file", str(run / "STOP")]
    write(run / "command.json", command)
    print(run.relative_to(ROOT).as_posix(), flush=True)
    execute(tree, command, run / "process.log")
    result = read(run / "result.json")
    verified = check_run(run)
    if result.get("status") != "succeeded" or not verified["valid"] or len(result["epochs"]) != params["epochs"]:
        raise RuntimeError(f"incomplete training run {run}, {verified['errors']}")
    write(run / "verified.json", verified)
    return run, result


def init_study(out, tree, kind):
    source = code_state(tree)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    if manifest.exists():
        previous = read(manifest)
        if previous["source"]["code_hash"] != source["code_hash"] or previous["kind"] != kind:
            raise ValueError("source or study differs from the recorded run")
    else:
        write(manifest, {"schema_version": 1, "source": source, "kind": kind, "weights_sha256": hash_file(WEIGHTS),
                         "started_unix_s": time.time(), "data": "mycar/data", "python": sys.version.split()[0]})
    return source


def group(rows, name):
    return [r for r in rows if r["method"] == name]


def stats(rows, key):
    values = [r[key] for r in rows]
    return f"{statistics.fmean(values):.5f} ± {statistics.stdev(values):.5f}" if len(values)>1 else f"{values[0]:.5f}"


def classification_study(out, tree):
    source = init_study(out, tree, "classification")
    cases = {
        "small": {"arch": "small"},
        "dropout": {"arch": "small", "dropout": .3},
        "augment-low-lr": {"arch": "small", "dropout": .3, "augment": True, "lr": .0001},
        "cached-extractor": {"freeze_stages": 5, "feature_cache": True},
        "augmented-extractor": {"freeze_stages": 5, "augment": True},
        "fine-tune-last-two": {"freeze_stages": 5, "unfreeze_schedule": "0:5,1:3", "backbone_lr_mult": .1,
                               "augment": True, "lr": .0001},
    }
    rows = []
    for dataset in ("donkey", "cifar10"):
        budget = {"max_samples": 3000, "val_size": 400, "test_size": 400} if dataset == "donkey" else {
                  "max_samples": 5000, "val_size": 1000, "test_size": 1000}
        common = {"arch": "resnet18", "dataset": dataset, "data": str(ROOT / "mycar/data"),
                  "cifar_root": str(ROOT / "runs/datasets/cifar10"), "weights": str(WEIGHTS), **budget,
                  "epochs": 2, "batch": 16, "image_size": 224, "num_workers": 0, "lr": .001, "device": "cuda"}
        for name, settings in cases.items():
            for seed in (1, 2):
                params = {**common, **settings, "seed": seed}
                run, result = attempt(tree, source, out/dataset/name/f"s{seed}", params, classification=True)
                last = result["epochs"][-1]
                rows.append({"dataset": dataset, "method": name, "seed": seed, "run": run.relative_to(ROOT).as_posix(),
                             "train_accuracy": last["train_accuracy"], "val_accuracy": last["val_accuracy"],
                             "gap": last["train_accuracy"]-last["val_accuracy"], "test_accuracy": result["test"]["accuracy"],
                             "epochs": result["epochs"]})
                write(out/"summary.json", rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 6, figsize=(17, 5.8), sharex=True, sharey=True, constrained_layout=True)
    lines = ["# 分类迁移的六档递进实验", "", "每档使用两个种子，输入 224×224，batch 16，训练 2 轮。"
             "道路数据按时间段划分，转向分档边界只由训练标签确定。CIFAR-10 的三个平衡子集全部来自官方训练档案，"
             "规模为 3,000/1,000/1,000。折线表示各轮训练与验证准确率的种子均值。", "", "![训练与验证曲线](curves.png)", "",
             "| 数据 | 方法 | 训练准确率 | 验证准确率 | 训练减验证 |", "| --- | --- | --- | --- | --- |"]
    for d, dataset in enumerate(("donkey","cifar10")):
        for i, name in enumerate(cases):
            selected = [r for r in rows if (r["dataset"],r["method"])==(dataset,name)]
            lines.append(f"| {dataset} | {name} | {stats(selected,'train_accuracy')} | {stats(selected,'val_accuracy')} | {stats(selected,'gap')} |")
            for field, style in (("train_accuracy","o-"),("val_accuracy","s--")):
                values = [statistics.mean(r["epochs"][e][field] for r in selected) for e in range(2)]
                axes[d,i].plot([1,2], values, style, label=field)
            axes[d,i].set(title=name if d==0 else "", xticks=[1,2], ylim=(0,1.02))
            axes[d,i].grid(alpha=.2)
        axes[d,0].set_ylabel(dataset + " accuracy")
    axes[-1,-1].legend(fontsize=7)
    fig.savefig(out/"curves.png",dpi=140)
    plt.close(fig)
    lines += ["", "small 从零训练，dropout 加入 0.3 随机失活，augment-low-lr 同时增加图像增强并将 lr 降至 0.0001。"
              "后面三档使用 ImageNet 权重。cached-extractor 只训练任务头，augmented-extractor 每轮重新增强图像。"
              "fine-tune-last-two 在第二轮释放末两段，任务头 lr=0.0001、骨干倍率 0.1。", "",
              "summary.json 和每个 attempt 的 config、metrics、result 与 checkpoint 位于本目录。复现命令为 "
              "`python -m benchmark.transfer_study classification --out runs/stage3/classify-repro`。"]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def split_for(phase, out):
    if phase == "formal":
        path = ROOT / "runs/autopilot/splits/donkey-all-val1000-test1000-tb50g5-s42.json"
        split = read(path)
        if [len(split["splits"][k]) for k in ("train","val","test")] != [7680,1000,1000]:
            raise ValueError("formal split differs from formal-full-002")
        return path
    path = out / "pilot-split.json"
    if not path.exists():
        rows = load_donkey_index(ROOT/"mycar/data", fixed_throttle=.5, angle_scale=1.)[:2000]
        splits = build_time_block_split(frame_numbers([str(r[0]) for r in rows]), val_size=200, test_size=200,
                                        block_size=50, gap=5, seed=42)
        save_split(path,splits,source_count=len(rows),kind="time_block",seed=42,meta={"block_size":50,"gap":5,"purged":splits["purged"]})
    return path


def donkey_phase(out, tree, source, phase):
    directory = out / phase
    directory.mkdir(exist_ok=True)
    path = split_for(phase,out)
    seeds = (1,2,3) if phase=="formal" else (1,2)
    common = {"device":"cuda","dtype":"float32","base_width":64,"batch":2,"lr":.001,"num_workers":0,
              "data":str(ROOT/"mycar/data"),"split_file":str(path), "max_samples":0 if phase=="formal" else 2000,
              "epochs":3 if phase=="formal" else 2,"conv_backend":"cuda_native_register","image_size":"120x160",
              "phase_timing_every":20,"no_tensorboard":True,"checkpoint_every":0,"initial_validation":True}
    cases = {"scratch-cifar":{"stem":"cifar"}, "scratch-imagenet":{"stem":"imagenet","compat":"torchvision","input_norm":"imagenet"}}
    for level in (0,2,3,4):
        cases[f"pretrained-k{level}"] = {"stem":"imagenet","pretrained":str(WEIGHTS),"freeze_stages":level,"input_norm":"imagenet"}
    rows = []
    def measure(name, settings, seed):
        params = {**common,**settings,"seed":seed}
        run, result = attempt(tree,source,directory/name/f"s{seed}",params)
        steps = [r for r in (json.loads(t) for t in (run/"metrics.jsonl").read_text(encoding="utf-8").splitlines()) if r["kind"]=="step"][3:]
        rows.append({"method":name,"seed":seed,"run":run.relative_to(ROOT).as_posix(),"params":params,
                     "initial":result["initial_validation"]["val_angle_mae"],"final":result["metrics"]["val_angle_mae_final"],
                     "best_checkpoint_val":result["metrics"]["val_angle_mae"],"epochs":result["epochs"],
                     "step_ms":statistics.median(r["step_total_ms"] for r in steps),
                     "samples_per_s":sum(r["batch_size"] for r in steps)*1000/sum(r["step_total_ms"] for r in steps),
                     "pool_mib":result["memory"]["peak_pool_mb"], "steps_per_epoch":read(run/"config.json")["steps_per_epoch"]})
        write(directory/"summary.json",rows)
    for seed in seeds:
        for name, settings in cases.items():
            measure(name,settings,seed)
    best = min((name for name in cases if name.startswith("pretrained-")),
               key=lambda name:statistics.mean(r["final"] for r in group(rows,name)))
    tuned = best+"-layer-lr"
    for seed in seeds:
        measure(tuned, {**cases[best],"backbone_lr_mult":.1},seed)
    eligible = [*list(cases)[2:],tuned]
    winner = min(eligible,key=lambda name:statistics.mean(r["final"] for r in group(rows,name)))
    chosen = min(group(rows,winner),key=lambda r:r["best_checkpoint_val"])
    locked = {"selection":"mean final validation MAE across paired seeds, then lowest validation MAE at best checkpoint",
              "method":winner,"seed":chosen["seed"],"run":chosen["run"],"split_sha256":hash_file(path)}
    write(directory/"locked.json",locked)
    evaluations = directory/"evaluation.json"
    if not evaluations.exists():
        items = [{"key":f"{r['method']}-s{r['seed']}", "args":r["params"],"checkpoint":str(ROOT/r["run"]/"checkpoint")} for r in rows]
        items.append({"key":"selected-transfer-best","args":chosen["params"],"checkpoint":str(ROOT/chosen["run"]/"best")})
        write(directory/"test-items.json",{"split":"test","items":items})
        execute(tree,[sys.executable,"-X","utf8","-m","apps.train.evaluate_donkey","--items",str(directory/"test-items.json"),
                      "--out",str(evaluations),"--run-dir",str(directory/"evaluation-run")],directory/"evaluation.log")
    evaluation = read(evaluations)
    if evaluation.get("failures"):
        raise RuntimeError(f"{phase}: incomplete test evaluations")
    test = {v["key"]:v["angle_mae"] for v in evaluation["results"]}
    targets = {r["seed"]:r["final"] for r in group(rows,"scratch-cifar")}
    for r in rows:
        r["test"] = test[f"{r['method']}-s{r['seed']}"]
        trace = [r["initial"]]+[e["val_angle_mae"] for e in r["epochs"]]
        r["baseline_target"] = targets[r["seed"]]
        r["epoch_to_target"] = next((n for n,v in enumerate(trace) if v<=targets[r["seed"]]),None)
        r["step_to_target"] = r["epoch_to_target"]*r["steps_per_epoch"] if r["epoch_to_target"] is not None else None
    write(directory/"summary.json",rows)
    write(directory/"selected-model.json",{"id":"transfer-"+winner,"path":chosen["run"]+"/best",
                                          "config":chosen["run"]+"/config.json","offline_angle_mae":test["selected-transfer-best"]})
    return rows


def best_epoch_rows(rows):
    """Best validation epoch per run, and the first epoch reaching the same-seed manual baseline's best value."""
    target = {r["seed"]: min(e["val_angle_mae"] for e in r["epochs"]) for r in group(rows, "scratch-cifar")}
    result = []
    for r in rows:
        trace = [e["val_angle_mae"] for e in r["epochs"]]
        best = min(trace)
        reach = next((n + 1 for n, value in enumerate(trace) if value <= target[r["seed"]]), None)
        result.append({"method": r["method"], "seed": r["seed"], "best": best, "best_epoch": trace.index(best) + 1,
                       "target": target[r["seed"]], "reach_epoch": reach,
                       "reach_step": reach * r["steps_per_epoch"] if reach is not None else None})
    return result


def donkey_best_epoch_section(rows):
    """Markdown comparing the runs at their best validation epoch (the last epoch degrades for most methods)."""
    best = best_epoch_rows(rows)
    methods = list(dict.fromkeys(r["method"] for r in best))
    last = statistics.fmean(r["final"] for r in group(rows, "scratch-cifar"))
    lines = ["## 按最佳验证轮的补充统计", "",
             f"学习率 1e-3、batch 2 时多数配置在第 3 轮验证误差回升，人工基线 scratch-cifar 末轮均值为 {last:.5f}。"
             "上表的“最终验证 MAE”与到达时间都以末轮为准，到达目标因此偏宽。下表取每个种子 3 轮内的最佳验证角度 MAE，"
             "到达时间以同 seed 人工基线的最佳值为目标，3 轮内未到达记为“未达”。训练保存的 best checkpoint 按验证 loss"
             "（转向与油门的 MSE）选择，其角度 MAE 可能与逐轮最小值不同。本节只用验证集，不增加测试集评估。", "",
             "| 配置 | 各 seed 最佳验证 MAE（轮） | 均值 ± 样本标准差 | 达到同 seed 基线最佳的 epoch / step |",
             "| --- | --- | --- | --- |"]
    for method in methods:
        g = [r for r in best if r["method"] == method]
        values = [r["best"] for r in g]
        reach = ", ".join(f"{r['reach_epoch']}/{r['reach_step']}" if r["reach_epoch"] is not None else "未达" for r in g)
        per_seed = ", ".join("{:.5f}（{}）".format(r["best"], r["best_epoch"]) for r in g)
        lines.append(f"| {method} | {per_seed} | "
                     f"{statistics.fmean(values):.5f} ± {statistics.stdev(values):.5f} | {reach} |")
    lines += ["", "同结构的 scratch-imagenet 与预训练配置按最佳验证 MAE 配对，差值取预训练减从零训练，"
              "95% 区间用 t 分布（自由度 2）。", "", "| 配置 | 配对差均值 | 95% 区间 | 负迁移判定 |", "| --- | --- | --- | --- |"]
    scratch = {r["seed"]: r["best"] for r in best if r["method"] == "scratch-imagenet"}
    for method in [m for m in methods if m.startswith("pretrained-")]:
        diff = [r["best"] - scratch[r["seed"]] for r in best if r["method"] == method]
        mean = statistics.fmean(diff)
        margin = 4.303 * statistics.stdev(diff) / math.sqrt(len(diff))
        lines.append(f"| {method} | {mean:+.5f} | [{mean - margin:+.5f}, {mean + margin:+.5f}] | "
                     f"{'负迁移' if mean - margin > 0 else '未检出'} |")
    return lines


def donkey_report(out,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    lines = ["# DonkeyCar 全量迁移对照","","先运行 2,000 帧、2 个种子的试验。全量阶段沿用 formal-full-002 的"
             "7,680/1,000/1,000 时间段划分，3 个共同种子、3 轮、batch 2。测试评估在按验证集锁定方法后进行。","",
             "| 配置 | 初始验证 MAE | 最终验证 MAE | 最终测试 MAE | 达到同 seed 基线的 epoch / step | 中位 step ms | samples/s | 池峰值 MiB |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for method in methods:
        g=group(rows,method)
        reach=", ".join(f"{r['epoch_to_target']}/{r['step_to_target']}" if r["epoch_to_target"] is not None else "未达" for r in g)
        lines.append(f"| {method} | {stats(g,'initial')} | {stats(g,'final')} | {stats(g,'test')} | {reach} | "
                     f"{statistics.median(r['step_ms'] for r in g):.2f} | {statistics.median(r['samples_per_s'] for r in g):.2f} | "
                     f"{statistics.median(r['pool_mib'] for r in g):.1f} |")
        for ax,key in zip(axes,("loss","val_angle_mae")):
            ax.plot(range(1,4),[statistics.mean(r["epochs"][i][key] for r in g) for i in range(3)],"o-",label=method)
    axes[0].set_ylabel("Training MSE"); axes[1].set_ylabel("Validation angle MAE")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.set_xticks([1,2,3]); ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.savefig(out/"curves.png",dpi=150); plt.close(fig)
    lines+=["","![训练与验证曲线](curves.png)","","下表比较同结构、同预算的 scratch-imagenet 与各预训练配置。"
            "差值取预训练减从零训练的最终验证 MAE。三组配对差使用 t 分布的 95% 区间，区间整体大于零时标记负迁移。","",
            "| 配置 | 配对差均值 | 95% 区间 | 负迁移判定 |","| --- | --- | --- | --- |"]
    baseline={r["seed"]:r["final"] for r in group(rows,"scratch-imagenet")}
    for method in methods[2:]:
        diff=[r["final"]-baseline[r["seed"]] for r in group(rows,method)]
        mean=statistics.mean(diff); margin=4.303*statistics.stdev(diff)/math.sqrt(3)
        lines.append(f"| {method} | {mean:+.5f} | [{mean-margin:+.5f}, {mean+margin:+.5f}] | {'负迁移' if mean-margin>0 else '未检出'} |")
    lines+=["",*donkey_best_epoch_section(rows)]
    locked=read(out/"formal/locked.json")
    lines+=["",f"按验证集锁定 {locked['method']}，闭环评估使用该配置 seed {locked['seed']} 的最佳验证 checkpoint。"
            "表中终值使用最后一轮 checkpoint。一次性测试清单包含部署候选的单独记录。","","原始结果分别在 pilot/ 与 formal/，每个 attempt 保存命令、完整 "
            "config、metrics、resources、result 和 checkpoint。summary.json 保留每个种子的训练曲线、初始 MAE、到达基线的步数和内存口径。",
            "复现命令为 `python -m benchmark.transfer_study donkey --tree <实验代码树> --out runs/stage3/donkey-repro`。"]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def lora_study(out, tree):
    source = init_study(out, tree, "lora")
    split = split_for("pilot", out)
    settings = {"device": "cuda", "dtype": "float32", "batch": 8, "epochs": 2, "seed": 1,
                "lr": .001, "data": str(ROOT / "mycar/data"), "max_samples": 2000, "split_file": str(split),
                "pretrained": str(WEIGHTS), "image_size": "120x160", "input_norm": "imagenet",
                "conv_backend": "cuda_native_register", "phase_timing_every": 20, "no_tensorboard": True,
                "checkpoint_every": 0, "initial_validation": True}
    cases = {"head": {"freeze_stages": 5}, **{f"lora-r{r}": {"lora_rank": r} for r in (4, 8, 16)},
             "freeze-3": {"freeze_stages": 3}, "full": {"freeze_stages": 0}}
    rows = []
    for name, config in cases.items():
        run, result = attempt(tree, source, out/name, {**settings, **config, "export_onnx": name == "lora-r4"})
        transfer = result["transfer"]
        if not transfer["frozen_unchanged"]:
            raise RuntimeError(f"frozen tensors changed in {name}")
        rows.append({"method": name, "trainable": transfer["freeze"]["trainable_values"],
                     "initial_mae": result["initial_validation"]["val_angle_mae"],
                     "final_mae": result["metrics"]["val_angle_mae_final"], "epochs": result["epochs"],
                     "run": run.relative_to(ROOT).as_posix(), "lora": transfer.get("lora")})
        write(out/"summary.json", rows)
    counts = read(split)["splits"]
    lines = ["# ResNet18 低秩微调对照", "",
             f"取前 2,000 帧按时间段切分，训练 {len(counts['train'])} 帧，验证 {len(counts['val'])} 帧。"
             "六种方式使用同一组 ImageNet 权重、任务头初值、样本顺序和预算。batch 8，学习率 0.001，"
             "训练 2 轮，seed 1。表中使用最后一轮验证集转向 MAE。", "",
             "| 配置 | 可训练参数 | 初始验证 MAE | 最终验证 MAE |", "| --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['trainable']:,} | {row['initial_mae']:.6f} | {row['final_mae']:.6f} |")
    lines.extend(["", "LoRA 固定原权重、偏置和 BN 统计量，更新 layer4 的五个卷积及回归头。"
                  "缩放系数等于 1。各组均逐字节校验冻结状态。单个种子的结果用于比较参数预算和此次验证表现。",
                  "", "复现命令为 `python -m benchmark.transfer_study lora --out runs/stage3/lora-repro`。"
                  "本目录 summary.json 和各 attempt 保存参数量、逐轮指标、训练命令与 checkpoint。"])
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind",choices=("classification","donkey","lora"))
    parser.add_argument("--tree",type=Path,default=ROOT)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args(argv)
    out=(ROOT/args.out).resolve() if args.out else ARCHIVE/{"classification":"classify-001","donkey":"donkey-001","lora":"lora-001"}[args.kind]
    if not out.is_relative_to(ROOT):
        parser.error("output must be inside the workspace")
    tree=args.tree.resolve()
    if args.kind=="classification":
        classification_study(out,tree)
    elif args.kind=="lora":
        lora_study(out,tree)
    else:
        source=init_study(out,tree,"donkey")
        donkey_phase(out,tree,source,"pilot")
        donkey_report(out,donkey_phase(out,tree,source,"formal"))
    if code_state(tree)["code_hash"] != read(out/"manifest.json")["source"]["code_hash"]:
        raise RuntimeError("training checkout changed during the study")


if __name__=="__main__":
    main()
