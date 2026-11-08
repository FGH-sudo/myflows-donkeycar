"""Compare completed runs only within the same architecture, data partition, budget and seed."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from ..files import read_json
from ..provenance import canonical, file_hash


def comparison_record(result, config, *, split_hash, code, source):
    samples, data = result.get("samples") or {}, result.get("data") or {}
    value = (result.get("metrics") or {}).get("val_angle_mae_final")
    if (not split_hash or not code or not data.get("data_hash") or not data.get("val_hash")
            or config.get("seed") is None or not config.get("batch") or not config.get("lr")
            or not config.get("epochs") or len(result.get("epochs") or []) != config["epochs"]
            or not samples.get("expected_trained") or samples.get("trained") != samples["expected_trained"]
            or not isinstance(value, (float, int)) or not math.isfinite(value)):
        return None
    pretrained = bool(config.get("pretrained"))
    arch = config.get("arch", "resnet18")
    tv = pretrained or arch != "resnet18" or config.get("compat") == "torchvision"
    norm = config.get("input_norm", "auto")
    norm = ("imagenet" if pretrained else "none") if norm == "auto" else norm
    controls = {k: config.get(k, default) for k, default in
                (("batch", None), ("epochs", None), ("lr", None), ("dtype", "float32"),
                 ("conv_backend", "auto"), ("dropout", 0.), ("weight_decay", 0.),
                 ("augment", False), ("mixup", False), ("cutmix", False), ("graph_opt", False),
                 ("initializer", "kaiming_normal"), ("distributed", "none"), ("workers", 1))}
    controls.update(arch=arch, compat="torchvision" if tv else "myflows", stem="imagenet" if tv else config.get("stem", "cifar"),
                    width=64 if tv else config.get("base_width", 64), image_size=config.get("image_size") or "120x160",
                    norm=norm, train=samples.get("train"), trained=samples["trained"], split_hash=split_hash,
                    data_hash=data["data_hash"], val_hash=data["val_hash"], code=code)
    transfer = result.get("transfer") or {}
    treatment = None
    if pretrained:
        report = transfer.get("load_report") or {}
        weights = report.get("npz_sha256")
        if not weights or transfer.get("frozen_unchanged") is False:
            return None
        treatment = (weights, config.get("freeze_stages", 0), config.get("backbone_lr_mult", 1.),
                     config.get("unfreeze_schedule"), config.get("stage_lr_mults"),
                     config.get("lora_rank", 0), config.get("lora_alpha"))
    return {"controls": controls, "treatment": treatment, "seed": config["seed"], "mae": value, "source": source}


def paired_estimates(records):
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in records:
        if record is not None:
            groups[canonical(record["controls"])][record["treatment"]][record["seed"]].append(record)
    estimates = []
    for key, treatments in groups.items():
        scratch = treatments.get(None, {})
        for treatment, runs in treatments.items():
            seeds = sorted(runs.keys() & scratch.keys())
            if treatment is None or len(seeds) < 3:
                continue
            from scipy.stats import t

            differences = [statistics.fmean(r["mae"] for r in runs[s]) -
                           statistics.fmean(r["mae"] for r in scratch[s]) for s in seeds]
            mean = statistics.fmean(differences)
            margin = float(t.ppf(.975, len(seeds)-1)) * statistics.stdev(differences) / math.sqrt(len(seeds))
            identifier = hashlib.sha256((key + canonical(treatment)).encode()).hexdigest()[:12]
            estimates.append({"id": identifier, "controls": json.loads(key), "treatment": treatment,
                              "seeds": seeds, "paired_differences": differences, "mean": mean,
                              "low": mean-margin, "high": mean+margin, "negative": mean-margin > 0.,
                              "sources": [r["source"] for s in seeds for r in scratch[s] + runs[s]]})
    return estimates


def matching_records(root, ev):
    split = (ev.state.get("provenance") or {}).get("split") or {}
    records = []
    for trial in ev.valid():
        args = trial.get("resolved_args") or {}
        run = Path(ev.summary(trial).get("run_dir") or "")
        raw = read_json(run/"result.json") if run.is_dir() and ev.summary(trial).get("run_dir") else None
        if raw and raw.get("status") == "succeeded":
            records.append(comparison_record(raw, {**args, **(raw.get("config") or {})}, split_hash=split.get("split_hash"),
                                            code=trial.get("code_hash"), source=f"{trial['trial_id']}/result.json"))
    archive = root / "docs/experiments/semester_2026_fall/transfer/donkey-001"
    if not split.get("split_hash") or not split.get("val_hash"):
        return records
    manifest = read_json(archive/"manifest.json") or {}
    code = (manifest.get("source") or {}).get("code_hash")
    fixed = ev.spec.get("fixed") or {}
    weights = {ev.summary(trial).get("transfer", {}).get("npz_sha256") for trial in ev.valid()
               if ev.summary(trial).get("transfer")}
    weights.discard(None)
    for phase in ("pilot", "formal"):
        rows = read_json(archive/phase/"summary.json") or []
        if not rows:
            continue
        first = rows[0]["params"]
        if (fixed.get("arch", "resnet18") != "resnet18" or first["epochs"] != fixed.get("epochs")
                or file_hash(root/first["split_file"]) != split["split_hash"]):
            continue
        for row in rows:
            path = (root/row["run"]/"result.json").resolve()
            if not path.is_relative_to(archive.resolve()):
                continue
            raw = read_json(path) or {}
            if raw.get("status") == "succeeded" and raw.get("data", {}).get("val_hash") == split["val_hash"]:
                record = comparison_record(raw, {**row["params"], **raw["config"]}, split_hash=split["split_hash"],
                                           code=code, source=path.relative_to(root.resolve()).as_posix())
                if record and (record["treatment"] is None or record["treatment"][0] in weights):
                    records.append(record)
    return records
