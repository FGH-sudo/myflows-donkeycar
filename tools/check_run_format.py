"""Validate v1 training artifacts; accept unversioned v0 archives without rewriting them."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MyFlows.monitoring.schema import SCHEMA_VERSION, schema_version

RESOURCE_NUMBERS = ("cpu_utilization_pct", "ram_used_bytes", "ram_total_bytes", "gpu_utilization_pct",
                    "gpu_memory_mib", "gpu_device_memory_mib", "gpu_device_memory_peak_mib",
                    "gpu_temperature_c", "gpu_power_w", "gpu_power_limit_w", "process_rss_total_bytes")
STEP_NUMBERS = ("loss", "data_load_ms", "train_step_ms", "step_time_ms", "batch_size", "samples_per_sec")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_record(row, kind: str) -> list[str]:
    errors = []
    if not isinstance(row, dict):
        return ["record must be an object"]
    version = schema_version(row)
    if type(version) is not int or version not in (0, SCHEMA_VERSION):
        return [f"unsupported schema_version: {version!r}"]
    if version == 0:
        return errors

    def require(key, typ):
        if not isinstance(row.get(key), typ):
            errors.append(f"{key}: expected {typ.__name__}")

    def number(key, *, nullable=False, nonnegative=True):
        value = row.get(key)
        if value is None and nullable:
            missing = row.get("missing")
            if not isinstance(missing, dict) or not isinstance(missing.get(key), str) or not missing[key]:
                errors.append(f"{key}: null requires a reason in missing")
        elif not _finite(value) or (nonnegative and value < 0):
            errors.append(f"{key}: expected a finite {'nonnegative ' if nonnegative else ''}number")

    if kind in ("metrics", "resources", "steps"):
        number("monotonic_s")
        number("unix_s")
    if kind == "config":
        # Task-specific configuration is additive and can be a single or distributed training run.
        if not row.get("kind") and not row.get("task"):
            errors.append("kind or task is required")
    elif kind == "metrics":
        if row.get("kind") == "step":
            number("step")
            for key in STEP_NUMBERS:
                number(key, nonnegative=key != "loss")
            for key in ("forward_ms", "backward_ms", "optimizer_ms", "step_total_ms", "samples_per_sec_total"):
                if key in row:
                    number(key)
            if "step_total_ms" in row and all(_finite(row.get(k)) for k in ("step_total_ms", "step_time_ms", "data_load_ms")):
                if not math.isclose(row["step_total_ms"], row["step_time_ms"] + row["data_load_ms"], rel_tol=1e-6):
                    errors.append("step_total_ms must equal data_load_ms + step_time_ms")
        elif row.get("kind") == "epoch":
            number("epoch")
            number("loss", nonnegative=False)
            number("learning_rate")
            number("epoch_time_s")
        else:
            errors.append("kind must be step or epoch")
    elif kind == "resources":
        for key in RESOURCE_NUMBERS:
            number(key, nullable=True)
        number("interval_s")
        if _finite(row.get("interval_s")) and row["interval_s"] <= 0:
            errors.append("interval_s must be positive")
        require("gpus", list)
        require("missing", dict)
        if row.get("sampling_status") not in ("ok", "partial", "unavailable", "error"):
            errors.append("sampling_status is missing or invalid")
        rss = row.get("process_rss_bytes")
        if rss is not None and (not isinstance(rss, dict) or any(not _finite(v) or v < 0 for v in rss.values())):
            errors.append("process_rss_bytes must map PIDs to nonnegative byte counts")
        for gpu in row.get("gpus", []) if isinstance(row.get("gpus"), list) else []:
            if not isinstance(gpu, dict):
                errors.append("gpus entries must be objects")
                continue
            for key in ("utilization_pct", "memory_used_mib", "memory_total_mib", "temperature_c", "power_w", "power_limit_w"):
                if gpu.get(key) is None and not (gpu.get("missing") or {}).get(key):
                    errors.append(f"gpus[].{key}: null requires a reason")
    elif kind == "steps":
        for key in ("rank", "epoch", "step", "start_s", "end_s", "step_wall_s", "data_s", "n_samples"):
            number(key)
        if _finite(row.get("end_s")) and _finite(row.get("start_s")) and row["end_s"] < row["start_s"]:
            errors.append("end_s precedes start_s")
    elif kind == "result":
        if row.get("status") not in ("succeeded", "stopped", "failed"):
            errors.append("invalid result status")
        number("exit_code")
        require("environment", dict)
        if isinstance(row.get("environment"), dict):
            for key in ("python", "platform", "packages", "gpu", "cuda", "native_source_sha256", "missing"):
                if key not in row["environment"]:
                    errors.append(f"environment.{key} is required")
    return errors


def _reject_constant(value):
    raise ValueError(f"non-JSON numeric constant {value}")


def check_run(directory: str | Path, *, allow_incomplete: bool = False) -> dict:
    directory = Path(directory)
    errors, warnings, files = [], [], {}
    result = None
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        config = {}
    legacy_config = isinstance(config, dict) and bool(config) and schema_version(config) == 0
    specs = [(directory / name, kind) for name, kind in (
        ("config.json", "config"), ("metrics.jsonl", "metrics"), ("resources.jsonl", "resources"), ("result.json", "result"))]
    specs += [(p, "steps") for p in sorted(directory.glob("rank-*/steps.jsonl"))]
    for path, kind in specs:
        if not path.is_file():
            continue
        rel = path.relative_to(directory).as_posix()
        versions, count = set(), 0
        try:
            with path.open(encoding="utf-8-sig") as handle:
                texts = enumerate(handle, 1) if path.suffix == ".jsonl" else [(1, handle.read())]
                for line, raw in texts:
                    if not raw.strip():
                        continue
                    try:
                        if path.suffix == ".jsonl" and not raw.endswith("\n"):
                            raise ValueError("incomplete JSONL line")
                        row = json.loads(raw, parse_constant=_reject_constant)
                    except ValueError as exc:
                        errors.append(f"{rel}:{line}: {exc}")
                        continue
                    count += 1
                    if isinstance(row, dict):
                        versions.add(str(schema_version(row)))
                        if kind == "result":
                            result = row
                    if kind == "result" and legacy_config and isinstance(row, dict) and schema_version(row) == 1 and "environment" not in row:
                        warnings.append(f"{rel}: historical result v1 envelope under v0 config, environment not available")
                    else:
                        errors.extend(f"{rel}:{line}: {error}" for error in validate_record(row, kind))
        except OSError as exc:
            errors.append(f"{rel}: {exc}")
        files[rel] = {"records": count, "versions": sorted(versions)}
        if "0" in versions:
            warnings.append(f"{rel}: legacy v0, field completeness is not enforced")
    if not files:
        errors.append("no run artifacts found")
    if not allow_incomplete:
        if result is None:
            errors.append("result.json is required for a completed run")
        elif result.get("status") in ("succeeded", "stopped"):
            for name in ("config.json", "metrics.jsonl"):
                if name not in files or not files[name]["records"]:
                    errors.append(f"{name} is missing or empty")
        config_path = directory / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if (config.get("monitoring") or {}).get("enabled") and not files.get("resources.jsonl", {}).get("records"):
                errors.append("monitoring is enabled but resources.jsonl is missing or empty")
        except (OSError, ValueError, AttributeError):
            pass
    return {"schema_version": SCHEMA_VERSION, "valid": not errors, "files": files, "errors": errors, "warnings": warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true", help="validate a running or legacy archive")
    args = parser.parse_args(argv)
    report = check_run(args.run_dir, allow_incomplete=args.allow_incomplete)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
