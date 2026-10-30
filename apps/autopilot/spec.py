"""CampaignSpec: defaults, validation and resolution into apps.train arguments (spec §10)."""

from __future__ import annotations

import copy
import itertools
import math
import re
import time
from pathlib import Path

from apps.console.presets import MAX_FREEZE_STAGES, SINGLE_ARGS, ValidationError, pretrained_path, validate_single

SCHEMA_VERSION = 3
TASKS = {
    "donkey_resnet18_single": {
        "label": "DonkeyCar ResNet18 转向回归",
        "metrics": {"val_angle_mae": "minimize", "val_loss": "minimize"},
    },
}
# Accuracy search: CandidateList/TPE sample these; everything else is fixed or decided by throughput.
ACCURACY_KEYS = ("lr", "batch")
# Decided by specialists from throughput, never by the accuracy ranking (spec §6.1).
THROUGHPUT_KEYS = ("num_workers", "conv_backend", "distributed", "workers", "transport")
# Regularisation joins the search at the TPE stage (E3).
TPE_KEYS = ("weight_decay", "dropout", "augment")
# Transfer learning (E2): searchable only after the capability check (``unavailable_keys``) — the campaign fixes
# ``pretrained`` and the converted ImageNet weights exist. 0 (nothing frozen) is apps.train's default.
TRANSFER_KEYS = ("freeze_stages",)
TRANSFER_DEFAULTS = {"freeze_stages": 0}
NO_PRETRAINED = "campaign 未设置 fixed.pretrained（冻结阶段只对 ImageNet 预训练权重有意义）"
# Static fallback for callers without a spec; ``unavailable_keys(spec, repo_root)`` is the real check.
UNAVAILABLE_KEYS = {"freeze_stages": NO_PRETRAINED}
SPECIALISTS = {
    "gpu_monitor": "GPU Monitor",
    "benchmark": "Benchmark",
    "operator_opt": "Operator Optimization",
    "ps_training": "PS Training",
    "transfer": "Transfer Learning",
}
STRATEGIES = ("candidate_list", "tpe")
MODES = {"dev": "联调", "formal": "正式"}
MAX_LLM_CALLS = 30
# Token cap per campaign, set from the first real acceptance (spec §13 layer 3, layer3-advisor-001 on 2026-10-26):
# largest background call 9,185 tokens × 30 calls × 1.5 headroom. An explicit null means no cap.
DEFAULT_MAX_TOTAL_TOKENS = 414_000
# Training defaults of every trial: no TensorBoard and checkpoints only at epoch ends (the best model is kept).
TRIAL_DEFAULTS = {"checkpoint_every": 0, "no_tensorboard": True}
# Campaign-level split settings: they shape the split file, they are not apps.train arguments.
SPLIT_KEYS = ("val_size", "split_seed", "test_size", "split_kind", "block_size")
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


class SpecError(ValueError):
    pass


def default_spec(mode: str = "dev") -> dict:
    seeds = [1, 2] if mode != "formal" else [1, 2, 3]
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": None,
        "title": "",
        "mode": mode,
        "task": "donkey_resnet18_single",
        "objective": {"metric": "val_angle_mae", "direction": "minimize"},
        "baseline": {"lr": 0.001, "batch": 2, "num_workers": 0, "seeds": seeds},
        # Formal campaigns: the whole data set, a time-block split and an independent test set (spec §13 GPU 5).
        "fixed": {"max_samples": 1000 if mode != "formal" else 0, "epochs": 3, "val_size": 100 if mode != "formal" else 1000,
                  "test_size": 0 if mode != "formal" else 1000, "split_kind": "random" if mode != "formal" else "time_block",
                  "block_size": 50, "split_seed": 42, "split_file": None,
                  "dtype": "float32", "device": "cuda", "conv_backend": "cuda_native_register",
                  "base_width": 64, "stem": "cifar"},
        "space": {"lr": [0.0001, 0.0003, 0.001], "batch": [1, 2, 4]},
        "strategy": {"name": "candidate_list", "max_candidates": 6, "seed": 0},
        "budget": {"max_trials": 12, "trial_timeout_s": 1800, "wall_time_s": 14400, "max_retries": 1,
                   "max_gpu_memory_mb": None},
        "online": {"divergence_windows": 3, "divergence_factor": 3.0, "window_steps": 10, "hang_timeout_s": 120,
                   "gpu_idle_pct": 35.0, "explosion_factor": 100.0},
        "confirm": {"top_k": 2, "seeds": None},
        "specialists": ["gpu_monitor", "benchmark"],
        "llm": {"enabled": True, "max_calls": MAX_LLM_CALLS, "max_total_tokens": DEFAULT_MAX_TOTAL_TOKENS},
        # E3: exact rollback from TrainingStates (saved every state_every steps) on NaN/divergence; OOM → batch/2
        "rollback": {"enabled": mode == "formal", "max_rollbacks": 2, "lr_factor": 0.5, "state_every": 100,
                     "oom_half_batch": True},
        "created_by": "user",
    }


def _merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _number(value, name: str, *, kind=float, minimum=None, maximum=None, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise SpecError(f"{name} 必须是数值")
    try:
        out = kind(value)
    except (TypeError, ValueError, OverflowError):
        raise SpecError(f"{name} 必须是{'整数' if kind is int else '数值'}：{value!r}") from None
    if not math.isfinite(out):
        raise SpecError(f"{name} 必须是有限数值：{value!r}")
    if kind is int and float(value) != out:
        raise SpecError(f"{name} 必须是整数：{value!r}")
    if minimum is not None and out < minimum:
        raise SpecError(f"{name} 不能小于 {minimum}")
    if maximum is not None and out > maximum:
        raise SpecError(f"{name} 不能大于 {maximum}")
    return out


def _cast_space_value(key: str, value):
    caster = SINGLE_ARGS[key]
    if caster is bool:
        if not isinstance(value, bool):
            raise SpecError(f"搜索空间 {key} 的取值必须是 true/false")
        return value
    return _number(value, f"搜索空间 {key}", kind=caster)


def transfer_status(fixed: dict | None, repo_root: Path) -> tuple[bool, str | None]:
    """Is transfer learning usable with these fixed settings: ``pretrained`` set and its weights converted."""
    fixed = fixed or {}
    if not fixed.get("pretrained"):
        return False, NO_PRETRAINED
    path = pretrained_path(fixed.get("arch", "resnet18"), fixed["pretrained"], repo_root)
    if not path.is_file():
        return False, f"预训练权重不存在：{path}（先运行 tools/convert_torchvision_weights.py）"
    return True, None


def unavailable_keys(spec: dict, repo_root: Path) -> dict:
    """Search keys this campaign cannot use, with the reason (spec §6.1: capability check)."""
    ok, why = transfer_status(spec.get("fixed"), repo_root)
    return {} if ok else {key: why for key in TRANSFER_KEYS}


def new_campaign_id(task: str) -> str:
    short = {"donkey_resnet18_single": "donkey-r18"}.get(task, task.replace("_", "-"))
    return f"{short}-{time.strftime('%Y%m%d-%H%M%S')}"


def validate_spec(raw: dict, repo_root: Path) -> dict:
    """Merge ``raw`` onto the defaults of its mode and check every field; returns the full spec."""
    if not isinstance(raw, dict):
        raise SpecError("campaign 配置必须是 JSON 对象")
    mode = raw.get("mode", "dev")
    if mode not in MODES:
        raise SpecError("mode 只能是 dev 或 formal")
    unknown = set(raw) - set(default_spec())
    if unknown:
        raise SpecError(f"不支持的字段：{', '.join(sorted(unknown))}")
    spec = _merge(default_spec(mode), raw)
    spec["schema_version"] = SCHEMA_VERSION

    if spec["campaign_id"] is None:
        spec["campaign_id"] = new_campaign_id(spec["task"])
    if not isinstance(spec["campaign_id"], str) or not _ID.match(spec["campaign_id"]):
        raise SpecError("campaign_id 只能由小写字母、数字、- 和 _ 组成，长度 3～64")
    if spec["task"] not in TASKS:
        raise SpecError(f"未知任务：{spec['task']}")
    metric, direction = spec["objective"].get("metric"), spec["objective"].get("direction")
    if metric not in TASKS[spec["task"]]["metrics"]:
        raise SpecError(f"任务 {spec['task']} 不支持目标指标 {metric}")
    if direction not in ("minimize", "maximize"):
        raise SpecError("objective.direction 只能是 minimize 或 maximize")

    base = spec["baseline"]
    base["lr"] = _number(base.get("lr"), "baseline.lr", minimum=1e-12)
    base["batch"] = _number(base.get("batch"), "baseline.batch", kind=int, minimum=1)
    base["num_workers"] = _number(base.get("num_workers", 0), "baseline.num_workers", kind=int, minimum=0)
    seeds = base.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise SpecError("baseline.seeds 必须是非空列表")
    base["seeds"] = [_number(s, "baseline.seeds", kind=int, minimum=0) for s in seeds]
    if len(set(base["seeds"])) != len(base["seeds"]):
        raise SpecError("baseline.seeds 不能重复")

    fixed = spec["fixed"]
    for key in list(fixed):
        if key in ("split_file",) + SPLIT_KEYS:
            continue
        if key not in SINGLE_ARGS:
            raise SpecError(f"fixed 中不支持的训练参数：{key}")
        if key in ACCURACY_KEYS or key == "seed" or key in base:
            raise SpecError(f"fixed 不能包含 {key}（由基线、搜索空间或 seed 决定）")
    fixed["max_samples"] = _number(fixed.get("max_samples", 0), "fixed.max_samples", kind=int, minimum=0)
    fixed["epochs"] = _number(fixed.get("epochs", 1), "fixed.epochs", kind=int, minimum=1)
    fixed["val_size"] = _number(fixed.get("val_size", 0), "fixed.val_size", kind=int, minimum=1)
    fixed["split_seed"] = _number(fixed.get("split_seed", 42), "fixed.split_seed", kind=int, minimum=0)
    fixed["test_size"] = _number(fixed.get("test_size", 0), "fixed.test_size", kind=int, minimum=0)
    fixed["block_size"] = _number(fixed.get("block_size", 50), "fixed.block_size", kind=int, minimum=1)
    if fixed.get("split_kind", "random") not in ("random", "time_block"):
        raise SpecError("fixed.split_kind 只能是 random 或 time_block")
    fixed["split_kind"] = fixed.get("split_kind", "random")
    if mode == "formal":
        # Spec §13, GPU acceptance 5: a formal campaign has 3 seeds, a time-block split and a held-out test set.
        if fixed["split_kind"] != "time_block" or fixed["test_size"] < 1:
            raise SpecError("正式 campaign 必须按时间段划分（split_kind=time_block）并留出测试集（test_size ≥ 1）")
        if len(base["seeds"]) < 3:
            raise SpecError("正式 campaign 至少需要 3 个基线 seed")

    space = spec["space"]
    if not isinstance(space, dict) or not space:
        raise SpecError("space 必须是非空对象")
    strategy = spec["strategy"]
    if strategy.get("name") not in STRATEGIES:
        raise SpecError(f"strategy.name 只能是 {', '.join(STRATEGIES)}")
    allowed = set(ACCURACY_KEYS) | (set(TPE_KEYS) if strategy["name"] == "tpe" else set()) | set(TRANSFER_KEYS)
    unavailable = unavailable_keys(spec, repo_root)
    for key, values in space.items():
        if key in unavailable:
            raise SpecError(f"搜索空间不能包含 {key}：{unavailable[key]}")
        if key in fixed:
            raise SpecError(f"{key} 不能同时出现在 fixed 与搜索空间中")
        if key not in allowed:
            extra = "（正则项只在 TPE 策略中搜索）" if key in TPE_KEYS else ""
            raise SpecError(f"搜索空间不支持 {key}{extra}；num_workers 等由吞吐量决定")
        if not isinstance(values, list) or not values:
            raise SpecError(f"搜索空间 {key} 必须是非空列表")
        space[key] = sorted({_cast_space_value(key, v) for v in values})
    for key in ACCURACY_KEYS:
        if key not in space:
            raise SpecError(f"搜索空间必须包含 {key}")
    if any(v <= 0 for v in space["lr"]) or any(v < 1 for v in space["batch"]):
        raise SpecError("搜索空间中的 lr 必须为正数，batch 必须为正整数")
    if any(not 0 <= v <= MAX_FREEZE_STAGES for v in space.get("freeze_stages", [])):
        raise SpecError(f"搜索空间中的 freeze_stages 必须在 0 到 {MAX_FREEZE_STAGES} 之间")
    grid = 1
    for key in (*ACCURACY_KEYS, *TRANSFER_KEYS):
        grid *= len(space.get(key, [None]))
    if "max_candidates" not in (raw.get("strategy") or {}):  # the default shrinks to a small grid
        strategy["max_candidates"] = min(int(strategy["max_candidates"]), max(1, grid))
    strategy["max_candidates"] = _number(strategy.get("max_candidates", 6), "strategy.max_candidates", kind=int,
                                         minimum=1, maximum=max(1, grid))
    strategy["seed"] = _number(strategy.get("seed", 0), "strategy.seed", kind=int, minimum=0)

    budget = spec["budget"]
    budget["max_trials"] = _number(budget.get("max_trials"), "budget.max_trials", kind=int, minimum=1, maximum=500)
    budget["trial_timeout_s"] = _number(budget.get("trial_timeout_s"), "budget.trial_timeout_s", minimum=1)
    budget["wall_time_s"] = _number(budget.get("wall_time_s"), "budget.wall_time_s", minimum=1)
    budget["max_retries"] = _number(budget.get("max_retries", 1), "budget.max_retries", kind=int, minimum=0,
                                    maximum=3)
    budget["max_gpu_memory_mb"] = _number(budget.get("max_gpu_memory_mb"), "budget.max_gpu_memory_mb",
                                          minimum=1, allow_none=True)

    online = spec["online"]
    online["divergence_windows"] = _number(online.get("divergence_windows"), "online.divergence_windows", kind=int,
                                           minimum=1)
    online["divergence_factor"] = _number(online.get("divergence_factor"), "online.divergence_factor", minimum=1.0)
    online["window_steps"] = _number(online.get("window_steps", 10), "online.window_steps", kind=int, minimum=1)
    online["hang_timeout_s"] = _number(online.get("hang_timeout_s"), "online.hang_timeout_s", minimum=1)
    online["gpu_idle_pct"] = _number(online.get("gpu_idle_pct", 35.0), "online.gpu_idle_pct", minimum=0,
                                     maximum=100)
    # Loss explosion (a divergence without a rising trend): EMA above this multiple of the first loss for
    # divergence_windows windows; 0 turns the check off.
    online["explosion_factor"] = _number(online.get("explosion_factor", 100.0), "online.explosion_factor", minimum=0)
    if 0 < online["explosion_factor"] < 2:
        raise SpecError("online.explosion_factor 必须为 0（关闭）或不小于 2")

    confirm = spec["confirm"]
    confirm["top_k"] = _number(confirm.get("top_k", 2), "confirm.top_k", kind=int, minimum=0, maximum=2)
    if confirm.get("seeds") is not None:
        if not isinstance(confirm["seeds"], list) or not confirm["seeds"]:
            raise SpecError("confirm.seeds 必须是非空列表或 null")
        confirm["seeds"] = [_number(s, "confirm.seeds", kind=int, minimum=0) for s in confirm["seeds"]]

    specialists = spec["specialists"]
    if not isinstance(specialists, list) or set(specialists) - set(SPECIALISTS):
        raise SpecError(f"specialists 只能取 {', '.join(SPECIALISTS)}")
    spec["specialists"] = list(dict.fromkeys(specialists))

    llm = spec["llm"]
    llm["enabled"] = bool(llm.get("enabled"))
    llm["max_calls"] = _number(llm.get("max_calls", MAX_LLM_CALLS), "llm.max_calls", kind=int, minimum=0,
                               maximum=MAX_LLM_CALLS)
    llm["max_total_tokens"] = _number(llm.get("max_total_tokens"), "llm.max_total_tokens", kind=int, minimum=1,
                                      allow_none=True)
    rollback = spec["rollback"]
    rollback["enabled"] = bool(rollback.get("enabled"))
    rollback["max_rollbacks"] = _number(rollback.get("max_rollbacks", 2), "rollback.max_rollbacks", kind=int,
                                        minimum=0, maximum=5)
    rollback["lr_factor"] = _number(rollback.get("lr_factor", 0.5), "rollback.lr_factor", minimum=0.01, maximum=0.99)
    rollback["state_every"] = _number(rollback.get("state_every", 100), "rollback.state_every", kind=int, minimum=1)
    rollback["oom_half_batch"] = bool(rollback.get("oom_half_batch", True))
    if spec["created_by"] not in ("user", "llm"):
        raise SpecError("created_by 只能是 user 或 llm")
    if not isinstance(spec.get("title") or "", str):
        raise SpecError("title 必须是字符串")

    # The baseline itself must be a valid training: device/backend/dtype combinations are checked here once.
    try:
        validate_single(resolve_args(spec, baseline_params(spec, base["seeds"][0]), check_split=False), repo_root)
    except ValidationError as exc:
        raise SpecError(f"基线配置无效：{exc}") from exc
    return spec


def baseline_params(spec: dict, seed: int) -> dict:
    base = spec["baseline"]
    return {"lr": base["lr"], "batch": base["batch"], "num_workers": base["num_workers"], "seed": int(seed)}


def resolve_args(spec: dict, params: dict, *, check_split: bool = True) -> dict:
    """apps.train arguments of one trial: fixed settings, trial parameters, then the trial defaults."""
    fixed = {k: v for k, v in spec["fixed"].items() if k not in SPLIT_KEYS and k != "split_file"}
    args = {**TRIAL_DEFAULTS, **fixed, **params}
    for key, default in TRANSFER_DEFAULTS.items():
        if args.get(key) == default:  # same run as leaving it out: keep fingerprints equal to the baseline's
            args.pop(key)
    if (spec.get("rollback") or {}).get("enabled") and args.get("distributed", "none") == "none":
        # After a divergence the healthy state lies a detection span (plus a trough window) back, and the run
        # goes on for a few steps until it is stopped: keep twice the span (the step-0 state is always kept).
        every = int(spec["rollback"]["state_every"])
        online = spec.get("online") or {}
        span = int(online.get("window_steps", 10)) * int(online.get("divergence_windows", 3))
        args.update(state_every=every, keep_states=max(2, -(-2 * span // every) + 2))
    split = spec["fixed"].get("split_file")
    if split:
        args["split_file"] = split
    elif check_split:
        raise SpecError("campaign 尚未生成固定的数据划分文件")
    else:
        args["val_size"] = spec["fixed"]["val_size"]
        args["split_seed"] = spec["fixed"]["split_seed"]
    return args


def grid(space: dict, keys=ACCURACY_KEYS) -> list[dict]:
    keys = [k for k in keys if k in space]
    return [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]


def in_space(spec: dict, key: str, value) -> bool:
    """Rule and LLM proposals may move inside the range of a searched key, not beyond it."""
    values = spec["space"].get(key)
    if values is None:
        return True
    if isinstance(values[0], bool):
        return value in values
    return min(values) <= value <= max(values)
