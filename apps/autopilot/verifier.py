"""Critic / Verifier (spec §6.5): pre-check before queueing, online rules while running, post-check after.

Rule proposals and LLM-initiated writes go through the same pre-check. Online clocks (timeout, hang) start
when the trial gets the GPU lock, so time spent queueing behind other training never counts.
"""

from __future__ import annotations

import bisect
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

from apps.common.gpu_lock import acquired_unix_s
from apps.console.presets import ValidationError, validate_single
from apps.console.readers import JsonlTail

from .spec import (ACCURACY_KEYS, THROUGHPUT_KEYS, TPE_KEYS, TRANSFER_KEYS, SpecError, in_space, unavailable_keys,
                   resolve_args)

# Failures caused by the configuration itself: never retried with the same configuration.
CONFIG_FAILURES = ("nan", "diverged", "oom", "timeout")
# Infrastructure failures: retried once (spec §6.2).
INFRA_FAILURES = ("hang", "crash", "error", "lock_timeout")
OOM_PATTERN = re.compile(r"OutOfMemoryError|out of memory|显存不足|cudaErrorMemoryAllocation", re.IGNORECASE)


def precheck(params: dict, *, spec: dict, repo_root: Path, knowledge, trials: list[dict], fingerprint: str | None,
             delta: dict | None = None, budget_state: dict | None = None, capabilities: dict | None = None) -> dict:
    """Is this trial worth queueing? Returns ``{"passed", "reasons", "warnings", "args"}``."""
    reasons, warnings = [], []
    allowed = set(ACCURACY_KEYS) | set(THROUGHPUT_KEYS) | set(TRANSFER_KEYS) | {"seed"}
    if spec["strategy"]["name"] == "tpe":
        allowed |= set(TPE_KEYS)
    unavailable = unavailable_keys(spec, repo_root)
    for key in (delta or params):
        if key in unavailable:
            reasons.append(f"能力不可用：{key}（{unavailable[key]}）")
        elif key not in allowed:
            reasons.append(f"参数 {key} 不在可调范围内（精度只搜 {'、'.join(ACCURACY_KEYS)}，"
                           f"吞吐参数 {'、'.join(THROUGHPUT_KEYS)}）")
    for key, value in (delta or {}).items():
        if key in spec["space"] and not in_space(spec, key, value):
            values = spec["space"][key]
            reasons.append(f"{key}={value} 超出搜索空间 [{min(values)}, {max(values)}]")
    for name, (ok, why) in (capabilities or {}).items():
        if not ok and name in (delta or {}):
            reasons.append(f"能力不可用：{name}（{why}）")
    args = None
    if not reasons:
        try:
            args = validate_single(resolve_args(spec, params), repo_root)
        except (ValidationError, SpecError) as exc:
            reasons.append(f"参数校验失败：{exc}")
    if args is not None:
        fact = knowledge.predicted_failure(spec["task"], args) if knowledge is not None else None
        if fact is not None:
            reasons.append(f"经验库预测会失败（{fact.get('outcome')}）：{fact.get('fact')}")
        if knowledge is not None:
            warnings += [f"经验库提示：{f.get('fact')}" for f in knowledge.warnings(spec["task"], args)]
    if fingerprint is not None:
        for trial in trials:
            if trial.get("fingerprint") != fingerprint:
                continue
            verdict = (trial.get("verdict") or {}).get("status")
            kind = (trial.get("verdict") or {}).get("kind")
            if trial.get("status") in ("queued", "running", "pending_confirmation"):
                reasons.append(f"重复的指纹：与排队中的 {trial['trial_id']} 相同")
            elif verdict in ("valid", "invalid") or (verdict == "failed" and kind in CONFIG_FAILURES):
                reasons.append(f"重复的指纹：{trial['trial_id']} 已用相同配置跑过（{verdict}"
                               f"{'·' + kind if kind else ''}）")
            if reasons and reasons[-1].startswith("重复"):
                break
    if budget_state is not None:
        if budget_state["trials_committed"] >= spec["budget"]["max_trials"]:
            reasons.append(f"预算：trial 数已达上限 {spec['budget']['max_trials']}")
        if budget_state["run_s"] >= spec["budget"]["wall_time_s"]:
            reasons.append("预算：总时长已用完")
    return {"passed": not reasons, "reasons": reasons, "warnings": warnings, "args": args}


@dataclass
class OnlineStop:
    kind: str       # nan | diverged | oom | hang | timeout
    message: str
    unix_s: float
    step: int | None = None        # the step row that triggered the stop
    since_step: int | None = None  # first step of the anomaly: a rollback goes to a state saved before it

    def to_dict(self) -> dict:
        out = {"kind": self.kind, "message": self.message, "unix_s": self.unix_s}
        out.update({k: v for k, v in (("step", self.step), ("since_step", self.since_step)) if v is not None})
        return out


class OnlineMonitor:
    """Incremental reads of metrics.jsonl / resources.jsonl / train.log about once a second."""

    def __init__(self, run_dir: Path, online: dict, trial_timeout_s: float, *, log_path: Path | None = None,
                 clock=time.time):
        self.run_dir = Path(run_dir)
        self.online = online
        self.trial_timeout_s = float(trial_timeout_s)
        self.clock = clock
        self.metrics = JsonlTail(self.run_dir / "metrics.jsonl")
        self.resources = JsonlTail(self.run_dir / "resources.jsonl")
        self.log_path = Path(log_path) if log_path else self.run_dir / "train.log"
        self._log_offset = 0
        self.acquired: float | None = None
        self.last_progress: float | None = None
        self.ema: float | None = None
        self.window_emas: list[float] = []
        self.window_end_steps: list[int] = []
        self.steps_seen = 0
        self.first_loss: float | None = None
        self.first_step: int | None = None
        self.first_step_unix: float | None = None
        self.step_unix: list[float] = []  # recent step rows, for the GPU-idle rule
        self.findings: list[dict] = []
        self._idle_reported = False

    def _new_log_text(self) -> str:
        try:
            with self.log_path.open("rb") as f:
                f.seek(self._log_offset)
                data = f.read()
        except OSError:
            return ""
        self._log_offset += len(data)
        return data.decode("utf-8", errors="replace")

    def check(self) -> OnlineStop | None:
        now = self.clock()
        if self.acquired is None:
            self.acquired = acquired_unix_s(self.run_dir)
            if self.acquired is None:
                return None  # still waiting for the GPU: no hang or timeout clock yet
            self.last_progress = self.acquired
        rows = self.metrics.poll()
        if rows:
            self.last_progress = now
        window = int(self.online.get("window_steps", 10))
        for row in rows:
            if row.get("kind") != "step":
                continue
            loss, step = row.get("loss"), row.get("step")
            step = step if isinstance(step, int) else None
            if loss is None or not isinstance(loss, (int, float)) or not math.isfinite(loss):
                return OnlineStop("nan", f"第 {row.get('step')} 步 loss 为 {loss}", now, step=step, since_step=step)
            self.step_unix.append(float(row.get("unix_s") or now))
            if self.first_step_unix is None:
                self.first_step_unix = self.step_unix[-1]
                self.first_loss, self.first_step = float(loss), step
            alpha = 2.0 / (window + 1.0)
            self.ema = float(loss) if self.ema is None else (1.0 - alpha) * self.ema + alpha * float(loss)
            self.steps_seen += 1
            if self.steps_seen % window == 0:
                self.window_emas.append(self.ema)
                self.window_end_steps.append(step if step is not None else self.steps_seen)
                stop = self._diverged(now)
                if stop is not None:
                    return stop
        if OOM_PATTERN.search(self._new_log_text()):
            return OnlineStop("oom", "训练日志出现显存不足错误", now)
        if now - self.last_progress > float(self.online["hang_timeout_s"]):
            return OnlineStop("hang", f"获得 GPU 锁后 {now - self.last_progress:.0f} s 没有新的 step 行"
                                      f"（阈值 {self.online['hang_timeout_s']} s）", now)
        if now - self.acquired > self.trial_timeout_s:
            return OnlineStop("timeout", f"获得 GPU 锁后已运行 {now - self.acquired:.0f} s，"
                                         f"超过单 trial 上限 {self.trial_timeout_s:.0f} s", now)
        self._gpu_idle(now)
        return None

    def _since(self, index: int) -> int | None:
        """First step after window ``index - 1`` ended (the run's first step for index 0)."""
        if index <= 0:
            return self.first_step
        return self.window_end_steps[index - 1] + 1

    def _diverged(self, now: float) -> OnlineStop | None:
        k = int(self.online["divergence_windows"])
        emas = self.window_emas
        step = self.window_end_steps[-1] if self.window_end_steps else None
        if len(emas) >= k + 1:
            recent = emas[-(k + 1):]
            rising = all(b > a for a, b in zip(recent, recent[1:]))
            initial = emas[0]
            factor = float(self.online["divergence_factor"])
            if rising and initial > 0 and recent[-1] > factor * initial:
                return OnlineStop("diverged", f"loss 的 EMA 连续 {k} 个窗口上升，{recent[-1]:.4g} 超过初始值 "
                                              f"{initial:.4g} 的 {factor:g} 倍", now, step=step,
                                  since_step=self._since(len(emas) - k))
        # Explosion: with Adam a too-large lr blows the loss up on the first updates and it stays orders of
        # magnitude above the start without a rising trend, which the rule above never sees.
        explode = float(self.online.get("explosion_factor", 100.0) or 0.0)
        if explode > 0 and self.first_loss and self.first_loss > 0 and len(emas) >= k:
            limit = explode * self.first_loss
            if all(e > limit for e in emas[-k:]):
                first = len(emas) - k
                while first > 0 and emas[first - 1] > limit:
                    first -= 1
                return OnlineStop("diverged", f"loss 爆炸：EMA 连续 {k} 个窗口都超过首步 loss {self.first_loss:.4g} 的 "
                                              f"{explode:g} 倍（当前 {emas[-1]:.4g}）", now, step=step,
                                  since_step=self._since(first))
        return None

    def _gpu_idle(self, now: float) -> None:
        self.resources.poll()
        if self._idle_reported or self.steps_seen < 5:
            return
        # Only samples taken while steps run back to back: model build and kernel compilation before the first
        # step, and the validation pass and checkpointing at an epoch boundary, idle the GPU without the training
        # loop being starved. A sample counts when the steps around it are at most `gap` apart (slow, data-bound
        # steps still count: the gap scales with the typical step interval).
        del self.step_unix[:-4096]
        times = self.step_unix
        recent = times[-51:]
        intervals = sorted(b - a for a, b in zip(recent, recent[1:]))
        if not intervals:
            return
        gap = max(2.0, 5.0 * intervals[len(intervals) // 2])

        def between_steps(t: float) -> bool:
            i = bisect.bisect_right(times, t)
            return 0 < i < len(times) and times[i] - times[i - 1] <= gap

        rows = [r for r in self.resources.rows[-400:] if r.get("unix_s") and between_steps(float(r["unix_s"]))]
        values = [r.get("gpu_utilization_pct") for r in rows[-20:] if r.get("gpu_utilization_pct") is not None]
        if len(values) < 10:
            return
        values.sort()
        median = values[len(values) // 2]
        threshold = float(self.online.get("gpu_idle_pct", 35.0))
        if median < threshold and now - times[-1] < 5.0:
            self._idle_reported = True
            self.findings.append({"kind": "gpu_idle", "message": f"step 仍在推进，但 GPU 利用率中位数只有 "
                                  f"{median:.0f}%（阈值 {threshold:.0f}%）", "gpu_util_median": median,
                                  "unix_s": now})


# Findings the online monitor records while a trial runs (stops are verdicts, not findings).
ONLINE_FINDING_KINDS = ("gpu_idle",)


class _RecordedTail:
    """Stand-in for JsonlTail that releases a finished run's rows up to the replay clock."""

    def __init__(self, path: Path, clock):
        from .files import read_jsonl

        self.rows: list[dict] = []
        self.recorded = read_jsonl(path)
        self._clock = clock
        self._next = 0

    def poll(self) -> list[dict]:
        now, new = self._clock(), []
        while self._next < len(self.recorded) and (self.recorded[self._next].get("unix_s") or 0) <= now:
            new.append(self.recorded[self._next])
            self._next += 1
        self.rows.extend(new)
        return new


def replay_online_findings(run_dir: Path, online: dict, trial_timeout_s: float) -> list[dict]:
    """The findings the current online rules raise on a finished attempt: its recorded metrics and GPU samples
    are fed to an OnlineMonitor in time order. A regenerated report uses it to re-check the findings recorded
    while the trial ran."""
    run_dir = Path(run_dir)
    clock = [0.0]
    monitor = OnlineMonitor(run_dir, online, trial_timeout_s, log_path=run_dir / "__replay_without_log__",
                            clock=lambda: clock[0])
    monitor.metrics = _RecordedTail(run_dir / "metrics.jsonl", monitor.clock)
    monitor.resources = _RecordedTail(run_dir / "resources.jsonl", monitor.clock)
    def times(tail):
        return [float(r["unix_s"]) for r in tail.recorded if isinstance(r.get("unix_s"), (int, float))]

    # One check per GPU sample, as the live monitor polls about once a second (steps are far denser).
    ticks = sorted(set(times(monitor.resources) or times(monitor.metrics)) | set(times(monitor.metrics)[-1:]))
    if not ticks:
        return []
    monitor.acquired = acquired_unix_s(run_dir) or ticks[0]
    monitor.last_progress = monitor.acquired
    for tick in ticks:
        clock[0] = tick
        monitor.check()  # a replayed stop is already the attempt's verdict; only findings are collected
    return monitor.findings


def postcheck(trial: dict, attempt: dict, result: dict | None, *, spec: dict, split: dict) -> dict:
    """Verdict of a finished attempt: only ``valid`` trials are ranked."""
    status = attempt.get("status")
    failure = attempt.get("failure")
    if status == "cancelled":
        return {"status": "cancelled", "kind": None, "reasons": [attempt.get("note") or "campaign 停止或用户取消"]}
    if failure:
        return {"status": "failed", "kind": failure.get("kind"), "reasons": [failure.get("message") or ""]}
    result = result or {}
    reasons = []
    if result.get("status") != "succeeded":
        reasons.append(f"训练状态为 {result.get('status')}（被停止或未跑完）")
    metrics = result.get("metrics") or {}
    metric = spec["objective"]["metric"]
    for key in {metric, "val_loss"}:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            reasons.append(f"指标 {key} 不是有限值：{value}")
    data = result.get("data") or {}
    if split.get("val_hash") and data.get("val_hash") != split["val_hash"]:
        reasons.append("验证集指纹与本 campaign 的固定划分不一致")
    samples = result.get("samples") or {}
    if split.get("train_count") is not None and samples.get("train") != split["train_count"]:
        reasons.append(f"训练样本数 {samples.get('train')} 与划分 {split['train_count']} 不一致")
    if samples.get("expected_trained") is not None and samples.get("trained") != samples["expected_trained"]:
        reasons.append(f"实际训练样本 {samples.get('trained')} 与配置应训练的 {samples['expected_trained']} 不一致")
    requested = (trial.get("resolved_args") or {}).get("conv_backend", "auto")
    actual = (result.get("backend") or {}).get("actual")
    if requested != "auto" and actual != requested:
        reasons.append(f"实际卷积后端 {actual} 与请求的 {requested} 不一致")
    transfer = result.get("transfer") or {}
    if transfer.get("frozen_unchanged") is False:
        reasons.append("冻结阶段的参数或 BN buffer 在训练中被改动（冻结失效）")
    report = transfer.get("load_report") or {}
    if report and (report.get("missing") or report.get("shape_mismatch") or report.get("unexpected")):
        reasons.append("预训练权重没有完整载入（有缺失、多余或形状不符的张量）")
    limit = spec["budget"].get("max_gpu_memory_mb")
    peak = (result.get("memory") or {}).get("peak_pool_mb")
    if limit and peak is not None and peak > limit:
        reasons.append(f"显存峰值 {peak:.0f} MB 超出预算 {limit:.0f} MB")
    return {"status": "invalid" if reasons else "valid", "kind": None, "reasons": reasons}
