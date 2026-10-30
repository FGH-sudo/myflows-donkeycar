"""AutoPilotService: campaigns, their driver threads, pending confirmations and recovery (spec §5, §9).

One driver thread per running campaign runs the closed loop

    evidence → findings → proposals → pre-check → queue → dispatch → run (online checks) → verdict → record

Trials are submitted to the console's JobManager, so they share one serial queue with manual jobs; the
training entry's GPU lock also serialises runs started from a terminal. LLM-initiated training starts wait in
``state.pending`` until a person confirms them.
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

import psutil

from apps.console.presets import TRAIN_MODULE

from . import reporter, rollback
from .agents import build_specialists
from .agents.base import Finding
from .agents.benchmark import noise
from .data import ensure_split
from .evidence import Evidence, summarize_attempt
from .executor import Executor
from .files import read_json, write_json
from .memory import CampaignStore, KnowledgeBase
from .planner import Planner
from .provenance import code_state, environment, fingerprint
from .scheduler import Scheduler, budget_state
from .spec import SpecError, resolve_args, validate_spec
from .state import drop_states
from .verifier import (CONFIG_FAILURES, INFRA_FAILURES, ONLINE_FINDING_KINDS, postcheck, precheck,
                       replay_online_findings)

ACTIVE = ("running", "pausing", "stopping", "finishing")
EDITABLE = ("draft", "paused")
PLANNER_KEYS = ("phase", "phase_log", "proposed", "explore_proposed", "confirm_plan", "chains", "adopted",
                "stop_reason", "noise", "findings", "best", "budget_used", "tpe")
KEEP_TOP = 2
# Scores kept models on the independent test set when the campaign's split has one (formal campaigns).
LLM_EVENT_KINDS = ("llm_action", "llm_call", "llm_error", "llm_skip", "llm_disabled")
EVAL_MODULE = "apps.train.evaluate_donkey"
EVAL_TIMEOUT_S = 3600.0


def machine_boot_time() -> float:
    """When this machine last booted (Unix seconds)."""
    return psutil.boot_time()


class CampaignError(Exception):
    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


def _check_split(spec: dict, split: dict) -> None:
    if not split.get("train_count") or not split.get("val_count"):
        raise CampaignError("训练集与验证集不能为空", 422)
    if spec["mode"] == "formal" and (split.get("kind") != "time_block" or not split.get("test_count")):
        raise CampaignError("正式 campaign 的实际划分必须为 time_block，且包含独立测试集", 422)


def initial_state() -> dict:
    return {"status": "draft", "phase": "baseline", "created_unix_s": time.time(), "started_unix_s": None,
            "finished_unix_s": None, "stop_requested": False, "pause_requested": False, "stop_reason": None,
            "provenance": {}, "adopted": {}, "findings": {}, "chains": [], "proposed": [], "pending": [],
            "explore_proposed": False, "confirm_plan": None, "budget_used": {}, "best": None, "noise": None,
            "llm": {"session_id": None, "calls": 0, "tokens": 0, "failures": 0, "consecutive_failures": 0,
                    "disabled_reason": None, "decision_points": []},
            "notes": [], "phase_log": [], "unavailable": {}}


class AutoPilotService:
    def __init__(self, root: Path, *, jobs, repo_root: Path, python: str | None = None,
                 module: str = TRAIN_MODULE, poll_s: float = 1.0, advisor=None, record_environment: bool = True,
                 eval_module: str = EVAL_MODULE):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = jobs
        self.repo_root = Path(repo_root)
        self.python = python or sys.executable
        self.executor = Executor(jobs, python=self.python, module=module, poll_s=poll_s)
        self.poll_s = float(poll_s)
        self.knowledge = KnowledgeBase(self.root / "knowledge.jsonl")
        self.knowledge.ensure_seeded()
        self.advisor = advisor
        self.record_environment = record_environment
        self.eval_module = eval_module
        self._stores: dict[str, CampaignStore] = {}
        self._drivers: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
        self._closing = threading.Event()

    # ---- lifecycle --------------------------------------------------------------------------
    def start(self) -> AutoPilotService:
        """Resume the drivers of campaigns that were running when the console stopped.

        A console restart re-attaches to the trials that kept running. After a machine restart no trial survived
        and nobody may be expecting the GPU to be busy, so a campaign last written before the boot is paused
        instead of starting training on its own; the user continues it from the console.
        """
        boot = machine_boot_time()
        for cid in self.campaign_ids():
            store = self.store(cid)
            try:
                doc = store.load()
            except FileNotFoundError:
                continue
            status = doc["state"].get("status")
            if status not in ACTIVE:
                continue
            if status in ("running", "pausing") and (doc.get("updated_unix_s") or 0) < boot:
                store.update_state(lambda s: s.update(status="paused", pause_requested=False))
                store.event("campaign", status="paused", source="reboot",
                            note="机器重启后暂停：开机时不自动开始训练，在控制台点“继续”接着跑")
                continue
            store.event("campaign", status="recovered", note="控制台重启后恢复")
            self._spawn(cid)
        return self

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop the drivers; running trials keep running and are re-attached on the next start."""
        self._closing.set()
        for thread in list(self._drivers.values()):
            thread.join(timeout=timeout)

    def campaign_ids(self) -> list[str]:
        return sorted(p.parent.name for p in self.root.glob("*/campaign.json"))

    def store(self, cid: str) -> CampaignStore:
        with self._lock:
            if cid not in self._stores:
                self._stores[cid] = CampaignStore(self.root, cid)
            return self._stores[cid]

    def _existing(self, cid: str) -> CampaignStore:
        store = self.store(cid)
        if not store.exists():
            raise CampaignError(f"campaign 不存在：{cid}", 404)
        return store

    # ---- queries ----------------------------------------------------------------------------
    def list_campaigns(self) -> list[dict]:
        out = []
        for cid in self.campaign_ids():
            try:
                out.append(self._summary(self.store(cid)))
            except (FileNotFoundError, KeyError):
                continue
        return sorted(out, key=lambda c: c.get("created_unix_s") or 0, reverse=True)

    def _summary(self, store: CampaignStore) -> dict:
        doc = store.load()
        spec, state = doc["spec"], doc["state"]
        trials = store.trials()
        ev = self._evidence(store, doc, trials)
        best = ev.best()
        used = budget_state(trials)
        return {"campaign_id": spec["campaign_id"], "title": spec.get("title"), "task": spec["task"],
                "mode": spec["mode"], "status": state["status"], "phase": state["phase"],
                "created_by": spec["created_by"], "created_unix_s": state.get("created_unix_s"),
                "started_unix_s": state.get("started_unix_s"), "finished_unix_s": state.get("finished_unix_s"),
                "objective": spec["objective"], "budget": spec["budget"], "used": used,
                "trials": len(trials), "valid": len(ev.valid()),
                "best": {"trial_id": best["trial_id"], "value": ev.objective(best), "params": best["params"]}
                if best else None,
                "pending": sum(1 for p in state.get("pending") or [] if p["status"] == "pending_confirmation"),
                "llm": {"enabled": spec["llm"]["enabled"], "calls": state["llm"].get("calls", 0),
                        "disabled_reason": state["llm"].get("disabled_reason")},
                "stop_reason": state.get("stop_reason")}

    def get_campaign(self, cid: str) -> dict:
        store = self._existing(cid)
        doc = store.load()
        out = self._summary(store)
        out.update(spec=doc["spec"], state={k: v for k, v in doc["state"].items() if k not in ("proposed",)})
        report = store.report_dir / "report.md"
        out["report_available"] = report.exists()
        return out

    def trials(self, cid: str) -> list[dict]:
        store = self._existing(cid)
        doc = store.load()
        trials = store.trials()
        ev = self._evidence(store, doc, trials)
        rows = {r["trial_id"]: r for r in reporter.leaderboard(ev)}
        return [{**t, "rank": rows.get(t["trial_id"], {}).get("rank"), "objective": ev.objective(t)} for t in trials]

    def findings(self, cid: str) -> dict:
        doc = self._existing(cid).load()
        return {"findings": list((doc["state"].get("findings") or {}).values()),
                "chains": doc["state"].get("chains") or [], "noise": doc["state"].get("noise"),
                "unavailable": doc["state"].get("unavailable") or {}}

    def events(self, cid: str, since: int = 0, kinds: tuple[str, ...] | None = None) -> list[dict]:
        return self._existing(cid).events(since, kinds)

    def pending(self, cid: str) -> list[dict]:
        return list(self._existing(cid).load()["state"].get("pending") or [])

    def report(self, cid: str) -> dict:
        store = self._existing(cid)
        path = store.report_dir / "report.md"
        if not path.exists():
            raise CampaignError("报告尚未生成", 404)
        return {"markdown": path.read_text(encoding="utf-8"), "path": str(path),
                "figures": sorted(p.name for p in (store.report_dir / "figures").glob("*.png")),
                "confirmation": read_json(store.report_dir / "confirmation.json")}

    # ---- campaign writes ----------------------------------------------------------------------
    def create_campaign(self, raw: dict, *, created_by: str = "user", source: str = "console") -> dict:
        raw = dict(raw or {})
        raw["created_by"] = created_by
        try:
            spec = validate_spec(raw, self.repo_root)
        except SpecError as exc:
            raise CampaignError(str(exc), 422) from exc
        store = self.store(spec["campaign_id"])
        if store.exists():
            raise CampaignError(f"campaign 已存在：{spec['campaign_id']}")
        fixed = spec["fixed"]
        try:
            if fixed.get("split_file"):
                from .data import describe_split

                split = describe_split(self.repo_root, fixed["split_file"], max_samples=fixed["max_samples"])
            else:
                split = ensure_split(self.repo_root, self.root / "splits", max_samples=fixed["max_samples"],
                                     val_size=fixed["val_size"], split_seed=fixed["split_seed"],
                                     kind=fixed.get("split_kind", "random"), test_size=fixed.get("test_size", 0),
                                     block_size=fixed.get("block_size", 50))
                fixed["split_file"] = split["split_file"]
        except (OSError, ValueError) as exc:
            raise CampaignError(f"无法准备数据划分：{exc}", 422) from exc
        _check_split(spec, split)
        state = initial_state()
        state["provenance"]["split"] = split
        if not spec["llm"]["enabled"]:
            state["llm"]["disabled_reason"] = "本 campaign 未启用 LLM"
        doc = {"schema_version": spec["schema_version"], "spec": spec, "state": state}
        store.dir.mkdir(parents=True, exist_ok=True)
        store.save(doc)
        store.event("campaign", status="created", created_by=created_by, source=source,
                    spec_summary={"objective": spec["objective"], "space": spec["space"], "budget": spec["budget"]})
        return self.get_campaign(spec["campaign_id"])

    def update_campaign(self, cid: str, patch: dict, *, source: str = "console") -> dict:
        store = self._existing(cid)
        doc = store.load()
        if doc["state"]["status"] not in EDITABLE:
            raise CampaignError(f"campaign 状态为 {doc['state']['status']}，只能在草稿或暂停时修改")
        patch = dict(patch or {})
        if patch.get("campaign_id", cid) != cid:
            raise CampaignError("不能修改 campaign_id", 422)
        locked = {"task", "objective", "fixed", "baseline"} if doc["state"].get("started_unix_s") else set()
        changed_locked = [k for k in locked if k in patch and patch[k] != doc["spec"].get(k)]
        if changed_locked:
            raise CampaignError(f"campaign 已开始运行，不能修改 {', '.join(changed_locked)}", 422)
        raw = {k: v for k, v in doc["spec"].items() if k != "schema_version"}
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(raw.get(key), dict):
                raw[key] = {**raw[key], **value}
            else:
                raw[key] = value
        try:
            spec = validate_spec(raw, self.repo_root)
        except SpecError as exc:
            raise CampaignError(str(exc), 422) from exc
        if spec["fixed"] != doc["spec"]["fixed"]:
            raise CampaignError("修改数据或固定参数请新建 campaign", 422)
        _check_split(spec, doc["state"]["provenance"].get("split") or {})
        doc = store.load()
        doc["spec"] = spec
        store.save(doc)
        store.event("campaign", status="updated", source=source, fields=sorted(patch))
        return self.get_campaign(cid)

    def start_campaign(self, cid: str, *, source: str = "console") -> dict:
        store = self._existing(cid)
        doc = store.load()
        status = doc["state"]["status"]
        if status in ACTIVE and cid in self._drivers and self._drivers[cid].is_alive():
            return self.get_campaign(cid)
        if status not in ("draft", "paused") and status not in ACTIVE:
            raise CampaignError(f"campaign 已结束（{status}），不能再启动")
        _check_split(doc["spec"], doc["state"]["provenance"].get("split") or {})

        def mark(state):
            state.update(status="running", stop_requested=False, pause_requested=False)
            state["started_unix_s"] = state.get("started_unix_s") or time.time()

        store.update_state(mark)
        store.event("campaign", status="running", source=source)
        self._spawn(cid)
        return self.get_campaign(cid)

    def pause_campaign(self, cid: str, *, source: str = "console") -> dict:
        store = self._existing(cid)
        doc = store.load()
        if doc["state"]["status"] not in ("running", "pausing"):
            raise CampaignError(f"campaign 状态为 {doc['state']['status']}，不能暂停")
        store.update_state(lambda s: s.update(pause_requested=True, status="pausing"))
        store.event("campaign", status="pausing", source=source, note="当前 trial 跑完后暂停")
        return self.get_campaign(cid)

    def stop_campaign(self, cid: str, *, source: str = "console") -> dict:
        store = self._existing(cid)
        doc = store.load()
        status = doc["state"]["status"]
        if status in ("completed", "stopped", "failed"):
            return self.get_campaign(cid)
        alive = cid in self._drivers and self._drivers[cid].is_alive()
        store.update_state(lambda s: s.update(stop_requested=True, status="stopping" if alive else s["status"]))
        store.event("campaign", status="stopping", source=source)
        if not alive:
            self._finalize(store, "stopped", "用户停止")
        return self.get_campaign(cid)

    def stop_trial(self, cid: str, trial_id: str, *, source: str = "console") -> dict:
        store = self._existing(cid)
        trial = store.load_trial(trial_id)
        if trial is None:
            raise CampaignError(f"trial 不存在：{trial_id}", 404)
        if trial["status"] == "queued":
            trial.update(status="cancelled", verdict={"status": "cancelled", "kind": None,
                                                      "reasons": [f"排队时被停止（{source}）"]})
            store.save_trial(trial)
        elif trial["status"] == "running":
            job_id = (trial.get("attempts") or [{}])[-1].get("job_id")
            if job_id:
                self.jobs.stop(job_id)
        else:
            raise CampaignError(f"trial {trial_id} 状态为 {trial['status']}，不在运行")
        store.event("trial_stop", trial_id=trial_id, source=source)
        return store.load_trial(trial_id)

    def add_note(self, cid: str, text: str, *, title: str | None = None, source: str = "llm") -> dict:
        """Add a report note. A note with the same title from the same source replaces the earlier one, so a
        model that rewrites its interpretation (too long, corrected) leaves one version in the report; the
        event log keeps every write."""
        store = self._existing(cid)
        note = {"text": str(text)[:8000], "title": title, "source": source, "unix_s": time.time()}
        replaced = []

        def put(state):
            notes = state.setdefault("notes", [])
            same = [i for i, n in enumerate(notes) if title and n.get("title") == title and n.get("source") == source]
            if same:
                notes[same[-1]] = note
                replaced.append(True)
            else:
                notes.append(note)

        store.update_state(put)
        store.event("note", source=source, title=title, chars=len(note["text"]), replaced=bool(replaced))
        return note

    # ---- trial submission (rules, users and the LLM share the pre-check) ----------------------
    def _params_from(self, store, doc: dict, delta: dict, base_trial: str | None) -> tuple[dict, str | None]:
        from .spec import baseline_params

        trials = store.trials()
        ev = self._evidence(store, doc, trials)
        base = None
        if base_trial:
            base = next((t for t in trials if t["trial_id"] == base_trial), None)
            if base is None:
                raise CampaignError(f"trial 不存在：{base_trial}", 404)
        else:
            base = ev.best()
        params = dict(base["params"]) if base else baseline_params(doc["spec"], doc["spec"]["baseline"]["seeds"][0])
        params.update(doc["state"].get("adopted") or {})
        params.update(delta)
        return params, base["trial_id"] if base else None

    def check_proposals(self, cid: str, proposals: list[dict]) -> list[dict]:
        """Pre-check proposals without queueing them (``{"delta", "base_trial"?}`` each)."""
        store = self._existing(cid)
        doc = store.load()
        out = []
        for item in proposals:
            delta = dict(item.get("delta") or {})
            params, base = self._params_from(store, doc, delta, item.get("base_trial"))
            out.append(self._precheck(store, doc, params, delta))
        return out

    def _precheck(self, store, doc: dict, params: dict, delta: dict | None) -> dict:
        spec = doc["spec"]
        trials = store.trials()
        split = doc["state"]["provenance"].get("split") or {}
        code = (doc["state"]["provenance"].get("code") or {}).get("code_hash")
        try:
            args = resolve_args(spec, params)
        except SpecError as exc:
            return {"passed": False, "reasons": [str(exc)], "warnings": [], "args": None, "fingerprint": None}
        fp = fingerprint(args, split.get("data_hash"), split.get("split_hash"), code)
        unavailable = doc["state"].get("unavailable") or {}
        result = precheck(params, spec=spec, repo_root=self.repo_root, knowledge=self.knowledge, trials=trials,
                          fingerprint=fp, delta=delta, budget_state=budget_state(trials),
                          capabilities={name: (False, why) for name, why in unavailable.items()})
        result["fingerprint"] = fp
        return result

    def _admit(self, store, doc: dict, *, phase: str, params: dict, delta: dict | None,
               origin: dict) -> tuple[dict | None, dict]:
        """Pre-check and queue one trial; returns ``(record or None, pre-check)``. Rejections are recorded."""
        check = self._precheck(store, doc, params, delta)
        store.event("proposal", phase=phase, params=params, origin=origin)
        store.event("precheck", phase=phase, passed=check["passed"], reasons=check["reasons"],
                    warnings=check["warnings"], source=origin.get("source"))
        summary = {k: check[k] for k in ("passed", "reasons", "warnings")}
        if not check["passed"]:
            return None, summary
        code = (doc["state"]["provenance"].get("code") or {}).get("code_hash")
        record = Scheduler(store, doc["spec"]).enqueue(
            phase=phase, params=params, args=check["args"], fingerprint=check["fingerprint"], origin=origin,
            precheck=summary)
        record["code_hash"] = code
        store.save_trial(record)
        return record, summary

    def submit_trials(self, cid: str, proposals: list[dict], *, source: str = "llm") -> list[dict]:
        """Queue trials directly (an LLM after dsh approval, or a person); each goes through the pre-check."""
        store = self._existing(cid)
        doc = store.load()
        if doc["state"]["status"] not in ("running", "paused", "draft", "pausing"):
            raise CampaignError(f"campaign 状态为 {doc['state']['status']}，不能再提交 trial")
        out = []
        for item in proposals:
            delta = dict(item.get("delta") or {})
            params, base = self._params_from(store, doc, delta, item.get("base_trial"))
            origin = {"strategy": "llm" if source.startswith("llm") else "user", "source": source,
                      "agent": source, "base_trial": base, "delta": delta, "kind": item.get("kind", "accuracy"),
                      "rationale": item.get("rationale") or "", "expected": item.get("expected") or "",
                      "finding_ids": item.get("finding_ids") or [], "pending_id": item.get("pending_id")}
            record, check = self._admit(store, doc, phase="exploit", params=params, delta=delta, origin=origin)
            out.append({"queued": record is not None, "trial_id": record["trial_id"] if record else None,
                        "params": params, "precheck": check})
        return out

    def request_trials(self, cid: str, proposals: list[dict], *, source: str = "llm") -> dict:
        """An LLM asks to run trials: the console shows a pending item; nothing runs until a person confirms."""
        store = self._existing(cid)
        doc = store.load()
        checks = self.check_proposals(cid, proposals)
        reasons = [r for c in checks for r in c["reasons"]]
        item = {"id": self._pending_id(doc), "kind": "trials", "campaign_id": cid, "source": source,
                "proposals": [{"delta": p.get("delta") or {}, "base_trial": p.get("base_trial"),
                               "rationale": p.get("rationale") or "", "expected": p.get("expected") or "",
                               "precheck": {k: c[k] for k in ("passed", "reasons", "warnings")}}
                              for p, c in zip(proposals, checks)],
                "precheck": {"passed": all(c["passed"] for c in checks), "reasons": reasons},
                "status": "pending_confirmation", "created_unix_s": time.time()}
        store.update_state(lambda s: s.setdefault("pending", []).append(item))
        store.event("pending", pending_id=item["id"], pending_kind="trials", source=source,
                    passed=item["precheck"]["passed"], count=len(proposals))
        return item

    def request_campaign_start(self, cid: str, *, source: str = "llm") -> dict:
        store = self._existing(cid)
        doc = store.load()
        if doc["state"]["status"] not in ("draft", "paused"):
            raise CampaignError(f"campaign 状态为 {doc['state']['status']}，不需要启动")
        item = {"id": self._pending_id(doc), "kind": "campaign_start", "campaign_id": cid, "source": source,
                "proposals": [], "precheck": {"passed": True, "reasons": []}, "status": "pending_confirmation",
                "created_unix_s": time.time()}
        store.update_state(lambda s: s.setdefault("pending", []).append(item))
        store.event("pending", pending_id=item["id"], pending_kind="campaign_start", source=source)
        return item

    @staticmethod
    def _pending_id(doc: dict) -> str:
        return f"pa-{len(doc['state'].get('pending') or []) + 1:03d}"

    def resolve_pending(self, cid: str, pid: str, *, confirm: bool, source: str = "console") -> dict:
        store = self._existing(cid)
        doc = store.load()
        item = next((p for p in doc["state"].get("pending") or [] if p["id"] == pid), None)
        if item is None:
            raise CampaignError(f"待确认项不存在：{pid}", 404)
        if item["status"] != "pending_confirmation":
            raise CampaignError(f"待确认项已处理（{item['status']}）")
        outcome = {}
        if confirm:
            if item["kind"] == "campaign_start":
                self.start_campaign(cid, source=f"{item['source']}→{source}")
            else:
                outcome["trials"] = self.submit_trials(
                    cid, [{**p, "pending_id": pid} for p in item["proposals"]], source=item["source"])
        status = "confirmed" if confirm else "rejected"

        def mark(state):
            for p in state.get("pending") or []:
                if p["id"] == pid:
                    p.update(status=status, resolved_unix_s=time.time(), resolved_by=source, outcome=outcome)

        store.update_state(mark)
        store.event("confirmation", pending_id=pid, status=status, source=source, outcome=outcome)
        return next(p for p in store.load()["state"]["pending"] if p["id"] == pid)

    # ---- evidence and diagnosis -----------------------------------------------------------------
    def _evidence(self, store, doc: dict, trials: list[dict] | None = None) -> Evidence:
        trials = store.trials() if trials is None else trials
        summaries = {t["trial_id"]: (t.get("result") or {}).get("summary") or {} for t in trials}
        return Evidence(spec=doc["spec"], state=doc["state"], trials=trials, summaries=summaries)

    def _diagnose(self, store, ev: Evidence, specialists) -> list[Finding]:
        findings, known = [], ev.state.setdefault("findings", {})
        unavailable = {}
        for specialist in specialists:
            ok, why = specialist.available(ev)
            if not ok:
                unavailable[specialist.name] = why
                continue
            try:
                found = specialist.diagnose(ev)
            except Exception as exc:  # a broken rule must not stop the campaign
                store.event("note", source="planner", title=f"{specialist.name} 诊断出错", text=str(exc))
                continue
            for finding in found:
                findings.append(finding)
                if finding.id not in known:
                    known[finding.id] = {**finding.to_dict(), "first_seen_unix_s": time.time()}
                    store.event("finding", **finding.to_dict())
                else:  # a conclusion moves as more evidence arrives: keep the latest wording and data
                    known[finding.id] = {**finding.to_dict(), "first_seen_unix_s": known[finding.id].get("first_seen_unix_s")}
        for fid, data in known.items():
            if data.get("agent") == "verifier":
                findings.append(Finding(**{k: data[k] for k in ("id", "agent", "kind", "message", "trial_id",
                                                                "severity", "evidence", "data") if k in data}))
        ev.state["unavailable"] = unavailable
        return findings

    # ---- the driver ----------------------------------------------------------------------------
    def _spawn(self, cid: str) -> None:
        with self._lock:
            thread = self._drivers.get(cid)
            if thread is not None and thread.is_alive():
                return
            thread = threading.Thread(target=self._drive_safe, args=(cid,), name=f"autopilot-{cid}", daemon=True)
            self._drivers[cid] = thread
            thread.start()

    def _drive_safe(self, cid: str) -> None:
        store = self.store(cid)
        try:
            self._drive(store)
        except Exception as exc:
            store.event("campaign", status="error", error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc())
            try:
                self._finalize(store, "failed", f"驱动线程出错：{exc}")
            except Exception:
                pass

    def _commit(self, store, state: dict) -> None:
        store.update_state(lambda s: s.update({k: state[k] for k in PLANNER_KEYS if k in state}))

    def _drive(self, store: CampaignStore) -> None:
        doc = store.load()
        spec = doc["spec"]
        _check_split(spec, doc["state"]["provenance"].get("split") or {})
        specialists = build_specialists(spec["specialists"], repo_root=self.repo_root)
        tpe = None
        if spec["strategy"]["name"] == "tpe":
            from .strategies.tpe import TPEStrategy

            tpe = TPEStrategy(spec, knowledge=self.knowledge)
        planner = Planner(spec, specialists=specialists, tpe=tpe)
        scheduler = Scheduler(store, spec)
        if doc["state"].get("status") == "finishing":  # interrupted while writing the report: only finish it
            self._finalize(store, doc["state"].get("finishing_as") or "stopped", doc["state"].get("stop_reason"))
            return
        provenance = {"code": code_state(self.repo_root)}
        if self.record_environment and not doc["state"]["provenance"].get("environment"):
            provenance["environment"] = environment()
        store.update_state(lambda s: s["provenance"].update(provenance))
        idle_rounds = 0
        while not self._closing.is_set():
            doc = store.load()
            state = doc["state"]
            if state.get("stop_requested"):
                self._finalize(store, "stopped", state.get("stop_reason") or "用户停止")
                return
            trials = store.trials()
            running = [t for t in trials if t["status"] == "running"]
            if running:
                self._run_trial(store, running[0])
                continue
            if state.get("pause_requested"):
                store.update_state(lambda s: s.update(status="paused", pause_requested=False))
                store.event("campaign", status="paused")
                return
            ev = self._evidence(store, doc, trials)
            ev.state["noise"] = noise(ev)
            findings = self._diagnose(store, ev, specialists)
            phase_before = ev.state.get("phase")
            proposals = planner.next(ev, findings)
            if ev.state.get("phase") != phase_before:
                store.event("phase", phase=ev.state["phase"], previous=phase_before,
                            note=(ev.state.get("phase_log") or [{}])[-1].get("note"))
                self._decision_point(store, ev, f"phase:{phase_before}->{ev.state['phase']}")
            best = ev.best()
            ev.state["best"] = {"trial_id": best["trial_id"], "value": ev.objective(best)} if best else None
            ev.state["budget_used"] = budget_state(trials)
            self._commit(store, ev.state)
            for p in proposals:
                origin = {"strategy": p.strategy, "agent": p.agent, "source": "rule", "base_trial": p.base_trial,
                          "finding_ids": p.finding_ids, "rationale": p.rationale, "expected": p.expected,
                          "delta": p.delta, "kind": p.kind}
                self._admit(store, doc, phase=p.phase, params=p.params, delta=p.delta or None, origin=origin)
            trials = store.trials()
            used = budget_state(trials)
            if used["trials_started"] >= spec["budget"]["max_trials"] or used["run_s"] >= spec["budget"]["wall_time_s"]:
                if not any(t["status"] == "running" for t in trials):
                    why = "trial 数" if used["trials_started"] >= spec["budget"]["max_trials"] else "总时长"
                    self._finalize(store, "completed", f"预算用完（{why}）")
                    return
            trial = scheduler.dispatch(trials, (doc["state"]["provenance"].get("split") or {}))
            if trial is None:
                doc = store.load()
                if doc["state"].get("phase") == "done":
                    self._finalize(store, "completed", doc["state"].get("stop_reason") or "所有阶段完成")
                    return
                idle_rounds = idle_rounds + 1 if not proposals else 0
                if idle_rounds > len(("baseline", "explore", "exploit", "confirm")) + 2:
                    self._finalize(store, "completed", "没有可运行的候选")
                    return
                continue
            idle_rounds = 0
            self._run_trial(store, trial)

    def _run_trial(self, store: CampaignStore, trial: dict) -> None:
        doc = store.load()
        spec = doc["spec"]
        split = doc["state"]["provenance"].get("split") or {}

        def should_stop():
            return bool(store.load()["state"].get("stop_requested"))

        def on_finding(item: dict):
            fid = f"verifier:{item['kind']}:{item['trial_id']}"
            finding = {"id": fid, "agent": "verifier", "kind": item["kind"], "message": item["message"],
                       "trial_id": item["trial_id"], "severity": "warning", "evidence": [], "data": item}
            store.update_state(lambda s: s.setdefault("findings", {}).setdefault(fid, finding))
            store.event("finding", **finding)

        next_attempt = {"cause": "first", "overrides": None}
        while True:
            attempts = trial.get("attempts") or []
            if attempts and attempts[-1]["status"] == "running":
                attempt = attempts[-1]
            else:
                attempt = self.executor.submit(store, trial, len(attempts) + 1, **next_attempt)
            attempt = self.executor.wait(store, trial, attempt, spec, should_stop=should_stop, on_finding=on_finding,
                                         abort=self._closing.is_set)
            if attempt["status"] == "running":
                return  # shutting down; the next start re-attaches to the job
            failure = attempt.get("failure") or {}
            # Retries count only infrastructure failures; rollbacks have their own limit.
            retries_left = spec["budget"]["max_retries"] - sum(
                1 for a in trial["attempts"] if a.get("cause") == "retry")
            paging = self._paging_share(attempt) if failure.get("kind") == "hang" else None
            if paging is not None and retries_left > 0:
                # A hang with the GPU memory full is WDDM paging: the same configuration would hang again.
                store.event("retry_skipped", trial_id=trial["trial_id"], attempt=attempt["attempt"],
                            kind="hang", reason=f"卡死时显存占整张 GPU 的 {paging:.0%}，是换页而不是基础设施故障，"
                                                "重试同一配置没有意义")
                retries_left = 0
            if attempt["status"] == "failed" and failure.get("kind") in INFRA_FAILURES and retries_left > 0 \
                    and not should_stop():
                store.event("retry", trial_id=trial["trial_id"], attempt=attempt["attempt"],
                            kind=failure.get("kind"), message=failure.get("message"))
                next_attempt = {"cause": "retry", "overrides": attempt.get("overrides")}
                continue
            if attempt["status"] == "failed" and failure.get("kind") in rollback.ROLLBACK_KINDS and not should_stop():
                plan, why = rollback.plan(trial, attempt, spec)
                if plan is not None:
                    trial.setdefault("rollbacks", []).append({**plan, "failed_attempt": attempt["attempt"]})
                    store.save_trial(trial)
                    store.event("rollback", trial_id=trial["trial_id"], attempt=attempt["attempt"],
                                failure=failure.get("kind"), from_step=plan["from_step"],
                                from_attempt=plan["from_attempt"], lr_before=plan["lr_before"], lr=plan["lr"],
                                reason=why)
                    next_attempt = {"cause": "rollback",
                                    "overrides": {"resume_state": plan["resume_state"], "lr": plan["lr"]}}
                    continue
                if rollback.settings(spec)["enabled"]:
                    store.event("rollback_skipped", trial_id=trial["trial_id"], attempt=attempt["attempt"],
                                failure=failure.get("kind"), reason=why)
            break
        self._drop_states(trial)
        run_dir = Path(attempt["run_dir"])
        result = read_json(run_dir / "result.json")
        verdict = postcheck(trial, attempt, result, spec=spec, split=split)
        summary = summarize_attempt(run_dir)
        trial_result = {
            "schema_version": 1, "run_id": f"autopilot/{store.campaign_id}/{trial['trial_id']}/attempt-{attempt['attempt']}",
            "trial_id": trial["trial_id"], "attempt": attempt["attempt"], "phase": trial["phase"],
            "fingerprint": trial["fingerprint"], "origin": trial["origin"], "resolved_config": trial["resolved_args"],
            "command": (read_json(run_dir / "command.json") or {}).get("command"), "status": attempt["status"],
            "verdict": verdict, "metrics": (result or {}).get("metrics") or {},
            "resources": {"memory": (result or {}).get("memory"), **{k: summary.get(k) for k in (
                "gpu_util_median", "data_share", "mem_peak_mb", "mem_share", "host_overhead_share", "throttled")}},
            "timing": {"run_s": attempt.get("run_s"), "lock_wait_s": attempt.get("lock_wait_s"),
                       **((result or {}).get("timing") or {})},
            "paths": {"run_dir": str(run_dir), "result": str(run_dir / "result.json"),
                      "metrics": str(run_dir / "metrics.jsonl"), "resources": str(run_dir / "resources.jsonl"),
                      "log": str(run_dir / "train.log")},
            "failure": attempt.get("failure"), "job_id": attempt.get("job_id"),
        }
        write_json(run_dir / "verdict.json", verdict)
        write_json(run_dir / "trial_result.json", trial_result)
        trial.update(status="done" if verdict["status"] != "cancelled" else "cancelled", verdict=verdict,
                     finished_unix_s=time.time(),
                     result={"metrics": trial_result["metrics"], "summary": summary, "timing": trial_result["timing"],
                             "data": (result or {}).get("data"), "memory": (result or {}).get("memory"),
                             "samples": (result or {}).get("samples"), "backend": (result or {}).get("backend"),
                             "epochs": (result or {}).get("epochs")})
        store.save_trial(trial)
        store.event("verdict", trial_id=trial["trial_id"], attempt=attempt["attempt"], **verdict,
                    objective=(trial_result["metrics"] or {}).get(spec["objective"]["metric"]))
        if verdict["status"] == "failed" and verdict.get("kind") == "oom":
            fact = self.knowledge.record_oom(spec["task"], trial["resolved_args"], evidence=str(run_dir),
                                             code_hash=trial.get("code_hash"))
            store.event("note", source="memory", title="经验库新增显存不足记录", text=fact.get("fact"))
        if verdict["status"] == "failed" and verdict.get("kind") == "oom":
            self._derive_after_oom(store, trial)
        if verdict["status"] == "failed" and verdict.get("kind") in ("nan", "diverged", "oom"):
            self._decision_point(store, None, f"failure:{verdict['kind']}:{trial['trial_id']}")

    def _evaluate_test_set(self, store: CampaignStore, ev: Evidence) -> None:
        """Score the baseline and the best configuration (every seed) on the held-out test split, once, at the end.

        The test set never takes part in ranking or confirmation; this is the final, independent check.
        """
        from .planner import Planner

        planner = Planner(ev.spec)
        baseline = [t for t in ev.valid() if planner._is_baseline(t)]
        others = [t for t in ev.ranked() if not planner._is_baseline(t)]
        groups = {"baseline": baseline}
        if others:
            key = ev.accuracy_key(others[0])
            groups["best"] = [t for t in ev.valid() if ev.accuracy_key(t) == key]
        items = []
        for group, trials in groups.items():
            for t in trials:
                attempt = (t.get("attempts") or [{}])[-1]
                items.append({"key": t["trial_id"], "group": group, "args": t["resolved_args"],
                              "checkpoint": str(Path(attempt.get("run_dir", "")) / "checkpoints" / "best")})
        if not items:
            return
        out_dir = store.report_dir / "test-eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "items.json", {"split": "test", "items": items})
        job = self.jobs.submit(job_type="single", label=f"AutoPilot · {store.campaign_id} · 测试集评估",
                               task="donkey_resnet18_single", run_dir=out_dir,
                               command=[self.python, "-m", self.eval_module, "--items", str(out_dir / "items.json"),
                                        "--out", str(out_dir / "evaluation.json"), "--run-dir", "{run_dir}"],
                               extra={"campaign_id": store.campaign_id, "record_resources": False})
        store.event("test_eval", status="started", job_id=job["id"], items=len(items))
        deadline = time.time() + EVAL_TIMEOUT_S
        while time.time() < deadline and not self._closing.is_set():
            current = self.jobs.get(job["id"])
            if current is None or current["status"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(self.poll_s)
        payload = read_json(out_dir / "evaluation.json") or {}
        by_key = {r["key"]: r for r in payload.get("results") or []}
        summary = {"split": "test", "job_id": job["id"], "failures": payload.get("failures") or [], "groups": {}}
        for group, trials in groups.items():
            values = {t["trial_id"]: by_key[t["trial_id"]]["angle_mae"] for t in trials if t["trial_id"] in by_key}
            summary["groups"][group] = {
                "trials": [t["trial_id"] for t in trials], "angle_mae": values,
                "mean": sum(values.values()) / len(values) if values else None,
                "samples": next((by_key[k]["samples"] for k in values), None),
                "params": {k: v for k, v in (trials[0].get("params") or {}).items() if k != "seed"} if trials else {}}
        store.update_state(lambda s: s.update(test_evaluation=summary))
        store.event("test_eval", status="done" if payload else "failed", job_id=job["id"],
                    groups={g: v["mean"] for g, v in summary["groups"].items()}, failures=len(summary["failures"]))

    @staticmethod
    def _paging_share(attempt: dict) -> float | None:
        """The attempt's GPU memory share when it is high enough to explain a hang by paging, else None."""
        from .agents.gpu_monitor import PAGING_MEMORY

        summary = summarize_attempt(Path(attempt["run_dir"]))
        shares = [v for v in (summary.get("mem_share"), summary.get("device_mem_share")) if v is not None]
        return max(shares) if shares and max(shares) > PAGING_MEMORY else None

    @staticmethod
    def _drop_states(trial: dict) -> None:
        """TrainingStates exist for rollbacks within the trial; once it is decided they are only disk usage."""
        drop_states(trial)

    def _derive_after_oom(self, store, trial: dict) -> None:
        """E3: an OOM trial derives one trial with half the batch (the knowledge base already blocks the original)."""
        doc = store.load()
        delta = rollback.oom_proposal(trial, doc["spec"])
        if delta is None or (trial.get("origin") or {}).get("strategy") == "rollback":
            return
        params = {**trial["params"], **delta}
        origin = {"strategy": "rollback", "agent": "verifier", "source": "rule", "base_trial": trial["trial_id"],
                  "delta": delta, "kind": "accuracy", "finding_ids": [],
                  "rationale": f"{trial['trial_id']} 显存不足：batch {trial['params'].get('batch')} → {delta['batch']}",
                  "expected": "不再 OOM；精度需重新评估"}
        phase = trial["phase"] if trial["phase"] in ("explore", "exploit") else "exploit"
        record, check = self._admit(store, doc, phase=phase, params=params, delta=delta, origin=origin)
        store.event("oom_derived", trial_id=trial["trial_id"], derived=record["trial_id"] if record else None,
                    delta=delta, precheck=check)

    def _decision_point(self, store, ev, reason: str) -> None:
        """Hook for the LLM advisor (after the baseline, after explore batches, after OOM/NaN failures)."""
        if self.advisor is None:
            return
        try:
            self.advisor.consult(self, store, reason)
        except Exception as exc:
            store.event("llm_error", reason=reason, error=f"{type(exc).__name__}: {exc}")

    # ---- the end of a campaign -------------------------------------------------------------------
    def _finalize(self, store: CampaignStore, status: str, reason: str | None) -> None:
        now = time.time()
        for trial in store.trials():
            if trial["status"] == "queued":
                trial.update(status="skipped", skip_reason=f"campaign 结束（{reason}）", finished_unix_s=now)
                store.save_trial(trial)

        def mark(state):
            state.update(status="finishing", finishing_as=status, finished_unix_s=now, stop_requested=False,
                         pause_requested=False)
            if reason and not state.get("stop_reason"):
                state["stop_reason"] = reason
            for item in state.get("pending") or []:
                if item["status"] == "pending_confirmation":
                    item.update(status="expired", resolved_unix_s=now)

        store.update_state(mark)
        doc = store.load()
        ev = self._evidence(store, doc)
        ev.state["noise"] = noise(ev)
        self._commit(store, ev.state)
        if status == "completed" and ((doc["state"].get("provenance") or {}).get("split") or {}).get("test_count"):
            self._evaluate_test_set(store, ev)
            doc = store.load()
            ev = self._evidence(store, doc)
        self._prune_checkpoints(store, ev)
        if self.advisor is not None:
            self._decision_point(store, ev, "report")
            doc = store.load()
            ev = self._evidence(store, doc)
        ev.state["status"] = status
        path = self._write_report(store, ev)
        # The final status is set last, so a finished campaign always has its report.
        store.update_state(lambda s: s.update(status=status, report={"path": str(path),
                                                                      "generated_unix_s": time.time()}))
        store.event("campaign", status=status, reason=reason, report=str(path))

    def _write_report(self, store: CampaignStore, ev: Evidence) -> Path:
        specialists = build_specialists(ev.spec["specialists"], repo_root=self.repo_root)
        status_map = {s.name: s.available(ev) for s in specialists}
        findings = list((ev.state.get("findings") or {}).values())
        sections = []
        for specialist in specialists:
            render = getattr(specialist, "report_section", None)
            if render is not None and status_map[specialist.name][0]:
                try:
                    text = render(ev, [f for f in findings if f.get("agent") == specialist.name])
                except Exception as exc:  # a broken section must not cost the report
                    text = f"### {specialist.label}\n\n- 生成失败：{type(exc).__name__}: {exc}\n"
                if text:
                    sections.append(text)
        return reporter.write(ev, store, llm_events=store.events(0, LLM_EVENT_KINDS), specialists_status=status_map,
                              sections=sections)

    def regenerate_report(self, cid: str) -> dict:
        """Re-render a finished campaign's report with the current rules.

        The specialists diagnose the recorded evidence again (findings are refreshed in place; no trial is
        started, recorded chains and decisions stay as they were), then report.md is written again.
        """
        store = self._existing(cid)
        doc = store.load()
        if doc["state"]["status"] not in ("completed", "stopped", "failed"):
            raise CampaignError("只有已结束的 campaign 能重新生成报告", 409)
        # The evidence summaries are recomputed from the run directories too (they are derived data).
        for trial in store.trials():
            attempts = trial.get("attempts") or []
            if trial.get("result") and attempts and Path(attempts[-1]["run_dir"]).is_dir():
                trial["result"]["summary"] = summarize_attempt(Path(attempts[-1]["run_dir"]))
                store.save_trial(trial)
        self._recheck_online_findings(store, doc)
        doc = store.load()
        ev = self._evidence(store, doc)
        ev.state["noise"] = noise(ev)
        known = ev.state.setdefault("findings", {})
        for specialist in build_specialists(doc["spec"]["specialists"], repo_root=self.repo_root):
            if not specialist.available(ev)[0]:
                continue
            try:
                found = specialist.diagnose(ev)
            except Exception as exc:  # a broken rule must not cost the report
                store.event("note", source="reporter", title=f"{specialist.name} 诊断出错", text=str(exc))
                continue
            for finding in found:
                first = (known.get(finding.id) or {}).get("first_seen_unix_s", time.time())
                known[finding.id] = {**finding.to_dict(), "first_seen_unix_s": first}
        store.update_state(lambda s: s.update(findings=known, noise=ev.state["noise"]))
        path = self._write_report(store, ev)
        store.update_state(lambda s: s.update(report={"path": str(path), "generated_unix_s": time.time(),
                                                      "regenerated": True}))
        store.event("note", source="reporter", title="重新生成报告", text="按当前规则重新诊断已记录的证据并重写报告")
        return self.report(cid)

    def _recheck_online_findings(self, store: CampaignStore, doc: dict) -> None:
        """Replay each trial's recorded attempts with the current online rules. A recorded online finding that no
        attempt reproduces stays in the record, marked retracted with the reason; the report lists it apart."""
        spec = doc["spec"]
        reproduced: dict[str, set] = {}
        for trial in store.trials():
            for attempt in trial.get("attempts") or []:
                run_dir = Path(attempt.get("run_dir") or "")
                if attempt.get("run_dir") and (run_dir / "metrics.jsonl").is_file():
                    found = replay_online_findings(run_dir, spec["online"], spec["budget"]["trial_timeout_s"])
                    reproduced.setdefault(trial["trial_id"], set()).update(f["kind"] for f in found)
        retracted = []

        def mark(state):
            for f in (state.get("findings") or {}).values():
                if f.get("agent") != "verifier" or f.get("kind") not in ONLINE_FINDING_KINDS                         or f.get("trial_id") not in reproduced:
                    continue
                if f["kind"] in reproduced[f["trial_id"]]:
                    f.pop("retracted", None)
                else:
                    f["retracted"] = "按当前在线规则重放该 trial 记录的 step 与 GPU 采样后不再出现"
                    retracted.append(f["id"])

        store.update_state(mark)
        if retracted:
            store.event("note", source="reporter", title="撤回在线发现",
                        text=f"重放后不再成立：{', '.join(sorted(retracted))}")

    def _prune_checkpoints(self, store: CampaignStore, ev: Evidence) -> None:
        """Keep the best models of the baseline and the top-2 trials; others keep logs and metrics only."""
        keep = {t["trial_id"] for t in ev.trials if t.get("phase") == "baseline"}
        keep |= {t["trial_id"] for t in ev.ranked()[:KEEP_TOP]}
        removed = 0
        for trial in ev.trials:
            for attempt in trial.get("attempts") or []:
                ckpt = Path(attempt["run_dir"]) / "checkpoints"
                if not ckpt.is_dir():
                    continue
                if trial["trial_id"] in keep:
                    for path in ckpt.glob("checkpoint.*"):  # resume state; the best model stays
                        path.unlink(missing_ok=True)
                else:
                    shutil.rmtree(ckpt, ignore_errors=True)
                    removed += 1
        if removed:
            store.event("note", source="executor", title="清理断点",
                        text=f"保留基线与前 {KEEP_TOP} 名的最佳模型，删除其余 {removed} 个尝试的断点")
