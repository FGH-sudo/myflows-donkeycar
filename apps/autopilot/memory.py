"""AutoPilot Memory (spec §6.4): working memory, trial registry, event log and the knowledge store.

Layout (``runs/autopilot``)::

    knowledge.jsonl                      reusable facts (not in git)
    <campaign_id>/campaign.json          spec + working memory
    <campaign_id>/events.jsonl           every proposal, precheck, dispatch, verdict, LLM action ...
    <campaign_id>/trials/<id>/trial.json trial record (attempts, verdict, summary)
    <campaign_id>/trials/<id>/attempt-<n>/  run directory of one attempt (written by apps.train)
    <campaign_id>/report/                report.md, leaderboard.csv, figures/

Every JSON write goes through a temporary file and ``os.replace``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from .files import append_jsonl, read_json, read_jsonl, write_json

TRIAL_SCHEMA_VERSION = 1
KNOWLEDGE_SEMANTICS_VERSION = 1


class CampaignStore:
    """Files of one campaign. Thread-safe: the driver thread and HTTP handlers share it."""

    def __init__(self, root: Path, campaign_id: str):
        self.root = Path(root)
        self.campaign_id = campaign_id
        self.dir = self.root / campaign_id
        self.campaign_path = self.dir / "campaign.json"
        self.events_path = self.dir / "events.jsonl"
        self.trials_dir = self.dir / "trials"
        self.report_dir = self.dir / "report"
        self._lock = threading.RLock()
        self._seq = None
        self.listeners: list[Callable[[dict], None]] = []

    def exists(self) -> bool:
        return self.campaign_path.exists()

    # ---- campaign document ----------------------------------------------------------
    def load(self) -> dict:
        with self._lock:
            doc = read_json(self.campaign_path)
            if doc is None:
                raise FileNotFoundError(self.campaign_path)
            return doc

    def save(self, doc: dict) -> None:
        with self._lock:
            doc["updated_unix_s"] = time.time()
            write_json(self.campaign_path, doc)

    def update_state(self, fn: Callable[[dict], None]) -> dict:
        """Read-modify-write of ``state`` under the store lock; returns the new document."""
        with self._lock:
            doc = self.load()
            fn(doc["state"])
            self.save(doc)
            return doc

    # ---- events ---------------------------------------------------------------------
    def event(self, kind: str, /, **fields) -> dict:
        """Append one event; a field named ``kind`` (a finding's or a verdict's) is stored as ``<kind>_kind``."""
        if "kind" in fields:
            fields[f"{kind}_kind"] = fields.pop("kind")
        with self._lock:
            if self._seq is None:
                rows = read_jsonl(self.events_path)
                self._seq = rows[-1]["seq"] if rows else 0
            self._seq += 1
            row = {"seq": self._seq, "unix_s": time.time(), "kind": kind, **fields}
            append_jsonl(self.events_path, row)
        for listener in list(self.listeners):
            try:
                listener(row)
            except Exception:
                pass
        return row

    def events(self, since_seq: int = 0, kinds: tuple[str, ...] | None = None) -> list[dict]:
        rows = [r for r in read_jsonl(self.events_path) if r.get("seq", 0) > since_seq]
        return [r for r in rows if kinds is None or r.get("kind") in kinds]

    # ---- trials ---------------------------------------------------------------------
    def trial_dir(self, trial_id: str) -> Path:
        return self.trials_dir / trial_id

    def attempt_dir(self, trial_id: str, attempt: int) -> Path:
        return self.trial_dir(trial_id) / f"attempt-{attempt}"

    def next_trial_id(self) -> str:
        with self._lock:
            existing = [p.name for p in self.trials_dir.glob("t*") if p.is_dir()] if self.trials_dir.exists() else []
            numbers = [int(name[1:]) for name in existing if name[1:].isdigit()]
            trial_id = f"t{(max(numbers) + 1) if numbers else 1:03d}"
            self.trial_dir(trial_id).mkdir(parents=True, exist_ok=True)
            return trial_id

    def save_trial(self, record: dict) -> None:
        with self._lock:
            record["updated_unix_s"] = time.time()
            write_json(self.trial_dir(record["trial_id"]) / "trial.json", record)

    def load_trial(self, trial_id: str) -> dict | None:
        return read_json(self.trial_dir(trial_id) / "trial.json")

    def trials(self) -> list[dict]:
        if not self.trials_dir.exists():
            return []
        records = [read_json(p / "trial.json") for p in sorted(self.trials_dir.iterdir()) if p.is_dir()]
        return sorted((r for r in records if r), key=lambda r: r["trial_id"])


# ---- knowledge store ---------------------------------------------------------------------
# Resource facts carried over from earlier reports (spec §6.4). Convergence facts start empty: earlier
# results used the old Adam.
SEED_FACTS = [
    {"id": "seed-cifar-w64-batch16", "kind": "resource", "task": "donkey_resnet18_single",
     "model": {"stem": "cifar", "base_width": 64}, "when": {"batch_min": 16, "workers_max": 1},
     "outcome": "tight",
     "fact": "cifar stem、w64、batch 16 时显存吃紧：layer1 每个卷积的 im2col 缓冲约 708 MB",
     "source": "docs/experiments/semester_2026_fall（第 19 讲 GEMM 报告）"},
    {"id": "seed-two-w64-workers-batch4", "kind": "resource", "task": "donkey_resnet18_single",
     "model": {"stem": "cifar", "base_width": 64}, "when": {"workers_min": 2, "per_worker_batch_min": 4},
     "outcome": "oom",
     "fact": "两个 w64 worker 各用 batch 4 时超过 8 GB 显存：WDDM 会悄悄换页、看起来像卡死，"
             "需要按 Worker 限制显存池（--worker-memory-mb），限制后直接显存不足",
     "source": "分布式正式训练联调（80d02af）"},
    {"id": "seed-socket-json-width", "kind": "resource", "task": "donkey_resnet18_single",
     "model": {}, "when": {"transport": "socket_json", "base_width_min": 33}, "outcome": "rejected",
     "fact": "模型宽度大于 32 时 Socket JSON 单条消息超过 64 MB 上限，传输被拒绝",
     "source": "apps/console/presets.py validate_single"},
]


def _model_of(args: dict) -> dict:
    return {"stem": args.get("stem", "cifar"), "base_width": int(args.get("base_width", 64))}


def _workers(args: dict) -> int:
    return int(args.get("workers", 1)) if args.get("distributed", "none") != "none" else 1


def fact_matches(fact: dict, args: dict) -> bool:
    model = _model_of(args)
    for key, value in (fact.get("model") or {}).items():
        if model.get(key) != value:
            return False
    when = fact.get("when") or {}
    workers = _workers(args)
    batch = int(args.get("batch", 1))
    checks = {
        "batch_min": lambda v: batch >= v,
        "workers_min": lambda v: workers >= v,
        "workers_max": lambda v: workers <= v,
        "workers": lambda v: workers == v,
        "per_worker_batch_min": lambda v: batch / max(1, workers) >= v,
        "transport": lambda v: args.get("transport") == v and workers > 1,
        "base_width_min": lambda v: model["base_width"] >= v,
        "distributed": lambda v: args.get("distributed", "none") == v,
        "dtype": lambda v: args.get("dtype", "float32") == v,
        "max_samples": lambda v: True,
    }
    return all(checks.get(key, lambda _v: True)(value) for key, value in when.items())


class KnowledgeBase:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def ensure_seeded(self) -> None:
        with self._lock:
            if self.path.exists():
                return
            now = time.time()
            for fact in SEED_FACTS:
                append_jsonl(self.path, {**fact, "semantics_version": KNOWLEDGE_SEMANTICS_VERSION,
                                         "code_hash": None, "created_unix_s": now, "origin": "import"})

    def facts(self, kind: str | None = None) -> list[dict]:
        rows = read_jsonl(self.path)
        return [r for r in rows if (kind is None or r.get("kind") == kind)
                and r.get("semantics_version") == KNOWLEDGE_SEMANTICS_VERSION]

    def add(self, fact: dict) -> dict:
        fact = {"semantics_version": KNOWLEDGE_SEMANTICS_VERSION, "created_unix_s": time.time(), **fact}
        with self._lock:
            if any(r.get("id") == fact.get("id") for r in read_jsonl(self.path)):
                return fact
            append_jsonl(self.path, fact)
        return fact

    def predicted_failure(self, task: str, args: dict) -> dict | None:
        """A fact that says this configuration runs out of memory (or is refused) on this machine."""
        for fact in self.facts("resource"):
            if fact.get("task") in (None, task) and fact.get("outcome") in ("oom", "rejected") \
                    and fact_matches(fact, args):
                return fact
        return None

    def warnings(self, task: str, args: dict) -> list[dict]:
        return [f for f in self.facts("resource") if f.get("task") in (None, task) and f.get("outcome") == "tight"
                and fact_matches(f, args)]

    def record_oom(self, task: str, args: dict, *, evidence: str, code_hash: str | None) -> dict:
        model = _model_of(args)
        workers = _workers(args)
        when = {"batch_min": int(args.get("batch", 1)), "workers": workers,
                "distributed": args.get("distributed", "none"), "dtype": args.get("dtype", "float32")}
        fact_id = f"oom-{model['stem']}-w{model['base_width']}-b{when['batch_min']}-{when['distributed']}{workers}"
        return self.add({"id": fact_id, "kind": "resource", "task": task, "model": model, "when": when,
                         "outcome": "oom", "fact": f"{model['stem']} stem、w{model['base_width']}、batch "
                         f"{when['batch_min']}（{workers} 个进程）训练时显存不足", "source": evidence,
                         "code_hash": code_hash, "origin": "trial"})
