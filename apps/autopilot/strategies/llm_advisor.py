"""LLMAdvisor: one headless dsh run per decision point (spec §6.1, §8.5, §8.6).

Decision points: after the baseline, after each explore batch, after OOM/NaN/divergence failures, and once
at the end for the report explanation. The model reads the campaign through the read-only tools and may
answer with ``request_trials`` (pending items a person confirms) and ``write_note``; it cannot start
training. One background session per campaign, resumed with ``--session-id``.

Degradation: skipped (and recorded) when dsh is unavailable or no key is configured; at most
``spec.llm.max_calls`` calls per campaign; three consecutive failures turn the LLM off for the campaign; an
optional ``max_total_tokens`` does the same once exceeded. Decision points run in a background thread so
training continues meanwhile; the report explanation runs inline, after any call still in flight.
"""

from __future__ import annotations

import threading
import time

from ..dsh import prompts
from ..spec import MAX_LLM_CALLS

MAX_CONSECUTIVE_FAILURES = 3


class LLMAdvisor:
    def __init__(self, host, *, run_async: bool = True):
        self.host = host
        self.run_async = run_async
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    # ---- entry point (AutoPilotService._decision_point) ----------------------------------------------
    def consult(self, service, store, reason: str) -> None:
        doc = store.load()
        spec, llm = doc["spec"], doc["state"]["llm"]
        if not spec["llm"].get("enabled") or llm.get("disabled_reason"):
            return
        max_calls = int(spec["llm"].get("max_calls") or MAX_LLM_CALLS)
        if llm.get("calls", 0) >= max_calls:
            self._disable(store, f"已达到后台调用上限（{max_calls} 次）")
            return
        if not self.host.available:
            store.event("llm_skip", reason=reason, why=self.host.status().get("reason") or "dsh 不可用")
            return
        if not self.host.key_configured():
            store.event("llm_skip", reason=reason,
                        why="没有 DeepSeek API Key 或账号登录（在 AutoPilot 助手页的 dsh 里添加 Key 或登录）")
            return
        cid = store.campaign_id
        with self._lock:
            running = self._threads.get(cid)
            busy = running is not None and running.is_alive()
        if reason == "report":
            if busy:
                running.join(timeout=self.host.headless_timeout_s + 30)
            self._run(store, reason)
            return
        if busy:
            store.event("llm_skip", reason=reason, why="上一次顾问调用仍在进行")
            return
        if not self.run_async:
            self._run(store, reason)
            return
        thread = threading.Thread(target=self._run_safe, args=(store, reason), name=f"llm-advisor-{cid}", daemon=True)
        with self._lock:
            self._threads[cid] = thread
        thread.start()

    def wait(self, campaign_id: str, timeout: float | None = None) -> None:
        thread = self._threads.get(campaign_id)
        if thread is not None:
            thread.join(timeout=timeout)

    # ---- one call --------------------------------------------------------------------------------------
    def _run_safe(self, store, reason: str) -> None:
        try:
            self._run(store, reason)
        except Exception as exc:
            store.event("llm_error", reason=reason, error=f"{type(exc).__name__}: {exc}")

    def _run(self, store, reason: str) -> None:
        cid = store.campaign_id
        session_id = store.load()["state"]["llm"].get("session_id")
        store.update_state(lambda s: s["llm"].update(calls=int(s["llm"].get("calls", 0)) + 1))
        store.event("llm_call", status="started", reason=reason, session_id=session_id)
        result = self.host.run_headless(prompts.build(cid, reason), campaign=cid, session_id=session_id)
        if not result["ok"] and session_id and "does not exist" in (result.get("error") or ""):
            # The session store was reset (e.g. a new dsh home): start a new background session.
            result = self.host.run_headless(prompts.build(cid, reason), campaign=cid, session_id=None)
        tokens = int((result.get("usage") or {}).get("total_tokens") or 0)
        limit = store.load()["spec"]["llm"].get("max_total_tokens")

        def mark(state):
            llm = state["llm"]
            llm["session_id"] = result.get("session_id") or llm.get("session_id")
            llm["tokens"] = int(llm.get("tokens") or 0) + tokens
            if result["ok"]:
                llm["consecutive_failures"] = 0
            else:
                llm["failures"] = int(llm.get("failures") or 0) + 1
                llm["consecutive_failures"] = int(llm.get("consecutive_failures") or 0) + 1
                if llm["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES and not llm.get("disabled_reason"):
                    llm["disabled_reason"] = f"连续 {MAX_CONSECUTIVE_FAILURES} 次调用失败，本 campaign 关闭 LLM"
            if limit and llm["tokens"] >= int(limit) and not llm.get("disabled_reason"):
                llm["disabled_reason"] = f"token 用量达到上限（{llm['tokens']} ≥ {limit}）"
            llm.setdefault("decision_points", []).append(
                {"reason": reason, "ok": result["ok"], "unix_s": time.time(), "tokens": tokens,
                 "tools": [c.get("tool") for c in result.get("tool_calls") or []]})

        before = store.load()["state"]["llm"].get("disabled_reason")
        doc = store.update_state(mark)
        store.event("llm_call", status="ok" if result["ok"] else "failed", reason=reason,
                    duration_s=result.get("duration_s"), usage=result.get("usage"),
                    tool_calls=result.get("tool_calls"), text=(result.get("text") or "")[:2000],
                    error=result.get("error"), session_id=result.get("session_id"))
        after = doc["state"]["llm"].get("disabled_reason")
        if after and after != before:
            store.event("llm_disabled", why=after)

    @staticmethod
    def _disable(store, why: str) -> None:
        store.update_state(lambda s: s["llm"].update(disabled_reason=why))
        store.event("llm_disabled", why=why)
