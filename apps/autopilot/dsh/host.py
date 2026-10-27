"""DshHost: the AutoPilot-only DeepSeek Harness runtime (spec §8.1, §8.2, §8.6, §9).

- Locates the pinned source clone (``MYFLOWS_DSH_ROOT`` or ``<repo>/../dsh-myflows``), checks its commit,
  build output and Node.js version.
- Keeps two AutoPilot-only profiles under ``runs/autopilot/dsh-home`` (never ``~/.dsh``): each lists
  dsh-base, its mode bundle and our ``@myflows/dsh-autopilot`` bundle, which is linked (junction/symlink)
  into the profile's ``node_modules`` from ``apps/autopilot/dsh/bundle``; each profile's own patch layer is
  rewritten from ``bundle/profiles/*.patch.yml`` on every start.
- Checks the composed trees (``--dump-config``) of both profiles: our rows present, no tool row that could
  reach the shell, files, terminal or network, and no skipped bundle or dangling patch.
- Runs the Web UI (``myflows-autopilot-web``) on 127.0.0.1:8792 for the assistant page and the headless
  one-shot runs of the Planner advisor. Only processes this host started are ever stopped.

The model credential (a DeepSeek API key or a DeepSeek account login) lives only in the AutoPilot dsh home's
credential store, written by the dsh web page.
Ambient ``DEEPSEEK_*``/``DSH_*`` variables are scrubbed from every dsh process, so neither the person's own
dsh configuration nor a key in the console's environment leaks in.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

EXPECTED_COMMIT = "687f282e86"
MIN_NODE = (22, 19)
DEFAULT_PORT = 8792
BUNDLE_NAME = "@myflows/dsh-autopilot"
BUNDLE_DIR = Path(__file__).with_name("bundle")
TOOLS_SERVER = Path(__file__).with_name("tools_server.py")
PROFILES = {"web": "myflows-autopilot-web", "headless": "myflows-autopilot-headless"}
MODE_BUNDLES = {"web": "@deepseek-ai/dsh-web-app", "headless": "@deepseek-ai/dsh-headless"}
GUARD_PACKAGE = "@myflows/dsh-autopilot/guard"
MCP_PACKAGE = "@deepseek-ai/dsh-mcp-client"
# The DeepSeek account login dsh stores after "登录" (credentialKey("deepseek-account-platform", "default")).
ACCOUNT_RECORD = "deepseek-account-platform/default"
# Headless runs default to the API-key route (deepseek-official). With only an account login they are switched to
# the account route, which never falls back to a key; deepseek-flash is in that route's built-in catalog.
ACCOUNT_ROUTE_PATCH = """# Written by MyFlows AutoPilot: headless runs use the DeepSeek account login (no API key stored).
- id: agent-default-model
  config:
    provider: deepseek-account
    model: deepseek-flash
"""

# Packages whose rows must not be mounted: model tools beyond the AutoPilot MCP server, and the context rows
# that would read the workspace or the harness checkout.
_FORBIDDEN = re.compile(
    r"^@deepseek-ai/dsh-(tool-|agent-tool-|mcp-resources|skill|plan-mode|workflow-|ptc-|terminal|command-goal|"
    r"goal|subagent|agent-instructions|web-search|web-fetch|web-deepseek|harness-dev-context|"
    r"api-terminal-controller|client-ui-sidebar-terminal)")
# Same prefixes, but services or settings pages rather than tools.
_ALLOWED = {"@deepseek-ai/dsh-tool-call-timeout-policy", "@deepseek-ai/dsh-tool-cordis/host",
            "@deepseek-ai/dsh-tool-subagent/model-selection-settings"}
# What the Web profile's only agent preset may mount.
_PRESET_PACKAGES = {"cordis:group", "@deepseek-ai/dsh-compaction-basic", "@deepseek-ai/dsh-command-compact",
                    "@deepseek-ai/dsh-persona"}
_WARNING = re.compile(r"not found|skipping|skipped|incompatible|cannot resolve|error", re.IGNORECASE)
_SCRUB = re.compile(r"^(DEEPSEEK_|DSH_|ANTHROPIC_|OPENAI_|NODE_OPTIONS$)", re.IGNORECASE)
_URL_LINE = re.compile(r"^dsh web: (https?://\S+)")
WEB_LOG_LINES = 200


def _sha(paths: list[Path], extra: str = "") -> str:
    h = hashlib.sha256(extra.encode())
    for p in sorted(paths):
        h.update(str(p.relative_to(BUNDLE_DIR)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _yaml_rows(text: str) -> list:
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("tag:yaml.org,2002:js", lambda loader, node: {"$js": loader.construct_scalar(node)})
    return yaml.load(text, Loader=Loader) or []


def _walk(rows, path=()):
    """Effective rows: an explicitly disabled row hides its subtree; a ``!!js`` switch counts as enabled."""
    for row in rows or []:
        if not isinstance(row, dict) or "id" not in row or row.get("disabled") is True:
            continue
        yield path + (str(row["id"]),), row
        config = row.get("config")
        if isinstance(config, list):
            yield from _walk(config, path + (str(row["id"]),))
        elif isinstance(config, dict) and isinstance(config.get("plugins"), list):
            yield from _walk(config["plugins"], path + (str(row["id"]),))


def check_composition(dump: str, mode: str, stderr: str = "") -> list[str]:
    """Problems in one profile's composed tree (``--dump-config`` output); empty when it is as designed."""
    errors = [f"警告：{line.strip()}" for line in stderr.splitlines() if line.strip() and _WARNING.search(line)]
    try:
        rows = _yaml_rows(dump)
    except Exception as exc:
        return errors + [f"无法解析组合导出：{type(exc).__name__}: {exc}"]
    effective = list(_walk(rows))
    by_id = {path: row for path, row in effective}
    mcp = by_id.get(("mcp-autopilot",))
    if mcp is None or mcp.get("name") != MCP_PACKAGE:
        errors.append("缺少 mcp-autopilot 行（AutoPilot 补丁层没有生效）")
    elif (mcp.get("config") or {}).get("serverName") != "autopilot":
        errors.append("mcp-autopilot 的 serverName 不是 autopilot")
    guard = by_id.get(("autopilot-guard",))
    if guard is None or guard.get("name") != GUARD_PACKAGE:
        errors.append("缺少 autopilot-guard 守卫行")
    elif (guard.get("config") or {}).get("profile") != mode:
        errors.append(f"守卫的 profile 不是 {mode}")
    tools = by_id.get(("tools",))
    if tools is not None and (tools.get("config") or {}).get("mode") not in (None, "native"):
        errors.append("tools 不是 native 模式")
    for path, row in effective:
        name = str(row.get("name") or "")
        if _FORBIDDEN.match(name) and name not in _ALLOWED:
            errors.append(f"组合中仍有被禁止的行：{'/'.join(path)}（{name}）")
        if len(path) > 1 and path[0].startswith("preset-") and name not in _PRESET_PACKAGES:
            errors.append(f"agent 预设中有未允许的插件：{'/'.join(path)}（{name}）")
    presets = [path[0] for path in by_id if len(path) == 1 and path[0].startswith("preset-")]
    if mode == "web" and presets != ["preset-standard"]:
        errors.append(f"Web profile 应只启用 preset-standard，实际为 {presets or '无'}")
    return errors


def _node_version(node: str) -> tuple[int, ...] | None:
    try:
        out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.match(r"v(\d+)\.(\d+)\.(\d+)", out)
    return tuple(int(x) for x in m.groups()) if m else None


def _git_head(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _is_junction(path: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        return os.lstat(path).st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT
    except (OSError, AttributeError):
        return False


def _link(target: Path, link: Path) -> None:
    """Point ``link`` at ``target`` (a junction on Windows, which needs no privilege; a symlink elsewhere)."""
    target = target.resolve()
    if os.path.lexists(link):
        if os.path.realpath(link) == str(target):
            return
        if link.is_symlink():
            link.unlink()
        elif _is_junction(link):
            os.rmdir(link)  # removes the junction itself, never the directory it points to
        else:
            shutil.rmtree(link)  # a plain copy inside our own dsh home
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target.resolve()), str(link))
    else:
        link.symlink_to(target.resolve(), target_is_directory=True)


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _kill_tree(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    from apps.console.jobs import kill_tree

    try:
        kill_tree(proc.pid, timeout=5.0)
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


class DshHost:
    def __init__(self, *, repo_root: Path, autopilot_root: Path, python: str | None = None,
                 dsh_root: Path | None = None, port: int = DEFAULT_PORT, node: str = "node",
                 launcher: list[str] | None = None, extra_env: dict | None = None,
                 web_ready_timeout_s: float = 120.0, headless_timeout_s: float = 300.0):
        self.repo_root = Path(repo_root)
        self.autopilot_root = Path(autopilot_root)
        self.python = python or sys.executable
        env_root = os.environ.get("MYFLOWS_DSH_ROOT")
        self.dsh_root = Path(dsh_root or env_root or self.repo_root.parent / "dsh-myflows")
        self.port = int(port)
        self.node = node
        # Tests replace ``node apps/cli/lib/bin.js`` with a fake (tests/system/fixtures/fake_dsh.py).
        self.launcher = list(launcher) if launcher else None
        self.extra_env = dict(extra_env or {})
        self.web_ready_timeout_s = float(web_ready_timeout_s)
        self.headless_timeout_s = float(headless_timeout_s)
        self.home = (self.autopilot_root / "dsh-home").resolve()
        self.workspace = (self.autopilot_root / "dsh-workspace").resolve()
        self.token = secrets.token_urlsafe(32)
        self.console_url: str | None = None
        self.context_campaign: str | None = None
        self._lock = threading.RLock()
        self._prepared = threading.Event()
        self._preparing = False
        self._clone: dict = {}
        self._composition: dict = {"ok": False, "errors": [], "checked": False}
        self._reason: str | None = "正在检查 dsh 运行时"
        self._web: subprocess.Popen | None = None
        self._web_state = "stopped"
        self._web_url: str | None = None
        self._web_error: str | None = None
        self._web_log: deque[str] = deque(maxlen=WEB_LOG_LINES)
        self._web_stopping = False
        self._headless: set[subprocess.Popen] = set()
        self._closing = False

    # ---- lifecycle ------------------------------------------------------------------------------------
    def start(self, console_url: str | None = None) -> None:
        """Remember the console URL and check the runtime in the background (the Web UI starts lazily)."""
        if console_url:
            self.console_url = console_url
        with self._lock:
            if self._preparing or self._prepared.is_set():
                return
            self._preparing = True
        threading.Thread(target=self.prepare, name="dsh-prepare", daemon=True).start()

    def shutdown(self) -> None:
        self._closing = True
        self.stop_web()
        for proc in list(self._headless):
            _kill_tree(proc)

    def prepare(self, force: bool = False) -> bool:
        """Check the clone, write both profiles and check their composition; the result is cached."""
        with self._lock:
            try:
                self._reason = None
                self._clone = self._check_clone()
                if self._clone.get("error"):
                    self._reason = self._clone["error"]
                    return False
                self.ensure_profiles()
                self._composition = self._check_compositions(force=force)
                if not self._composition["ok"]:
                    self._reason = "组合检查失败：" + "；".join(self._composition["errors"][:3])
                    return False
                return True
            except Exception as exc:  # any failure turns the LLM layer off, never the console
                self._reason = f"dsh 检查出错：{type(exc).__name__}: {exc}"
                return False
            finally:
                self._preparing = False
                self._prepared.set()

    @property
    def available(self) -> bool:
        return self._prepared.is_set() and self._reason is None

    # ---- clone and profiles -----------------------------------------------------------------------------
    def command(self) -> list[str]:
        if self.launcher:
            return list(self.launcher)
        return [self.node, str(self.dsh_root / "apps" / "cli" / "lib" / "bin.js")]

    def _check_clone(self) -> dict:
        if self.launcher:
            return {"path": " ".join(self.launcher), "commit": None, "expected": EXPECTED_COMMIT, "built": True,
                    "node": None, "fake": True}
        info = {"path": str(self.dsh_root), "commit": None, "expected": EXPECTED_COMMIT, "built": False, "node": None}
        if not (self.dsh_root / "package.json").exists():
            info["error"] = f"找不到 dsh 克隆：{self.dsh_root}（可用 MYFLOWS_DSH_ROOT 指定）"
            return info
        info["commit"] = _git_head(self.dsh_root)
        if not info["commit"] or not info["commit"].startswith(EXPECTED_COMMIT):
            info["error"] = f"dsh 克隆版本不符：{(info['commit'] or '未知')[:10]}，需要 {EXPECTED_COMMIT}"
            return info
        info["built"] = (self.dsh_root / "apps" / "cli" / "lib" / "bin.js").exists()
        if not info["built"]:
            info["error"] = "dsh 克隆缺少构建产物 apps/cli/lib/bin.js，请在克隆中执行构建"
            return info
        version = _node_version(self.node)
        info["node"] = ".".join(map(str, version)) if version else None
        if version is None or version[:2] < MIN_NODE:
            info["error"] = f"需要 Node.js ≥ {MIN_NODE[0]}.{MIN_NODE[1]}，当前为 {info['node'] or '未找到'}"
        return info

    def profile_dir(self, mode: str) -> Path:
        return self.home / "profiles" / PROFILES[mode]

    def ensure_profiles(self) -> None:
        """Write both AutoPilot profiles; existing ones are brought back to the template on every start."""
        self.home.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        for mode in PROFILES:
            d = self.profile_dir(mode)
            d.mkdir(parents=True, exist_ok=True)
            manifest = {"name": f"dsh-profile-{PROFILES[mode]}", "private": True, "dependencies": {},
                        "dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-base", MODE_BUNDLES[mode], BUNDLE_NAME]}}}
            self._write_if_changed(d / "package.json", json.dumps(manifest, indent=2) + "\n")
            template = (BUNDLE_DIR / "profiles" / f"{mode}.patch.yml").read_text(encoding="utf-8")
            header = "# Generated by MyFlows DshHost from apps/autopilot/dsh/bundle/profiles; edits are overwritten.\n"
            self._write_if_changed(d / "cordis.patch.yml", header + template)
            _link(BUNDLE_DIR, d / "node_modules" / "@myflows" / "dsh-autopilot")

    @staticmethod
    def _write_if_changed(path: Path, text: str) -> None:
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")

    def _composition_key(self) -> str:
        files = [p for p in BUNDLE_DIR.rglob("*") if p.is_file()]
        return _sha(files, extra=f"{self._clone.get('commit')}|{self.command()}|{Path(__file__).read_bytes()!r}")

    def _check_compositions(self, force: bool = False) -> dict:
        cache = self.home / "composition.json"
        key = self._composition_key()
        if not force and cache.exists():
            try:
                cached = json.loads(cache.read_text(encoding="utf-8"))
                if cached.get("key") == key and cached.get("ok"):
                    return {**cached, "cached": True}
            except ValueError:
                pass
        errors: list[str] = []
        for mode in PROFILES:
            try:
                out = subprocess.run([*self.command(), "--profile", PROFILES[mode], "--dump-config"],
                                     cwd=self.workspace, env=self.env(source=f"llm:{mode}"), capture_output=True,
                                     timeout=180, encoding="utf-8", errors="replace")
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f"{PROFILES[mode]}：无法导出组合（{exc}）")
                continue
            if out.returncode != 0:
                errors.append(f"{PROFILES[mode]}：导出组合失败（退出码 {out.returncode}）{out.stderr.strip()[-300:]}")
                continue
            errors += [f"{PROFILES[mode]}：{e}" for e in check_composition(out.stdout, mode, out.stderr)]
        result = {"key": key, "ok": not errors, "errors": errors, "checked": True, "checked_unix_s": time.time()}
        cache.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        return result

    # ---- environment and key -------------------------------------------------------------------------------
    def env(self, *, source: str = "llm:web", campaign: str | None = None) -> dict:
        env = {k: v for k, v in os.environ.items() if not _SCRUB.match(k)}
        env.update({
            "DSH_HOME": str(self.home), "DSH_TELEMETRY_DISABLED": "1", "NO_COLOR": "1",
            "MYFLOWS_ROOT": str(self.repo_root), "MYFLOWS_PYTHON": self.python,
            "MYFLOWS_AUTOPILOT_SERVER": str(TOOLS_SERVER), "MYFLOWS_CONSOLE_URL": self.console_url or "",
            "MYFLOWS_AUTOPILOT_TOKEN": self.token, "MYFLOWS_AUTOPILOT_SOURCE": source,
            "MYFLOWS_AUTOPILOT_CAMPAIGN": campaign or "", "PYTHONIOENCODING": "utf-8",
        })
        env.update(self.extra_env)
        return env

    def credentials(self) -> dict:
        """Which DeepSeek credentials the AutoPilot dsh home holds; only presence is checked, values are never read out.

        ``api_key``: the ``DEEPSEEK_API_KEY`` reference (dsh "添加 API Key"), the environment or ``.env``;
        ``account``: a DeepSeek account login (dsh "登录"), the ``deepseek-account-platform/default`` grant
        record, which dsh uses for inference when no key is set.
        """
        api_key = bool(self.extra_env.get("DEEPSEEK_API_KEY"))
        account = False
        path = self.home / ".credentials.yaml"
        if path.exists():
            try:
                import yaml

                document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                api_key = api_key or bool((document.get("refs") or {}).get("DEEPSEEK_API_KEY"))
                record = (document.get("records") or {}).get(ACCOUNT_RECORD) or {}
                account = record.get("kind") == "grant" and bool((record.get("payload") or {}).get("token"))
            except Exception:
                pass
        env_file = self.home / ".env"
        if not api_key and env_file.exists():
            api_key = any(line.split("=", 1)[0].strip() == "DEEPSEEK_API_KEY" and line.split("=", 1)[1].strip()
                          for line in env_file.read_text(encoding="utf-8").splitlines() if "=" in line)
        return {"api_key": api_key, "account": account}

    def key_configured(self) -> bool:
        """Whether dsh can call the model: a DeepSeek API key or a DeepSeek account login is stored."""
        found = self.credentials()
        return found["api_key"] or found["account"]

    # ---- status -----------------------------------------------------------------------------------------------
    def status(self, ensure_web: bool = False) -> dict:
        if not self._prepared.is_set():
            if not self._preparing:
                self.start()
            return {"available": False, "llm_ready": False, "checking": True, "reason": "正在检查 dsh 运行时…",
                    "web": {"state": "stopped", "url": None}}
        if ensure_web and self.available and self._web_state in ("stopped", "exited", "failed") \
                and not self._web_error_sticky():
            self.start_web()
        found = self.credentials() if self.available else {"api_key": False, "account": False}
        key = found["api_key"] or found["account"]
        return {
            "available": self.available, "llm_ready": self.available and key, "reason": self._reason,
            "key_configured": key, "credentials": found, "clone": self._clone,
            "composition": {"ok": self._composition.get("ok", False), "errors": self._composition.get("errors", []),
                            "cached": self._composition.get("cached", False)},
            "web": {"state": self._web_state, "url": self._web_url if self._web_state == "running" else None,
                    "pid": self._web.pid if self._web and self._web.poll() is None else None,
                    "error": self._web_error, "port": self.port},
            "profiles": PROFILES, "home": str(self.home), "context_campaign": self.context_campaign,
        }

    def _web_error_sticky(self) -> bool:
        # After a failed start the page shows the error and a restart button instead of retrying every poll.
        return self._web_state == "failed"

    # ---- the Web UI -------------------------------------------------------------------------------------------
    def start_web(self) -> dict:
        with self._lock:
            if not self.available:
                return {"state": "unavailable", "error": self._reason}
            if self._web is not None and self._web.poll() is None:
                return {"state": self._web_state}
            self._stop_leftover_web()
            if _port_open(self.port):
                self._web_state, self._web_error = "failed", f"端口 {self.port} 已被其他进程占用（不是本控制台启动的）"
                return {"state": self._web_state, "error": self._web_error}
            cmd = [*self.command(), "--profile", PROFILES["web"], *self._route_patch(), "--host", "127.0.0.1",
                   "--port", str(self.port), "--no-open"]
            self._web_log.clear()
            self._web_url, self._web_error, self._web_stopping = None, None, False
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            try:
                proc = subprocess.Popen(cmd, cwd=self.workspace, env=self.env(source="llm:web"),
                                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        encoding="utf-8", errors="replace", **kwargs)
            except OSError as exc:
                self._web_state, self._web_error = "failed", f"无法启动 dsh 网页：{exc}"
                return {"state": self._web_state, "error": self._web_error}
            self._web, self._web_state = proc, "starting"
            self._record_web_pid(proc.pid)
            threading.Thread(target=self._pump_web, args=(proc,), name="dsh-web-log", daemon=True).start()
            threading.Thread(target=self._watch_web, args=(proc,), name="dsh-web-watch", daemon=True).start()
            return {"state": self._web_state}

    def _record_web_pid(self, pid: int) -> None:
        import psutil

        try:
            created = psutil.Process(pid).create_time()
        except psutil.Error:
            return
        (self.home / "web.pid.json").write_text(json.dumps({"pid": pid, "create_time": created}), encoding="utf-8")

    def _stop_leftover_web(self) -> None:
        """A Web process a previous console started (and did not stop, e.g. it was killed) holds the port.
        It is stopped only when its pid and creation time match what that console recorded."""
        import psutil

        path = self.home / "web.pid.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            proc = psutil.Process(int(record["pid"]))
            if abs(proc.create_time() - float(record["create_time"])) > 0.01:
                return
        except (OSError, ValueError, KeyError, psutil.Error):
            return
        if self._web is not None and self._web.pid == proc.pid:
            return
        from apps.console.jobs import kill_tree

        kill_tree(proc.pid, timeout=5.0)
        path.unlink(missing_ok=True)

    def _pump_web(self, proc: subprocess.Popen) -> None:
        log_path = self.home / "web.log"
        with open(log_path, "w", encoding="utf-8") as log, proc.stdout:
            for line in proc.stdout:
                m = _URL_LINE.match(line.strip())
                if m and proc is self._web:
                    # The URL carries the Web UI's launch token: kept in memory for the iframe, never logged.
                    self._web_url = m.group(1)
                    self._web_state = "running"
                    line = "dsh web: ready (launch URL withheld)\n"
                self._web_log.append(line.rstrip())
                log.write(line)
                log.flush()

    def _watch_web(self, proc: subprocess.Popen) -> None:
        deadline = time.time() + self.web_ready_timeout_s
        while proc.poll() is None:
            if self._web_state == "starting" and time.time() > deadline and proc is self._web:
                self._web_error = f"dsh 网页 {self.web_ready_timeout_s:.0f} 秒内没有就绪"
                _kill_tree(proc)
                self._web_state = "failed"
                return
            time.sleep(0.2)
        if proc is not self._web:
            return
        if self._web_stopping or self._closing:
            self._web_state = "stopped"
            return
        tail = "\n".join(list(self._web_log)[-8:])
        self._web_state = "exited" if self._web_state == "running" else "failed"
        self._web_error = f"dsh 网页进程已退出（退出码 {proc.returncode}）" + (f"：{tail[-600:]}" if tail else "")

    def stop_web(self) -> None:
        with self._lock:
            proc = self._web
            self._web_stopping = True
        _kill_tree(proc)
        with self._lock:
            if proc is self._web:
                self._web_state, self._web_url = "stopped", None

    def restart_web(self) -> dict:
        if not self.available:
            self.prepare(force=True)
        self.stop_web()
        self._web_state, self._web_error = "stopped", None
        self.start_web()
        return self.status()

    # ---- headless one-shot runs -----------------------------------------------------------------------------
    def run_headless(self, prompt: str, *, campaign: str | None = None, session_id: str | None = None,
                     timeout_s: float | None = None) -> dict:
        """One ``myflows-autopilot-headless --json`` run; the transcript is summarized, never stored raw."""
        timeout_s = float(timeout_s or self.headless_timeout_s)
        started = time.time()
        out = {"ok": False, "session_id": session_id, "text": "", "tool_calls": [], "usage": {}, "error": None,
               "returncode": None, "duration_s": None}
        if not self.available:
            out["error"] = self._reason or "dsh 不可用"
            return out
        cmd = [*self.command(), "--profile", PROFILES["headless"], *self._route_patch(), "--json"]
        if session_id:
            cmd += ["--session-id", session_id]
        cmd.append(prompt)
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        try:
            proc = subprocess.Popen(cmd, cwd=self.workspace, env=self.env(source="llm:headless", campaign=campaign),
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    encoding="utf-8", errors="replace", **kwargs)
        except OSError as exc:
            out["error"] = f"无法启动 dsh：{exc}"
            return out
        self._headless.add(proc)
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            stdout, stderr = proc.communicate()
            out["error"] = f"超时（{timeout_s:.0f} 秒）"
        finally:
            self._headless.discard(proc)
        out["returncode"] = proc.returncode
        out["duration_s"] = round(time.time() - started, 2)
        self._parse_transcript(stdout, out)
        if out["error"] is None and proc.returncode != 0:
            tail = (stderr or "").strip().splitlines()[-3:]
            out["error"] = out.get("stream_error") or f"dsh 退出码 {proc.returncode}" + (f"：{' / '.join(tail)}" if tail else "")
        out["ok"] = out["error"] is None
        return out

    def _route_patch(self) -> list[str]:
        """``--patch`` overlay that moves the default model to the account route when only a login is stored, for
        headless runs and for new chats in the Web page (which otherwise start with no model selected)."""
        found = self.credentials()
        if found["api_key"] or not found["account"]:
            return []
        path = self.home / "account-route.patch.yml"
        self._write_if_changed(path, ACCOUNT_ROUTE_PATCH)
        return ["--patch", str(path)]

    @staticmethod
    def _parse_transcript(stdout: str, out: dict) -> None:
        calls: dict[str, dict] = {}
        usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "steps": 0, "complete": True}
        texts = []
        for line in (stdout or "").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("type")
            if kind == "session":
                out["session_id"] = event.get("sessionId") or out["session_id"]
            elif kind == "tool_call":
                calls[event.get("callId")] = {"tool": event.get("tool"), "status": "called"}
            elif kind == "tool_result":
                item = calls.setdefault(event.get("callId"), {"tool": None})
                item["status"] = event.get("status")
                if event.get("status") == "error":
                    item["error"] = str(event.get("result") or "")[:300]
            elif kind == "text":
                texts.append(str(event.get("text") or ""))
            elif kind == "final":
                out["text"] = str(event.get("text") or "")
            elif kind == "error":
                out["stream_error"] = str(event.get("message") or "")[:500]
            elif kind == "status" and event.get("phase") == "step_end":
                usage["steps"] += 1
                u = event.get("usage")
                if isinstance(u, dict):
                    usage["input_tokens"] += int(u.get("inputTokens") or 0)
                    usage["output_tokens"] += int(u.get("outputTokens") or 0)
                    usage["reasoning_tokens"] += int(u.get("reasoningTokens") or 0)
                else:
                    usage["complete"] = False
        if not out["text"] and texts:
            out["text"] = texts[-1]
        out["tool_calls"] = list(calls.values())
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        out["usage"] = usage

    # ---- the advisor ------------------------------------------------------------------------------------------
    def advisor(self, service):
        from ..strategies.llm_advisor import LLMAdvisor

        return LLMAdvisor(self)
