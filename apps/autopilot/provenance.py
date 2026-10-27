"""Code, data and split fingerprints: a trial's identity is sha256(config, data, split, code) (spec §6.2)."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

CODE_SUFFIXES = (".py", ".cu", ".cuh", ".cpp", ".h", ".proto", ".js", ".mjs")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: str | Path) -> str | None:
    try:
        return sha256_bytes(Path(path).read_bytes())
    except OSError:
        return None


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def fingerprint(resolved_config: dict, data_hash: str | None, split_hash: str | None, code_hash: str | None) -> str:
    payload = {"config": resolved_config, "data": data_hash, "split": split_hash, "code": code_hash}
    return sha256_bytes(canonical(payload).encode("utf-8"))


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    return result.stdout.decode("utf-8", errors="replace")


def _repo_state(directory: Path) -> dict:
    """HEAD plus a hash of uncommitted code changes (tracked diff and untracked code files)."""
    head = _git(["rev-parse", "HEAD"], directory)
    if head is None:
        return {"head": None, "dirty_hash": None, "dirty_files": []}
    patterns = [f"*{suffix}" for suffix in CODE_SUFFIXES]
    diff = _git(["diff", "HEAD", "--", *patterns], directory) or ""
    untracked = [p for p in (_git(["ls-files", "--others", "--exclude-standard", "--", *patterns], directory)
                             or "").splitlines() if p and not p.startswith(("runs/", "docs/"))]
    digest = hashlib.sha256(diff.encode("utf-8"))
    for rel in sorted(untracked):
        digest.update(rel.encode("utf-8"))
        digest.update((file_hash(directory / rel) or "").encode("ascii"))
    changed = (_git(["diff", "HEAD", "--name-only", "--", *patterns], directory) or "").splitlines()
    dirty = sorted(set(changed) | set(untracked))
    return {"head": head.strip(), "dirty_hash": digest.hexdigest() if dirty else None, "dirty_files": dirty[:50]}


def code_state(repo_root: Path) -> dict:
    """Both repositories (root and the nested MyFlows) and one combined ``code_hash``."""
    repo_root = Path(repo_root)
    repos = {"root": _repo_state(repo_root), "MyFlows": _repo_state(repo_root / "MyFlows")}
    combined = canonical({name: [state["head"], state["dirty_hash"]] for name, state in repos.items()})
    return {"repos": repos, "code_hash": sha256_bytes(combined.encode("utf-8"))}


def data_hash(data_dir: Path, catalog: str = "catalog_generated.catalog") -> str | None:
    """The tub's manifest and catalog (labels and file list); the images themselves are not hashed."""
    data_dir = Path(data_dir)
    parts = [file_hash(data_dir / name) for name in ("manifest.json", catalog, f"{catalog}_manifest")]
    if not any(parts):
        images = sorted(p.name for p in (data_dir / "images").glob("*.jpg")) if (data_dir / "images").is_dir() else []
        if not images:
            return None
        return sha256_bytes("\n".join(images).encode("utf-8"))
    return sha256_bytes(canonical(parts).encode("utf-8"))


def environment() -> dict:
    packages = {}
    from importlib.metadata import PackageNotFoundError, version

    for name in ("numpy", "cupy-cuda12x", "optuna", "fastapi"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    gpu = None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=15, check=False)
        gpu = result.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages, "gpu": gpu}
