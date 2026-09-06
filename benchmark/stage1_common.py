"""Reproducible artifacts shared by the stage-one command-line experiments."""

import contextlib
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(*arrays):
    sha = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        sha.update(str((array.shape, array.dtype.str)).encode())
        sha.update(array.tobytes())
    return sha.hexdigest()


def command(args, cwd=ROOT):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30)
    if result.returncode:
        raise RuntimeError(f"{args}: {result.stderr}")
    return result.stdout


def source_state():
    result = {}
    for name, directory in (("root", ROOT), ("MyFlows", ROOT / "MyFlows")):
        if not (directory / ".git").exists():
            origin_path = ROOT / ".stage1-source-origin.json"
            if not origin_path.exists():
                raise RuntimeError(f"{directory} has neither Git metadata nor a restored source manifest")
            origin = json.loads(origin_path.read_text(encoding="utf-8"))[name]
            candidates = set(origin["files_sha256"])
            for parent, directories, filenames in os.walk(directory):
                directories[:] = [d for d in directories if d not in
                                  (".git", ".venv", ".codex", ".agents", "__pycache__", "experiments", "DonkeySimWin")
                                  and not (name == "root" and Path(parent) == directory and d == "MyFlows")]
                for filename in filenames:
                    p = Path(parent) / filename
                    if p.suffix in (".py", ".cu", ".md", ".txt", ".pdf"):
                        candidates.add(p.relative_to(directory).as_posix())
            files = {}
            for p in sorted(candidates):
                path = (directory / p).resolve()
                if not path.is_relative_to(directory.resolve()):
                    raise ValueError("restored source manifest path escapes its repository")
                if path.is_file():
                    files[p] = digest(path)
            result[name] = {"head": None, "origin_head": origin.get("head") or origin.get("origin_head"),
                            "status": "restored source export without Git metadata", "diff": None,
                            "origin_status": origin["status"], "files_sha256": files,
                            "untracked": sorted(set(files) - set(origin["files_sha256"]))}
            continue
        paths = command(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], directory)
        files = {}
        for relative in sorted(set(paths.split("\0")) - {""}):
            path = directory / relative
            if path.is_file() and not relative.startswith("docs/experiments/"):
                files[relative] = digest(path)
        result[name] = {
            "head": command(["git", "rev-parse", "HEAD"], directory).strip(),
            "status": command(["git", "status", "--short"], directory),
            "diff": command(["git", "diff", "--binary", "HEAD"], directory),
            "untracked": command(["git", "ls-files", "--others", "--exclude-standard"], directory).splitlines(),
            "files_sha256": files,
        }
    return result


def environment():
    packages = {}
    for name in ("numpy", "cupy-cuda12x", "torch", "scikit-learn", "onnx", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        gpu = command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,temperature.gpu,power.draw", "--format=csv"])
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        gpu = str(exc)
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": packages, "nvidia_smi": gpu,
            "installed_distributions": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
            "venv_config": cfg.read_text() if cfg.exists() else None,
            "thread_environment": {k: os.environ.get(k) for k in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class RunArtifacts:
    def __init__(self, out_dir, config):
        self.path = Path(out_dir).resolve()
        self.config = config
        self.results = {}
        self.timings = []
        self.notes = []
        self.manifest = {}

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.stack = contextlib.ExitStack()
        log = self.stack.enter_context((self.path / "stdout.log").open("w", encoding="utf-8"))
        self.stack.enter_context(contextlib.redirect_stdout(_Tee(sys.stdout, log)))
        self.stack.enter_context(contextlib.redirect_stderr(_Tee(sys.stderr, log)))
        try:
            write_json(self.path / "config.json", self.config)
            self.manifest = {"schema_version": 1, "run_id": self.path.name,
                             "command": [sys.executable, *sys.argv],
                             "environment": environment(), "source": source_state()}
            print(f"run_id={self.path.name}", flush=True)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc is not None:
                traceback.print_exception(exc_type, exc, tb)
            self.results.update(status="failed" if exc else "passed",
                                elapsed_s=time.perf_counter() - self.started)
            if exc is not None:
                self.results["error"] = f"{exc_type.__name__}: {exc}"
            write_json(self.path / "results.json", self.results)
            fields = sorted({key for row in self.timings for key in row}) or ["case", "status"]
            with (self.path / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.timings)
            (self.path / "report.md").write_text(
                f"# {self.path.name}\n\nStatus: {self.results['status']}\n\n" +
                "\n\n".join(self.notes) + "\n\nSee config.json, results.json and timings.csv for raw evidence.\n",
                encoding="utf-8")
            print(f"status={self.results['status']}", flush=True)
        finally:
            self.stack.close()
        self.manifest["artifacts_sha256"] = {
            str(p.relative_to(self.path)): digest(p) for p in sorted(self.path.rglob("*"))
            if p.is_file() and p.name != "manifest.json"}
        write_json(self.path / "manifest.json", self.manifest)
        return False


def error_metrics(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {"max_abs": float(diff.max()),
            "max_rel": float((diff / np.maximum(np.abs(expected), 1e-6)).max()),
            "allclose": bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-3))}
