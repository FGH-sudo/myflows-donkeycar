"""Repository-relative file entries of distributed experiment configs.

Archived configs (frozen_task_configs.json, candidate_config.json) store data and BN statistics
paths relative to the repository root. Older archives recorded absolute Windows paths of one
checkout (``<checkout>\\.codex\\...``); those are re-rooted onto the current checkout when they
do not exist here, so the same archive works from another clone, from WSL or on Linux.
"""

import re
from pathlib import Path, PurePosixPath, PureWindowsPath

REPO_ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ('prepared_dir', 'bn_statistics', 'init_checkpoint')
# Top-level repository directories such entries point into.
_ANCHORS = ('.codex', 'docs', 'runs', 'mycar', 'build')


def repo_path(value):
    """Absolute path of a config file entry in this checkout."""
    text = str(value)
    native = Path(text)
    if native.is_absolute() and native.exists():
        return str(native)
    windows = '\\' in text or re.match(r'^[A-Za-z]:', text)
    parts = (PureWindowsPath(text) if windows else PurePosixPath(text)).parts
    anchor = next((i for i, part in enumerate(parts) if part in _ANCHORS), None)
    if anchor is not None:
        return str(REPO_ROOT.joinpath(*parts[anchor:]))
    return str(native if native.is_absolute() else REPO_ROOT / native)


def repo_relative(value):
    """POSIX path relative to the repository root, or the path unchanged when it lies outside."""
    try:
        return Path(repo_path(value)).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(value)


def resolve_config_paths(config):
    """Copy of ``config`` with its file entries as absolute paths of this checkout."""
    return {key: repo_path(value) if key in PATH_KEYS and value else value for key, value in config.items()}


def relative_config_paths(config):
    """Copy of ``config`` with its file entries relative to the repository root, for archiving."""
    return {key: repo_relative(value) if key in PATH_KEYS and value else value for key, value in config.items()}
