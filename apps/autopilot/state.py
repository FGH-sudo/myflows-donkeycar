"""TrainingStates of a trial (spec §12, E3): the campaign-side view of what apps.train saved.

The format itself (transactional writes, checksums, manifest) lives in ``MyFlows.utils.checkpoint``; this
module finds the verified states across a trial's attempts and removes them once the trial is decided.
A state that fails verification (missing file, checksum mismatch, half-written directory) is never listed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from MyFlows.utils.checkpoint import TrainingStateError, list_training_states, verify_training_state

STATES_DIR = "states"


def states_dir(attempt: dict) -> Path:
    return Path(attempt["run_dir"]) / STATES_DIR


def trial_states(trial: dict) -> list[dict]:
    """Every verified state of the trial, oldest first: ``{path, step, state, attempt}``."""
    out = []
    for attempt in trial.get("attempts") or []:
        for state in list_training_states(states_dir(attempt)):
            out.append({**state, "attempt": attempt["attempt"]})
    return sorted(out, key=lambda s: (s["step"], s["attempt"]))


def verify(path: str | Path) -> bool:
    """True when the state at ``path`` still matches its manifest (checked again right before a resume)."""
    try:
        verify_training_state(Path(path))
        return True
    except (TrainingStateError, OSError, ValueError):
        return False


def drop_states(trial: dict) -> int:
    """Delete the trial's states (they only serve rollbacks inside the trial); returns how many attempts had some."""
    dropped = 0
    for attempt in trial.get("attempts") or []:
        directory = states_dir(attempt)
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
            dropped += 1
    return dropped
