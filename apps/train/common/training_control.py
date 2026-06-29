# -*- coding: utf-8 -*-
"""Small training-control helpers shared by training entrypoints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStoppingResult:
    improved: bool
    should_stop: bool
    best: float
    bad_epochs: int


class EarlyStopping:
    """Patience-based early stopping for metrics where lower is better."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.bad_epochs = 0

    def update(self, value: float) -> EarlyStoppingResult:
        current = float(value)
        if self.best is None or current < self.best - self.min_delta:
            self.best = current
            self.bad_epochs = 0
            return EarlyStoppingResult(True, False, self.best, self.bad_epochs)
        self.bad_epochs += 1
        return EarlyStoppingResult(False, self.bad_epochs >= self.patience, self.best, self.bad_epochs)


class BestScore:
    """Track the best score for metrics where lower is better."""

    def __init__(self):
        self.best: float | None = None

    def update(self, value: float) -> bool:
        current = float(value)
        if self.best is None or current < self.best:
            self.best = current
            return True
        return False
