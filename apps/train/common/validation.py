# -*- coding: utf-8 -*-
"""Generic validation-loop helpers for fixed-batch MyFlows training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from MyFlows.data.pipeline import MultiprocessDataLoader


@dataclass
class ValidationResult:
    mean_loss: float
    batches: int
    samples: int


def run_loss_validation(
    dataset: Sequence,
    batch_size: int,
    *,
    load_fn: Callable,
    step_fn: Callable,
    num_workers: int = 0,
) -> ValidationResult | None:
    """Iterate validation data and let ``step_fn`` perform model-specific forward."""

    if not dataset:
        return None
    loader = MultiprocessDataLoader(
        dataset,
        int(batch_size),
        num_workers=int(num_workers),
        shuffle=False,
        load_fn=load_fn,
    )
    total_loss = 0.0
    batches = 0
    samples = 0
    for x_items, y_items in loader:
        loss_value, real_count = step_fn(x_items, y_items)
        total_loss += float(loss_value)
        batches += 1
        samples += int(real_count)
    if batches == 0:
        return None
    return ValidationResult(total_loss / batches, batches, samples)
