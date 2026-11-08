"""Estimate target similarity with a frozen-feature linear probe inside the training split."""

import numpy as np

from MyFlows.core.device import asnumpy
from MyFlows.core.graph import Graph
from apps.common.splits import build_time_block_split, frame_numbers


def probe_indices(samples):
    frames = frame_numbers([row[0] for row in samples])
    if len(frames) < 16:
        return [], []
    groups = build_time_block_split(frames, val_size=len(frames)//4, block_size=min(16, len(frames)//8),
                                    gap=2, seed=917)
    rng = np.random.default_rng(917)
    return [sorted(rng.choice(groups[key], min(limit, len(groups[key])), replace=False).tolist())
            for key, limit in (("train", 96), ("val", 32))]


def linear_probe(model, input_node, samples, load):
    fit, holdout = probe_indices(samples)
    info = {"train_samples": len(samples), "fit_indices": fit, "holdout_indices": holdout,
            "source": "train", "metric": "1 - probe_mae / median_predictor_mae"}
    if min(len(fit), len(holdout)) < 4:
        return {**info, "score": None, "reason": "too few disjoint training frames"}
    feature = model._last_transfer_features
    graph = Graph(feature)
    batch = input_node.value.shape[0]
    original_input = input_node.value
    selected = fit + holdout
    values = []
    model.eval()
    try:
        for start in range(0, len(selected), batch):
            rows = [load(samples[i])[0] for i in selected[start:start+batch]]
            input_node.value = np.stack(rows + [rows[-1]] * (batch-len(rows)))
            graph.forward()
            values.append(asnumpy(feature.value)[:len(rows)].copy())
    finally:
        input_node.value = original_input
        model.train(True)
    features = np.concatenate(values).astype(np.float64)
    targets = np.asarray([samples[i][1] for i in selected], np.float64)
    n = len(fit)
    centered = features - features[:n].mean(axis=0)
    kernel = centered[:n] @ centered[:n].T
    penalty = max(float(np.trace(kernel))/n*.01, 1e-12)
    kernel.flat[::n+1] += penalty
    y = targets[:n]
    dual = np.linalg.solve(kernel, y-y.mean())
    estimate = (centered[n:] @ centered[:n].T) @ dual + y.mean()
    error = float(np.mean(np.abs(estimate - targets[n:])))
    baseline = float(np.mean(np.abs(np.median(y) - targets[n:])))
    return {**info, "ridge": penalty, "probe_mae": error, "constant_mae": baseline,
            "score": 1-error/baseline if baseline > 0 and np.isfinite(error) else None}
