"""TPEStrategy (spec §6.1, E3): Optuna's TPE over the accuracy space, with the regularisation keys added.

Each ``ask`` rebuilds an in-memory study from the campaign's history (every valid trial, baseline included,
becomes a completed Optuna trial), so the strategy keeps no state of its own and survives console restarts.
The space is conditional: ``lr`` (log scale between the listed extremes) and ``batch`` always; one
regularisation branch — none, ``weight_decay`` or ``dropout`` (each only when listed in the space) — with its
own value only on that branch; ``augment`` independently when listed. Every suggestion lies inside the space
(the same range check as the pre-check) and never repeats a configuration already tried.
"""

from __future__ import annotations

import math

from ..agents.base import Proposal
from ..spec import ACCURACY_KEYS, TPE_KEYS, TRANSFER_KEYS, baseline_params

MAX_DRAWS = 64
REGULARIZERS = ("weight_decay", "dropout")


def _accuracy_view(params: dict) -> tuple:
    keys = (*ACCURACY_KEYS, *TPE_KEYS, *TRANSFER_KEYS)
    return tuple((k, params.get(k)) for k in keys if params.get(k) not in (None, 0, 0.0, False))


class TPEStrategy:
    name = "tpe"

    def __init__(self, spec: dict, *, knowledge=None):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.spec = spec
        self.knowledge = knowledge
        self.space = spec["space"]
        self.seed = int(spec["strategy"].get("seed", 0))
        self.regularizers = [k for k in REGULARIZERS if k in self.space]

    # ---- the search space ------------------------------------------------------------------------
    def _lr_bounds(self) -> tuple[float, float]:
        values = self.space["lr"]
        return float(min(values)), float(max(values))

    def _distributions(self) -> dict:
        from optuna.distributions import CategoricalDistribution, FloatDistribution

        low, high = self._lr_bounds()
        out = {"lr": FloatDistribution(low, high, log=True) if high > low else CategoricalDistribution([low]),
               "batch": CategoricalDistribution(sorted(int(b) for b in self.space["batch"]))}
        if self.regularizers:
            out["regularizer"] = CategoricalDistribution(["none", *self.regularizers])
            for key in self.regularizers:
                values = [float(v) for v in self.space[key] if float(v) > 0]
                if values:
                    lo, hi = min(values), max(values)
                    out[key] = FloatDistribution(lo, hi, log=True) if hi > lo else CategoricalDistribution([lo])
        if "augment" in self.space:
            out["augment"] = CategoricalDistribution(sorted({bool(v) for v in self.space["augment"]}))
        if "freeze_stages" in self.space:
            out["freeze_stages"] = CategoricalDistribution(sorted(int(v) for v in self.space["freeze_stages"]))
        return out

    def _suggest(self, trial) -> dict:
        low, high = self._lr_bounds()
        lr = trial.suggest_float("lr", low, high, log=True) if high > low else low
        params = {"lr": float(f"{lr:.3g}"), "batch": int(trial.suggest_categorical(
            "batch", sorted(int(b) for b in self.space["batch"])))}
        if self.regularizers:
            branch = trial.suggest_categorical("regularizer", ["none", *self.regularizers])
            if branch != "none":
                values = [float(v) for v in self.space[branch] if float(v) > 0]
                if values:
                    lo, hi = min(values), max(values)
                    value = trial.suggest_float(branch, lo, hi, log=True) if hi > lo else lo
                    params[branch] = float(f"{value:.3g}")
        if "augment" in self.space:
            params["augment"] = bool(trial.suggest_categorical("augment", sorted({bool(v) for v in self.space["augment"]})))
        if "freeze_stages" in self.space:
            params["freeze_stages"] = int(trial.suggest_categorical(
                "freeze_stages", sorted(int(v) for v in self.space["freeze_stages"])))
        return params

    def _history_params(self, params: dict) -> dict | None:
        """A finished trial as Optuna parameters, or None when it lies outside the space TPE models."""
        low, high = self._lr_bounds()
        lr, batch = params.get("lr"), params.get("batch")
        if lr is None or not (low * (1 - 1e-9) <= float(lr) <= high * (1 + 1e-9)) or batch not in self.space["batch"]:
            return None
        out = {"lr": min(max(float(lr), low), high), "batch": int(batch)}
        if self.regularizers:
            active = [k for k in self.regularizers if float(params.get(k) or 0) > 0]
            if len(active) > 1:
                return None
            out["regularizer"] = active[0] if active else "none"
            if active:
                values = [float(v) for v in self.space[active[0]] if float(v) > 0]
                value = float(params[active[0]])
                if not values or not (min(values) <= value <= max(values)):
                    return None
                out[active[0]] = value
        if "augment" in self.space:
            out["augment"] = bool(params.get("augment", False))
            if out["augment"] not in {bool(v) for v in self.space["augment"]}:
                return None
        if "freeze_stages" in self.space:
            out["freeze_stages"] = int(params.get("freeze_stages", 0))
            if out["freeze_stages"] not in {int(v) for v in self.space["freeze_stages"]}:
                return None
        return out

    # ---- ask ----------------------------------------------------------------------------------------
    def ask(self, ev) -> Proposal | None:
        import optuna

        distributions = self._distributions()
        import warnings

        with warnings.catch_warnings():  # multivariate + group (conditional spaces) are marked experimental
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            sampler = optuna.samplers.TPESampler(seed=self.seed + len(ev.trials), multivariate=True, group=True,
                                                 n_startup_trials=3)
        study = optuna.create_study(direction="minimize" if ev.minimize else "maximize", sampler=sampler)
        warm = 0
        for trial in ev.valid():
            params = self._history_params(trial.get("params") or {})
            if params is None:
                continue
            used = {k: distributions[k] for k in params}
            study.add_trial(optuna.trial.create_trial(params=params, distributions=used, value=ev.objective(trial)))
            warm += 1
        tried = {_accuracy_view(t.get("params") or {}) for t in ev.trials}
        tried.add(_accuracy_view(baseline_params(self.spec, self.spec["baseline"]["seeds"][0])))
        for _ in range(MAX_DRAWS):
            trial = study.ask()
            params = self._suggest(trial)
            if _accuracy_view(params) in tried or not self._inside(params):
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                continue
            # A branch not taken is simply absent: the baseline has no regularisation, apps.train defaults to 0.
            return Proposal(
                delta=dict(params), strategy=self.name, agent="planner", phase="explore", kind="accuracy",
                rationale=f"TPE（Optuna，{warm} 个历史 trial 预热）建议："
                          + "、".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()),
                expected=f"在已有结果的基础上改进 {ev.metric}")
        return None

    def _inside(self, params: dict) -> bool:
        from ..spec import in_space

        return all(in_space(self.spec, k, v) for k, v in params.items() if k in self.space) and all(
            not isinstance(v, float) or math.isfinite(v) for v in params.values())
