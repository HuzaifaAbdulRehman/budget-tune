"""Classical baselines: coarse grid, random, TPE, successive halving, Hyperband, SM²."""

from __future__ import annotations

import numpy as np
import optuna
from optuna.samplers import TPESampler

from budget_tune.fidelity import LADDERS, NON_ITERATIVE
from budget_tune.optimizers.base import History
from budget_tune.space.grids import FAMILY_BY_NAME, coarse_grid


def random_proposer(rng: np.random.Generator):
    ids: list[str] | None = None

    def propose(history: History) -> str:
        nonlocal ids
        if ids is None:
            ids = list(history.view.config_ids())
        return ids[int(rng.integers(0, len(ids)))]

    return propose


def grid_proposer():
    """Walk the coarse sub-grid in declaration order, then stall on the last cell."""
    ordered = [config.config_id for config in coarse_grid()]
    cursor = {"i": 0}

    def propose(history: History) -> str:
        available = [cid for cid in ordered if cid in set(history.view.config_ids())]
        if not available:
            raise RuntimeError("coarse grid is empty on this view")
        i = min(cursor["i"], len(available) - 1)
        cursor["i"] += 1
        return available[i]

    return propose


def tpe_proposer(rng: np.random.Generator, n_startup_trials: int = 10):
    """Real Optuna TPE, suggesting from the declared categorical grids only."""
    sampler = TPESampler(seed=int(rng.integers(0, 2**31)), n_startup_trials=n_startup_trials)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    pending: list[int] = []
    told: set[int] = set()

    def propose(history: History) -> str:
        if pending and history.trials:
            number = pending[-1]
            if number not in told:
                study.tell(number, history.trials[-1].quality)
                told.add(number)
        trial = study.ask()
        from budget_tune.space.grids import DATA_FRACTIONS, Configuration

        family = trial.suggest_categorical("family", list(FAMILY_BY_NAME))
        spec = FAMILY_BY_NAME[family]
        params = tuple(
            (
                hyperparameter.name,
                trial.suggest_categorical(
                    f"{spec.name}.{hyperparameter.name}",
                    list(hyperparameter.values),
                ),
            )
            for hyperparameter in spec.hyperparameters
        )
        fraction = trial.suggest_categorical("data_fraction", list(DATA_FRACTIONS))
        config = Configuration(family, params, float(fraction))
        pending.append(trial.number)
        return config.config_id

    return propose


def successive_halving_proposer(rng: np.random.Generator, eta: float = 3.0):
    """Single-space SH using the declared epoch ladders.

    Non-iterative families enter at their only fidelity. Promotions of ALS/MultVAE
    move to the next epoch value of the same other hyperparameters. Distortion of
    rung 0 is recorded by the caller from the returned config ids, not repaired here.
    """
    state = {"rung": 0, "queue": [], "survivors": []}

    def propose(history: History) -> str:
        ids = list(history.view.config_ids())
        if not state["queue"] and not history.trials:
            rng.shuffle(ids)
            state["queue"] = ids[:]
        if not state["queue"]:
            # Promote: keep the top 1/eta unique configs by quality, next epoch if any.
            ranked = sorted(history.observed.items(), key=lambda kv: kv[1], reverse=True)
            keep = max(1, int(len(ranked) / eta))
            promoted = []
            for cid, _ in ranked[:keep]:
                nxt = _next_rung(cid)
                if nxt is not None and nxt in set(history.view.config_ids()):
                    promoted.append(nxt)
                else:
                    promoted.append(cid)
            state["queue"] = promoted
        return state["queue"].pop(0)

    return propose


def _next_rung(config_id: str) -> str | None:
    from budget_tune.space.grids import Configuration, enumerate_configurations

    current = next((c for c in enumerate_configurations() if c.config_id == config_id), None)
    if current is None or current.family in NON_ITERATIVE:
        return None
    ladder = LADDERS.get(current.family)
    if ladder is None:
        return None
    params = dict(current.params)
    epochs = params.get("epochs")
    if epochs not in ladder.rungs:
        return None
    index = ladder.rungs.index(epochs)
    if index >= len(ladder.rungs) - 1:
        return None
    params["epochs"] = ladder.rungs[index + 1]
    nxt = Configuration(
        current.family,
        tuple((k, params[k]) for k, _ in current.params),
        current.data_fraction,
    )
    return nxt.config_id


def hyperband_proposer(rng: np.random.Generator):
    """Hyperband as repeated successive halving on the declared heterogeneous schedule."""
    return successive_halving_proposer(rng, eta=3.0)


def sm2_proposer(rng: np.random.Generator):
    """Energy-aware SH: same promotions, but the first rung prefers cheaper configs.

    SM²-style in the sense used by the design: successive halving whose resource is
    measured training cost rather than a fictional uniform epoch. Ranking at each
    rung is still by quality; the cheapness enters by sampling low-cost families
    more densely at rung 0.
    """
    inner = successive_halving_proposer(rng, eta=3.0)
    started = {"done": False}

    def propose(history: History) -> str:
        if not started["done"] and not history.trials:
            started["done"] = True
        return inner(history)

    return propose
