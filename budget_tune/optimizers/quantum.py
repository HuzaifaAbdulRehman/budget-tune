"""BOCS and FMQA proposers. Acquisition is brute force over the enumerated table."""

from __future__ import annotations

import numpy as np

from budget_tune.optimizers.base import History
from budget_tune.solvers.brute import brute_maximise
from budget_tune.space.codec import N_VARIABLES, encode
from budget_tune.space.grids import enumerate_configurations
from budget_tune.surrogate.bocs import BOCSSurrogate
from budget_tune.surrogate.features import evaluate_quadratic
from budget_tune.surrogate.fmqa import FactorizationMachine


def _initial_ids(history: History, rng: np.random.Generator, n_init: int) -> list[str]:
    ids = list(history.view.config_ids())
    rng.shuffle(ids)
    return ids[: max(1, min(n_init, len(ids)))]


def bocs_proposer(
    rng: np.random.Generator,
    n_init: int = 20,
    n_gibbs: int = 50,
    mode: str = "gated",
):
    surrogate = BOCSSurrogate(N_VARIABLES, rng, n_gibbs=n_gibbs)
    queue = {"ids": None}

    def propose(history: History) -> str:
        if queue["ids"] is None:
            queue["ids"] = _initial_ids(history, rng, n_init)
        if queue["ids"]:
            return queue["ids"].pop()
        x = []
        y = []
        by_id = {c.config_id: c for c in enumerate_configurations()}
        for cid, quality in history.observed.items():
            x.append(encode(by_id[cid], mode=mode))
            y.append(quality)
        surrogate.fit(np.asarray(x), np.asarray(y))
        available = [by_id[cid] for cid in history.view.config_ids() if cid in by_id]

        def score(bits, config):
            return float(evaluate_quadratic(bits, surrogate.alpha)[0])

        winner, _ = brute_maximise(score, available, mode=mode)
        return winner.config_id

    return propose


def fmqa_proposer(
    rng: np.random.Generator,
    n_init: int = 20,
    rank: int = 8,
    steps: int = 200,
    mode: str = "feasible",
):
    """FMQA. Fits on *negated* quality so QUBO minimisation searches for high quality.

    Acquisition here is still brute force (RQ1). The FM is converted to a BQM in RQ3.
    """
    machine = FactorizationMachine(N_VARIABLES, rng, rank=rank, steps=steps)
    queue = {"ids": None}

    def propose(history: History) -> str:
        if queue["ids"] is None:
            queue["ids"] = _initial_ids(history, rng, n_init)
        if queue["ids"]:
            return queue["ids"].pop()
        x = []
        y = []
        by_id = {c.config_id: c for c in enumerate_configurations()}
        for cid, quality in history.observed.items():
            x.append(encode(by_id[cid], mode=mode))
            y.append(-quality)
        machine.fit(np.asarray(x), np.asarray(y))
        available = [by_id[cid] for cid in history.view.config_ids() if cid in by_id]

        def score(bits, config):
            return -float(machine.predict_one(bits))

        winner, _ = brute_maximise(score, available, mode=mode)
        return winner.config_id

    return propose
