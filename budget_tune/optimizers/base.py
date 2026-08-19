"""Equal-cost HPO loop over a :class:`SearchView`.

Every method — classical or QUBO — goes through :func:`run`. Training cost is a table
lookup; re-proposals are charged zero, identically for every method; optimiser overhead
is measured live around ``propose``. The test split is unreachable: this module never
names a reporting column.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from budget_tune.benchmark.schema import SearchView
from budget_tune.space.grids import (
    Configuration,
    configuration_from_row,
    enumerate_configurations,
)

QUALITY = "val_ndcg_at_10"
COST = "train_cpu_seconds"


@dataclass
class Trial:
    """One propose-and-lookup step."""

    config_id: str
    quality: float
    train_cpu_seconds_charged: float
    overhead_cpu_seconds: float
    cumulative_cpu_seconds: float
    best_quality: float
    duplicate: bool
    n_unique: int


@dataclass
class History:
    """What a proposer is allowed to know."""

    view: SearchView
    trials: list[Trial] = field(default_factory=list)
    observed: dict[str, float] = field(default_factory=dict)
    configs: dict[str, Configuration] = field(default_factory=dict)

    @property
    def n_fits(self) -> int:
        return sum(not trial.duplicate for trial in self.trials)

    @property
    def n_solves(self) -> int:
        return len(self.trials)


Proposer = Callable[[History], str]


def _all_configs(view: SearchView) -> dict[str, Configuration]:
    by_id = {config.config_id: config for config in enumerate_configurations()}
    # Restrict to ids the view actually has, in case a test uses a subset.
    return {cid: by_id[cid] for cid in view.config_ids() if cid in by_id} or {
        row.config_id: configuration_from_row(row) for _, row in view.frame.iterrows()
    }


def run(
    name: str,
    view: SearchView,
    propose: Proposer,
    budget_cpu_seconds: float,
    *,
    seed: int,
    max_steps: int | None = None,
) -> dict:
    """Run ``propose`` until cumulative CPU-seconds hit the budget.

    Returns a JSON-serialisable record. ``max_steps`` is a safety cap for tests, not a
    budget: the comparison axis is CPU-seconds.
    """
    if budget_cpu_seconds <= 0:
        raise ValueError("budget must be positive")
    history = History(view=view, configs=_all_configs(view))
    spent = 0.0
    best = float("-inf")
    steps = 0
    while spent < budget_cpu_seconds:
        if max_steps is not None and steps >= max_steps:
            break
        started = time.process_time()
        config_id = propose(history)
        overhead = time.process_time() - started
        if config_id not in history.view.frame.config_id.values:
            raise KeyError(f"{name} proposed unknown config_id {config_id!r}")
        duplicate = config_id in history.observed
        if duplicate:
            quality = history.observed[config_id]
            charged = 0.0
        else:
            row = history.view.lookup(config_id)
            quality = float(row[QUALITY])
            charged = float(row[COST])
            history.observed[config_id] = quality
        spent += charged + overhead
        best = max(best, quality)
        trial = Trial(
            config_id=config_id,
            quality=quality,
            train_cpu_seconds_charged=charged,
            overhead_cpu_seconds=overhead,
            cumulative_cpu_seconds=spent,
            best_quality=best,
            duplicate=duplicate,
            n_unique=len(history.observed),
        )
        history.trials.append(trial)
        steps += 1
        if not history.observed:
            continue
    return {
        "method": name,
        "dataset": view.dataset,
        "seed": seed,
        "budget_cpu_seconds": budget_cpu_seconds,
        "spent_cpu_seconds": spent,
        "n_trials": len(history.trials),
        "n_unique": len(history.observed),
        "n_duplicates": sum(trial.duplicate for trial in history.trials),
        "n_surrogate_fits": history.n_fits,
        "n_acquisition_solves": history.n_solves,
        "best_quality": best if history.trials else None,
        "best_config_id": (
            max(history.observed, key=history.observed.get) if history.observed else None
        ),
        "trials": [trial.__dict__ for trial in history.trials],
    }


def checkpoint(record: dict, fractions: tuple[float, ...] = (0.25, 0.50, 1.0)) -> dict:
    """Best quality at pre-registered fractions of the budget. Descriptive, not a test."""
    budget = float(record["budget_cpu_seconds"])
    out = {}
    for fraction in fractions:
        limit = fraction * budget
        best = None
        for trial in record["trials"]:
            if trial["cumulative_cpu_seconds"] <= limit:
                best = trial["best_quality"]
        out[str(fraction)] = best
    return out
