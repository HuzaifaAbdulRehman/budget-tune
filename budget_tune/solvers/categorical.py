"""Simulated annealing in the categorical domain (BOCS Appendix A).

The neighbourhood of a configuration is every configuration that differs in exactly one
hyperparameter assignment — including family or data fraction. Two one-hot-feasible
states that differ in one block are adjacent under this move set and are *never*
adjacent under single-bit flips. Attributed to Baptista & Poloczek, not claimed as a
contribution of this project.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from budget_tune.space.codec import encode
from budget_tune.space.grids import DATA_FRACTIONS, FAMILIES, FAMILY_BY_NAME, Configuration


def _neighbours(config: Configuration) -> list[Configuration]:
    out: list[Configuration] = []
    for spec in FAMILIES:
        if spec.name == config.family:
            continue
        params = tuple((h.name, h.values[0]) for h in spec.hyperparameters)
        out.append(Configuration(spec.name, params, config.data_fraction))
    for fraction in DATA_FRACTIONS:
        if fraction != config.data_fraction:
            out.append(Configuration(config.family, config.params, float(fraction)))
    params = dict(config.params)
    for hyperparameter in FAMILY_BY_NAME[config.family].hyperparameters:
        for value in hyperparameter.values:
            if params[hyperparameter.name] == value:
                continue
            updated = dict(params)
            updated[hyperparameter.name] = value
            out.append(
                Configuration(
                    config.family,
                    tuple(
                        (h.name, updated[h.name])
                        for h in FAMILY_BY_NAME[config.family].hyperparameters
                    ),
                    config.data_fraction,
                )
            )
    return out


def categorical_sa(
    score: Callable[[np.ndarray, Configuration], float],
    start: Configuration,
    rng: np.random.Generator,
    *,
    steps: int = 200,
    temperature: float = 1.0,
    cool: float = 0.95,
    mode: str = "feasible",
    minimise: bool = True,
) -> tuple[Configuration, float]:
    """Minimise (default) or maximise ``score`` by one-block moves."""

    def energy(config: Configuration) -> float:
        value = float(score(encode(config, mode=mode), config))
        return value if minimise else -value

    current = start
    current_e = energy(current)
    best = current
    best_e = current_e
    temp = temperature
    for _ in range(steps):
        temp *= cool
        options = _neighbours(current)
        proposal = options[int(rng.integers(0, len(options)))]
        proposed_e = energy(proposal)
        if proposed_e <= current_e or rng.random() < np.exp(
            (current_e - proposed_e) / max(temp, 1e-12)
        ):
            current, current_e = proposal, proposed_e
            if current_e < best_e:
                best, best_e = current, current_e
    return best, (best_e if minimise else -best_e)
