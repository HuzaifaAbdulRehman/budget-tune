"""Exact acquisition search by enumerating canonical configurations.

RQ1 solves the acquisition by brute force. With 471 cells that is exact and instant, so
the comparison measures surrogates rather than confounding them with a heuristic solver.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from budget_tune.space.codec import encode
from budget_tune.space.grids import Configuration, enumerate_configurations


def brute_maximise(
    score: Callable[[np.ndarray, Configuration], float],
    configs: Sequence[Configuration] | None = None,
    *,
    mode: str = "gated",
) -> tuple[Configuration, float]:
    """Return the canonical configuration that maximises ``score(bits, config)``."""
    best_config = None
    best = -np.inf
    for config in configs if configs is not None else enumerate_configurations():
        value = float(score(encode(config, mode=mode), config))
        if value > best:
            best = value
            best_config = config
    if best_config is None:
        raise ValueError("no configurations to search")
    return best_config, best


def brute_minimise(
    score: Callable[[np.ndarray, Configuration], float],
    configs: Sequence[Configuration] | None = None,
    *,
    mode: str = "gated",
) -> tuple[Configuration, float]:
    def negated(bits, config):
        return -float(score(bits, config))

    config, value = brute_maximise(negated, configs, mode=mode)
    return config, -value
