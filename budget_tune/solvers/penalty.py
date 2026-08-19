"""Penalty-encoded samplers wrapped around the companion, with tabu's timeout pinned.

RQ3 compares these to brute force and to categorical SA. They live behind a function so
tests that do not have the companion still import ``budget_tune.solvers``.
"""

from __future__ import annotations

import numpy as np


def sample_penalty_neal(
    bqm, rng: np.random.Generator, num_reads: int = 50, num_sweeps: int = 1000
):
    from budget_tune.companion import ensure_importable

    ensure_importable("feasible_rerank")
    from neal import SimulatedAnnealingSampler

    seed = int(rng.integers(0, 2**31))
    return SimulatedAnnealingSampler().sample(
        bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed
    )


def sample_penalty_tabu(bqm, rng: np.random.Generator, timeout: float = 1.0):
    """Tabu with an explicit timeout. The dwave default is 20 ms wall-clock."""
    from budget_tune.companion import ensure_importable

    ensure_importable("feasible_rerank")
    from dwave.samplers import TabuSampler

    seed = int(rng.integers(0, 2**31))
    return TabuSampler().sample(bqm, timeout=timeout, seed=seed)


def sample_penalty_sb(bqm, rng: np.random.Generator, **kwargs):
    from budget_tune.companion import ensure_importable

    ensure_importable("feasible_rerank")
    from qubo_rerank.solvers.bifurcation import SimulatedBifurcationSampler

    return SimulatedBifurcationSampler().sample(bqm, **kwargs)


def best_sample_bits(response, d: int) -> np.ndarray:
    """The lowest-energy sample as a dense ``{0,1}^d`` vector over variables ``0..d-1``."""
    record = next(iter(response.data(fields=["sample", "energy"], sorted_by="energy")))
    sample = record.sample
    return np.array([int(sample[i]) for i in range(d)], dtype=np.int8)
