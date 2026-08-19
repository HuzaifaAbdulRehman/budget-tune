"""Turn a second-order surrogate into a minimisation BQM.

HPO maximises quality. QUBO minimises. The conversion is one place: the linear and
quadratic coefficients are negated, and a test asserts that a solver's returned energy
equals the recomputed objective of its returned bits.
"""

from __future__ import annotations

import dimod
import numpy as np

from budget_tune.surrogate.features import unpack_quadratic


def quadratic_to_bqm(
    intercept: float,
    linear: np.ndarray,
    pairwise: np.ndarray,
    *,
    minimise: bool = True,
) -> dimod.BinaryQuadraticModel:
    """Build a BINARY BQM from ``f(x) = c + h·x + x^T Q x`` (strict upper triangle of Q).

    Args:
        intercept: constant term. Stored as the BQM offset.
        linear: length-``d`` vector.
        pairwise: ``(d, d)`` with pairwise[i, j] = coefficient of ``x_i x_j`` for ``i < j``.
            The diagonal is ignored (binary ``x² = x`` belongs in ``linear``).
        minimise: if True, negate the polynomial so a QUBO solver maximises the original.
    """
    linear = np.asarray(linear, dtype=float).reshape(-1)
    pairwise = np.asarray(pairwise, dtype=float)
    d = linear.size
    if pairwise.shape != (d, d):
        raise ValueError(f"pairwise must be {(d, d)}; got {pairwise.shape}")

    sign = -1.0 if minimise else 1.0
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    for i in range(d):
        bqm.add_variable(i, sign * float(linear[i]))
    for i in range(d):
        for j in range(i + 1, d):
            coeff = float(pairwise[i, j] + pairwise[j, i])
            if coeff:
                bqm.add_quadratic(i, j, sign * coeff)
    bqm.offset = sign * float(intercept)
    return bqm


def alpha_to_bqm(
    alpha: np.ndarray, d: int, *, minimise: bool = True
) -> dimod.BinaryQuadraticModel:
    """Pack a BOCS-style coefficient vector into a BQM."""
    intercept, linear, pairwise = unpack_quadratic(alpha, d)
    return quadratic_to_bqm(intercept, linear, pairwise, minimise=minimise)


def surrogate_energy(
    bits, intercept: float, linear: np.ndarray, pairwise: np.ndarray
) -> float:
    """``c + h·x + Σ_{i<j} Q_ij x_i x_j``, independent of dimod."""
    x = np.asarray(bits, dtype=float).reshape(-1)
    linear = np.asarray(linear, dtype=float).reshape(-1)
    pairwise = np.asarray(pairwise, dtype=float)
    value = float(intercept) + float(linear @ x)
    for i in range(x.size):
        if x[i] == 0:
            continue
        for j in range(i + 1, x.size):
            value += float(pairwise[i, j] + pairwise[j, i]) * x[i] * x[j]
    return value


def bqm_energy(bqm: dimod.BinaryQuadraticModel, bits) -> float:
    """Evaluate ``bqm`` on a dense bit vector with variables ``0..d-1``."""
    x = np.asarray(bits, dtype=int).reshape(-1)
    sample = {i: int(x[i]) for i in range(x.size)}
    return float(bqm.energy(sample))
