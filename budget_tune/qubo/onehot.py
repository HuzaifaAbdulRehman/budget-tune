"""One-hot block penalties, with the constant offset that is easy to drop.

For a block with values ``a``::

    P (Σ_a x_a − 1)² = P (Σ_a x_a + 2 Σ_{a<b} x_a x_b − 2 Σ_a x_a + 1)

using ``x² = x``. Linear ``−P`` per variable, quadratic ``+2P`` per pair, constant
``+P`` per block. With ``J`` blocks the offset is ``P J`` and must sit in the BQM's
offset field; dropping it makes a feasible assignment's energy disagree with the
objective by exactly that amount, which is a silent off-by-``P`` that looks like a
different but still plausible QUBO.
"""

from __future__ import annotations

import dimod
import numpy as np

from budget_tune.space.codec import BLOCKS


def onehot_penalty(
    strength: float, variables: list | None = None
) -> dimod.BinaryQuadraticModel:
    """Penalty BQM for every block in the flat encoding.

    Args:
        strength: ``P``. Recomputed every acquisition iteration; never a global constant.
        variables: optional names for the ``d`` bits. Defaults to ``0..d-1``.
    """
    n = BLOCKS[-1].stop
    labels = list(range(n) if variables is None else variables)
    if len(labels) != n:
        raise ValueError(f"expected {n} variable labels; got {len(labels)}")

    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    for label in labels:
        bqm.add_variable(label, 0.0)

    offset = 0.0
    p = float(strength)
    for block in BLOCKS:
        members = labels[block.start : block.stop]
        for var in members:
            bqm.add_linear(var, -p)
        for i, u in enumerate(members):
            for v in members[i + 1 :]:
                bqm.add_quadratic(u, v, 2.0 * p)
        offset += p
    bqm.offset += offset
    return bqm


def penalty_strength(objective: dimod.BinaryQuadraticModel, margin: float = 2.0) -> float:
    """``P`` above the largest single-flip swing of ``objective``.

    Same bound as the companion's ``suggest_strength``: ``max_v (|h_v| + Σ_u |J_uv|)``.
    Under Thompson sampling the coefficients change every iteration, so this is called
    every iteration.
    """
    max_swing = 0.0
    for variable in objective.variables:
        swing = abs(objective.get_linear(variable))
        swing += sum(abs(bias) for bias in objective.adj[variable].values())
        max_swing = max(max_swing, swing)
    return margin * max(max_swing, 1e-12)


def dense_onehot_energy(bits, strength: float) -> float:
    """Independent expansion of ``P Σ_blocks (Σx − 1)²``, for tests."""
    vector = np.asarray(bits, dtype=float).reshape(-1)
    total = 0.0
    for block in BLOCKS:
        total += float(strength) * (vector[block.start : block.stop].sum() - 1.0) ** 2
    return total
