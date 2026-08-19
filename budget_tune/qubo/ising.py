"""QUBO → Ising with the algebra written down, because a factor-of-two here is silent.

``x = (s + 1) / 2`` with ``s ∈ {−1, +1}``. For ``E = x^T Q x + h·x + c`` (Q symmetric,
diagonal folded into h already or not — we take dimod's linear/quadratic split):

    x_i x_j = (s_i s_j + s_i + s_j + 1) / 4

The companion project once shipped a ``J/8`` vs ``J/4`` error that still produced a
perfectly plausible Ising model. Tests compare against this expansion, not against dimod's
own converter alone.
"""

from __future__ import annotations

import dimod
import numpy as np


def bqm_to_ising(bqm: dimod.BinaryQuadraticModel) -> tuple[dict, dict, float]:
    """Return ``(h, J, offset)`` for spins ``s = 2x − 1``.

    ``h`` is linear in ``s``, ``J`` is the upper-triangle coupling, ``offset`` is the
    constant so that Ising energy equals BQM energy on corresponding assignments.
    """
    h: dict = {v: 0.0 for v in bqm.variables}
    j: dict = {}
    offset = float(bqm.offset)

    for v, bias in bqm.linear.items():
        h[v] = h.get(v, 0.0) + 0.5 * float(bias)
        offset += 0.5 * float(bias)

    for (u, v), bias in bqm.quadratic.items():
        q = float(bias)
        j[(u, v)] = j.get((u, v), 0.0) + 0.25 * q
        h[u] = h.get(u, 0.0) + 0.25 * q
        h[v] = h.get(v, 0.0) + 0.25 * q
        offset += 0.25 * q

    return h, j, offset


def ising_energy(h: dict, j: dict, offset: float, bits) -> float:
    """Ising energy of a *binary* assignment, converting ``s = 2x − 1``."""
    x = np.asarray(bits, dtype=float).reshape(-1)
    spins = {i: 2.0 * x[i] - 1.0 for i in range(x.size)}
    energy = float(offset)
    for v, bias in h.items():
        energy += float(bias) * spins[int(v)]
    for (u, v), bias in j.items():
        energy += float(bias) * spins[int(u)] * spins[int(v)]
    return energy


def dimod_ising_energy(bqm: dimod.BinaryQuadraticModel, bits) -> float:
    """dimod's own conversion, used as a second check rather than the definition."""
    ising = bqm.change_vartype(dimod.SPIN, inplace=False)
    x = np.asarray(bits, dtype=int).reshape(-1)
    sample = {i: (1 if x[i] else -1) for i in range(x.size)}
    return float(ising.energy(sample))
