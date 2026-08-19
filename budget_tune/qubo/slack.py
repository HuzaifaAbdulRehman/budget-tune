"""Inequality constraints as a penalty with binary slack.

``Σ_i c_i x_i ≤ τ`` becomes ``Σ_i c_i x_i + s = τ`` with ``s = δ Σ_b 2^b y_b ≥ 0``.
Four costs this encoding actually has, none of them free:

* ``δ`` discretises the slack, so the constraint is approximate;
* the ``y_b`` add variables;
* incrementing ``s`` by one unit can require several bit flips — a second barrier of
  the same species as one-hot;
* ``Q`` is a second penalty weight that interacts with the one-hot ``P``.
"""

from __future__ import annotations

import math

import dimod
import numpy as np


def slack_bits(tau: float, delta: float) -> int:
    """How many slack bits are needed to reach ``tau`` in steps of ``delta``."""
    if delta <= 0:
        raise ValueError(f"delta must be positive; got {delta}")
    if tau < 0:
        raise ValueError(f"tau must be non-negative; got {tau}")
    if tau == 0:
        return 0
    return int(math.floor(math.log2(tau / delta))) + 1


def slack_inequality(
    costs: np.ndarray,
    tau: float,
    strength: float,
    *,
    delta: float | None = None,
    x_labels: list | None = None,
) -> dimod.BinaryQuadraticModel:
    """Penalty BQM for ``cost·x + slack = τ``.

    Slack variables are named ``("slack", b)`` so they cannot collide with integer
    configuration bits. ``delta`` defaults to a 32nd of ``tau`` (or 1.0 when tau is 0).
    """
    costs = np.asarray(costs, dtype=float).reshape(-1)
    d = costs.size
    labels = list(range(d) if x_labels is None else x_labels)
    if len(labels) != d:
        raise ValueError("x_labels length must match costs")

    if delta is None:
        delta = (tau / 32.0) if tau > 0 else 1.0
    n_slack = slack_bits(tau, delta)
    q = float(strength)

    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    for label in labels:
        bqm.add_variable(label, 0.0)
    slack_labels = [("slack", b) for b in range(n_slack)]
    for label in slack_labels:
        bqm.add_variable(label, 0.0)

    # (Σ c_i x_i + δ Σ 2^b y_b − τ)²
    # Expand: linear and quadratic in the combined affine form.
    coeffs = {labels[i]: float(costs[i]) for i in range(d)}
    for b, label in enumerate(slack_labels):
        coeffs[label] = float(delta) * (2**b)

    # Q (Σ a_k z_k − τ)² = Q [ Σ_k a_k² z_k + 2 Σ_{k<ℓ} a_k a_ℓ z_k z_ℓ − 2τ Σ a_k z_k + τ² ]
    # using z² = z.
    keys = list(coeffs)
    for k in keys:
        a = coeffs[k]
        bqm.add_linear(k, q * (a * a - 2.0 * tau * a))
    for i, u in enumerate(keys):
        for v in keys[i + 1 :]:
            bqm.add_quadratic(u, v, q * 2.0 * coeffs[u] * coeffs[v])
    bqm.offset += q * (tau**2)
    return bqm


def slack_value(sample: dict, delta: float) -> float:
    """Decode the slack bits in a dimod sample."""
    total = 0.0
    for (kind, b), bit in sample.items():
        if kind == "slack":
            total += float(delta) * (2 ** int(b)) * int(bit)
    return total
