"""Quadratic feature map shared by BOCS, the oracle ridge, and the QUBO packer.

A second-order polynomial on ``d`` bits has ``p = 1 + d + C(d, 2)`` coefficients,
ordered as intercept, then linear terms, then pairwise terms ``(i, j)`` with ``i < j``.
"""

from __future__ import annotations

import numpy as np


def n_quadratic(d: int) -> int:
    return 1 + d + d * (d - 1) // 2


def pairwise_index(d: int, i: int, j: int) -> int:
    """Index of ``x_i x_j`` (``i < j``) in the packed coefficient vector."""
    if not 0 <= i < j < d:
        raise ValueError(f"need 0 <= i < j < {d}; got i={i}, j={j}")
    # skip intercept + d linears; then pairs (0,1)..(0,d-1), (1,2)..
    return 1 + d + (i * (2 * d - i - 1)) // 2 + (j - i - 1)


def design_matrix(x: np.ndarray) -> np.ndarray:
    """Rows of observations ``(n, d)`` → ``(n, p)`` quadratic features including intercept."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n, d = x.shape
    p = n_quadratic(d)
    phi = np.empty((n, p), dtype=float)
    phi[:, 0] = 1.0
    phi[:, 1 : 1 + d] = x
    col = 1 + d
    for i in range(d):
        for j in range(i + 1, d):
            phi[:, col] = x[:, i] * x[:, j]
            col += 1
    return phi


def unpack_quadratic(alpha: np.ndarray, d: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Split a packed coefficient vector into intercept, linear, pairwise matrix."""
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    p = n_quadratic(d)
    if alpha.size != p:
        raise ValueError(f"alpha has length {alpha.size}, expected {p} for d={d}")
    intercept = float(alpha[0])
    linear = alpha[1 : 1 + d].copy()
    pairwise = np.zeros((d, d), dtype=float)
    col = 1 + d
    for i in range(d):
        for j in range(i + 1, d):
            pairwise[i, j] = alpha[col]
            col += 1
    return intercept, linear, pairwise


def pack_quadratic(intercept: float, linear: np.ndarray, pairwise: np.ndarray) -> np.ndarray:
    """Inverse of :func:`unpack_quadratic`."""
    linear = np.asarray(linear, dtype=float).reshape(-1)
    pairwise = np.asarray(pairwise, dtype=float)
    d = linear.size
    alpha = np.zeros(n_quadratic(d), dtype=float)
    alpha[0] = intercept
    alpha[1 : 1 + d] = linear
    col = 1 + d
    for i in range(d):
        for j in range(i + 1, d):
            alpha[col] = pairwise[i, j] + pairwise[j, i]
            col += 1
    return alpha


def evaluate_quadratic(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """``Phi(x) @ alpha`` for one or many rows."""
    phi = design_matrix(x)
    return phi @ np.asarray(alpha, dtype=float).reshape(-1)
