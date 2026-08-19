"""Bayesian horseshoe Gibbs sampler (Makalic & Schmidt, arXiv:1508.03884).

BOCS's surrogate is a quadratic polynomial with this prior, not an arbitrary sparse
regression. The sampler is the whole method: Thompson sampling draws one posterior
coefficient vector per iteration.
"""

from __future__ import annotations

import numpy as np


def _fastmvg_rue(phi: np.ndarray, ptp: np.ndarray, alpha: np.ndarray, d_diag: np.ndarray, rng):
    """Rue's sampler for N(mu, S) with S = inv(Phi'Phi + D^{-1}), small p."""
    p = phi.shape[1]
    dinv = np.diag(1.0 / np.maximum(np.diag(d_diag), 1e-18))
    matrix = ptp + dinv
    matrix = 0.5 * (matrix + matrix.T)
    try:
        chol = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        jitter = (np.abs(np.diag(matrix)).max() + 1.0) * 1e-12
        chol = np.linalg.cholesky(matrix + jitter * np.eye(p))
    rhs = phi.T @ alpha
    mid = np.linalg.solve(chol, rhs)
    mean = np.linalg.solve(chol.T, mid)
    noise = np.linalg.solve(chol.T, rng.standard_normal(p))
    return mean + noise


def _fastmvg_bhattacharya(phi: np.ndarray, alpha: np.ndarray, d_diag: np.ndarray, rng):
    """Bhattacharya–Chakraborty–Mallick sampler, large p."""
    n, p = phi.shape
    d = np.diag(d_diag)
    u = rng.standard_normal(p) * np.sqrt(np.maximum(d, 0.0))
    v = phi @ u + rng.standard_normal(n)
    dpt = phi.T * d[:, None]
    w = np.linalg.solve(phi @ dpt + np.eye(n), alpha - v)
    return u + dpt @ w


def horseshoe_sample(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 200,
    burnin: int = 100,
    thin: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw posterior samples of ``(intercept, beta)``.

    ``x`` is the design matrix *without* an intercept column (BOCS builds quadratic
    features first, then this sampler puts the constant back as ``b0 = mean(y)`` after
    centering, matching Makalic & Schmidt). Returns ``beta`` of shape ``(p, n_samples)``
    and ``b0`` of shape ``(n_samples,)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, p = x.shape
    if y.size != n:
        raise ValueError("x and y length disagree")
    if n == 0 or p == 0:
        raise ValueError("empty design")

    y_centered = y - y.mean()
    xtx = x.T @ x

    beta = np.zeros((p, n_samples))
    b0 = np.full(n_samples, y.mean())
    sigma2 = 1.0
    lambda2 = rng.uniform(size=p)
    tau2 = 1.0
    nu = np.ones(p)
    xi = 1.0

    stored = 0
    iteration = 0
    while stored < n_samples:
        sigma = np.sqrt(max(sigma2, 1e-18))
        d_star = np.diag(tau2 * lambda2)
        if p > n and p > 200:
            b = _fastmvg_bhattacharya(x / sigma, y_centered / sigma, sigma2 * d_star, rng)
        else:
            b = _fastmvg_rue(x / sigma, xtx / sigma2, y_centered / sigma, sigma2 * d_star, rng)

        residual = y_centered - x @ b
        shape = (n + p) / 2.0
        scale = float(residual @ residual) / 2.0 + float(
            np.sum(b**2 / np.maximum(lambda2, 1e-18)) / tau2 / 2.0
        )
        sigma2 = 1.0 / rng.gamma(shape, 1.0 / max(scale, 1e-18))

        scale_l = 1.0 / nu + b**2 / 2.0 / max(tau2 * sigma2, 1e-18)
        lambda2 = 1.0 / rng.exponential(1.0 / np.maximum(scale_l, 1e-18))

        shape_t = (p + 1.0) / 2.0
        scale_t = 1.0 / xi + float(
            np.sum(b**2 / np.maximum(lambda2, 1e-18)) / 2.0 / max(sigma2, 1e-18)
        )
        tau2 = 1.0 / rng.gamma(shape_t, 1.0 / max(scale_t, 1e-18))

        nu = 1.0 / rng.exponential(1.0 / (1.0 + 1.0 / np.maximum(lambda2, 1e-18)))
        xi = 1.0 / rng.exponential(1.0 / (1.0 + 1.0 / max(tau2, 1e-18)))

        iteration += 1
        if iteration > burnin and iteration % thin == 0:
            beta[:, stored] = b
            stored += 1

    return beta, b0
