"""BOCS surrogate: second-order polynomial, horseshoe prior, Thompson sampling.

Baptista & Poloczek, ICML 2018. The acquisition optimiser in *this* project is brute
force over canonical configurations for RQ1 (the space is enumerable) and categorical-
domain SA for RQ3. The SDP relaxation is not used: it would need cvxpy, which is not
a campaign dependency, and brute force is exact at this scale.
"""

from __future__ import annotations

import numpy as np

from budget_tune.surrogate.features import design_matrix, evaluate_quadratic, n_quadratic
from budget_tune.surrogate.horseshoe import horseshoe_sample


class BOCSSurrogate:
    """Fit ``f(x) = α₀ + Σ α_j x_j + Σ_{i<j} α_{ij} x_i x_j`` and draw posterior α."""

    def __init__(
        self,
        d: int,
        rng: np.random.Generator,
        n_gibbs: int = 200,
        burnin: int = 100,
    ) -> None:
        if d < 1:
            raise ValueError(f"d must be positive; got {d}")
        self.d = d
        self.rng = rng
        self.n_gibbs = n_gibbs
        self.burnin = burnin
        self.alpha = np.zeros(n_quadratic(d))

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Replace ``self.alpha`` with the last Gibbs draw. Returns that draw."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"x must be (n, {self.d}); got {x.shape}")
        if x.shape[0] != y.size:
            raise ValueError("x and y length disagree")
        if x.shape[0] < 2:
            # A single point cannot identify a quadratic; keep a constant fit.
            self.alpha = np.zeros(n_quadratic(self.d))
            self.alpha[0] = float(y.mean()) if y.size else 0.0
            return self.alpha

        phi = design_matrix(x)
        # Drop the intercept column; horseshoe_sample recentres y.
        features = phi[:, 1:]
        # Drop all-zero columns so the sampler's covariance stays defined.
        keep = np.any(np.abs(features) > 1e-15, axis=0)
        if not keep.any():
            self.alpha = np.zeros(n_quadratic(self.d))
            self.alpha[0] = float(y.mean())
            return self.alpha

        beta, intercepts = horseshoe_sample(
            features[:, keep],
            y,
            self.rng,
            n_samples=1,
            burnin=self.n_gibbs,
            thin=1,
        )
        packed = np.zeros(n_quadratic(self.d))
        packed[0] = float(intercepts[-1])
        packed[1:][keep] = beta[:, -1]
        self.alpha = packed
        return self.alpha

    def predict(self, x: np.ndarray) -> np.ndarray:
        return evaluate_quadratic(x, self.alpha)
