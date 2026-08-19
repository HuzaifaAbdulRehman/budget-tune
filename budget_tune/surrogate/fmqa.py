"""FMQA surrogate: a rank-K factorization machine trained with Adam.

Kitai et al., Phys. Rev. Research 2, 013319 (2020), equation (3)::

    f(q) = Σ_i w_i q_i + Σ_i Σ_j Σ_k v_{ik} v_{jk} q_i q_j

with factorization size ``K = 8`` (LIBFM default, the value they freeze). There is no
global intercept in that equation; a constant is absorbed into the QUBO offset as zero.
Training minimises squared error on the *negated* quality, so a QUBO minimiser searches
for high quality. Initialization is small Gaussian, matching LIBFM's scale.
"""

from __future__ import annotations

import numpy as np


class FactorizationMachine:
    """Binary factorization machine with a QUBO view of its prediction."""

    def __init__(
        self,
        d: int,
        rng: np.random.Generator,
        rank: int = 8,
        steps: int = 400,
        lr: float = 0.05,
        init_std: float = 0.1,
    ) -> None:
        if d < 1 or rank < 1:
            raise ValueError("d and rank must be positive")
        self.d = d
        self.rank = rank
        self.rng = rng
        self.steps = steps
        self.lr = lr
        self.w = np.zeros(d)
        self.v = rng.normal(0.0, init_std, size=(d, rank))

    def predict_one(self, q: np.ndarray) -> float:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self.d:
            raise ValueError(f"expected {self.d} bits; got {q.size}")
        linear = float(self.w @ q)
        # Σ_i Σ_j <v_i, v_j> q_i q_j = ||Σ_i v_i q_i||²
        mixed = self.v.T @ q
        return linear + float(mixed @ mixed)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return np.array([self.predict_one(x)])
        return np.array([self.predict_one(row) for row in x])

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Adam on squared error. ``y`` is already the quantity to *predict* (minimised)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"x must be (n, {self.d}); got {x.shape}")
        n = x.shape[0]
        if n != y.size:
            raise ValueError("x and y length disagree")
        if n == 0:
            return

        m_w = np.zeros_like(self.w)
        v_w = np.zeros_like(self.w)
        m_v = np.zeros_like(self.v)
        v_v = np.zeros_like(self.v)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for step in range(1, self.steps + 1):
            pred = self.predict(x)
            err = pred - y
            # dL/dw_i = (2/n) Σ_n err_n q_{n i}
            grad_w = (2.0 / n) * (err @ x)
            # d pred / d v_{ik} = 2 q_i (Σ_j v_{jk} q_j)
            hidden = x @ self.v  # (n, K)
            grad_v = np.zeros_like(self.v)
            for k in range(self.rank):
                # (2/n) Σ_n err_n * 2 q_{n i} * hidden_{n k}
                grad_v[:, k] = (4.0 / n) * (x * (err * hidden[:, k])[:, None]).sum(axis=0)

            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w**2
            m_v = beta1 * m_v + (1 - beta1) * grad_v
            v_v = beta2 * v_v + (1 - beta2) * grad_v**2
            mwhat = m_w / (1 - beta1**step)
            vwhat = v_w / (1 - beta2**step)
            mvhat = m_v / (1 - beta1**step)
            vvhat = v_v / (1 - beta2**step)
            self.w -= self.lr * mwhat / (np.sqrt(vwhat) + eps)
            self.v -= self.lr * mvhat / (np.sqrt(vvhat) + eps)

    def as_quadratic(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Intercept 0, linear ``w + diag(VV^T)`` folded? Keep the FM expansion explicit.

        ``f(q) = w·q + q^T (V V^T) q``. For binary q, the diagonal of ``VV^T`` is a
        linear contribution. The QUBO quadratic part is the off-diagonal of ``VV^T``.
        """
        gram = self.v @ self.v.T
        linear = self.w + np.diag(gram)
        pairwise = gram.copy()
        np.fill_diagonal(pairwise, 0.0)
        return 0.0, linear, pairwise

    def to_bqm(self, *, minimise: bool = False):
        """QUBO whose energy equals :meth:`predict_one` when ``minimise`` is False."""
        from budget_tune.qubo.acquisition import quadratic_to_bqm

        intercept, linear, pairwise = self.as_quadratic()
        return quadratic_to_bqm(intercept, linear, pairwise, minimise=minimise)
