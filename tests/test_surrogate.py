"""Surrogate invariants: feature packing, FM equals QUBO, ridge ceiling, horseshoe draws."""

from __future__ import annotations

import numpy as np

from budget_tune.qubo.acquisition import bqm_energy
from budget_tune.surrogate.bocs import BOCSSurrogate
from budget_tune.surrogate.features import (
    design_matrix,
    evaluate_quadratic,
    n_quadratic,
    pack_quadratic,
    unpack_quadratic,
)
from budget_tune.surrogate.fmqa import FactorizationMachine
from budget_tune.surrogate.ridge import argmin_regret, fit_ridge_quadratic


class TestFeatures:
    def test_pack_unpack_round_trip(self):
        intercept, linear, pairwise = 1.5, np.array([0.1, -0.2, 0.3]), np.zeros((3, 3))
        pairwise[0, 1] = 0.5
        pairwise[1, 2] = -0.25
        alpha = pack_quadratic(intercept, linear, pairwise)
        c, h, q = unpack_quadratic(alpha, 3)
        assert c == intercept
        np.testing.assert_allclose(h, linear)
        np.testing.assert_allclose(np.triu(q, 1), np.triu(pairwise, 1))

    def test_design_matches_manual_quadratic(self):
        x = np.array([[1, 0, 1]])
        phi = design_matrix(x)
        assert phi.shape == (1, n_quadratic(3))
        # intercept, x0, x1, x2, x0x1, x0x2, x1x2
        np.testing.assert_array_equal(phi[0], [1, 1, 0, 1, 0, 1, 0])


class TestFMQA:
    def test_prediction_agrees_with_the_bqm(self):
        rng = np.random.default_rng(2)
        fm = FactorizationMachine(6, rng, rank=3, steps=1)
        q = rng.integers(0, 2, size=6)
        pred = fm.predict_one(q)
        bqm = fm.to_bqm(minimise=False)
        assert abs(bqm_energy(bqm, q) - pred) < 1e-9

    def test_fit_reduces_error_on_a_quadratic_target(self):
        rng = np.random.default_rng(3)
        d = 8
        true = rng.normal(size=d)
        x = rng.integers(0, 2, size=(40, d)).astype(float)
        y = x @ true
        fm = FactorizationMachine(d, rng, rank=4, steps=150, lr=0.05)
        before = np.mean((fm.predict(x) - y) ** 2)
        fm.fit(x, y)
        after = np.mean((fm.predict(x) - y) ** 2)
        assert after < before


class TestBOCS:
    def test_constant_data_recovers_the_mean(self):
        rng = np.random.default_rng(4)
        d = 6
        x = rng.integers(0, 2, size=(12, d)).astype(float)
        y = np.full(12, 0.3)
        model = BOCSSurrogate(d, rng, n_gibbs=20, burnin=10)
        model.fit(x, y)
        pred = model.predict(x)
        assert abs(pred.mean() - 0.3) < 0.15


class TestRidgeCeiling:
    def test_exact_quadratic_has_zero_regret(self):
        rng = np.random.default_rng(5)
        d = 5
        alpha_true = rng.normal(size=n_quadratic(d))
        x = np.array(list(np.ndindex(*([2] * d))), dtype=float)
        y = evaluate_quadratic(x, alpha_true)
        fit = fit_ridge_quadratic(x, y, alphas=(1e-6, 1e-3, 1.0), n_splits=4, rng=rng)
        regret = argmin_regret(x, y, fit["alpha"], maximise=True)
        assert regret["regret"] < 1e-6
        assert fit["in_sample_r2"] > 0.99
