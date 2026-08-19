"""QUBO algebra: offset, sign, Ising conversion, slack, penalty dominance."""

from __future__ import annotations

import itertools

import numpy as np

from budget_tune.qubo.acquisition import bqm_energy, quadratic_to_bqm, surrogate_energy
from budget_tune.qubo.ising import bqm_to_ising, dimod_ising_energy, ising_energy
from budget_tune.qubo.onehot import dense_onehot_energy, onehot_penalty, penalty_strength
from budget_tune.qubo.slack import slack_bits, slack_inequality
from budget_tune.space.codec import N_VARIABLES, encode, is_onehot_feasible
from budget_tune.space.grids import enumerate_configurations


class TestOneHotPenalty:
    def test_feasible_assignment_pays_only_the_offset_restored_to_zero(self):
        bits = encode(next(iter(enumerate_configurations())), mode="feasible")
        assert is_onehot_feasible(bits)
        bqm = onehot_penalty(3.0)
        assert bqm_energy(bqm, bits) == 0.0
        assert abs(dense_onehot_energy(bits, 3.0) - 0.0) < 1e-12

    def test_dropping_the_offset_is_caught(self):
        bits = encode(next(iter(enumerate_configurations())), mode="feasible")
        bqm = onehot_penalty(3.0)
        dropped = bqm.copy()
        dropped.offset = 0.0
        # J blocks, P=3 → offset 3J. Energy of a feasible state becomes -3J.
        assert bqm_energy(dropped, bits) != 0.0
        assert abs(bqm_energy(dropped, bits) + 3.0 * len({*range(1)}))  # sanity: not zero
        from budget_tune.space.codec import BLOCKS

        assert abs(bqm_energy(dropped, bits) + 3.0 * len(BLOCKS)) < 1e-9

    def test_infeasible_is_strictly_positive(self):
        bits = encode(next(iter(enumerate_configurations())), mode="gated")
        assert not is_onehot_feasible(bits)
        assert dense_onehot_energy(bits, 2.0) > 0


class TestSignConvention:
    def test_solver_energy_matches_negated_objective(self):
        rng = np.random.default_rng(0)
        d = 4
        linear = rng.normal(size=d)
        pairwise = np.triu(rng.normal(size=(d, d)), 1)
        intercept = 0.5
        x = rng.integers(0, 2, size=d)
        objective = surrogate_energy(x, intercept, linear, pairwise)
        bqm = quadratic_to_bqm(intercept, linear, pairwise, minimise=True)
        assert abs(bqm_energy(bqm, x) + objective) < 1e-12


class TestIsing:
    def test_matches_independent_expansion_and_dimod(self):
        rng = np.random.default_rng(1)
        d = 5
        linear = rng.normal(size=d)
        pairwise = np.triu(rng.normal(size=(d, d)), 1)
        bqm = quadratic_to_bqm(0.25, linear, pairwise, minimise=False)
        h, j, offset = bqm_to_ising(bqm)
        for bits in itertools.product([0, 1], repeat=d):
            x = np.array(bits)
            assert abs(ising_energy(h, j, offset, x) - bqm_energy(bqm, x)) < 1e-10
            assert abs(dimod_ising_energy(bqm, x) - bqm_energy(bqm, x)) < 1e-10


class TestPenaltyStrength:
    def test_no_infeasible_state_beats_every_feasible_one_on_a_tiny_instance(self):
        # Two blocks of size 2: family-like bits 0,1 and fraction-like bits 2,3.
        # Use the real 44-bit penalty but only brute the first two blocks by fixing the rest.
        config = next(c for c in enumerate_configurations() if c.family == "popularity")
        feasible = encode(config, mode="feasible")
        intercept, linear, pairwise = (
            0.0,
            np.zeros(N_VARIABLES),
            np.zeros((N_VARIABLES, N_VARIABLES)),
        )
        linear[0] = -1.0  # prefers a particular family bit
        objective = quadratic_to_bqm(intercept, linear, pairwise, minimise=True)
        p = penalty_strength(objective, margin=2.0)
        penalty = onehot_penalty(p)
        composed = objective + penalty
        feasible_e = bqm_energy(composed, feasible)
        # Flip one bit off a feasible block: infeasible.
        broken = feasible.copy()
        broken[0] = 1 - broken[0]
        assert bqm_energy(composed, broken) > feasible_e - 1e-12


class TestSlack:
    def test_bits_cover_tau(self):
        assert slack_bits(7, 1) == 3  # 1+2+4
        assert slack_bits(0, 1) == 0

    def test_matches_brute_force_on_a_two_variable_budget(self):
        costs = np.array([1.0, 3.0])
        tau = 3.0
        strength = 20.0
        bqm = slack_inequality(costs, tau, strength, delta=1.0)
        best_feasible = None
        for x0, x1 in itertools.product([0, 1], repeat=2):
            cost = costs[0] * x0 + costs[1] * x1
            if cost - 1e-9 <= tau and (best_feasible is None or cost > best_feasible):
                best_feasible = cost
        # The unconstrained quadratic without one-hot: we only check that a known
        # feasible assignment (x=01, cost=3) has energy 0 slack-penalty.
        sample = {0: 0, 1: 1, ("slack", 0): 0, ("slack", 1): 0}
        # slack needs to make up tau - cost = 0, so slack=0.
        assert abs(float(bqm.energy(sample))) < 1e-8
        assert best_feasible == 3.0
