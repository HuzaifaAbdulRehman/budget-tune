"""Acquisition functions as BQMs: one-hot penalties, slack inequalities, Ising checks."""

from budget_tune.qubo.acquisition import quadratic_to_bqm, surrogate_energy
from budget_tune.qubo.ising import bqm_to_ising, ising_energy
from budget_tune.qubo.onehot import onehot_penalty, penalty_strength
from budget_tune.qubo.slack import slack_inequality

__all__ = [
    "bqm_to_ising",
    "ising_energy",
    "onehot_penalty",
    "penalty_strength",
    "quadratic_to_bqm",
    "slack_inequality",
    "surrogate_energy",
]
