"""Surrogates: quadratic features, horseshoe BOCS, FMQA, the RQ0 ridge ceiling."""

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

__all__ = [
    "BOCSSurrogate",
    "FactorizationMachine",
    "argmin_regret",
    "design_matrix",
    "evaluate_quadratic",
    "fit_ridge_quadratic",
    "n_quadratic",
    "pack_quadratic",
    "unpack_quadratic",
]
