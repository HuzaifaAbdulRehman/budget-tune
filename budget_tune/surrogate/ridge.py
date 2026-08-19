"""Cross-validated ridge quadratic — the RQ0 ceiling, not an optimiser.

An unregularised least-squares fit on E1 is underdetermined (``p = 991``, ``n = 471``)
and would report ``R² = 1`` by interpolating. This module fits ridge with the regulariser
chosen by cross-validation.

**Two different regrets live here, and only one of them is held out.** Cross-validation
picks the regularisation strength; the returned coefficients then come from a fit on *every*
row. :func:`argmin_regret` scores that fit on the same rows, so its number says whether a
quadratic of this width can *represent* the ranking -- it does not say whether a surrogate
could find the best cell without having seen it. An earlier version of this docstring
claimed the opposite. :func:`held_out_regret` answers the second question by picking within
a fold the fit never saw, which is the quantity ``docs/design.md`` asked for.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from budget_tune.surrogate.features import design_matrix, evaluate_quadratic

#: Log-spaced ridge grid, frozen as a meta-parameter. Tuned only on the meta catalogue.
RIDGE_ALPHAS: tuple[float, ...] = tuple(10.0**exp for exp in range(-4, 5))


def fit_ridge_quadratic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    n_splits: int = 5,
    rng: np.random.Generator | None = None,
) -> dict:
    """Return held-out R², the chosen α, packed coefficients, and in-sample predictions."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    phi = design_matrix(x)
    n = phi.shape[0]
    if n < 3:
        alpha = np.zeros(phi.shape[1])
        alpha[0] = float(y.mean()) if n else 0.0
        return {
            "alpha": alpha,
            "ridge_alpha": None,
            "cv_r2": None,
            "in_sample_r2": None,
            "predictions": evaluate_quadratic(x, alpha) if n else np.array([]),
        }

    splits = min(n_splits, n)
    seed = int(rng.integers(0, 2**31)) if rng is not None else 0
    cv = KFold(n_splits=splits, shuffle=True, random_state=seed)
    ridge_alphas = np.asarray(alphas, dtype=float)
    model = RidgeCV(alphas=ridge_alphas, cv=cv, fit_intercept=False)
    model.fit(phi, y)
    pred = model.predict(phi)
    fold_scores = []
    for train_idx, test_idx in cv.split(phi):
        if len(train_idx) < 3:
            continue
        inner = min(3, len(train_idx))
        fold = RidgeCV(alphas=ridge_alphas, cv=inner, fit_intercept=False)
        fold.fit(phi[train_idx], y[train_idx])
        fold_scores.append(r2_score(y[test_idx], fold.predict(phi[test_idx])))
    return {
        "alpha": np.asarray(model.coef_, dtype=float),
        "ridge_alpha": float(model.alpha_),
        "cv_r2": float(np.mean(fold_scores)) if fold_scores else None,
        "in_sample_r2": float(r2_score(y, pred)),
        "predictions": pred,
    }


def argmin_regret(
    x: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    *,
    maximise: bool = True,
) -> dict:
    """Regret of the surrogate's argmin/argmax against the true best observed row.

    The surrogate is evaluated on the *same* rows it could propose — the enumerated
    table — so this is the RQ0 number: can a quadratic even point at a good cell?
    """
    pred = evaluate_quadratic(x, alpha)
    true_best = float(y.max() if maximise else y.min())
    pick = int(pred.argmax() if maximise else pred.argmin())
    picked_true = float(y[pick])
    return {
        "true_best": true_best,
        "picked_index": pick,
        "picked_true": picked_true,
        "regret": abs(true_best - picked_true),
        "regret_normalised": (
            abs(true_best - picked_true)
            / abs(true_best - float(y.min() if maximise else y.max()))
            if np.ptp(y) > 0
            else 0.0
        ),
    }


def held_out_regret(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    n_splits: int = 5,
    rng: np.random.Generator | None = None,
    maximise: bool = True,
) -> dict:
    """Regret of an argmax the surrogate was never allowed to see.

    For each fold: fit on the other folds, predict the held-out rows, take the surrogate's
    best *among those rows*, and compare its true value against the best truly available in
    that fold. This is the RQ0 question as ``docs/design.md`` posed it -- can a quadratic
    point at a good cell it has not already been shown?

    Two regrets are reported per fold and they answer different questions:

    * ``fold_regret`` -- against the best cell in that fold. What the surrogate gave up
      among the options it was choosing between.
    * ``global_regret`` -- against the best cell in the whole table. Larger by construction,
      because most folds do not contain the global optimum; useful only as an upper bound.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    phi = design_matrix(x)
    n = phi.shape[0]
    splits = min(n_splits, n)
    if n < 6 or splits < 2:
        return {"folds": [], "n_folds": 0, "note": f"too few rows ({n}) to hold any out"}

    seed = int(rng.integers(0, 2**31)) if rng is not None else 0
    cv = KFold(n_splits=splits, shuffle=True, random_state=seed)
    ridge_alphas = np.asarray(alphas, dtype=float)
    true_best = float(y.max() if maximise else y.min())
    spread = float(np.ptp(y))

    folds = []
    for train_idx, test_idx in cv.split(phi):
        if len(train_idx) < 3 or len(test_idx) < 1:
            continue
        inner = min(3, len(train_idx))
        model = RidgeCV(alphas=ridge_alphas, cv=inner, fit_intercept=False)
        model.fit(phi[train_idx], y[train_idx])
        pred = model.predict(phi[test_idx])
        local = int(pred.argmax() if maximise else pred.argmin())
        picked = float(y[test_idx][local])
        fold_best = float(y[test_idx].max() if maximise else y[test_idx].min())
        folds.append(
            {
                "n_held_out": int(len(test_idx)),
                "picked_true": picked,
                "fold_best": fold_best,
                "fold_regret": abs(fold_best - picked),
                "global_regret": abs(true_best - picked),
                "fold_contains_global_best": bool(abs(fold_best - true_best) < 1e-12),
            }
        )

    if not folds:
        return {"folds": [], "n_folds": 0, "note": "no usable folds"}

    fold_regrets = [f["fold_regret"] for f in folds]
    global_regrets = [f["global_regret"] for f in folds]
    return {
        "folds": folds,
        "n_folds": len(folds),
        "mean_fold_regret": float(np.mean(fold_regrets)),
        "max_fold_regret": float(max(fold_regrets)),
        "mean_fold_regret_normalised": (
            float(np.mean(fold_regrets) / spread) if spread > 0 else 0.0
        ),
        "mean_global_regret": float(np.mean(global_regrets)),
        "folds_finding_their_own_best": int(sum(1 for f in folds if f["fold_regret"] < 1e-12)),
        "true_best": true_best,
    }
