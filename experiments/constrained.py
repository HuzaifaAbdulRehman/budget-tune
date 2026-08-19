"""RQ2: constrained selection — post-filter vs scalarisation vs slack QUBO.

``maximise validation quality s.t. train CPU-seconds ≤ τ``. The feasible set is known
exactly because the table is enumerated. When the space is enumerable, post-filtering
is exact and free; if the QUBO loses here it loses on the axis it was supposed to win.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from budget_tune.benchmark.schema import load_search
from budget_tune.qubo.acquisition import alpha_to_bqm
from budget_tune.qubo.onehot import onehot_penalty, penalty_strength
from budget_tune.qubo.slack import slack_inequality
from budget_tune.space.codec import N_VARIABLES, decode, encode
from budget_tune.space.grids import configuration_from_row
from budget_tune.surrogate.features import unpack_quadratic
from budget_tune.surrogate.ridge import fit_ridge_quadratic


def post_filter(frame: pd.DataFrame, tau: float) -> dict:
    feasible = frame[frame.train_cpu_seconds <= tau]
    if feasible.empty:
        return {"config_id": None, "quality": None, "cost": None, "feasible": False}
    best = feasible.loc[feasible.val_ndcg_at_10.idxmax()]
    return {
        "config_id": best.config_id,
        "quality": float(best.val_ndcg_at_10),
        "cost": float(best.train_cpu_seconds),
        "feasible": True,
    }


def scalarise(frame: pd.DataFrame, rho: float) -> dict:
    score = frame.val_ndcg_at_10 - rho * frame.train_cpu_seconds
    best = frame.loc[score.idxmax()]
    return {
        "config_id": best.config_id,
        "quality": float(best.val_ndcg_at_10),
        "cost": float(best.train_cpu_seconds),
        "score": float(score.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/rq2"))
    parser.add_argument(
        "--quantiles", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 1.0]
    )
    args = parser.parse_args()
    view = load_search(args.benchmark, args.dataset)
    frame = view.frame
    x, y, costs = [], [], []
    for _, row in frame.iterrows():
        config = configuration_from_row(row)
        x.append(encode(config, mode="gated"))
        y.append(float(row["val_ndcg_at_10"]))
        costs.append(float(row["train_cpu_seconds"]))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    costs = np.asarray(costs, dtype=float)
    quality_fit = fit_ridge_quadratic(x, y)
    # Linear cost model over bits: the slack encoding is Σ c_i x_i ≤ τ, which is
    # misspecified for a gated one-hot of a CASH space. That misspecification is the
    # thing RQ2 measures against exact post-filter.
    cost_fit = fit_ridge_quadratic(x, costs)
    _, cost_linear, _ = unpack_quadratic(cost_fit["alpha"], N_VARIABLES)

    taus = [float(frame.train_cpu_seconds.quantile(q)) for q in args.quantiles]
    rows = []
    for q, tau in zip(args.quantiles, taus, strict=True):
        filtered = post_filter(frame, tau)
        rho = (frame.val_ndcg_at_10.max() / max(tau, 1e-9)) * 0.1
        scaled = scalarise(frame, rho)
        slack_row = {"config_id": None, "quality": None, "feasible": None}
        try:
            from budget_tune.solvers.penalty import best_sample_bits, sample_penalty_neal

            objective = alpha_to_bqm(quality_fit["alpha"], N_VARIABLES, minimise=True)
            strength = penalty_strength(objective, margin=2.0)
            composed = (
                objective
                + onehot_penalty(strength)
                + slack_inequality(cost_linear, tau, strength)
            )
            rng = np.random.default_rng(0)
            bits = best_sample_bits(
                sample_penalty_neal(composed, rng, num_reads=20, num_sweeps=200),
                N_VARIABLES,
            )
            recovered = decode(bits, repair="argmax")
            hit = frame[frame.config_id == recovered.config_id]
            if not hit.empty:
                slack_row = {
                    "config_id": recovered.config_id,
                    "quality": float(hit.val_ndcg_at_10.iloc[0]),
                    "feasible": float(hit.train_cpu_seconds.iloc[0]) <= tau,
                }
        except Exception as exc:  # noqa: BLE001 — diagnostic
            slack_row = {"config_id": None, "quality": None, "feasible": f"{exc}"}
        rows.append(
            {
                "dataset": args.dataset,
                "quantile": q,
                "tau": tau,
                "feasible_fraction": float((frame.train_cpu_seconds <= tau).mean()),
                "post_filter_config_id": filtered["config_id"],
                "post_filter_quality": filtered["quality"],
                "scalarise_config_id": scaled["config_id"],
                "scalarise_quality": scaled["quality"],
                "scalarise_feasible": scaled["cost"] <= tau,
                "slack_qubo_config_id": slack_row["config_id"],
                "slack_qubo_quality": slack_row["quality"],
                "slack_qubo_feasible": slack_row["feasible"],
            }
        )
    args.out.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(args.out / f"{args.dataset}_constrained.csv", index=False)
    (args.out / f"{args.dataset}_constrained.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
