"""RQ3: does penalty-encoded one-hot cost anything on these acquisition problems?

Harvest one quadratic from the enumerated table (the RQ0 ridge fit), convert it to a
QUBO with a one-hot penalty, and compare brute force / categorical SA / penalty neal
on recovered quality and feasibility-before-repair. A null result is in-scope: d=44
with blocks of 2–5 is much smaller than the companion's n=200, k=10 instance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from budget_tune.benchmark.schema import load_search
from budget_tune.qubo.acquisition import alpha_to_bqm, bqm_energy
from budget_tune.qubo.onehot import onehot_penalty, penalty_strength
from budget_tune.solvers.brute import brute_maximise
from budget_tune.solvers.categorical import categorical_sa
from budget_tune.space.codec import N_VARIABLES, decode, encode, is_onehot_feasible
from budget_tune.space.grids import configuration_from_row
from budget_tune.surrogate.features import evaluate_quadratic
from budget_tune.surrogate.ridge import fit_ridge_quadratic


def _table(view):
    x, y, configs = [], [], []
    for _, row in view.frame.iterrows():
        config = configuration_from_row(row)
        x.append(encode(config, mode="gated"))
        y.append(float(row["val_ndcg_at_10"]))
        configs.append(config)
    return np.asarray(x), np.asarray(y), configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--dataset", default="gift_cards")
    parser.add_argument("--out", type=Path, default=Path("results/rq3"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--penalty-reads", type=int, default=20)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    view = load_search(args.benchmark, args.dataset)
    x, y, configs = _table(view)
    fit = fit_ridge_quadratic(x, y, rng=rng)
    alpha = fit["alpha"]

    def score(bits, config):
        return float(evaluate_quadratic(bits, alpha)[0])

    brute_cfg, brute_val = brute_maximise(score, configs, mode="gated")
    sa_cfg, sa_val = categorical_sa(score, configs[0], rng, minimise=False, mode="feasible")

    objective = alpha_to_bqm(alpha, N_VARIABLES, minimise=True)
    strength = penalty_strength(objective, margin=2.0)
    composed = objective + onehot_penalty(strength)

    payload = {
        "dataset": args.dataset,
        "brute_config_id": brute_cfg.config_id,
        "brute_surrogate": brute_val,
        "categorical_sa_config_id": sa_cfg.config_id,
        "categorical_sa_surrogate": sa_val,
        "penalty_strength": strength,
        "cv_r2": fit["cv_r2"],
    }

    def _record(name: str, sampler) -> None:
        try:
            response = sampler()
            bits = best_sample_bits(response, N_VARIABLES)
            payload[f"{name}_feasible_before_repair"] = bool(is_onehot_feasible(bits))
            payload[f"{name}_energy"] = bqm_energy(composed, bits)
            recovered = decode(bits, repair="argmax")
            payload[f"{name}_config_id"] = recovered.config_id
            payload[f"{name}_surrogate"] = score(encode(recovered, mode="gated"), recovered)
        except Exception as exc:  # noqa: BLE001 — diagnostic, not control flow
            payload[name] = f"unavailable: {exc}"

    try:
        from budget_tune.solvers.penalty import (
            best_sample_bits,
            sample_penalty_neal,
            sample_penalty_sb,
            sample_penalty_tabu,
        )

        _record(
            "neal",
            lambda: sample_penalty_neal(
                composed, rng, num_reads=args.penalty_reads, num_sweeps=200
            ),
        )
        _record("tabu", lambda: sample_penalty_tabu(composed, rng, timeout=1.0))
        _record("sb", lambda: sample_penalty_sb(composed, rng))
    except Exception as exc:  # noqa: BLE001
        payload["penalty_samplers"] = f"unavailable: {exc}"

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.dataset}_barrier.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
