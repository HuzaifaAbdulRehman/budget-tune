"""RQ0: can a quadratic even point at a good configuration?

Cross-validated ridge over the enumerated search table, E1 (flat gated) and E2
(per family). Run after the campaign. Does not open the reporting split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from budget_tune.benchmark.schema import load_search
from budget_tune.space.codec import encode, encode_family
from budget_tune.space.grids import FAMILIES, configuration_from_row
from budget_tune.surrogate.ridge import argmin_regret, fit_ridge_quadratic


def _xy(frame: pd.DataFrame, mode: str, family: str | None = None):
    rows = frame if family is None else frame[frame.family == family]
    x = []
    y = []
    ids = []
    for _, row in rows.iterrows():
        config = configuration_from_row(row)
        bits = encode(config, mode="gated") if mode == "E1" else encode_family(config)
        x.append(bits)
        y.append(float(row["val_ndcg_at_10"]))
        ids.append(config.config_id)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), ids


def analyse_dataset(directory: Path, dataset: str, rng: np.random.Generator) -> dict:
    view = load_search(directory, dataset)
    frame = view.frame
    report: dict = {"dataset": dataset, "n": len(frame), "E1": {}, "E2": {}}

    x, y, ids = _xy(frame, "E1")
    fit = fit_ridge_quadratic(x, y, rng=rng)
    regret = argmin_regret(x, y, fit["alpha"], maximise=True)
    report["E1"] = {
        "variables": int(x.shape[1]),
        "ridge_alpha": fit["ridge_alpha"],
        "cv_r2": fit["cv_r2"],
        "in_sample_r2": fit["in_sample_r2"],
        "true_best": regret["true_best"],
        "picked_config_id": ids[regret["picked_index"]],
        "picked_true": regret["picked_true"],
        "regret": regret["regret"],
        "regret_normalised": regret["regret_normalised"],
    }

    for spec in FAMILIES:
        sub = frame[frame.family == spec.name]
        if sub.empty:
            continue
        xf, yf, idf = _xy(frame, "E2", family=spec.name)
        fit_f = fit_ridge_quadratic(xf, yf, rng=rng)
        regret_f = argmin_regret(xf, yf, fit_f["alpha"], maximise=True)
        report["E2"][spec.name] = {
            "n": int(len(sub)),
            "variables": int(xf.shape[1]),
            "ridge_alpha": fit_f["ridge_alpha"],
            "cv_r2": fit_f["cv_r2"],
            "in_sample_r2": fit_f["in_sample_r2"],
            "true_best": regret_f["true_best"],
            "picked_config_id": idf[regret_f["picked_index"]],
            "picked_true": regret_f["picked_true"],
            "regret": regret_f["regret"],
            "regret_normalised": regret_f["regret_normalised"],
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--out", type=Path, default=Path("results/rq0"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    search = pd.read_csv(args.benchmark / "search.csv")
    datasets = list(search.dataset.unique())
    reports = [analyse_dataset(args.benchmark, name, rng) for name in datasets]
    (args.out / "oracle_surrogate.json").write_text(
        json.dumps(reports, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(reports, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
