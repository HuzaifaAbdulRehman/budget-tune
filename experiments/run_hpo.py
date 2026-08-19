"""RQ1: equal-cost HPO over the enumerated table.

Meta-parameters are chosen on Gift Cards and then frozen. Headline catalogues are a
separate invocation. Acquisition for BOCS/FMQA is brute force. The reporting split is
not opened here; ``experiments.analyse`` reads it once afterwards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from budget_tune.benchmark.schema import load_search
from budget_tune.optimizers import (
    bocs_proposer,
    checkpoint,
    fmqa_proposer,
    grid_interleaved_proposer,
    grid_proposer,
    hyperband_proposer,
    random_proposer,
    run,
    sm2_proposer,
    successive_halving_proposer,
    tpe_proposer,
)

METHODS = (
    "grid",
    "random",
    "tpe",
    "successive_halving",
    "hyperband",
    "sm2",
    "bocs",
    "fmqa",
    # Appended, never inserted. Seeds are derived from METHODS.index(method), so moving an
    # existing entry would silently reseed every run already measured.
    "grid_interleaved",
)


def make_proposer(name: str, rng: np.random.Generator, n_init: int, n_startup: int):
    if name == "grid":
        return grid_proposer()
    if name == "grid_interleaved":
        return grid_interleaved_proposer()
    if name == "random":
        return random_proposer(rng)
    if name == "tpe":
        return tpe_proposer(rng, n_startup_trials=n_startup)
    if name == "successive_halving":
        return successive_halving_proposer(rng)
    if name == "hyperband":
        return hyperband_proposer(rng)
    if name == "sm2":
        return sm2_proposer(rng)
    if name == "bocs":
        return bocs_proposer(rng, n_init=n_init, n_gibbs=40)
    if name == "fmqa":
        return fmqa_proposer(rng, n_init=n_init, steps=80)
    raise KeyError(name)


def budget_for(view, fraction: float) -> float:
    """Declared budget: a fraction of the table's total training CPU-seconds."""
    total = float(view.frame["train_cpu_seconds"].sum())
    return max(total * fraction, float(view.frame["train_cpu_seconds"].min()) * 8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/hpo"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--budget-fraction", type=float, default=0.10)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--n-startup", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    view = load_search(args.benchmark, args.dataset)
    budget = budget_for(view, args.budget_fraction)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": args.dataset,
        "budget_cpu_seconds": budget,
        "budget_fraction": args.budget_fraction,
        "n_init": args.n_init,
        "n_startup": args.n_startup,
        "methods": args.methods,
        "seeds": args.seeds,
    }
    (args.out / f"{args.dataset}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    rows = []
    for seed in args.seeds:
        for method in args.methods:
            rng = np.random.default_rng(
                seed + 17 * (METHODS.index(method) if method in METHODS else 0)
            )
            record = run(
                method,
                view,
                make_proposer(method, rng, args.n_init, args.n_startup),
                budget,
                seed=seed,
                max_steps=args.max_steps,
            )
            record["checkpoints"] = checkpoint(record)
            path = args.out / f"{args.dataset}_{method}_seed{seed}.json"
            path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
            rows.append(
                {
                    "dataset": args.dataset,
                    "method": method,
                    "seed": seed,
                    "best_quality": record["best_quality"],
                    "best_config_id": record["best_config_id"],
                    "n_unique": record["n_unique"],
                    "n_duplicates": record["n_duplicates"],
                    "spent_cpu_seconds": record["spent_cpu_seconds"],
                    "checkpoint_0.25": record["checkpoints"].get("0.25"),
                    "checkpoint_0.5": record["checkpoints"].get("0.5"),
                    "checkpoint_1.0": record["checkpoints"].get("1.0"),
                }
            )
            print(
                f"{method} seed={seed} best={record['best_quality']:.4f} "
                f"unique={record['n_unique']}"
            )
    # Merge rather than overwrite. Writing only the methods this invocation ran silently
    # deleted the other eight from the summary the first time a single method was re-run,
    # and `analyse` reads the summary -- so the whole comparison collapsed to one row set
    # while every per-seed JSON was still on disk and intact.
    summary_path = args.out / f"{args.dataset}_summary.csv"
    frame = pd.DataFrame(rows)
    if summary_path.exists():
        previous = pd.read_csv(summary_path)
        frame = pd.concat([previous, frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["dataset", "method", "seed"], keep="last")
    frame.sort_values(["method", "seed"]).to_csv(summary_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
