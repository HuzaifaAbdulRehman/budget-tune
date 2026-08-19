"""Budget accounting, leakage wall, and method wiring over a SearchView."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from budget_tune.benchmark.schema import LeakageError, SearchView
from budget_tune.optimizers import random_proposer, run
from budget_tune.space.grids import enumerate_configurations, hyperparameter_columns


def _tiny_view(n: int = 8) -> SearchView:
    rows = []
    for seed_quality, config in enumerate(enumerate_configurations()[:n]):
        row = {
            "config_id": config.config_id,
            "dataset": "synthetic",
            "family": config.family,
            "data_fraction": config.data_fraction,
            "val_ndcg_at_10": 0.1 + seed_quality / 100,
            "val_recall_at_10": 0.2,
            "val_exposure_parity": 0.3,
            "train_cpu_seconds": 1.0 + seed_quality,
            "train_wall_seconds": 1.0,
            "score_cpu_seconds": 0.1,
            "score_wall_seconds": 0.1,
            "select_cpu_seconds": 0.01,
            "serve_cpu_seconds_per_request": 0.001,
            "peak_rss_bytes": 1024,
            "model_bytes": 512,
            "n_train_interactions": 100,
            "n_eval_users": 50,
            "train_repeats": 1,
            "score_repeats": 1,
            "train_below_quantum": False,
            "score_below_quantum": False,
            "other_cores": 0.0,
        }
        row.update({col: None for col in hyperparameter_columns()})
        row.update({f"{config.family}.{name}": value for name, value in config.params})
        rows.append(row)
    frame = pd.DataFrame(rows)
    return SearchView(frame=frame, dataset="synthetic")


class TestBudget:
    def test_duplicates_are_charged_zero_training(self):
        view = _tiny_view()
        ids = view.config_ids()

        # Always propose the first id.
        def propose(history):
            return ids[0]

        record = run("stuck", view, propose, budget_cpu_seconds=0.5, seed=0, max_steps=4)
        charged = [t["train_cpu_seconds_charged"] for t in record["trials"]]
        assert charged[0] > 0
        assert all(c == 0.0 for c in charged[1:])
        assert record["n_unique"] == 1
        assert record["n_duplicates"] == len(record["trials"]) - 1

    def test_random_stays_inside_the_view(self):
        view = _tiny_view()
        rng = np.random.default_rng(0)
        record = run(
            "random", view, random_proposer(rng), budget_cpu_seconds=200.0, seed=0, max_steps=6
        )
        assert record["n_trials"] == 6
        assert record["best_config_id"] in set(view.config_ids())
        assert record["best_quality"] == max(t["quality"] for t in record["trials"])


class TestLeakage:
    def test_search_view_still_refuses_a_reporting_column(self):
        view = _tiny_view()
        with pytest.raises(LeakageError):
            view.column("test_ndcg_at_10")
