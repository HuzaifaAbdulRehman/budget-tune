"""H1: the cost–accuracy frontier of the enumerated table.

Independent of QUBO. Reads only the search side. Reports whether ``data_fraction``
moves cost more than the ordinary hyperparameters — the pilot said no for ALS/MultVAE;
this checks the full table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def frontier(frame: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated rows: no other row is both cheaper and more accurate."""
    cost = frame["train_cpu_seconds"].to_numpy()
    quality = frame["val_ndcg_at_10"].to_numpy()
    keep = []
    for i in range(len(frame)):
        dominated = np.any(
            (cost <= cost[i])
            & (quality >= quality[i])
            & ((cost < cost[i]) | (quality > quality[i]))
        )
        if not dominated:
            keep.append(i)
    return frame.iloc[keep].sort_values("train_cpu_seconds")


def fraction_cost_ratios(frame: pd.DataFrame) -> dict:
    """Median train CPU at f=1.0 over f=0.25, grouped by family."""
    out = {}
    for family, group in frame.groupby("family"):
        low = group[group.data_fraction == 0.25]["train_cpu_seconds"].median()
        high = group[group.data_fraction == 1.0]["train_cpu_seconds"].median()
        if low and low > 0:
            out[family] = float(high / low)
        else:
            out[family] = None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--out", type=Path, default=Path("results/h1"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    search = pd.read_csv(args.benchmark / "search.csv")
    payload = {}
    for dataset, frame in search.groupby("dataset"):
        front = frontier(frame)
        payload[dataset] = {
            "n": int(len(frame)),
            "quality_range": [
                float(frame.val_ndcg_at_10.min()),
                float(frame.val_ndcg_at_10.max()),
            ],
            "cost_range": [
                float(frame.train_cpu_seconds.min()),
                float(frame.train_cpu_seconds.max()),
            ],
            "frontier_size": int(len(front)),
            "fraction_cost_ratio_1_over_0_25": fraction_cost_ratios(frame),
            "frontier_config_ids": list(front.config_id),
        }
        front.to_csv(args.out / f"{dataset}_frontier.csv", index=False)
    (args.out / "frontier.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
