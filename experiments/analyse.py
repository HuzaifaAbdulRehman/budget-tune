"""Read the reporting split once and score the configurations HPO selected.

This is the only experiment script that may open ``report.csv``. The manifest records
the read so a split used once per project and a split used after every idea stay
distinguishable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from budget_tune.benchmark.schema import load_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--hpo", type=Path, default=Path("results/hpo"))
    parser.add_argument("--out", type=Path, default=Path("results/analyse"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = list(args.hpo.glob("*_summary.csv"))
    if not summaries:
        print("no HPO summaries found")
        return 1

    reads = 0
    frames = []
    for path in summaries:
        summary = pd.read_csv(path)
        dataset = str(summary.dataset.iloc[0])
        report = load_report(args.benchmark, dataset)
        reads += 1
        merged = summary.merge(
            report[["config_id", "test_ndcg_at_10", "test_recall_at_10"]],
            left_on="best_config_id",
            right_on="config_id",
            how="left",
            suffixes=("", "_report"),
        )
        frames.append(merged)
        merged.to_csv(args.out / f"{dataset}_selected.csv", index=False)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.out / "selected.csv", index=False)
    manifest = {"report_reads": reads, "n_rows": int(len(combined))}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"report.csv was read {reads} time(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
