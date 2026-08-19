"""Why the selection split and the reporting split disagree about which family wins.

The audit found that on Software every method except grid lands at test NDCG ~0.044 while
grid reaches 0.134, and that the same family swap appears on Gift Cards. That is not an
optimiser result: the two splits disagree about which *family* is best, and grid's advantage
comes from a candidate set that cannot reach the disputed cell.

This recomputes the disagreement from the enumerated table alone -- no retraining, no HPO
rerun -- so the report can cite a file instead of an argument.

**The mechanism is a hypothesis, not a result.** Leave-two-out places validation one
interaction after the training history and test two after. A first-order Markov model
conditions on the last training item, so adjacency would favour it on validation; the ratios
below fit that story (collapse on the short-history Amazon catalogues, none on ML-100K) but
this experiment cannot separate it from any other family-dependent effect. It is reported as
the reading that fits, and labelled as such.

Run::

    python -m experiments.split_bias
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from budget_tune.space.grids import coarse_grid

BENCHMARK = Path("results/benchmark")
OUT = Path("results/split_bias")


def joined() -> pd.DataFrame:
    """The enumerated table with both splits side by side.

    This module is one of the few permitted to read the reporting split: it exists to study
    the relationship between the two, which cannot be done from one of them.
    """
    search = pd.read_csv(BENCHMARK / "search.csv")
    report = pd.read_csv(BENCHMARK / "report.csv")
    keep = ["dataset", "config_id", "test_ndcg_at_10"]
    return search.merge(report[keep], on=["dataset", "config_id"], validate="one_to_one")


def grid_reach(dataset: str, order: list[str]) -> dict:
    """How far the grid baseline actually got through its own candidate list.

    The grid baseline enumerates :func:`coarse_grid` in a fixed order and stops when the
    CPU budget runs out. Families are enumerated in declaration order, so a budget that
    expires early does not sample the space evenly -- it truncates it by family.
    """
    path = Path(f"results/hpo/{dataset}_grid_seed0.json")
    if not path.exists():
        return {}
    trials = json.loads(path.read_text(encoding="utf-8"))["trials"]
    evaluated = [trial["config_id"] for trial in trials]
    positions = [order.index(cid) for cid in evaluated if cid in order]
    reached = sorted({cid.split("|")[0] for cid in evaluated})
    return {
        "grid_candidates": len(order),
        "grid_evaluated": len(evaluated),
        "grid_last_position": (max(positions) + 1) if positions else 0,
        "grid_families_reached": reached,
        "grid_families_never_reached": [
            family
            for family in dict.fromkeys(cid.split("|")[0] for cid in order)
            if family not in reached
        ],
        "grid_evaluated_ids": set(evaluated),
    }


def per_catalogue(frame: pd.DataFrame, grid_ids: set[str]) -> list[dict]:
    order = [config.config_id for config in coarse_grid()]
    rows = []
    for dataset, group in frame.groupby("dataset"):
        val_best = group.loc[group.val_ndcg_at_10.idxmax()]
        test_best = group.loc[group.test_ndcg_at_10.idxmax()]

        # Can the grid baseline's candidate set even reach the validation optimum? If not,
        # its apparent test advantage is exclusion rather than judgement.
        in_grid = group[group.config_id.isin(grid_ids)]
        grid_val_best = in_grid.loc[in_grid.val_ndcg_at_10.idxmax()]
        reach = grid_reach(dataset, order)
        evaluated = reach.pop("grid_evaluated_ids", set())

        rows.append(
            {
                "dataset": dataset,
                "spearman_val_test": float(
                    group[["val_ndcg_at_10", "test_ndcg_at_10"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
                "val_argmax_family": val_best.family,
                "val_argmax_config_id": val_best.config_id,
                "val_argmax_val": float(val_best.val_ndcg_at_10),
                "val_argmax_test": float(val_best.test_ndcg_at_10),
                "test_argmax_family": test_best.family,
                "test_argmax_config_id": test_best.config_id,
                "test_argmax_val": float(test_best.val_ndcg_at_10),
                "test_argmax_test": float(test_best.test_ndcg_at_10),
                "families_agree": bool(val_best.family == test_best.family),
                # Two different questions, and conflating them was the audit's own error:
                # the grid *contains* the validation optimum, and never evaluates it.
                "val_optimum_in_grid_candidates": bool(val_best.config_id in grid_ids),
                "val_optimum_position_in_grid": (
                    order.index(val_best.config_id) + 1
                    if val_best.config_id in order
                    else None
                ),
                "val_optimum_evaluated_by_grid": bool(val_best.config_id in evaluated),
                **reach,
                "grid_best_val_config_id": grid_val_best.config_id,
                "grid_best_val_family": grid_val_best.family,
                "grid_best_val_test": float(grid_val_best.test_ndcg_at_10),
                # What selecting on the reporting split would have chosen instead.
                "counterfactual_test_selection_family": test_best.family,
                "counterfactual_test_selection_gain": float(
                    test_best.test_ndcg_at_10 - val_best.test_ndcg_at_10
                ),
            }
        )
    return rows


def ratios(frame: pd.DataFrame) -> list[dict]:
    """Median test/val ratio per catalogue and family: how much each family keeps."""
    work = frame.copy()
    work["ratio"] = work.test_ndcg_at_10 / work.val_ndcg_at_10.replace(0, pd.NA)
    out = []
    for (dataset, family), group in work.groupby(["dataset", "family"]):
        out.append(
            {
                "dataset": dataset,
                "family": family,
                "n": int(len(group)),
                "median_val": float(group.val_ndcg_at_10.median()),
                "median_test": float(group.test_ndcg_at_10.median()),
                "median_test_over_val": float(group.ratio.median()),
            }
        )
    return out


def main() -> int:
    frame = joined()
    grid_ids = {config.config_id for config in coarse_grid()}

    catalogues = per_catalogue(frame, grid_ids)
    family_ratios = ratios(frame)

    ratio_frame = pd.DataFrame(family_ratios)
    markov = ratio_frame[ratio_frame.family == "markov"].set_index("dataset")
    others = (
        ratio_frame[ratio_frame.family != "markov"]
        .groupby("dataset")
        .median_test_over_val.median()
    )

    summary = {
        "source": "results/benchmark/search.csv + report.csv, joined on (dataset, config_id)",
        "n_cells_per_catalogue": int(len(frame) / frame.dataset.nunique()),
        "grid_candidate_set_size": len(grid_ids),
        "per_catalogue": catalogues,
        "family_ratios": family_ratios,
        "markov_vs_other_families": {
            dataset: {
                "markov_ratio": float(markov.loc[dataset, "median_test_over_val"]),
                "other_families_median_ratio": float(others.loc[dataset]),
                "markov_relative": float(
                    markov.loc[dataset, "median_test_over_val"] / others.loc[dataset]
                ),
            }
            for dataset in markov.index
        },
        "reading": (
            "Grid's apparent test advantage is not a property of its candidate set: the "
            "coarse grid contains the validation optimum on every catalogue and never "
            "evaluates it, because it enumerates in declaration order and the CPU budget "
            "expires inside the third family. Reordering FAMILIES would change grid's "
            "answer. "
            "Validation and test disagree about the winning family on the catalogues where "
            "markov wins validation. Markov keeps a far smaller share of its validation "
            "score on the short-history Amazon catalogues and none of the loss on ML-100K. "
            "Leave-two-out places validation one interaction after training and test two, so "
            "temporal adjacency would favour a first-order Markov model on validation. That "
            "is the reading these ratios fit; this experiment does not establish it."
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(family_ratios).to_csv(OUT / "family_ratios.csv", index=False)
    pd.DataFrame(catalogues).to_csv(OUT / "per_catalogue.csv", index=False)

    for row in catalogues:
        verdict = "AGREE" if row["families_agree"] else "DISAGREE"
        print(
            f"{row['dataset']:<14} val-argmax {row['val_argmax_family']:<11} "
            f"-> test {row['val_argmax_test']:.4f} | "
            f"test-argmax {row['test_argmax_family']:<11} "
            f"-> test {row['test_argmax_test']:.4f} | {verdict}"
        )
    print()
    print("grid baseline: contains the validation optimum vs evaluates it")
    for row in catalogues:
        reached = ",".join(row.get("grid_families_reached", []))
        print(
            f"   {row['dataset']:<14} in candidates="
            f"{str(row['val_optimum_in_grid_candidates']):<5} at position "
            f"{row['val_optimum_position_in_grid']}/{row.get('grid_candidates')} | "
            f"evaluated {row.get('grid_evaluated')} of them, reached {reached} | "
            f"evaluated the optimum={row['val_optimum_evaluated_by_grid']}"
        )
    print(f"\nwrote {OUT}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
