"""Recompute every committed numerical claim from artifacts.

Independent of the analysis scripts that produced the tables. A claim that cannot be
regenerated from raw files is not a claim.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class ClaimError(AssertionError):
    """A committed number disagrees with the raw artifact."""


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def check_fidelity(failures: list[str]) -> None:
    report = json.loads(
        (ROOT / "results/fidelity/fidelity_report.json").read_text(encoding="utf-8")
    )
    als = report["als"]
    vae = report["multvae"]
    check(
        abs(als["low_vs_high"]["spearman"] - 0.922) < 0.001, "ALS spearman drifted", failures
    )
    check(
        abs(vae["low_vs_high"]["spearman"] - 0.587) < 0.001,
        "MultVAE spearman drifted",
        failures,
    )
    check(als["halving"]["regret"] == 0.0, "ALS SH regret drifted", failures)
    check(vae["top_k_overlap"]["5"] == 0.2, "MultVAE top-5 overlap drifted", failures)
    csv = pd.read_csv(ROOT / "results/fidelity/fidelity.csv")
    check(len(csv) == 792, f"fidelity.csv has {len(csv)} rows, expected 792", failures)


def check_calibration(failures: list[str]) -> None:
    model = json.loads(
        (ROOT / "results/calibration/cost_model_threads1.json").read_text(encoding="utf-8")
    )
    check(abs(model["total_hours"] - 5.221) < 0.01, "campaign hour estimate drifted", failures)
    for catalogue, ratio in model["cpu_to_wall"].items():
        check(
            0.98 < ratio <= 1.01,
            f"{catalogue} cpu/wall {ratio} left the pinned band",
            failures,
        )


def check_space(failures: list[str]) -> None:
    from budget_tune.space.grids import binary_width, space_size

    check(space_size()["total"] == 471, "space is no longer 471", failures)
    check(binary_width()["variables"] == 44, "encoding width drifted", failures)
    check(binary_width()["surrogate_parameters"] == 991, "surrogate size drifted", failures)


def check_optional_campaign(failures: list[str]) -> None:
    search = ROOT / "results/benchmark/search.csv"
    if not search.exists():
        return
    frame = pd.read_csv(search)
    check(frame.dataset.nunique() >= 1, "benchmark has no datasets", failures)
    per = frame.groupby("dataset").size()
    for dataset, n in per.items():
        check(n == 471, f"{dataset} has {n} aggregated rows, expected 471", failures)
    runs = ROOT / "results/benchmark/search_runs.csv"
    if runs.exists():
        run_frame = pd.read_csv(runs)
        for dataset, n in run_frame.groupby("dataset").size().items():
            check(
                n == 1263,
                f"{dataset} has {n} per-seed rows, expected 1263",
                failures,
            )


def check_optional_rq0(failures: list[str]) -> None:
    path = ROOT / "results/rq0/oracle_surrogate.json"
    if not path.exists():
        return
    reports = json.loads(path.read_text(encoding="utf-8"))
    for report in reports:
        check("E1" in report and "regret" in report["E1"], "RQ0 missing E1 regret", failures)
        check(
            abs(float(report["E1"]["regret"])) < 1e-12,
            f"{report['dataset']} E1 regret drifted from 0",
            failures,
        )
        # The in-sample regret above says a quadratic can represent the ranking. The
        # held-out one says whether it can pick a cell it has not been shown, which is the
        # question docs/design.md asked. It is deliberately *not* required to be zero --
        # freezing it at a value would be inventing the result this experiment measures.
        e1 = report["E1"]
        check(
            "held_out_fold_regret" in e1 and e1["held_out_fold_regret"] is not None,
            f"{report['dataset']} E1 missing held_out_fold_regret",
            failures,
        )
        check(
            math.isfinite(float(e1.get("held_out_fold_regret", float("nan"))))
            and float(e1["held_out_fold_regret"]) >= 0.0,
            f"{report['dataset']} E1 held_out_fold_regret is not a finite non-negative number",
            failures,
        )
        check(
            int(e1.get("held_out_n_folds", 0)) >= 2,
            f"{report['dataset']} E1 held-out regret used < 2 folds",
            failures,
        )
        check(
            e1.get("regret_is_in_sample") is True,
            f"{report['dataset']} E1 no longer labels its in-sample regret as such",
            failures,
        )


def check_optional_hpo(failures: list[str]) -> None:
    for name in ("gift_cards", "ml100k", "luxury_beauty", "software"):
        path = ROOT / "results/hpo" / f"{name}_summary.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        check(
            len(frame) % 30 == 0 and len(frame) >= 240,
            f"{name} HPO has {len(frame)} rows, expected a multiple of 30 and >= 240",
            failures,
        )


def check_optional_split_bias(failures: list[str]) -> None:
    """The val/test family disagreement and the grid baseline's reach.

    These carry two of the report's sharper sentences -- that grid never evaluated a
    MultVAE or Markov configuration, and that the splits disagree about the winning family
    on exactly the catalogues where markov wins validation -- so they are regenerated here
    rather than trusted.
    """
    path = ROOT / "results/split_bias/summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))

    by_dataset = {row["dataset"]: row for row in summary["per_catalogue"]}
    check(
        len(by_dataset) == 4,
        f"split_bias covers {len(by_dataset)} catalogues, expected 4",
        failures,
    )

    for dataset, row in by_dataset.items():
        # The claim is that the grid *contains* the optimum and does not reach it -- two
        # different statements, and conflating them was the original error.
        check(
            row["val_optimum_in_grid_candidates"] is True,
            f"{dataset}: coarse grid no longer contains the validation optimum",
            failures,
        )
        check(
            "multvae" not in row["grid_families_reached"]
            and "markov" not in row["grid_families_reached"],
            f"{dataset}: grid now reaches {row['grid_families_reached']}; the report says "
            "it never evaluated a MultVAE or Markov configuration",
            failures,
        )

    disagree = {d for d, row in by_dataset.items() if not row["families_agree"]}
    check(
        disagree == {"software", "gift_cards"},
        f"val/test family disagreement is now {sorted(disagree)}, "
        "report says software and gift_cards",
        failures,
    )

    ratios = summary["markov_vs_other_families"]
    check(
        ratios["ml100k"]["markov_ratio"] > 1.0,
        "markov no longer keeps its full validation score on ml100k",
        failures,
    )
    for dataset in ("software", "luxury_beauty"):
        check(
            ratios[dataset]["markov_relative"] < 0.5,
            f"{dataset}: markov no longer collapses relative to the other families",
            failures,
        )


def check_optional_constrained(failures: list[str]) -> None:
    """RQ2: post-filter exact, slack-QUBO rarely optimal and sometimes infeasible."""
    paths = sorted((ROOT / "results/rq2").glob("*_constrained.json"))
    if not paths:
        return
    rows = [cell for path in paths for cell in json.loads(path.read_text(encoding="utf-8"))]
    check(len(rows) == 20, f"RQ2 has {len(rows)} cells, expected 20", failures)

    # Post-filter enumerates the feasible set, so it cannot be beaten by a feasible answer.
    beaten = [
        cell
        for cell in rows
        if cell.get("slack_qubo_feasible") is True
        and cell["slack_qubo_quality"] is not None
        and cell["slack_qubo_quality"] > cell["post_filter_quality"] + 1e-12
    ]
    check(
        not beaten,
        f"{len(beaten)} cells where a feasible QUBO beat exact post-filtering",
        failures,
    )

    matched = sum(
        1 for cell in rows if cell["slack_qubo_config_id"] == cell["post_filter_config_id"]
    )
    infeasible = sum(1 for cell in rows if cell.get("slack_qubo_feasible") is not True)
    check(
        matched == 1,
        f"slack-QUBO matched post-filter in {matched} cells, report says 1",
        failures,
    )
    check(
        infeasible == 3,
        f"slack-QUBO infeasible in {infeasible} cells, report says 3",
        failures,
    )


def check_optional_barrier(failures: list[str]) -> None:
    """RQ3: neal and tabu one-hot feasible before repair; tabu matched brute force."""
    paths = sorted((ROOT / "results/rq3").glob("*_barrier.json"))
    if not paths:
        return
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        name = report["dataset"]
        check(
            report["neal_feasible_before_repair"] is True
            and report["tabu_feasible_before_repair"] is True,
            f"{name}: a sampler is no longer one-hot feasible before repair",
            failures,
        )
        check(
            abs(report["tabu_surrogate"] - report["brute_surrogate"]) < 1e-12,
            f"{name}: tabu no longer recovers the brute-force surrogate argmax",
            failures,
        )


def check_optional_grid_interleaved(failures: list[str]) -> None:
    """The interleaved grid must still lose its test advantage on Software.

    The whole point of adding it was that ordering, not judgement, produced grid's test
    number. If this stops holding the report's grid section is wrong.
    """
    path = ROOT / "results/analyse/selected.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if "grid_interleaved" not in set(frame.method):
        return
    software = frame[frame.dataset == "software"]
    declaration = software[software.method == "grid"].test_ndcg_at_10.median()
    interleaved = software[software.method == "grid_interleaved"].test_ndcg_at_10.median()
    others = software[~software.method.isin(["grid", "grid_interleaved"])]
    check(
        declaration > 0.12,
        f"declaration-order grid Software test is {declaration:.4f}, report says ~0.134",
        failures,
    )
    check(
        abs(interleaved - others.test_ndcg_at_10.median()) < 0.01,
        f"interleaved grid Software test is {interleaved:.4f}, report says it joins the "
        "other methods near 0.044",
        failures,
    )


def main() -> int:
    failures: list[str] = []
    check_space(failures)
    check_fidelity(failures)
    check_calibration(failures)
    check_optional_campaign(failures)
    check_optional_rq0(failures)
    check_optional_hpo(failures)
    check_optional_split_bias(failures)
    check_optional_constrained(failures)
    check_optional_barrier(failures)
    check_optional_grid_interleaved(failures)
    print(f"{len(failures)} failures")
    for item in failures:
        print(f"  FAIL {item}")
    if failures:
        raise SystemExit(1)
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
