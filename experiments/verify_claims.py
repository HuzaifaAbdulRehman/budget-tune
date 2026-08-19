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


def main() -> int:
    failures: list[str] = []
    check_space(failures)
    check_fidelity(failures)
    check_calibration(failures)
    check_optional_campaign(failures)
    check_optional_rq0(failures)
    check_optional_hpo(failures)
    print(f"{len(failures)} failures")
    for item in failures:
        print(f"  FAIL {item}")
    if failures:
        raise SystemExit(1)
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
