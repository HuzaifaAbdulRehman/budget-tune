"""Schema invariants, and the structural guarantee that search cannot see the test split.

With an enumerated benchmark, test-set leakage is one column name away and would improve
every number it touched. Discipline is not a control for that. These tests make the
separation a property of the code: the column sets are disjoint, the view refuses reporting
columns by name rather than by absence, and no module outside the benchmark and report
packages so much as mentions one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from budget_tune.benchmark import schema
from budget_tune.benchmark.schema import (
    AGGREGATED_COLUMNS,
    REPORT_QUALITY_COLUMNS,
    REPORT_RUN_COLUMNS,
    SEARCH_RUN_COLUMNS,
    LeakageError,
    SearchView,
    aggregate,
    load_report,
    load_search,
    validate_runs,
    write,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "budget_tune"

#: Packages allowed to name a reporting column. Everything else in the package is
#: search-side by definition, including every optimiser, surrogate and solver added later,
#: so this test keeps working as the project grows rather than needing to be updated.
REPORT_AWARE = {"benchmark", "report"}


def _runs(n_seeds: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    search_rows, report_rows = [], []
    for config_id, family, fraction in (
        ("als|factors=16|frac=1.0", "als", 1.0),
        ("markov|order=1|frac=0.25", "markov", 0.25),
    ):
        for seed in range(n_seeds):
            search_rows.append(
                {
                    "config_id": config_id,
                    "dataset": "synthetic",
                    "family": family,
                    "data_fraction": fraction,
                    "seed": seed,
                    "val_ndcg_at_10": 0.10 + seed / 100,
                    "val_recall_at_10": 0.20,
                    "val_exposure_parity": 0.30,
                    "train_cpu_seconds": 1.0 + seed,
                    "train_wall_seconds": 1.0 + seed,
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
                    "other_cores": 0.02,
                }
            )
            report_rows.append(
                {
                    "config_id": config_id,
                    "dataset": "synthetic",
                    "family": family,
                    "data_fraction": fraction,
                    "seed": seed,
                    "test_ndcg_at_10": 0.05,
                    "test_recall_at_10": 0.10,
                }
            )
    return pd.DataFrame(search_rows), pd.DataFrame(report_rows)


class TestColumnSets:
    def test_search_and_report_columns_are_disjoint(self):
        assert not set(SEARCH_RUN_COLUMNS) & set(REPORT_QUALITY_COLUMNS)

    def test_report_runs_carry_no_cost_or_search_metric(self):
        """The reporting file exists to hold test metrics and nothing else.

        Cost columns duplicated into it would be a second, unvalidated copy of the cost
        axis, and the two could drift.
        """
        extra = set(REPORT_RUN_COLUMNS) - set(schema.RUN_KEY_COLUMNS)
        assert extra == set(REPORT_QUALITY_COLUMNS)


class TestValidation:
    def test_accepts_a_well_formed_frame(self):
        search, report = _runs()
        validate_runs(search, "search")
        validate_runs(report, "report")

    def test_rejects_missing_columns(self):
        search, _ = _runs()
        with pytest.raises(ValueError, match="missing"):
            validate_runs(search.drop(columns=["train_cpu_seconds"]), "search")

    def test_rejects_unknown_columns(self):
        """A misspelled column must fail rather than arrive as silent missing data."""
        search, _ = _runs()
        search["train_cpu_secondss"] = 1.0
        with pytest.raises(ValueError, match="unknown"):
            validate_runs(search, "search")

    def test_rejects_a_reporting_column_in_the_search_frame(self):
        search, _ = _runs()
        search["test_ndcg_at_10"] = 0.9
        with pytest.raises(LeakageError):
            validate_runs(search, "search")

    def test_rejects_duplicate_measurements(self):
        """One row per (configuration, seed), or the median is over the wrong sample."""
        search, _ = _runs()
        with pytest.raises(ValueError, match="duplicate"):
            validate_runs(pd.concat([search, search.head(1)]), "search")


class TestAggregation:
    def test_median_and_range(self):
        search, _ = _runs(n_seeds=3)
        out = aggregate(search, AGGREGATED_COLUMNS)

        row = out[out.config_id == "als|factors=16|frac=1.0"].iloc[0]
        # Seeds contribute 1.0, 2.0, 3.0 CPU-seconds.
        assert row["train_cpu_seconds"] == pytest.approx(2.0)
        assert row["train_cpu_seconds_spread"] == pytest.approx(2.0)
        assert row["n_seeds"] == 3

    def test_a_missing_seed_is_visible_rather_than_averaged_away(self):
        search, _ = _runs(n_seeds=3)
        out = aggregate(search.drop(search.index[0]), AGGREGATED_COLUMNS)
        counts = dict(zip(out.config_id, out.n_seeds, strict=True))
        assert counts["als|factors=16|frac=1.0"] == 2

    def test_one_row_per_configuration(self):
        search, _ = _runs()
        out = aggregate(search, AGGREGATED_COLUMNS)
        assert len(out) == search.config_id.nunique()


class TestFiles:
    def test_round_trip(self, tmp_path):
        search, report = _runs()
        write(tmp_path, search, report)

        for name in (schema.SEARCH_FILE, schema.SEARCH_RUNS_FILE,
                     schema.REPORT_FILE, schema.REPORT_RUNS_FILE):
            assert (tmp_path / name).exists()

        view = load_search(tmp_path, "synthetic")
        assert len(view) == 2
        assert set(view.config_ids()) == set(search.config_id)

        reported = load_report(tmp_path, "synthetic")
        assert set(reported.columns) >= set(REPORT_QUALITY_COLUMNS)

    def test_search_files_contain_no_test_metric(self, tmp_path):
        search, report = _runs()
        write(tmp_path, search, report)
        for name in (schema.SEARCH_FILE, schema.SEARCH_RUNS_FILE):
            text = (tmp_path / name).read_text(encoding="utf-8")
            for column in REPORT_QUALITY_COLUMNS:
                assert column not in text

    def test_mismatched_coverage_is_refused(self, tmp_path):
        """A configuration measured on one side and not the other cannot be reported.

        Finding that out at report time -- after the machine conditions the campaign ran
        under are gone -- means re-running the campaign, so it is caught at write time.
        """
        search, report = _runs()
        with pytest.raises(ValueError, match="different"):
            write(tmp_path, search, report.drop(report.index[0]))


class TestSearchView:
    def test_refuses_construction_with_reporting_columns(self):
        search, _ = _runs()
        frame = aggregate(search, AGGREGATED_COLUMNS)
        frame["test_ndcg_at_10"] = 0.5
        with pytest.raises(LeakageError):
            SearchView(frame=frame, dataset="synthetic")

    def test_refuses_a_reporting_column_by_name(self):
        """Refused by name, not merely absent.

        An optimiser asking for ``test_ndcg_at_10`` gets an error that says what it did
        wrong, rather than a ``KeyError`` that reads like a typo and invites a workaround.
        """
        search, _ = _runs()
        view = SearchView(frame=aggregate(search, AGGREGATED_COLUMNS), dataset="synthetic")
        with pytest.raises(LeakageError, match="reporting column"):
            view.column("test_ndcg_at_10")

    def test_lookup_and_best(self):
        search, _ = _runs()
        view = SearchView(frame=aggregate(search, AGGREGATED_COLUMNS), dataset="synthetic")
        row = view.lookup("markov|order=1|frac=0.25")
        assert row["family"] == "markov"
        assert view.best()["val_ndcg_at_10"] == view.column("val_ndcg_at_10").max()


class TestNoSearchSideCodeTouchesTheReportSplit:
    """The structural half of the guarantee, and it must survive the project growing.

    Rather than listing the modules that exist today, this walks the package and exempts
    only the two subpackages whose job is reporting. Every optimiser, surrogate and solver
    added later is covered the moment it is written.
    """

    def _offenders(self, needles: list[str]) -> list[str]:
        found = []
        for path in PACKAGE_ROOT.rglob("*.py"):
            relative = path.relative_to(PACKAGE_ROOT)
            if relative.parts and relative.parts[0] in REPORT_AWARE:
                continue
            text = path.read_text(encoding="utf-8")
            if any(needle in text for needle in needles):
                found.append(str(relative))
        return found

    def test_no_module_names_a_reporting_column(self):
        assert self._offenders(REPORT_QUALITY_COLUMNS) == []

    def test_no_module_names_a_reporting_file(self):
        assert self._offenders(schema.REPORT_FILES) == []

    def test_no_module_calls_load_report(self):
        assert self._offenders(["load_report"]) == []
