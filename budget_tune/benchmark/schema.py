"""The benchmark table: what is measured, how it is stored, and what may read it.

The benchmark is an enumeration of every canonical configuration, measured once per seed.
It holds the validation metric optimisers select on **and** the test metric the report
uses -- and those two must never travel together, because a tabular benchmark makes
test-set leakage a one-line mistake rather than a protocol violation someone has to
commit deliberately.

So the artifact is split by column, into four files:

===================  =====================================================================
``search_runs.csv``  one row per (configuration, seed): validation metrics, costs, memory
``search.csv``       the same aggregated over seeds -- what optimisers actually read
``report_runs.csv``  one row per (configuration, seed): **test** metrics only
``report.csv``       the same aggregated
===================  =====================================================================

Optimisers receive a :class:`SearchView`, which is constructed from the search files and
has no path to the report ones. ``tests/test_schema.py`` asserts three things that together
make the separation structural rather than a matter of anyone's discipline: the two column
sets are disjoint, no module outside ``benchmark/`` and ``report/`` mentions a report
column, and a :class:`SearchView` raises when asked for one.

**Aggregation is the median, and the spread is the range.** At three seeds a standard
deviation implies more than the data supports; ``max - min`` is what was actually observed.
Deterministic families are measured once and carry a spread of zero, which is only
legitimate because the test suite asserts their determinism rather than assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from budget_tune.space.grids import hyperparameter_columns

#: What a configuration is. Present in every file, and the join key between them.
IDENTITY_COLUMNS: list[str] = ["config_id", "dataset", "family", "data_fraction"]

#: Per-seed key.
RUN_KEY_COLUMNS: list[str] = [*IDENTITY_COLUMNS, "seed"]

#: Quality on the split optimisers are allowed to see.
SEARCH_QUALITY_COLUMNS: list[str] = [
    "val_ndcg_at_10",
    "val_recall_at_10",
    "val_exposure_parity",
]

#: The cost axis. ``cpu`` is primary and ``wall`` is the robustness axis -- a conclusion
#: that flips between them is reported as axis-dependent rather than as a conclusion, so
#: both are carried from the start rather than added when the question arises.
COST_COLUMNS: list[str] = [
    "train_cpu_seconds",
    "train_wall_seconds",
    "score_cpu_seconds",
    "score_wall_seconds",
    "select_cpu_seconds",
    "serve_cpu_seconds_per_request",
]

#: Resources that are not time.
RESOURCE_COLUMNS: list[str] = [
    "peak_rss_bytes",
    "model_bytes",
    "n_train_interactions",
    "n_eval_users",
]

#: Measurement provenance that must travel with the number to keep it interpretable. A
#: reading taken below the clock quantum is a tick count, not a duration, and a table that
#: did not say so would present it as one.
MEASUREMENT_COLUMNS: list[str] = [
    "train_repeats",
    "score_repeats",
    "train_below_quantum",
    "score_below_quantum",
    # CPU consumed by *other* processes while this row was measured, in cores. Recorded
    # rather than merely guarded on: a benchmark that stops when the machine gets busy but
    # keeps no record of how busy it was cannot answer "was this row measured on a quiet
    # machine?" -- which is exactly the question that could not be answered about the rows
    # this column was added for.
    "other_cores",
]

#: **Never** available to an optimiser, a surrogate, or a solver.
REPORT_QUALITY_COLUMNS: list[str] = ["test_ndcg_at_10", "test_recall_at_10"]

SEARCH_RUN_COLUMNS: list[str] = [
    *RUN_KEY_COLUMNS,
    *SEARCH_QUALITY_COLUMNS,
    *COST_COLUMNS,
    *RESOURCE_COLUMNS,
    *MEASUREMENT_COLUMNS,
]

REPORT_RUN_COLUMNS: list[str] = [*RUN_KEY_COLUMNS, *REPORT_QUALITY_COLUMNS]

#: Columns aggregated across seeds, each gaining a ``_spread`` companion.
AGGREGATED_COLUMNS: list[str] = [*SEARCH_QUALITY_COLUMNS, *COST_COLUMNS, *RESOURCE_COLUMNS]

SEARCH_FILE = "search.csv"
SEARCH_RUNS_FILE = "search_runs.csv"
REPORT_FILE = "report.csv"
REPORT_RUNS_FILE = "report_runs.csv"

#: Files an optimiser must never open. Named here so the leakage test has one list to
#: check against rather than a regular expression someone has to keep current.
REPORT_FILES: list[str] = [REPORT_FILE, REPORT_RUNS_FILE]


class LeakageError(RuntimeError):
    """Raised when search-side code reaches for a reporting column."""


def validate_runs(frame: pd.DataFrame, kind: str) -> None:
    """Check a per-seed frame against the schema, or raise saying exactly what is wrong.

    Raises rather than warns, and names the offending columns. A campaign that wrote a
    misspelled column would otherwise produce a table whose missing values look like
    measurement failures.
    """
    expected = {"search": SEARCH_RUN_COLUMNS, "report": REPORT_RUN_COLUMNS}[kind]
    allowed = set(expected) | set(hyperparameter_columns())

    # Leakage is checked first, and deliberately. A reporting column in a search frame is
    # also an *unknown* column, so ordering these the other way round reported the most
    # serious fault in the project as a spelling mistake.
    if kind == "search":
        leaked = [c for c in frame.columns if c in set(REPORT_QUALITY_COLUMNS)]
        if leaked:
            raise LeakageError(f"search frame carries reporting columns: {leaked}")

    missing = [column for column in expected if column not in frame.columns]
    unknown = [column for column in frame.columns if column not in allowed]
    if missing:
        raise ValueError(f"{kind} runs frame is missing columns: {missing}")
    if unknown:
        raise ValueError(f"{kind} runs frame has unknown columns: {unknown}")

    duplicated = frame.duplicated(subset=RUN_KEY_COLUMNS)
    if duplicated.any():
        raise ValueError(
            f"{kind} runs frame has {int(duplicated.sum())} duplicate "
            f"(config_id, dataset, seed) rows; each must be measured once"
        )


def aggregate(runs: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Median across seeds, with the observed range beside it.

    The seed count is kept per configuration rather than assumed constant, so a
    configuration that failed on one seed is visibly a two-seed row instead of quietly
    being averaged as if it were three.
    """
    keys = [*IDENTITY_COLUMNS, *[c for c in hyperparameter_columns() if c in runs.columns]]
    grouped = runs.groupby(IDENTITY_COLUMNS, sort=False, dropna=False)

    out = grouped[columns].median().reset_index()
    spread = (grouped[columns].max() - grouped[columns].min()).reset_index()
    spread = spread.rename(columns={c: f"{c}_spread" for c in columns})

    merged = out.merge(spread, on=IDENTITY_COLUMNS, validate="one_to_one")
    merged["n_seeds"] = grouped["seed"].nunique().to_numpy()

    identity = runs[keys].drop_duplicates(subset=IDENTITY_COLUMNS)
    return identity.merge(merged, on=IDENTITY_COLUMNS, validate="one_to_one")


def write(directory: str | Path, search_runs: pd.DataFrame, report_runs: pd.DataFrame) -> None:
    """Write all four files, validating first.

    Both per-seed frames are required together: a directory holding search results without
    the matching reporting rows is a benchmark whose configurations cannot all be reported
    on, and discovering that at report time -- after the campaign hardware conditions are
    gone -- is too late.
    """
    validate_runs(search_runs, "search")
    validate_runs(report_runs, "report")

    search_keys = set(map(tuple, search_runs[RUN_KEY_COLUMNS].to_numpy().tolist()))
    report_keys = set(map(tuple, report_runs[RUN_KEY_COLUMNS].to_numpy().tolist()))
    if search_keys != report_keys:
        raise ValueError(
            "search and report runs cover different (config, dataset, seed) sets: "
            f"{len(search_keys - report_keys)} search-only, "
            f"{len(report_keys - search_keys)} report-only"
        )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    search_runs.to_csv(directory / SEARCH_RUNS_FILE, index=False)
    report_runs.to_csv(directory / REPORT_RUNS_FILE, index=False)
    aggregate(search_runs, AGGREGATED_COLUMNS).to_csv(directory / SEARCH_FILE, index=False)
    aggregate(report_runs, REPORT_QUALITY_COLUMNS).to_csv(directory / REPORT_FILE, index=False)


@dataclass(frozen=True)
class SearchView:
    """Everything an optimiser is allowed to know about the benchmark.

    Constructed only from the search files. There is no attribute, method or argument by
    which the reporting split can be reached, so an optimiser cannot select on test
    performance even by mistake -- which is the point, since with an enumerated table the
    mistake would be a single column name and would improve every number it touched.
    """

    frame: pd.DataFrame
    dataset: str

    def __post_init__(self) -> None:
        leaked = [c for c in self.frame.columns if c in set(REPORT_QUALITY_COLUMNS)]
        if leaked:
            raise LeakageError(
                f"SearchView was handed reporting columns {leaked}. The search and report "
                "splits are stored separately; something has joined them."
            )

    def __len__(self) -> int:
        return len(self.frame)

    def column(self, name: str) -> pd.Series:
        """One column, refusing reporting columns by name rather than by absence."""
        if name in set(REPORT_QUALITY_COLUMNS):
            raise LeakageError(
                f"{name!r} is a reporting column and is not available during search"
            )
        if name not in self.frame.columns:
            raise KeyError(f"unknown column {name!r}; have {sorted(self.frame.columns)}")
        return self.frame[name]

    def lookup(self, config_id: str) -> pd.Series:
        """The row for one configuration -- an optimiser's single evaluation primitive."""
        rows = self.frame[self.frame.config_id == config_id]
        if len(rows) != 1:
            raise KeyError(f"expected exactly one row for {config_id!r}, found {len(rows)}")
        return rows.iloc[0]

    def config_ids(self) -> list[str]:
        return list(self.frame.config_id)

    def best(self, metric: str = "val_ndcg_at_10") -> pd.Series:
        """The optimum of the enumerated space on a search-side metric.

        Available to *analysis* -- normalised regret needs it -- and to optimisers only in
        the sense that they could enumerate the same table themselves, which at this size
        they can. That is the honest position: with 471 configurations the benchmark is
        exhaustively solvable, and the report says so rather than implying the optimum was
        hard to find.
        """
        return self.frame.loc[self.column(metric).idxmax()]


def load_search(directory: str | Path, dataset: str) -> SearchView:
    """Load the search side of a benchmark directory."""
    directory = Path(directory)
    frame = pd.read_csv(directory / SEARCH_FILE)
    frame = frame[frame.dataset == dataset].reset_index(drop=True)
    if frame.empty:
        raise KeyError(f"no rows for dataset {dataset!r} in {directory / SEARCH_FILE}")
    return SearchView(frame=frame, dataset=dataset)


def load_report(directory: str | Path, dataset: str) -> pd.DataFrame:
    """Load the reporting split. **Only the report scripts may call this.**

    Every call is a use of the test set. The manifest records how many times it happened,
    because a test split read once per project and a test split read after every idea are
    different experiments and only one of them is the one described in the design.
    """
    directory = Path(directory)
    frame = pd.read_csv(directory / REPORT_FILE)
    frame = frame[frame.dataset == dataset].reset_index(drop=True)
    if frame.empty:
        raise KeyError(f"no rows for dataset {dataset!r} in {directory / REPORT_FILE}")
    return frame
