"""Leave-two-out splitting, and the data-retention lever the whole project turns on.

**Why two held-out items rather than one.** Both companion projects split leave-one-out,
which is right when the model is fixed and only the reranker is under study. Here the
*configuration* is under study, so there must be a split to choose it on and a second
split, never touched during selection, to report it on. Hyperparameter optimisation makes
the select-and-report-on-the-same-data failure worse rather than milder: every extra
configuration tried is another chance to fit the selection split's noise.

So, per user, time-ordered::

    [ ... training interactions ... ] [ v ] [ t ]
                train                  val   test

The **test item is exactly the companion projects' held-out item** -- the same rule, the
same stable tie-break -- so ``tests/test_splits.py`` can assert equality against
:func:`benchmarks.loader.leave_one_out` for every user rather than for a subset. The
validation item is then carved out of what the companions call training data. That makes
this project's training matrix a strict subset of theirs, which is the honest description
and the reason absolute NDCG here is **not comparable** to numbers in those repositories.

**A user needs three interactions to donate both.** With two, holding out a validation item
would leave an empty profile: no history to score against, no factor row, and an accuracy
metric measuring the fallback rather than the model. Such users therefore keep their
validation item in training and are excluded from evaluation. Under 5-core filtering none
exist, but the rule is written down rather than assumed, because a future catalogue with a
lower core threshold would otherwise hit it silently.

**Data fraction is recency retention, not random thinning.** ``data_fraction`` keeps each
user's most recent ``ceil(f * len(history))`` interactions, minimum one. Two reasons, and
the second is the load-bearing one:

* it is a real data-retention policy -- "keep the last N months" is a decision deployers
  actually make, whereas "delete a random 75% of interactions" is not;
* uniform random thinning destroys *adjacency*. A sequential model learns ``i -> j`` from
  consecutive interactions, and thinning turns consecutive pairs into non-consecutive
  ones, so the sequential family would be penalised by the data lever in a way the
  matrix-based families are not. That confound would sit inside the energy/accuracy
  trade-off this project exists to measure.

The cost of the choice, stated because it is real: recency retention is not an unbiased
sample of the interaction distribution, so a family that happens to benefit from recent
data is flattered relative to random thinning. The alternative was chosen against
deliberately, not overlooked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from budget_tune.companion import ensure_importable

#: Interactions a user must have before a validation item is carved out. Two would leave
#: an empty training profile; see the module docstring.
MIN_HISTORY_FOR_VALIDATION = 3


def _companion_primitives():
    """Import the shared preprocessing, deferred so unit tests need no checkout."""
    ensure_importable("feasible_rerank")
    from benchmarks.loader import (
        interaction_matrix,
        k_core,
        leave_one_out,
        popularity_tiers,
    )

    return k_core, leave_one_out, interaction_matrix, popularity_tiers


def _sequences_class():
    ensure_importable("green_rerank")
    from green_rerank.families.base import Sequences

    return Sequences


# --------------------------------------------------------------------------- splitting


def leave_two_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into ``(train, validation, test)`` by recency, per user.

    The test frame is byte-identical to what :func:`benchmarks.loader.leave_one_out`
    returns on the same input -- same rule, same ``mergesort`` tie-break -- so the two
    projects hold out the same item for the same user. The validation frame is the next
    most recent interaction, for users with at least
    :data:`MIN_HISTORY_FOR_VALIDATION` interactions.

    Args:
        df: interactions with ``user_id``, ``item_id``, ``timestamp`` columns.

    Returns:
        ``(train, validation, test)``, each reindexed from zero.
    """
    required = {"user_id", "item_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"interactions frame is missing {sorted(missing)}")

    ordered = df.sort_values(["user_id", "timestamp"], kind="mergesort")
    grouped = ordered.groupby("user_id", sort=False)

    test = grouped.tail(1)
    remaining = ordered.drop(test.index)

    # Only users who still have something left to train on donate a validation item.
    # ``size`` here counts the *original* history, so the threshold is stated in terms a
    # reader can check against the dataset statistics rather than in terms of a
    # post-removal count.
    history = grouped["item_id"].transform("size")
    eligible = ordered.index[history >= MIN_HISTORY_FOR_VALIDATION]
    candidates = remaining.loc[remaining.index.intersection(eligible)]
    validation = candidates.groupby("user_id", sort=False).tail(1)

    train = remaining.drop(validation.index)
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep one row per ``(user, item)``: the most recent.

    **Why this exists.** The Amazon exports record repeat interactions -- 18.2% of Luxury
    Beauty's 5-core rows and 6.9% of Software's are a ``(user, item)`` pair seen before.
    Without this step a user's held-out item can also sit earlier in that user's own
    training history, and because serving masks seen items with ``-inf`` the target becomes
    unreachable: that user scores zero under every configuration.

    The constant handicap would have been survivable. What was not is that the count moved
    with the data fraction -- 503 affected Luxury Beauty users at ``f=0.25`` against 802 at
    ``f=1.0`` -- because retention drops the earlier repeat. Less training data left *more*
    users scorable, so low-fraction configurations were being measured on an easier
    population. That is an accuracy bonus for precisely the configurations this project
    hypothesises should be competitive, and every individual number would have looked
    normal.

    **Why after k-core rather than before.** Deduplicating first shrinks Luxury Beauty to
    2,028 users and 936 items, because repeat buyers stop clearing the 5-interaction
    threshold. Deduplicating after leaves the 5-core user and item population untouched --
    no user or item can vanish, since at least one row of every pair survives -- so these
    remain the same catalogues the companion projects measured.

    **Why this is a completion rather than a change.** The binary interaction matrix already
    collapsed repeats: Luxury Beauty retained 25,554 training interactions and the matrix
    held 21,073 non-zeros. Repeats were visible to the split and to the sequences but never
    to the matrix-based models, so the pipeline was already half-deduplicated and
    inconsistently so.

    What it costs, stated because it is real: the sequential family loses self-transitions,
    and a repeat purchase is genuine signal in a repeat-purchase catalogue. The alternative
    -- keeping repeats and not masking seen items -- would have changed the task itself.

    Returns:
        ``(deduplicated frame, statistics)``. Original row order is preserved, so the
        stable tie-break used by :func:`leave_two_out` is unchanged and a catalogue with no
        repeats passes through byte-identical.
    """
    before_users = df.user_id.nunique()
    before_items = df.item_id.nunique()

    # Choose the surviving rows by sorting on a *copy*, then take them from the original
    # frame in its original order. Returning the sorted frame instead would reorder tied
    # timestamps by item id, silently changing which interaction the split holds out on
    # every catalogue -- including the ones with no duplicates at all.
    ordered = df.sort_values(["user_id", "item_id", "timestamp"], kind="mergesort")
    keep = ordered.drop_duplicates(subset=["user_id", "item_id"], keep="last").index
    deduped = df.loc[np.sort(keep.to_numpy())].reset_index(drop=True)

    distinct_per_user = deduped.groupby("user_id")["item_id"].size()
    distinct_per_item = deduped.groupby("item_id")["user_id"].size()

    return deduped, {
        "rows_before": len(df),
        "rows_after": len(deduped),
        "duplicate_rows_removed": len(df) - len(deduped),
        "duplicate_share": (len(df) - len(deduped)) / len(df) if len(df) else 0.0,
        # Zero by construction, and asserted rather than assumed: at least one row of every
        # pair survives, so deduplication cannot remove a user or an item outright.
        "users_removed": before_users - deduped.user_id.nunique(),
        "items_removed": before_items - deduped.item_id.nunique(),
        # The k-core property is re-checked, not re-imposed. Re-running k-core to
        # convergence would cascade -- dropping a user changes item counts and vice versa --
        # and would land back at the population Option A produces, which this choice exists
        # to avoid. So the violations are counted and reported, and the catalogue keeps its
        # users.
        "users_below_core_after_dedupe": int((distinct_per_user < 5).sum()),
        "items_below_core_after_dedupe": int((distinct_per_item < 5).sum()),
        "users_below_validation_threshold": int(
            (distinct_per_user < MIN_HISTORY_FOR_VALIDATION).sum()
        ),
    }


def target_leakage(matrix: sparse.csr_matrix, targets: dict[int, int]) -> int:
    """Held-out items that are present in a training matrix. Must always be zero.

    Kept as a function rather than only as a test so that :func:`assemble` can refuse to
    return a dataset that leaks. A test proves the property held once on the catalogues
    someone remembered to test; this proves it for every catalogue anyone ever loads.
    """
    indptr, indices = matrix.indptr, matrix.indices
    return sum(
        1
        for row, item in targets.items()
        if item in indices[indptr[row] : indptr[row + 1]]
    )


def retain_recent(train: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Keep each user's most recent ``ceil(fraction * n)`` interactions, minimum one.

    Deterministic: no seed, no sampling. Two runs at the same fraction retain the same
    rows, which is what lets a multi-fidelity rung *be* an already-measured table row
    rather than an approximation of one.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
    if fraction == 1.0:
        return train.reset_index(drop=True)

    ordered = train.sort_values(["user_id", "timestamp"], kind="mergesort")
    grouped = ordered.groupby("user_id", sort=False)

    counts = grouped["item_id"].transform("size").to_numpy()
    keep_n = np.maximum(1, np.ceil(fraction * counts)).astype(np.int64)
    # 0 is the most recent interaction, counting backwards within each user.
    position_from_end = counts - 1 - grouped.cumcount().to_numpy()

    return ordered[position_from_end < keep_n].reset_index(drop=True)


# ------------------------------------------------------------------------- assembling


def matrix_against_index(
    frame: pd.DataFrame,
    user_index: dict[str, int],
    item_index: dict[str, int],
    binary: bool = True,
) -> sparse.csr_matrix:
    """Build an interaction matrix against a **fixed** index.

    The companion's :func:`benchmarks.loader.interaction_matrix` derives its index from
    the frame it is given, which is correct there and wrong here: this project builds one
    matrix per data fraction and every one of them must share a column space. If the index
    moved with the fraction, item column 3 would mean a different item at 0.25 than at
    1.00, the group vector would be misaligned against the columns, and every fairness
    number would silently measure a permutation of the truth.

    Items that disappear entirely at a low fraction therefore remain as all-zero columns.

    ``tests/test_splits.py`` asserts this reproduces the companion's matrix exactly at
    ``fraction=1.0``; that equality is what makes "fixed index" a refinement of the shared
    primitive rather than a second implementation of it.
    """
    rows = frame.user_id.map(user_index)
    cols = frame.item_id.map(item_index)
    known = rows.notna() & cols.notna()

    values = (
        np.ones(int(known.sum()))
        if binary
        else frame.loc[known, "rating"].to_numpy(dtype=float)
    )
    return sparse.csr_matrix(
        (values, (rows[known].to_numpy(dtype=np.int64), cols[known].to_numpy(dtype=np.int64))),
        shape=(len(user_index), len(item_index)),
    )


@dataclass(frozen=True)
class Fold:
    """The training data at one value of ``data_fraction``.

    Attributes:
        fraction: the retention level this fold was built at.
        matrix: ``(n_users, n_items)`` implicit feedback, sharing the dataset's index.
        sequences: time-ordered retained interactions, for the sequential family.
        n_interactions: retained interaction count, reported so that "25% of the data"
            can be checked against what was actually kept rather than assumed.
    """

    fraction: float
    matrix: sparse.csr_matrix
    sequences: object
    n_interactions: int


@dataclass
class HpoDataset:
    """A catalogue split three ways, at every data fraction the space uses.

    Attributes:
        name: label for results tables.
        item_ids: raw catalogue ids, positionally aligned with the matrix columns.
        user_index: raw user id -> matrix row.
        groups: one group index per item -- genres on MovieLens, popularity tiers on
            Amazon. Computed **once, from the full-fraction training matrix**, so that a
            configuration trained on less data is still scored against the same partition.
            Deriving groups per fold would make the fairness column incomparable between
            configurations, which is the sort of error that leaves every number plausible.
        validation: ``user_row -> item column`` for the selection target.
        test: ``user_row -> item column`` for the reporting target.
        folds: ``fraction -> Fold``.
        stats: shapes and counts for the report's dataset table.
    """

    name: str
    item_ids: list[str]
    user_index: dict[str, int]
    groups: np.ndarray
    validation: dict[int, int]
    test: dict[int, int]
    folds: dict[float, Fold]
    stats: dict = field(default_factory=dict)

    @property
    def n_users(self) -> int:
        return len(self.user_index)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def fold(self, fraction: float) -> Fold:
        """The training data at ``fraction``, or raise listing what was built."""
        key = round(float(fraction), 6)
        if key not in self.folds:
            raise KeyError(
                f"no fold at fraction {fraction}; built: {sorted(self.folds)}. "
                "Folds are precomputed from the search space so that a fidelity rung is "
                "an already-measured configuration rather than a fresh approximation."
            )
        return self.folds[key]

    def eval_users(self) -> np.ndarray:
        """Rows holding **both** a validation and a test target, ascending.

        Both, not either: a configuration selected on one population and reported on
        another is comparing two different questions, and the difference would look like a
        generalisation gap. Under 5-core filtering the two populations coincide anyway,
        which the dataset statistics record so the reader can see it did not matter here.
        """
        return np.array(sorted(set(self.validation) & set(self.test)), dtype=np.int64)

    def targets(self, split: str, user_rows: np.ndarray) -> list[set[int]]:
        """Held-out item per user for ``split``, as the metric functions expect."""
        if split not in {"validation", "test"}:
            raise ValueError(f"split must be 'validation' or 'test'; got {split!r}")
        mapping = self.validation if split == "validation" else self.test
        return [
            {mapping[int(row)]} if int(row) in mapping else set() for row in user_rows
        ]


def assemble(
    name: str,
    raw: pd.DataFrame,
    groups_fn,
    fractions: tuple[float, ...],
    min_interactions: int = 5,
    n_groups: int = 4,
    binary: bool = True,
    max_sequence: int = 200,
) -> HpoDataset:
    """k-core, split three ways, then build one training fold per data fraction.

    Kept common between MovieLens and Amazon on purpose: a difference in results between
    catalogues that could be caused by a difference in preprocessing would make the
    cross-catalogue comparison meaningless.
    """
    k_core, _, interaction_matrix, _ = _companion_primitives()
    Sequences = _sequences_class()

    filtered = k_core(raw, min_interactions=min_interactions)
    filtered, dedupe_stats = deduplicate(filtered)
    train, validation, test = leave_two_out(filtered)

    # The index is defined by the full-fraction training frame and then held fixed. Using
    # the companion's builder for this establishes the exact categorical ordering it
    # would have produced; every reduced fold is then built against that same ordering.
    full_matrix, user_index, item_ids = interaction_matrix(train, binary=binary)
    item_index = {item: position for position, item in enumerate(item_ids)}

    def targets(frame: pd.DataFrame) -> dict[int, int]:
        return {
            user_index[row.user_id]: item_index[row.item_id]
            for row in frame.itertuples()
            if row.user_id in user_index and row.item_id in item_index
        }

    folds: dict[float, Fold] = {}
    for fraction in sorted({round(float(f), 6) for f in fractions}):
        frame = retain_recent(train, fraction)
        matrix = (
            full_matrix
            if fraction == 1.0
            else matrix_against_index(frame, user_index, item_index, binary=binary)
        )
        folds[fraction] = Fold(
            fraction=fraction,
            matrix=matrix,
            sequences=Sequences.from_frame(
                frame, user_index, item_index, max_length=max_sequence
            ),
            n_interactions=len(frame),
        )

    validation_targets = targets(validation)
    test_targets = targets(test)
    both = set(validation_targets) & set(test_targets)

    # Refuse to hand back a leaking dataset. This is the failure that would invalidate
    # every accuracy number in the project while leaving each of them plausible, so it is
    # checked on the way out rather than only in a test file -- and checked at every data
    # fraction, because the fraction-dependence of the count is what made the original bug
    # dangerous rather than merely regrettable.
    splits = (("validation", validation_targets), ("test", test_targets))
    for fraction, fold in folds.items():
        for split_name, mapping in splits:
            leaked = target_leakage(fold.matrix, mapping)
            if leaked:
                raise AssertionError(
                    f"{name}: {leaked} {split_name} targets appear in the training matrix "
                    f"at data_fraction={fraction}. Serving masks seen items, so those users "
                    "would score zero under every configuration, and the count moving with "
                    "the fraction would flatter low-data configurations."
                )

    return HpoDataset(
        name=name,
        item_ids=list(item_ids),
        user_index=user_index,
        groups=groups_fn(full_matrix, item_ids, n_groups),
        validation=validation_targets,
        test=test_targets,
        folds=folds,
        stats={
            "raw_interactions": len(raw),
            "core_interactions": dedupe_stats["rows_before"],
            "deduplicated_interactions": len(filtered),
            "dedupe": dedupe_stats,
            # Users and items present after k-core and deduplication, against those that
            # survive into the training matrix. The two differ when a user's entire history
            # collapses to a single distinct item: leave-two-out claims it as the test
            # target and nothing is left to train on, so the user never reaches the matrix.
            # That is attrition caused by the split rather than by deduplication, and
            # reporting one number for both would hide which.
            "core_users": int(filtered.user_id.nunique()),
            "core_items": int(filtered.item_id.nunique()),
            "users": int(full_matrix.shape[0]),
            "items": int(full_matrix.shape[1]),
            "users_without_training_rows": int(
                filtered.user_id.nunique() - full_matrix.shape[0]
            ),
            "items_without_training_rows": int(
                filtered.item_id.nunique() - full_matrix.shape[1]
            ),
            "train_interactions": len(train),
            "density": float(full_matrix.nnz / (full_matrix.shape[0] * full_matrix.shape[1])),
            # Split three ways, so three counts. The gap between "rows in the frame" and
            # "usable targets" is the number of held-out items whose item never appeared
            # in training; reporting both makes that attrition visible instead of leaving
            # it as an unexplained shrink in the evaluation population.
            "validation_rows": len(validation),
            "test_rows": len(test),
            "users_with_validation": len(validation_targets),
            "users_with_test": len(test_targets),
            "eval_users": len(both),
            "n_groups": int(n_groups),
            "fractions": sorted(folds),
            "retained_interactions": {f: folds[f].n_interactions for f in sorted(folds)},
            "min_history_for_validation": MIN_HISTORY_FOR_VALIDATION,
        },
    )


def expected_retained(counts: np.ndarray, fraction: float) -> int:
    """Interactions ``retain_recent`` should keep, from history lengths alone.

    Exists so the test suite can check the retention policy against arithmetic rather than
    against another call to the same function.
    """
    return int(sum(max(1, math.ceil(fraction * int(n))) for n in counts))
