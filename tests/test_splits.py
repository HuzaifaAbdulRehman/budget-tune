"""Invariants for the leave-two-out split and the data-retention lever.

The tests that earn their place here are the ones that would catch a *silent* error: a
split that leaks a held-out item into training, a retention policy whose fidelity rungs are
not nested, or an item index that moves between folds so that column 3 means different
items at different data fractions. Every one of those leaves a benchmark that looks
entirely normal and is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from budget_tune.companion import ensure_importable
from budget_tune.data.splits import (
    MIN_HISTORY_FOR_VALIDATION,
    expected_retained,
    leave_two_out,
    matrix_against_index,
    retain_recent,
)
from tests.conftest import require_companion

pytestmark = pytest.mark.companion


def _rows(frame: pd.DataFrame) -> set[tuple]:
    return set(map(tuple, frame[["user_id", "item_id", "timestamp"]].to_numpy().tolist()))


class TestLeaveTwoOut:
    def test_test_split_is_identical_to_the_companions(self, interactions, strict):
        """The reporting item must be the companions' held-out item, for every user.

        Not "similar": identical. The three repositories claim to be studying the same
        catalogues, and a different tie-break would make that claim false while leaving
        every table plausible. Asserting equality against the shared primitive is what
        makes leave-two-out a refinement of leave-one-out rather than a second protocol.
        """
        require_companion("feasible_rerank", strict)
        from benchmarks.loader import leave_one_out

        _, companion_test = leave_one_out(interactions)
        _, _, test = leave_two_out(interactions)

        assert _rows(test) == _rows(companion_test)
        assert len(test) == interactions.user_id.nunique()

    def test_train_plus_validation_is_the_companions_training_set(self, interactions, strict):
        require_companion("feasible_rerank", strict)
        from benchmarks.loader import leave_one_out

        companion_train, _ = leave_one_out(interactions)
        train, validation, _ = leave_two_out(interactions)

        assert _rows(train) | _rows(validation) == _rows(companion_train)
        assert len(train) + len(validation) == len(companion_train)

    def test_the_three_splits_partition_the_input(self, interactions):
        train, validation, test = leave_two_out(interactions)
        assert len(train) + len(validation) + len(test) == len(interactions)
        assert not (_rows(train) & _rows(validation))
        assert not (_rows(train) & _rows(test))
        assert not (_rows(validation) & _rows(test))

    def test_short_histories_donate_a_test_item_but_not_a_validation_item(self, interactions):
        """A two-interaction user must keep a profile.

        Carving a validation item out of a two-interaction history leaves nothing to score
        against, so the accuracy number would measure the fallback path. Under 5-core no
        such user survives, which is exactly why this is tested on a hand-made frame --
        the rule would otherwise be unexercised until a catalogue with a lower threshold
        arrived.
        """
        train, validation, test = leave_two_out(interactions)

        counts = interactions.groupby("user_id").size()
        short = {u for u, n in counts.items() if n < MIN_HISTORY_FOR_VALIDATION}
        assert short, "fixture must contain a user below the threshold"

        assert short & set(test.user_id) == short
        assert not short & set(validation.user_id)
        for user in short:
            assert (train.user_id == user).sum() == counts[user] - 1

    def test_ordering_is_by_recency(self, interactions):
        train, validation, test = leave_two_out(interactions)
        for user in validation.user_id:
            train_max = train.loc[train.user_id == user, "timestamp"].max()
            val_ts = validation.loc[validation.user_id == user, "timestamp"].iloc[0]
            test_ts = test.loc[test.user_id == user, "timestamp"].iloc[0]
            # Not strict: tied timestamps are legitimate and are broken by input order,
            # matching the companions' stable mergesort.
            assert train_max <= val_ts <= test_ts

    def test_tied_timestamps_split_deterministically(self):
        """Ties must not depend on anything that varies between runs."""
        tied = pd.DataFrame(
            [
                {"user_id": "a", "item_id": f"i{i}", "rating": 1.0, "timestamp": 7}
                for i in range(5)
            ]
        )
        first = leave_two_out(tied)
        second = leave_two_out(tied)
        for a, b in zip(first, second, strict=True):
            pd.testing.assert_frame_equal(a, b)

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="missing"):
            leave_two_out(pd.DataFrame({"user_id": ["a"], "item_id": ["i"]}))


class TestRetention:
    def test_full_fraction_is_the_identity(self, interactions):
        train, _, _ = leave_two_out(interactions)
        assert _rows(retain_recent(train, 1.0)) == _rows(train)

    def test_counts_match_arithmetic(self, interactions):
        """Checked against the ceiling rule, not against another call to the same code."""
        train, _, _ = leave_two_out(interactions)
        counts = train.groupby("user_id").size().to_numpy()
        for fraction in (0.25, 0.5, 1.0):
            assert len(retain_recent(train, fraction)) == expected_retained(counts, fraction)

    def test_every_user_keeps_at_least_one_interaction(self, interactions):
        train, _, _ = leave_two_out(interactions)
        kept = retain_recent(train, 0.01)
        assert set(kept.user_id) == set(train.user_id)
        assert (kept.groupby("user_id").size() >= 1).all()

    def test_retention_keeps_the_most_recent(self, interactions):
        train, _, _ = leave_two_out(interactions)
        kept = retain_recent(train, 0.5)
        for user, group in kept.groupby("user_id"):
            dropped = train[(train.user_id == user) & ~train.item_id.isin(group.item_id)]
            if len(dropped):
                assert dropped.timestamp.max() <= group.timestamp.min()

    def test_rungs_are_nested(self, interactions):
        """0.25 must be a subset of 0.5, which must be a subset of the full data.

        This is what makes a multi-fidelity rung an honest probe of the configuration
        above it. Random thinning would satisfy the counts and break the nesting, and the
        resulting Hyperband would be promoting on evidence that is not a subset of what it
        promotes to -- invisible in any output.
        """
        train, _, _ = leave_two_out(interactions)
        quarter = _rows(retain_recent(train, 0.25))
        half = _rows(retain_recent(train, 0.5))
        assert quarter <= half <= _rows(train)

    def test_invalid_fractions_raise(self, interactions):
        train, _, _ = leave_two_out(interactions)
        for bad in (0.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="fraction"):
                retain_recent(train, bad)


@pytest.fixture
def repeats() -> pd.DataFrame:
    """Interactions containing repeat ``(user, item)`` pairs, as the Amazon exports do.

    Structured so every branch of the bug is present:

    * user ``a`` buys ``i1`` early and again as their **most recent** interaction, so the
      test target is a repeat of something in their history;
    * user ``b`` repeats ``i2`` as their **second-most-recent**, so the *validation* target
      is the repeat;
    * user ``c`` has an early repeat that a low data fraction would drop, which is the
      fraction-dependence that made the original bug dangerous;
    * user ``d`` has no repeats at all and must pass through untouched.
    """
    rows = [
        ("a", "i1", 10),
        ("a", "i2", 20),
        ("a", "i3", 30),
        ("a", "i1", 40),
        ("b", "i1", 11),
        ("b", "i2", 21),
        ("b", "i3", 31),
        ("b", "i2", 41),
        ("b", "i4", 51),
        ("c", "i4", 12),
        ("c", "i4", 22),
        ("c", "i1", 32),
        ("c", "i2", 42),
        ("c", "i3", 52),
        ("d", "i1", 13),
        ("d", "i2", 23),
        ("d", "i3", 33),
    ]
    return pd.DataFrame(
        [{"user_id": u, "item_id": i, "rating": 1.0, "timestamp": t} for u, i, t in rows]
    )


class TestDeduplication:
    """Regression tests for repeat ``(user, item)`` interactions.

    The bug these cover was not caught by the existing suite because the leakage test ran
    against a synthetic catalogue in which every item is unique per user -- which is also
    why it survived undetected in the companion projects.
    """

    def test_removes_only_repeats_and_keeps_the_most_recent(self, repeats):
        from budget_tune.data.splits import deduplicate

        deduped, stats = deduplicate(repeats)

        assert stats["duplicate_rows_removed"] == 3
        assert len(deduped) == len(repeats) - 3
        assert not deduped.duplicated(subset=["user_id", "item_id"]).any()

        # Of a's two i1 rows, the one at t=40 survives.
        kept = deduped[(deduped.user_id == "a") & (deduped.item_id == "i1")]
        assert len(kept) == 1
        assert kept.timestamp.iloc[0] == 40

    def test_no_user_or_item_disappears(self, repeats):
        """The whole point of deduplicating after k-core rather than before."""
        from budget_tune.data.splits import deduplicate

        deduped, stats = deduplicate(repeats)
        assert stats["users_removed"] == 0
        assert stats["items_removed"] == 0
        assert set(deduped.user_id) == set(repeats.user_id)
        assert set(deduped.item_id) == set(repeats.item_id)

    def test_a_catalogue_without_repeats_passes_through_unchanged(self, interactions):
        """MovieLens and Gift Cards must be byte-identical, tie-break included.

        Deduplication sorts by ``(user, item, timestamp)`` to choose survivors. Returning
        that sorted frame instead of the original rows would reorder tied timestamps by
        item id, changing which interaction is held out on *every* catalogue -- including
        the ones with nothing to deduplicate.
        """
        from budget_tune.data.splits import deduplicate

        deduped, stats = deduplicate(interactions)
        assert stats["duplicate_rows_removed"] == 0
        pd.testing.assert_frame_equal(deduped, interactions.reset_index(drop=True))

    def test_original_row_order_is_preserved(self, repeats):
        """Surviving rows must keep their relative order in the input frame.

        Checked against the input rather than against a global sort: the fixture is grouped
        by user, so its timestamps are not globally ascending and asserting that they were
        tested nothing about order preservation.
        """
        from budget_tune.data.splits import deduplicate

        columns = ["user_id", "item_id", "timestamp"]
        deduped, _ = deduplicate(repeats)

        survivors = [tuple(row) for row in deduped[columns].to_numpy().tolist()]
        kept = set(survivors)
        original_order = [
            tuple(row) for row in repeats[columns].to_numpy().tolist() if tuple(row) in kept
        ]
        assert survivors == original_order

    def test_targets_no_longer_appear_in_training(self, repeats):
        """The property the whole change exists to establish."""
        from budget_tune.data.splits import deduplicate

        deduped, _ = deduplicate(repeats)
        train, validation, test = leave_two_out(deduped)

        train_pairs = set(map(tuple, train[["user_id", "item_id"]].to_numpy().tolist()))
        for split in (validation, test):
            pairs = set(map(tuple, split[["user_id", "item_id"]].to_numpy().tolist()))
            assert not (pairs & train_pairs)

    def test_without_deduplication_the_targets_do_leak(self, repeats):
        """The bug, pinned. If this ever stops failing, the fixture has lost its teeth.

        Asserting that the *unfixed* path is broken is what keeps the regression test
        honest: without it, a fixture that quietly stopped containing repeats would leave
        every test above passing for the wrong reason.
        """
        train, validation, test = leave_two_out(repeats)
        train_pairs = set(map(tuple, train[["user_id", "item_id"]].to_numpy().tolist()))

        val_pairs = set(map(tuple, validation[["user_id", "item_id"]].to_numpy().tolist()))
        test_pairs = set(map(tuple, test[["user_id", "item_id"]].to_numpy().tolist()))
        assert (val_pairs | test_pairs) & train_pairs

    def test_leak_count_would_have_moved_with_the_data_fraction(self, repeats):
        """The reason this was a bias and not merely a handicap.

        Retention drops the earlier repeat, so a smaller fraction leaves *more* users
        scorable and flatters low-data configurations -- an accuracy bonus for exactly the
        configurations the project hypothesises should be competitive.
        """
        train, _, test = leave_two_out(repeats)
        test_pairs = set(map(tuple, test[["user_id", "item_id"]].to_numpy().tolist()))

        counts = {}
        for fraction in (0.25, 1.0):
            kept = retain_recent(train, fraction)
            kept_pairs = set(map(tuple, kept[["user_id", "item_id"]].to_numpy().tolist()))
            counts[fraction] = len(test_pairs & kept_pairs)
        assert counts[0.25] < counts[1.0]

    def test_core_and_eligibility_are_rechecked_not_reimposed(self, repeats):
        """Deduplication can push a user below the core threshold; that is reported.

        Re-running k-core to convergence would cascade -- dropping a user changes item
        counts and vice versa -- and would land back at the smaller population that
        deduplicating *before* k-core produces, which is the outcome this choice exists to
        avoid. So the violation is counted and the catalogue keeps its users.
        """
        from budget_tune.data.splits import deduplicate

        _, stats = deduplicate(repeats)
        assert stats["users_below_core_after_dedupe"] == 4  # every user, on this tiny frame
        assert stats["users_below_validation_threshold"] == 0


class TestFixedIndex:
    def test_matches_the_companion_matrix_at_full_data(self, interactions, strict):
        """A fixed index must be a refinement of the shared builder, not a rewrite of it."""
        require_companion("feasible_rerank", strict)
        from benchmarks.loader import interaction_matrix

        train, _, _ = leave_two_out(interactions)
        reference, user_index, item_ids = interaction_matrix(train, binary=True)
        item_index = {item: i for i, item in enumerate(item_ids)}

        ours = matrix_against_index(train, user_index, item_index, binary=True)

        assert ours.shape == reference.shape
        np.testing.assert_array_equal(ours.toarray(), reference.toarray())


@pytest.fixture(scope="module")
def dataset():
    """The synthetic catalogue, assembled once for the end-to-end invariants."""
    ensure_importable("green_rerank")
    from budget_tune.data.catalogues import synthetic

    return synthetic(fractions=(0.25, 0.5, 1.0))


class TestAssembledDataset:
    """End-to-end invariants on the synthetic catalogue, which needs no download."""

    def test_held_out_items_never_appear_in_any_training_fold(self, dataset):
        """The leakage test. If this fails, every accuracy number in the project is void."""
        for fraction, fold in dataset.folds.items():
            matrix = fold.matrix.tocsr()
            for user_row, item in dataset.validation.items():
                assert matrix[user_row, item] == 0, f"validation item leaked at {fraction}"
            for user_row, item in dataset.test.items():
                assert matrix[user_row, item] == 0, f"test item leaked at {fraction}"

    def test_held_out_items_never_appear_in_any_sequence(self, dataset):
        """Sequences are a second path into the model, and need the same guarantee.

        The matrix-based families read the matrix; the sequential family reads this. A
        check on only one of them would leave the other free to leak.
        """
        for fold in dataset.folds.values():
            for user_row, history in fold.sequences.by_user.items():
                assert dataset.validation.get(user_row) not in history
                assert dataset.test.get(user_row) not in history

    def test_every_fold_shares_one_index(self, dataset):
        """Shapes and column meanings must not move with the data fraction.

        If they did, the group vector would be misaligned against the columns at some
        fractions and the exposure-parity column would measure a permutation of the truth
        -- in range, responsive to changes, and wrong.
        """
        shapes = {fold.matrix.shape for fold in dataset.folds.values()}
        assert len(shapes) == 1
        assert shapes.pop() == (dataset.n_users, dataset.n_items)
        assert len(dataset.groups) == dataset.n_items

    def test_folds_are_nested_as_matrices(self, dataset):
        smaller = dataset.fold(0.25).matrix.toarray()
        larger = dataset.fold(1.0).matrix.toarray()
        assert (smaller <= larger).all()
        assert dataset.fold(0.25).n_interactions < dataset.fold(1.0).n_interactions

    def test_evaluation_population_is_identical_across_fractions(self, dataset):
        """Selection and reporting must score the same users at every fidelity."""
        users = dataset.eval_users()
        assert len(users) > 0
        assert set(users) <= set(dataset.validation)
        assert set(users) <= set(dataset.test)

    def test_targets_are_one_item_per_user(self, dataset):
        users = dataset.eval_users()
        for split in ("validation", "test"):
            targets = dataset.targets(split, users)
            assert all(len(t) == 1 for t in targets)

    def test_validation_and_test_targets_differ(self, dataset):
        for user in dataset.eval_users():
            assert dataset.validation[int(user)] != dataset.test[int(user)]

    def test_unknown_fraction_raises_listing_what_exists(self, dataset):
        with pytest.raises(KeyError, match="built"):
            dataset.fold(0.75)

    def test_no_duplicate_survives_into_any_matrix(self, dataset):
        """After deduplication the retained count and the non-zero count must agree.

        They did not before: Luxury Beauty retained 25,554 training interactions against a
        matrix holding 21,073 non-zeros, and that gap *was* the bug -- repeats visible to
        the split and the sequences but invisible to the matrix-based models. Equality here
        detects any surviving duplicate directly, without needing to know where it came
        from.
        """
        for fold in dataset.folds.values():
            assert fold.matrix.nnz == fold.n_interactions

    def test_the_task_is_identical_at_every_data_fraction(self, dataset):
        """Comparable task definition at every data fraction.

        Three things must hold at every fraction so that a low-fraction configuration
        is the same experiment on less recent data, not a different population: the
        same users are scorable, against the same targets, with none of them masked.
        """
        from budget_tune.data.splits import target_leakage

        users = dataset.eval_users()
        for fraction, fold in dataset.folds.items():
            assert target_leakage(fold.matrix, dataset.validation) == 0, fraction
            assert target_leakage(fold.matrix, dataset.test) == 0, fraction
            scorable = [
                row
                for row in users
                if dataset.validation[int(row)] not in fold.matrix[int(row)].indices
            ]
            assert len(scorable) == len(users), fraction

    def test_stats_report_the_attrition(self, dataset):
        stats = dataset.stats
        usable = min(stats["users_with_validation"], stats["users_with_test"])
        assert stats["eval_users"] <= usable
        assert stats["dedupe"]["users_removed"] == 0
        assert stats["dedupe"]["items_removed"] == 0
        assert stats["retained_interactions"][0.25] < stats["retained_interactions"][1.0]


@pytest.fixture(scope="module")
def luxury_beauty():
    """The repeat-heavy catalogue, loaded once for the whole module."""
    from budget_tune.data import catalogues

    try:
        catalogues.resolve("luxury_beauty")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"luxury_beauty unavailable: {exc}")
    return catalogues.load("luxury_beauty", fractions=(0.25, 0.5, 1.0))


@pytest.mark.data
class TestRepeatHeavyCatalogue:
    """The test that would have caught this, on the catalogue that has the problem.

    Luxury Beauty carries repeats in 18.2% of its 5-core rows. The synthetic catalogue
    cannot exercise this at all -- every item is unique per user there -- which is exactly
    why the companion projects' leakage test passed while the bug was live.
    """

    def test_no_target_leaks_at_any_fraction(self, luxury_beauty):
        from budget_tune.data.splits import target_leakage

        for fraction, fold in luxury_beauty.folds.items():
            assert target_leakage(fold.matrix, luxury_beauty.validation) == 0, fraction
            assert target_leakage(fold.matrix, luxury_beauty.test) == 0, fraction

    def test_the_population_survived_deduplication(self, luxury_beauty):
        """Option B's justification, and the one place it is not free.

        Deduplication itself removes no user and no item -- at least one row of every pair
        survives. But seven Luxury Beauty users turn out to have bought a *single* distinct
        item repeatedly, and once the repeats collapse their entire history is one
        interaction, which leave-two-out claims as the test target. They have no training
        row left and so do not appear in the matrix at all.

        That is attrition caused by the split, not by deduplication, and it is recorded
        separately for exactly that reason: 3,589 users clear 5-core, 3,582 have anything
        to train on.
        """
        stats = luxury_beauty.stats
        assert stats["dedupe"]["users_removed"] == 0
        assert stats["dedupe"]["items_removed"] == 0
        assert stats["dedupe"]["duplicate_rows_removed"] == 5948

        assert stats["core_users"] == 3589
        assert stats["users"] == 3582
        assert stats["users_without_training_rows"] == 7

    def test_retained_and_nonzero_counts_agree(self, luxury_beauty):
        for fold in luxury_beauty.folds.values():
            assert fold.matrix.nnz == fold.n_interactions
