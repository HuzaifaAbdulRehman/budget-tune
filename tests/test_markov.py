"""The sequential Markov family, checked against an independent oracle.

This is the only model this project implements itself, so it is the only one whose bugs are
this project's alone. The oracle below is written from the definition in the module
docstring -- plain Python dictionaries, explicit loops, no shared code with the
implementation -- because a test that reuses the implementation's own machinery can only
find crashes, not wrong answers.

Two of these tests exist because of a specific mistake the design nearly made. The first
draft specified *additive* smoothing, which cannot change a ranking when every user is
scored from a single context: three grid cells would have produced three byte-identical
rows and the space would have carried a dead axis. ``test_smoothing_changes_the_ranking``
is the regression test for that, and it would fail against the original design.
"""

from __future__ import annotations

import numpy as np
import pytest

from budget_tune.companion import ensure_importable
from tests.conftest import require_companion

pytestmark = pytest.mark.companion


@pytest.fixture(scope="module")
def sequences_class(strict):
    require_companion("green_rerank", strict)
    ensure_importable("green_rerank")
    from green_rerank.families.base import Sequences

    return Sequences


@pytest.fixture
def histories() -> dict[int, list[int]]:
    """Hand-made histories with the structure the tests need.

    User 3 repeats a bigram so that a second-order context has more than one observation;
    user 4 has a single interaction so there is no transition to learn from it; user 5 is
    absent entirely, which is the empty-history path.
    """
    return {
        0: [0, 1, 2, 3],
        1: [1, 2, 0, 2, 3],
        2: [3, 2, 1, 0],
        3: [0, 1, 2, 0, 1, 4],
        4: [2],
    }


def model_matrix(n_users: int, n_items: int):
    """An empty interaction matrix of the right shape.

    The Markov model reads sequences, not the matrix; the matrix only supplies the shape
    and the seen-item mask. Built empty and by construction rather than by assignment,
    because assigning into a CSR matrix is both slow and a warning.
    """
    from scipy import sparse

    return sparse.csr_matrix((n_users, n_items), dtype=float)


@pytest.fixture
def hand_built(sequences_class):
    """Fit the model on explicit histories, for tests that compute the answer by hand."""

    def build(histories: dict[int, list[int]], n_items: int, **kwargs):
        from budget_tune.families.markov import SequentialMarkov

        n_users = max(histories) + 1
        model = SequentialMarkov(**kwargs)
        return model.fit(
            model_matrix(n_users, n_items),
            sequences_class(by_user=dict(histories), max_length=200),
        )

    return build


@pytest.fixture
def fitted(histories, sequences_class):
    from scipy import sparse

    n_items = 5
    n_users = 6
    rows, cols = [], []
    for user, history in histories.items():
        for item in history:
            rows.append(user)
            cols.append(item)
    matrix = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_users, n_items)
    )

    def build(**kwargs):
        from budget_tune.families.markov import SequentialMarkov

        model = SequentialMarkov(**kwargs)
        model.fit(matrix, sequences_class(by_user=dict(histories), max_length=200))
        return model, matrix

    return build


# --------------------------------------------------------------------------- the oracle


DECAY = 0.9


def oracle_scores(
    histories: dict[int, list[int]],
    n_items: int,
    user_rows: list[int],
    order: int,
    smoothing: float,
    decay: bool,
) -> np.ndarray:
    """A second implementation, written from the definition rather than from the code."""

    def weights(n: int) -> list[float]:
        if n <= 0:
            return []
        return [(DECAY ** (n - 1 - t)) if decay else 1.0 for t in range(n)]

    popularity = [0.0] * n_items
    unigram: dict[int, dict[int, float]] = {}
    bigram: dict[tuple[int, int], dict[int, float]] = {}

    for history in histories.values():
        for item in history:
            popularity[item] += 1.0
        w1 = weights(len(history) - 1)
        for t in range(len(history) - 1):
            row = unigram.setdefault(history[t], {})
            row[history[t + 1]] = row.get(history[t + 1], 0.0) + w1[t]
        w2 = weights(len(history) - 2)
        for t in range(len(history) - 2):
            key = (history[t], history[t + 1])
            row = bigram.setdefault(key, {})
            row[history[t + 2]] = row.get(history[t + 2], 0.0) + w2[t]

    total = sum(popularity)
    p0 = (
        [c / total for c in popularity]
        if total > 0
        else [1.0 / n_items] * n_items
    )

    def normalise(counts: dict[int, float], fallback: list[float]) -> list[float]:
        z = sum(counts.values())
        if z <= 0:
            return fallback
        return [counts.get(k, 0.0) / z for k in range(n_items)]

    out = np.zeros((len(user_rows), n_items))
    for position, user in enumerate(user_rows):
        history = histories.get(user, [])
        if not history:
            out[position] = p0
            continue

        last = history[-1]
        p1 = normalise(unigram.get(last, {}), p0)
        lower = [(1 - smoothing) * a + smoothing * b for a, b in zip(p1, p0, strict=True)]

        if order == 2 and len(history) >= 2:
            key = (history[-2], last)
            p2 = normalise(bigram[key], p1) if key in bigram else p1
            out[position] = [
                (1 - smoothing) * a + smoothing * b for a, b in zip(p2, lower, strict=True)
            ]
        else:
            out[position] = lower
    return out


# ---------------------------------------------------------------------------- the tests


class TestAgainstOracle:
    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.5])
    @pytest.mark.parametrize("decay", [False, True])
    def test_scores_match_the_oracle(self, fitted, histories, order, smoothing, decay):
        model, matrix = fitted(order=order, smoothing=smoothing, decay=decay)
        users = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)

        got = model.score_users(matrix, users, exclude_seen=False)
        want = oracle_scores(histories, 5, list(users), order, smoothing, decay)

        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


class TestInvariants:
    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.5])
    def test_scores_are_a_distribution(self, fitted, order, smoothing):
        """Both interpolation mixtures sum to one, so the raw scores must too.

        Worth pinning: a mixture whose weights do not sum to one still produces a valid
        ranking, so the error would never surface as a wrong recommendation -- only as a
        surrogate fitted to scores on a scale that drifts with the smoothing setting.
        """
        model, matrix = fitted(order=order, smoothing=smoothing)
        scores = model.score_users(matrix, np.array([0, 1, 2, 3, 4, 5]), exclude_seen=False)
        np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-12)

    def test_smoothing_changes_the_ranking(self, hand_built):
        """The dead-axis regression test -- see the module docstring.

        Built rather than borrowed from the shared fixture, and the first attempt shows
        why. Scoring an arbitrary user proved nothing: that user's last item had no
        outgoing transitions anywhere in the data, so every setting fell through to
        popularity and produced the same ranking. The test passed the *implementation* and
        failed to test the *property*. Here the numbers are chosen so the flip is forced:

            P1(. | 0) = [0, 2/3, 1/3]     item 1 is twice as likely to follow item 0
            P0        = [1/7, 3/28, 3/4]  item 2 dominates the catalogue

        so at ``s=0`` item 1 leads, and by ``s=0.5`` popularity has overtaken it.
        """
        model = hand_built(
            {0: [0, 1, 0, 1, 0, 2], 1: [2] * 20, 2: [1, 0]}, 3, order=1, smoothing=0.0
        )
        sharp = model.score_users(model_matrix(3, 3), np.array([2]), exclude_seen=False)[0]
        assert list(np.argsort(-sharp, kind="stable")) == [1, 2, 0]

        smoothed_model = hand_built(
            {0: [0, 1, 0, 1, 0, 2], 1: [2] * 20, 2: [1, 0]}, 3, order=1, smoothing=0.5
        )
        smoothed = smoothed_model.score_users(
            model_matrix(3, 3), np.array([2]), exclude_seen=False
        )[0]
        assert list(np.argsort(-smoothed, kind="stable")) == [2, 1, 0]

    def test_decay_favours_recent_transitions(self, hand_built):
        """A repeated early transition must lose ground to a single recent one.

        User 0 leaves item 0 for item 1 twice, early, and for item 2 once, last. Decay
        cannot reverse the ranking here -- two transitions still outweigh one -- but it
        must move the ratio, which is the property being claimed. Asserting on the ratio
        rather than on the order is deliberate: an order assertion would pass unchanged if
        the decay weights were applied to the wrong end of the history.
        """
        histories = {0: [0, 1, 0, 1, 0, 2], 1: [1, 0]}
        matrix = model_matrix(2, 3)

        plain = hand_built(histories, 3, order=1, decay=False).score_users(
            matrix, np.array([1]), exclude_seen=False
        )[0]
        decayed = hand_built(histories, 3, order=1, decay=True).score_users(
            matrix, np.array([1]), exclude_seen=False
        )[0]

        assert plain[1] > plain[2]
        np.testing.assert_allclose(plain, [0.0, 2 / 3, 1 / 3], atol=1e-12)
        # 0.9^4 + 0.9^2 = 1.4661 against 1.0, normalised.
        np.testing.assert_allclose(decayed, [0.0, 1.4661 / 2.4661, 1.0 / 2.4661], atol=1e-9)
        assert decayed[2] / decayed[1] > plain[2] / plain[1]

    def test_unseen_bigram_falls_through_to_first_order(self, fitted, histories):
        """Backoff must produce a full ranking, never a degenerate all-zero row."""
        model, matrix = fitted(order=2, smoothing=0.0)
        # User 2's history ends 1 -> 0, a bigram context that never appears as a prefix of
        # a triple anywhere in the fixture.
        scores = model.score_users(matrix, np.array([2]), exclude_seen=False)[0]
        first_order, _ = fitted(order=1, smoothing=0.0)
        expected = first_order.score_users(matrix, np.array([2]), exclude_seen=False)[0]
        np.testing.assert_allclose(scores, expected, atol=1e-12)

    def test_empty_history_falls_back_to_popularity(self, fitted, histories):
        model, matrix = fitted(order=2, smoothing=0.0)
        scores = model.score_users(matrix, np.array([5]), exclude_seen=False)[0]

        counts = np.zeros(5)
        for history in histories.values():
            for item in history:
                counts[item] += 1
        np.testing.assert_allclose(scores, counts / counts.sum(), atol=1e-12)

    def test_single_interaction_user_has_no_transition_to_learn_from(self, fitted):
        """Order 2 with a one-item history must not read past the start of the sequence."""
        model, matrix = fitted(order=2, smoothing=0.0)
        scores = model.score_users(matrix, np.array([4]), exclude_seen=False)[0]
        assert np.isfinite(scores).all()
        np.testing.assert_allclose(scores.sum(), 1.0, atol=1e-12)


class TestDeclaredProperties:
    def test_is_deterministic_as_the_space_declares(self, fitted, matrix_free=None):
        """The space marks this family deterministic, so it is measured once per cell.

        That is only legitimate if it is true: a stochastic family measured once would put
        a single draw into the benchmark with a spread column of zero, which is a claim
        about noise rather than a measurement of it.
        """
        from budget_tune.space.grids import FAMILY_BY_NAME

        assert FAMILY_BY_NAME["markov"].deterministic

        first, matrix = fitted(order=2, smoothing=0.1, decay=True)
        second, _ = fitted(order=2, smoothing=0.1, decay=True)
        users = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(
            first.score_users(matrix, users), second.score_users(matrix, users)
        )

    def test_model_bytes_counts_the_whole_model(self, fitted):
        """A dict-backed bigram index would report a fraction of the memory in use.

        Memory is one of the resources this project claims to measure, so a family that
        under-reports it would corrupt the constrained-selection experiment specifically.
        """
        model, _ = fitted(order=2)
        assert model.model_bytes > 0

        by_hand = sum(
            part.nbytes
            for part in (
                model._c1.data, model._c1.indices, model._c1.indptr,
                model._c2.data, model._c2.indices, model._c2.indptr,
                model._bigram_keys, model._popularity,
            )
        )
        assert model.model_bytes == by_hand

    def test_serving_never_recommends_a_seen_item(self, fitted):
        model, matrix = fitted(order=1)
        scores = model.score_users(matrix, np.array([0]), exclude_seen=True)[0]
        for item in matrix[0].indices:
            assert scores[item] == -np.inf

    def test_rejects_invalid_settings(self, sequences_class):
        from budget_tune.families.markov import SequentialMarkov

        with pytest.raises(ValueError, match="order"):
            SequentialMarkov(order=3)
        with pytest.raises(ValueError, match="smoothing"):
            SequentialMarkov(smoothing=1.0)

    def test_requires_sequences(self, sequences_class):
        """Deriving an order from the interaction matrix would mean inventing one."""
        from scipy import sparse

        from budget_tune.families.markov import SequentialMarkov

        with pytest.raises(ValueError, match="sequential"):
            SequentialMarkov().fit(sparse.csr_matrix((2, 3)), None)
