"""A sequential Markov recommender: the one model family this project implements itself.

Four of the five families are inherited from green-rerank. This one is new, and it is here
for two reasons. It is the cheapest way to put *order* into the search space -- every other
family scores from co-occurrence and is blind to what a user did last -- and it is the
model class the recommender literature this work is aimed at actually builds on
(semantic-enhanced Markov models for sequential e-commerce recommendation).

**The model.** Transitions are counted from consecutive interactions in each user's
training history, then interpolated with lower-order distributions::

    order 1:  P(k | j)    = (1-s) P1(k|j)   + s P0(k)
    order 2:  P(k | i,j)  = (1-s) P2(k|i,j) + s [ (1-s) P1(k|j) + s P0(k) ]

where ``P0`` is the popularity distribution, ``P1`` the first-order transition
distribution, ``P2`` the second-order one, and ``s`` is ``smoothing``.

**Why interpolation rather than additive smoothing, which is what the design first said.**
Additive smoothing would have been a dead grid axis. Scoring one user uses exactly one
context, so adding a constant to every count and renormalising is a strictly monotone
transform of the counts -- the ranking, and therefore NDCG and recall, would be *identical*
at every smoothing value. Three grid cells would have produced three identical rows, and
the fault would have shown up as a suspiciously flat sensitivity plot rather than as an
error. Interpolation mixes distributions with different normalisers, so ``s`` genuinely
moves the ranking.

Both mixtures sum to one by construction, which is the invariant the tests pin: the scores
are a probability distribution over the catalogue before seen items are masked out.

**Backoff falls through, it does not fail.** An unseen bigram context uses the first-order
distribution; an unseen unigram context uses popularity; a user with no history at all gets
popularity. So the model always returns a full ranking, and a cell of the grid can never
degenerate into "no prediction".

Everything the model holds is a numpy or scipy array, including the bigram lookup, so
``model_bytes`` counts the whole model. A Python dict of contexts would have been simpler
and would have reported a fraction of the memory actually used -- and memory is one of the
resources this project claims to measure.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from budget_tune.companion import ensure_importable

ensure_importable("green_rerank")

from green_rerank.families.base import Family, Sequences  # noqa: E402

#: Geometric weight applied per step of transition age when ``decay`` is on. Fixed rather
#: than searched: the grid already carries the on/off decision, and a second continuous
#: axis would multiply the campaign without testing anything the on/off contrast does not.
DECAY_RATE = 0.9


class SequentialMarkov(Family):
    """Interpolated first- or second-order Markov chain over item transitions.

    Args:
        order: 1 uses the last item as context, 2 uses the last two.
        smoothing: interpolation weight toward the lower-order distribution, in [0, 1).
        decay: weight each transition by :data:`DECAY_RATE` raised to its age within the
            user's history, so recent behaviour counts for more.
        max_context: histories longer than this are truncated from the *recent* end when
            counting. Mirrors ``Sequences.max_length``; a window keeping the oldest
            interactions would train on history the user has moved on from.
    """

    name = "markov"
    iterative = False
    needs_sequences = True
    _model_attrs = ("_c1_data", "_c1_indices", "_c1_indptr", "_c2_data", "_c2_indices",
                    "_c2_indptr", "_bigram_keys", "_popularity")

    def __init__(
        self,
        order: int = 1,
        smoothing: float = 0.0,
        decay: bool = False,
        max_context: int = 200,
    ) -> None:
        super().__init__()
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2; got {order}")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1); got {smoothing}")
        self.order = int(order)
        self.smoothing = float(smoothing)
        self.decay = bool(decay)
        self.max_context = int(max_context)

        self._c1: sparse.csr_matrix | None = None
        self._c2: sparse.csr_matrix | None = None
        self._bigram_keys: np.ndarray | None = None
        self._popularity: np.ndarray | None = None
        self._sequences: Sequences | None = None

    # ------------------------------------------------------------------------ training

    def fit(
        self, matrix: sparse.csr_matrix, sequences: Sequences | None = None
    ) -> SequentialMarkov:
        sequences = self._require_sequences(sequences)
        n_items = matrix.shape[1]

        histories = [
            [int(i) for i in sequences.by_user.get(int(row), [])][-self.max_context :]
            for row in range(matrix.shape[0])
        ]

        self._popularity = self._popularity_counts(histories, n_items)
        self._c1 = self._first_order(histories, n_items)
        if self.order == 2:
            self._bigram_keys, self._c2 = self._second_order(histories, n_items)
        else:
            self._bigram_keys, self._c2 = None, None

        self._sequences = sequences
        self._n_items = n_items
        self._fitted = True
        return self

    def _weights(self, n_transitions: int) -> np.ndarray:
        """Weight per transition, most recent first in age terms.

        Age 0 is the most recent transition in the user's history. With decay off every
        transition weighs 1, so the counts are ordinary occurrence counts.
        """
        if n_transitions <= 0:
            return np.zeros(0, dtype=float)
        ages = np.arange(n_transitions - 1, -1, -1, dtype=float)
        return DECAY_RATE**ages if self.decay else np.ones(n_transitions, dtype=float)

    def _popularity_counts(self, histories: list[list[int]], n_items: int) -> np.ndarray:
        counts = np.zeros(n_items, dtype=float)
        for history in histories:
            for item in history:
                counts[item] += 1.0
        return counts

    def _first_order(self, histories: list[list[int]], n_items: int) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        for history in histories:
            weights = self._weights(len(history) - 1)
            for step in range(len(history) - 1):
                rows.append(history[step])
                cols.append(history[step + 1])
                values.append(float(weights[step]))
        return sparse.csr_matrix(
            (values, (rows, cols)), shape=(n_items, n_items), dtype=float
        )

    def _second_order(
        self, histories: list[list[int]], n_items: int
    ) -> tuple[np.ndarray, sparse.csr_matrix]:
        """Counts over observed bigram contexts only.

        Storing a row per *observed* context rather than per possible one keeps this
        tractable: the dense form would be ``n_items**2`` rows, which is 1.8 million at a
        1,366-item catalogue and rises with the square.
        """
        keys: list[int] = []
        targets: list[int] = []
        values: list[float] = []
        for history in histories:
            weights = self._weights(len(history) - 2)
            for step in range(len(history) - 2):
                keys.append(history[step] * n_items + history[step + 1])
                targets.append(history[step + 2])
                values.append(float(weights[step]))

        if not keys:
            return np.zeros(0, dtype=np.int64), sparse.csr_matrix((0, n_items), dtype=float)

        key_array = np.asarray(keys, dtype=np.int64)
        unique = np.unique(key_array)
        rows = np.searchsorted(unique, key_array)
        counts = sparse.csr_matrix(
            (values, (rows, np.asarray(targets, dtype=np.int64))),
            shape=(unique.size, n_items),
            dtype=float,
        )
        return unique, counts

    # ------------------------------------------------------------------------- serving

    def _distribution(self, counts_row: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        """Normalise a count vector, falling through when the context was never seen."""
        total = counts_row.sum()
        if total <= 0:
            return fallback
        return counts_row / total

    def _scores(self, matrix: sparse.csr_matrix, user_rows: np.ndarray) -> np.ndarray:
        assert self._c1 is not None and self._popularity is not None
        n_items = self._n_items
        smoothing = self.smoothing

        popularity_total = self._popularity.sum()
        p0 = (
            self._popularity / popularity_total
            if popularity_total > 0
            # An empty training set leaves nothing to prefer, and a uniform ranking is the
            # honest statement of that. Returning zeros would make the top-n an artefact
            # of the tie-break rule instead.
            else np.full(n_items, 1.0 / n_items, dtype=float)
        )

        scores = np.empty((len(user_rows), n_items), dtype=float)
        for position, row in enumerate(user_rows):
            history = [int(i) for i in (self._sequences.by_user.get(int(row), []) or [])]
            history = history[-self.max_context :]

            if not history:
                scores[position] = p0
                continue

            last = history[-1]
            p1 = self._distribution(self._c1[last].toarray().ravel(), p0)
            lower = (1.0 - smoothing) * p1 + smoothing * p0

            if self.order == 2 and len(history) >= 2 and self._bigram_keys is not None:
                key = history[-2] * n_items + last
                index = np.searchsorted(self._bigram_keys, key)
                seen = index < self._bigram_keys.size and self._bigram_keys[index] == key
                p2 = (
                    self._distribution(self._c2[index].toarray().ravel(), p1)
                    if seen
                    else p1
                )
                scores[position] = (1.0 - smoothing) * p2 + smoothing * lower
            else:
                scores[position] = lower

        return scores

    # --------------------------------------------------------------------- bookkeeping
    #
    # ``model_bytes`` sums the attributes named in ``_model_attrs``, and the base class
    # only understands arrays and sparse matrices. Exposing the CSR components as plain
    # attributes keeps the accounting honest without special-casing this family in the
    # base class.

    @property
    def _c1_data(self):
        return None if self._c1 is None else self._c1.data

    @property
    def _c1_indices(self):
        return None if self._c1 is None else self._c1.indices

    @property
    def _c1_indptr(self):
        return None if self._c1 is None else self._c1.indptr

    @property
    def _c2_data(self):
        return None if self._c2 is None else self._c2.data

    @property
    def _c2_indices(self):
        return None if self._c2 is None else self._c2.indices

    @property
    def _c2_indptr(self):
        return None if self._c2 is None else self._c2.indptr
