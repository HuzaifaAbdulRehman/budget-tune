"""Score a fitted model on one split. The measurement primitive the campaign is built from.

Two rules govern this module, both inherited and both learned the hard way in the companion
projects.

**Scoring happens outside every measured window.** The functions here are deliberately split
so a caller can measure serving and then compute metrics afterwards: metric computation is
O(k) per user in Python, which is noise against a training run and the *majority* of the
reading for a cheap retrieval. Folding them together corrupts exactly the cheap
configurations the energy comparison depends on.

**Which split is scored is an argument, never a default.** The campaign calls this twice per
configuration and writes the two results to separate files, so nothing downstream can join
them by accident. This module lives under ``benchmark/`` for that reason -- it is one of the
two packages permitted to name a reporting column.

NDCG is computed against the genuinely held-out interaction, following green-rerank: under
leave-two-out there is at most one relevant item per user, so the ideal ranking puts it first
and the ideal DCG is ``dcg([1.0])``. It is *not* computed against the model's own scores,
which would return 1.0 by construction for every family and answer a different question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from budget_tune.companion import ensure_importable
from budget_tune.data.splits import HpoDataset


def _metrics():
    ensure_importable("feasible_rerank")
    from qubo_rerank.metrics.fairness import exposure_parity
    from qubo_rerank.metrics.relevance import dcg

    return dcg, exposure_parity


@dataclass(frozen=True)
class Scores:
    """What one evaluation produced, per user and aggregated.

    Attributes:
        ndcg: mean NDCG@k against the held-out item.
        recall: mean Recall@k against the held-out item.
        exposure_parity: mean deviation from equal group exposure across the returned
            lists. Carried because the constrained-selection experiment needs a second
            objective that is not a cost, and because it is cheap to compute here.
        per_user_ndcg: retained so that a selected configuration can later be compared
            against another by a paired per-user test, which has far more power than
            comparing means over seeds.
        n_users: users scored.
    """

    ndcg: float
    recall: float
    exposure_parity: float
    per_user_ndcg: np.ndarray
    n_users: int


def recommend(model, matrix, user_rows: np.ndarray, k: int) -> np.ndarray:
    """Top-``k`` catalogue items per user. The whole of what a measured window covers.

    Separated from :func:`score` so that the caller measures this and scores the result
    afterwards. Seen items are masked by the family base class, so a held-out item that also
    appeared in training could never be returned -- which is why the split deduplicates.

    The campaign measures the two halves separately via :func:`score_catalogue` and
    :func:`select_top`; this is the convenience path for callers that do not need the split.
    """
    return model.recommend(matrix, user_rows, n=k, exclude_seen=True).items


def score_catalogue(model, matrix, user_rows: np.ndarray) -> np.ndarray:
    """Dense catalogue-wide scores with seen items masked. The family-specific half.

    Split from :func:`select_top` because green-rerank profiled selection as the *majority*
    of retrieval cost -- 99.9% of it for popularity -- and that half is identical code for
    every family. Reported as one number, the comparison between families would be dominated
    by work none of them does differently.
    """
    return model.score_users(matrix, user_rows, exclude_seen=True)


def select_top(model, scores: np.ndarray, user_rows: np.ndarray, k: int) -> np.ndarray:
    """Top-``k`` from precomputed scores. Shared by every family, deliberately."""
    return model.select(scores, user_rows, k).items


def score(
    dataset: HpoDataset,
    items: np.ndarray,
    user_rows: np.ndarray,
    split: str,
    k: int = 10,
) -> Scores:
    """Grade recommendation lists against a split's held-out items.

    Args:
        dataset: the catalogue, for its targets and group labels.
        items: ``(n_users, k)`` catalogue item indices, best first.
        user_rows: matrix rows aligned with ``items``.
        split: ``"validation"`` or ``"test"``. Required, never defaulted.
        k: list length to grade at.
    """
    dcg, exposure_parity = _metrics()
    targets = dataset.targets(split, user_rows)

    ideal = dcg([1.0])
    per_user_ndcg = np.zeros(len(user_rows), dtype=float)
    per_user_recall = np.zeros(len(user_rows), dtype=float)
    parities = []

    for position in range(len(user_rows)):
        selected = [int(i) for i in items[position][:k]]
        target = targets[position]

        gains = [1.0 if item in target else 0.0 for item in selected]
        per_user_ndcg[position] = dcg(gains) / ideal if target else 0.0
        per_user_recall[position] = 1.0 if any(gains) else 0.0

        # Group labels of the items actually returned, graded as a complete list. Matches
        # green-rerank's usage so the parity column means the same thing in both projects.
        parities.append(exposure_parity(dataset.groups[selected], list(range(len(selected)))))

    return Scores(
        ndcg=float(per_user_ndcg.mean()),
        recall=float(per_user_recall.mean()),
        exposure_parity=float(np.mean(parities)),
        per_user_ndcg=per_user_ndcg,
        n_users=len(user_rows),
    )
