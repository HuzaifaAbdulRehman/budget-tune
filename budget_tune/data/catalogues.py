"""Which catalogues this project measures on, and where to find them.

The registry of *what a catalogue is* -- its files, its group definition, its note -- lives
in green-rerank and is reused wholesale, including its path resolution. Only the roles are
declared here, because the roles are this project's alone:

* three **headline** catalogues carry every reported result;
* one **meta** catalogue exists solely to freeze every method's own settings before the
  headline catalogues are touched.

That second role is the only defence against a leak that no split can prevent. The search
space is enumerated, so the experimenters know where the optimum is while choosing TPE's
startup-trial count, Hyperband's ``eta``, the surrogates' priors and the penalty weights.
Choosing them on Gift Cards and freezing them removes that knowledge from every headline
number -- and it applies to the classical baselines exactly as much as to the QUBO methods,
since tuning one side carefully and leaving the other at defaults is the asymmetry that
made the companion project's first conclusion wrong.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from budget_tune.companion import ensure_importable
from budget_tune.data.splits import HpoDataset, assemble

#: Catalogues every reported result is measured on.
HEADLINE: tuple[str, ...] = ("ml100k", "luxury_beauty", "software")

#: Reserved for freezing method meta-parameters. Never appears in a results table.
META: str = "gift_cards"


def _green_rerank_catalogues():
    ensure_importable("green_rerank")
    from green_rerank import catalogues

    return catalogues


def resolve(name: str) -> Path:
    """Locate a catalogue's files, reusing green-rerank's search path."""
    return _green_rerank_catalogues().resolve(name)


def describe(name: str):
    """The catalogue's registry entry -- kind, grouping, and why it is included."""
    return _green_rerank_catalogues().get(name)


def available() -> list[str]:
    """Registered catalogues whose files are actually present."""
    return _green_rerank_catalogues().available()


def role(name: str) -> str:
    """``"headline"``, ``"meta"``, or ``"unused"``.

    Callers that write results consult this, so a run that would put the meta catalogue
    into a headline table fails rather than producing one.
    """
    if name in HEADLINE:
        return "headline"
    if name == META:
        return "meta"
    return "unused"


def _raw_frame_and_groups(name: str):
    """``(raw interactions, groups_fn)`` for a registered catalogue."""
    catalogue = describe(name)
    path = resolve(name)
    ensure_importable("feasible_rerank")

    if catalogue.kind == "movielens":
        from benchmarks.movielens import genre_groups, load_genres
        from benchmarks.movielens import load_ratings as load_ml_ratings

        genres = load_genres(path)

        def groups_fn(matrix, item_ids, n_groups):
            return genre_groups(genres, list(item_ids), n_groups)

        return load_ml_ratings(path), groups_fn

    if catalogue.kind == "amazon":
        import numpy as np
        from benchmarks.loader import load_ratings, popularity_tiers

        def groups_fn(matrix, item_ids, n_groups):
            popularity = np.asarray(matrix.sum(axis=0), dtype=float).ravel()
            return popularity_tiers(popularity, n_tiers=n_groups)

        return load_ratings(path), groups_fn

    raise ValueError(f"catalogue {name!r} has unknown kind {catalogue.kind!r}")


def load(
    name: str,
    fractions: tuple[float, ...],
    min_interactions: int = 5,
    n_groups: int = 4,
    max_sequence: int = 200,
) -> HpoDataset:
    """Load a registered catalogue, split leave-two-out, at the given data fractions.

    Not cached. The campaign loads each catalogue once and holds it; a cache keyed on the
    preprocessing arguments would be a second place for two runs that filtered differently
    to look comparable.
    """
    raw, groups_fn = _raw_frame_and_groups(name)
    return assemble(
        name=name,
        raw=raw,
        groups_fn=groups_fn,
        fractions=fractions,
        min_interactions=min_interactions,
        n_groups=n_groups,
        max_sequence=max_sequence,
    )


def synthetic(
    fractions: tuple[float, ...],
    n_users: int = 120,
    n_items: int = 60,
    blocks: int = 4,
    per_user: int = 9,
    seed: int = 0,
) -> HpoDataset:
    """A block-structured catalogue with a known answer, needing no download.

    Exists so the whole path -- splitting, retention, families, measurement, scoring --
    runs in CI where no dataset is present. A harness whose end-to-end path only executes
    on the one machine holding the downloads is a harness whose end-to-end path is
    effectively untested.

    Each user draws from one block of items, so a model that learns the block structure
    can find the held-out items and one that does not cannot. ``per_user`` is 9 by default
    so that every user clears :data:`~budget_tune.data.splits.MIN_HISTORY_FOR_VALIDATION`
    with room to spare at the lowest data fraction.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    per_block = n_items // blocks

    rows = []
    for user in range(n_users):
        block = user % blocks
        low, high = block * per_block, (block + 1) * per_block
        picked = rng.choice(np.arange(low, high), size=min(per_user, per_block), replace=False)
        for step, item in enumerate(picked):
            rows.append(
                {
                    "user_id": f"u{user}",
                    "item_id": f"i{int(item)}",
                    "rating": 1.0,
                    # Distinct timestamps per user so the recency order is unambiguous;
                    # the split's tie-break is tested separately on deliberately tied data.
                    "timestamp": step,
                }
            )

    def groups_fn(matrix, item_ids, n_groups):
        return np.array([int(item[1:]) // per_block for item in item_ids]) % n_groups

    return assemble(
        name="synthetic",
        raw=pd.DataFrame(rows),
        groups_fn=groups_fn,
        fractions=fractions,
        min_interactions=1,
        n_groups=blocks,
    )
