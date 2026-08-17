"""The epoch ladder, and the statistics that decide whether it is worth having.

Successive halving needs two things from a resource, and the calibration pilot only
established the first: epochs are cheap at low budgets (C1), and a ranking at a low budget
must be informative about the ranking at a high one (C2). This module holds the declared
ladder and the machinery for testing C2 — kept separate from the experiment script so the
schedule is a fixed constant that a script cannot quietly re-tune, and so the statistics can
be tested against hand-computed cases.

The schedule below is **frozen before the validation runs**. Choosing rungs or keep fractions
after seeing which of them performs best would be post-hoc selection of a baseline's own
hyperparameters, which is precisely the asymmetry that made the companion project's first
conclusion wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ladder:
    """A successive-halving schedule for one family.

    Attributes:
        family: the family this applies to.
        rungs: epoch budgets, ascending. Taken from the existing search grid; no rung
            introduces an epoch value the campaign would not otherwise measure.
        keep: fraction of configurations surviving each promotion, one entry shorter than
            ``rungs``. Not a single ``eta`` because the grids are not geometric -- ALS steps
            x3 then x2 -- and forcing one would require inventing an epoch value.
    """

    family: str
    rungs: tuple[int, ...]
    keep: tuple[float, ...]

    def __post_init__(self) -> None:
        if list(self.rungs) != sorted(self.rungs):
            raise ValueError(f"{self.family}: rungs must ascend; got {self.rungs}")
        if len(self.keep) != len(self.rungs) - 1:
            raise ValueError(
                f"{self.family}: {len(self.rungs)} rungs need {len(self.rungs) - 1} "
                f"keep fractions; got {len(self.keep)}"
            )
        if not all(0 < fraction < 1 for fraction in self.keep):
            raise ValueError(f"{self.family}: keep fractions must be in (0, 1)")

    def survivors(self, n: int) -> list[int]:
        """How many configurations reach each rung, starting from ``n``.

        At least one always survives: a schedule that discarded everything would report a
        regret against a configuration it never evaluated.
        """
        counts = [n]
        for fraction in self.keep:
            counts.append(max(1, int(counts[-1] * fraction)))
        return counts


#: Frozen schedule. Rungs come from the epoch grids in ``budget_tune.space.grids``.
LADDERS: dict[str, Ladder] = {
    "als": Ladder("als", rungs=(5, 15, 30), keep=(1 / 3, 1 / 2)),
    "multvae": Ladder("multvae", rungs=(10, 20), keep=(0.5,)),
}

#: Families with no iterative resource. Named explicitly rather than inferred from a missing
#: ``epochs`` key, so that adding a family forces a decision about its fidelity instead of
#: silently defaulting to "none".
NON_ITERATIVE: tuple[str, ...] = ("popularity", "itemknn", "markov")


def rank_agreement(low: np.ndarray, high: np.ndarray) -> dict:
    """Spearman and Kendall agreement between two rankings of the same configurations.

    Both are reported because they disagree in a way that matters here: Spearman is sensitive
    to how far a configuration moves, Kendall to how often any pair swaps. A fidelity that
    preserves the top of the ranking while shuffling the bottom scores badly on one and well
    on the other, and successive halving only cares about the top.
    """
    from scipy import stats

    spearman = stats.spearmanr(low, high)
    kendall = stats.kendalltau(low, high)
    return {
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "kendall": float(kendall.statistic),
        "kendall_p": float(kendall.pvalue),
        "n": int(len(low)),
    }


def top_k_overlap(low: np.ndarray, high: np.ndarray, k: int) -> float:
    """Fraction of the high-budget top ``k`` that the low-budget top ``k`` already contains."""
    if k > len(low):
        raise ValueError(f"k={k} exceeds the {len(low)} configurations available")
    best_low = set(np.argsort(-low, kind="stable")[:k].tolist())
    best_high = set(np.argsort(-high, kind="stable")[:k].tolist())
    return len(best_low & best_high) / k


def simulate_halving(scores: dict[int, np.ndarray], ladder: Ladder) -> dict:
    """Run the declared schedule on already-measured scores and report what it would cost.

    Args:
        scores: ``rung epochs -> validation score per configuration``, all arrays aligned to
            the same configuration order.
        ladder: the frozen schedule.

    Returns:
        The regret of the surviving best against the true best at the maximum rung, and the
        number of configurations that were discarded early but would have finished strong.

    Regret is the decision-relevant statistic. A ladder can have mediocre rank correlation and
    still lose nothing, if the configurations it discards were never going to win; and it can
    have respectable correlation and still discard the winner.
    """
    top = ladder.rungs[-1]
    alive = np.arange(len(scores[top]))
    counts = ladder.survivors(len(alive))

    discarded_at: dict[int, np.ndarray] = {}
    for step, rung in enumerate(ladder.rungs[:-1]):
        keep_n = counts[step + 1]
        order = np.argsort(-scores[rung][alive], kind="stable")
        survivors = alive[order[:keep_n]]
        discarded_at[rung] = np.setdiff1d(alive, survivors)
        alive = survivors

    final = scores[top]
    true_best = int(np.argmax(final))
    survivor_best = int(alive[np.argmax(final[alive])])

    # How many early casualties would have finished strong. Counted at the first rung
    # specifically: that is where successive halving throws away the most, and where a bad
    # fidelity does its damage.
    #
    # The comparison set is the top ``min(10, survivors)`` rather than a fixed top ten,
    # because a schedule cannot promote more configurations than it has slots. Against a
    # fixed ten, a *perfect* fidelity promoting four survivors would be charged six
    # discarded-then-strong -- a metric that punishes the schedule for its own keep fraction
    # and would have read as evidence against epoch fidelity.
    first_rung = ladder.rungs[0]
    strong_k = min(10, counts[1])
    strong = set(np.argsort(-final, kind="stable")[:strong_k].tolist())
    discarded_then_strong = int(len(strong & set(discarded_at[first_rung].tolist())))

    return {
        "survivors_per_rung": counts,
        "true_best_score": float(final[true_best]),
        "survivor_best_score": float(final[survivor_best]),
        "regret": float(final[true_best] - final[survivor_best]),
        "regret_normalised": float(
            (final[true_best] - final[survivor_best]) / (final.max() - final.min())
        )
        if final.max() > final.min()
        else 0.0,
        "found_true_best": bool(survivor_best == true_best),
        "discarded_then_strong": discarded_then_strong,
        "discarded_then_strong_k": int(strong_k),
        "score_spread": float(final.max() - final.min()),
    }
