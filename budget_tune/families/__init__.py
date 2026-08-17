"""Model families.

Four of the five are imported from green-rerank rather than rewritten; only the sequential
Markov family is new here. The import is deferred behind :func:`build` so that this package
can be imported -- and most of the test suite run -- without the companion checkouts or
torch being present.
"""

from __future__ import annotations


def build(family: str, **kwargs):
    """Construct a family by name, with its hyperparameters as keyword arguments.

    One place where a configuration becomes an object. The campaign, the tests and any
    later live-run experiment all go through it, so a hyperparameter renamed in the search
    space fails loudly here instead of being silently ignored by a constructor that accepts
    ``**kwargs``.
    """
    if family == "markov":
        from budget_tune.families.markov import SequentialMarkov

        return SequentialMarkov(**kwargs)

    from budget_tune.companion import ensure_importable

    ensure_importable("green_rerank")

    if family == "popularity":
        from green_rerank.families.classical import Popularity

        return Popularity(**kwargs)
    if family == "itemknn":
        from green_rerank.families.classical import ItemKNN

        return ItemKNN(**kwargs)
    if family == "als":
        from green_rerank.families.classical import ImplicitALS

        return ImplicitALS(**kwargs)
    if family == "multvae":
        from green_rerank.families.neural import MultVAE

        return MultVAE(**kwargs)

    raise KeyError(f"unknown family {family!r}")


__all__ = ["build"]
