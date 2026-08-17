"""Shared fixtures and the companion/data skip rules.

A test that quietly shrinks when a checkout is missing is worse than one that fails: the
suite still passes, and the invariants that needed real data were never checked. So the
skips are explicit, marked, and reported -- and ``--strict-companion`` turns them into
failures on the machine where the checkouts *are* present, which is the machine whose
results are published.
"""

from __future__ import annotations

import pandas as pd
import pytest

from budget_tune.companion import all_available, available, ensure_importable


def pytest_addoption(parser):
    parser.addoption(
        "--strict-companion",
        action="store_true",
        default=False,
        help="fail instead of skipping when a companion checkout or catalogue is missing",
    )


def pytest_collection_modifyitems(config, items):
    strict = config.getoption("--strict-companion")
    if strict:
        return
    for item in items:
        if "companion" in item.keywords and not all_available():
            item.add_marker(pytest.mark.skip(reason="companion checkout not found"))


@pytest.fixture(scope="session")
def strict(request) -> bool:
    return bool(request.config.getoption("--strict-companion"))


@pytest.fixture(scope="session")
def companions_present() -> bool:
    return all_available()


def require_companion(key: str, strict: bool) -> None:
    """Make a companion importable, or skip -- or fail under ``--strict-companion``.

    Importing is part of the job, not the caller's. A version of this that only checked
    availability let three tests reach ``import benchmarks.loader`` with nothing on
    ``sys.path``, so they failed with an unrelated ``ModuleNotFoundError`` instead of
    testing anything.
    """
    if available(key):
        ensure_importable(key)
        return
    message = f"{key} checkout not found"
    if strict:
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture
def interactions() -> pd.DataFrame:
    """A small hand-written interaction frame with known structure.

    Deliberately ragged: user ``d`` has exactly two interactions, which is the boundary
    case the validation split has to refuse, and user ``e`` has three, which is the
    smallest history that may donate one.
    """
    rows = [
        # user, item, timestamp
        ("a", "i1", 10), ("a", "i2", 20), ("a", "i3", 30), ("a", "i4", 40), ("a", "i5", 50),
        ("b", "i2", 15), ("b", "i3", 25), ("b", "i1", 35), ("b", "i4", 45),
        ("c", "i5", 11), ("c", "i4", 21), ("c", "i3", 31),
        ("d", "i1", 12), ("d", "i2", 22),
        ("e", "i3", 13), ("e", "i1", 23), ("e", "i5", 33),
    ]
    return pd.DataFrame(
        [
            {"user_id": u, "item_id": i, "rating": 1.0, "timestamp": t}
            for u, i, t in rows
        ]
    )
