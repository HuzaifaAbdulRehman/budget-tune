"""Locate the two companion checkouts and import from them.

This project sits downstream of both:

* ``qubo-rerank`` (*feasible-rerank*) supplies the QUBO solvers, BQM assembly, the
  relevance/fairness metrics, the paired-Wilcoxon machinery and the dataset primitives
  (k-core, the interaction matrix, popularity tiers, the MovieLens genre groups).
* ``rerank-green`` (*green-rerank*) supplies the CPU-second measurement session, its
  guards, and four of the five model families with their hyperparameters already exposed
  as constructor arguments.

Neither is vendored. Two implementations of "the same" metric that disagree is the exact
failure this family of projects is supposed to be above, and a copy that drifts is how it
happens. The cost of that decision is this module: a checkout that is missing must fail
here, with the paths that were tried, rather than fifty frames deep in an ``ImportError``.

Resolution order per companion: an environment override, then the sibling directory
beside this repository.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Companion:
    """One companion checkout: what it is called, where to look, how to recognise it.

    Attributes:
        key: short name used in error messages and manifests.
        env_var: environment variable that overrides the search.
        dirname: sibling directory name, which is not always the repository name --
            ``feasible-rerank`` sits on disk as ``qubo-rerank``, from before it was
            renamed.
        marker: a file that must exist for the directory to count. Chosen as a module
            this project actually imports, so a half-cloned or renamed directory fails
            here rather than at the first attribute access.
    """

    key: str
    env_var: str
    dirname: str
    marker: str


COMPANIONS: dict[str, Companion] = {
    c.key: c
    for c in (
        Companion(
            key="feasible_rerank",
            env_var="BUDGET_TUNE_FEASIBLE_RERANK",
            dirname="qubo-rerank",
            marker="qubo_rerank/metrics/relevance.py",
        ),
        Companion(
            key="green_rerank",
            env_var="BUDGET_TUNE_GREEN_RERANK",
            dirname="rerank-green",
            marker="green_rerank/measure/session.py",
        ),
    )
}


class CompanionNotFound(RuntimeError):
    """A companion checkout could not be located."""


def candidate_paths(key: str) -> list[Path]:
    """Where a companion might be, most explicit first."""
    companion = _get(key)
    found: list[Path] = []
    override = os.environ.get(companion.env_var)
    if override:
        found.append(Path(override).expanduser())
    found.append(REPO_ROOT.parent / companion.dirname)
    return found


def _get(key: str) -> Companion:
    try:
        return COMPANIONS[key]
    except KeyError:
        raise KeyError(f"unknown companion {key!r}; known: {sorted(COMPANIONS)}") from None


@lru_cache(maxsize=4)
def companion_root(key: str) -> Path:
    """The checkout for ``key``, or raise listing every path that was tried."""
    companion = _get(key)
    tried = candidate_paths(key)
    for path in tried:
        if (path / companion.marker).exists():
            return path
    raise CompanionNotFound(
        f"could not find the {companion.key} companion. It supplies infrastructure this "
        "project reuses rather than reimplements.\nTried:\n  "
        + "\n  ".join(str(p) for p in tried)
        + f"\nSet {companion.env_var} to the checkout to override."
    )


@lru_cache(maxsize=4)
def ensure_importable(key: str) -> Path:
    """Put a companion on ``sys.path`` and return its root. Idempotent."""
    root = companion_root(key)
    if str(root) not in sys.path:
        # Appended, never prepended: this project's own modules must win any name
        # collision. Both companions carry a top-level ``experiments/`` and so does this
        # one, and a prepended path would silently route our imports into theirs.
        sys.path.append(str(root))
    return root


def ensure_all_importable() -> dict[str, Path]:
    """Put both companions on the path. Returns ``key -> root`` for the manifest."""
    return {key: ensure_importable(key) for key in COMPANIONS}


def available(key: str) -> bool:
    """Whether a companion can be found, for skipping tests that need real data."""
    try:
        companion_root(key)
    except CompanionNotFound:
        return False
    return True


def all_available() -> bool:
    return all(available(key) for key in COMPANIONS)


def revisions() -> dict[str, str]:
    """Git revision of each companion, for the manifest.

    A dirty working tree is reported as ``abc1234-dirty``, because that is not a revision
    anyone can return to and it must not look like one. Follows green-rerank's manifest
    convention so the three repositories' provenance records are directly comparable.
    """
    import subprocess

    out: dict[str, str] = {}
    for key in COMPANIONS:
        try:
            root = companion_root(key)
        except CompanionNotFound:
            out[key] = "not-found"
            continue
        try:
            rev = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout.strip()
            out[key] = f"{rev}-dirty" if dirty else rev
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            out[key] = "unknown"
    return out
