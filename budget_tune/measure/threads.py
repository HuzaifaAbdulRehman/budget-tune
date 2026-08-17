"""Thread pinning, which has to happen before numpy is imported.

**Why this is mandatory rather than tidy.** ``time.process_time`` sums CPU time over every
thread, so a BLAS matmul spread across four cores bills four CPU-seconds per wall-second.
The families in this space do not thread alike: ALS is dense linear algebra and MultVAE is
torch, both of which parallelise; ItemKNN is scipy sparse and the Markov family is a Python
loop, neither of which does. Comparing them on CPU-seconds without pinning would measure how
well each family happens to parallelise as much as how much work it does, and the cost axis
is the axis this project's conclusions hang from.

Pinning also makes the two axes agree. At one thread CPU-seconds and wall-seconds measure
the same thing, so the robustness check the design promises -- repeat the analysis on wall
time and see whether any conclusion flips -- becomes a check on measurement noise rather
than a check on thread counts.

**The ordering constraint is real.** OpenMP, MKL and OpenBLAS read their thread counts once,
when the shared library loads, which happens on ``import numpy``. Setting the variables
afterwards is silently ignored: the process keeps whatever it started with and the run
reports a pinned thread count it never had. :func:`pin` therefore refuses to run once numpy
is present, rather than leaving that failure to be discovered in a results table.

Usage, at the very top of an experiment script, above every other import::

    from budget_tune.measure.threads import pin
    pin(1)
"""

from __future__ import annotations

import os
import sys

#: The libraries that read a thread count from the environment at load time.
THREAD_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class ThreadPinningError(RuntimeError):
    """Pinning was attempted too late to take effect."""


def pin(threads: int = 1, allow_late: bool = False) -> dict[str, str]:
    """Pin every numerical backend to ``threads``, before numpy loads.

    Args:
        threads: threads per backend. One by default, which is the campaign's declared
            value: it makes CPU-seconds and wall-seconds coincide and removes the
            family-dependent parallelism confound described above.
        allow_late: permit pinning after numpy is imported. Only for tests that check the
            refusal, and for the deliberate multi-thread comparison in the calibration
            pilot, which re-execs a fresh process for each thread count.

    Returns:
        The variables as set, for the manifest.

    Raises:
        ThreadPinningError: numpy is already imported, so the setting would be recorded but
            not applied.
    """
    if threads < 1:
        raise ValueError(f"threads must be at least 1; got {threads}")

    if "numpy" in sys.modules and not allow_late:
        raise ThreadPinningError(
            "numpy is already imported, so BLAS has read its thread count and this call "
            "would change nothing but the manifest. Call pin() at the very top of the "
            "entry-point script, above every other import."
        )

    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(threads)

    # torch keeps its own pool and reads it at call time rather than at import, so it is
    # set separately -- and only if torch is already loaded, since importing it here would
    # cost seconds in every process that has no neural family to run.
    if "torch" in sys.modules:  # pragma: no cover - depends on import order
        sys.modules["torch"].set_num_threads(threads)

    return declared()


def declared() -> dict[str, str]:
    """The thread settings currently in the environment, for the manifest."""
    return {variable: os.environ.get(variable, "<unset>") for variable in THREAD_VARIABLES}


def verify(threads: int) -> dict[str, object]:
    """Check that the backends actually see the pinned count.

    Environment variables are a request, not a guarantee: a wheel built against a different
    OpenMP runtime can ignore them. This reports what the libraries say about themselves so
    a run records the thread count it *had* rather than the one it asked for.
    """
    report: dict[str, object] = {"requested": threads, "environment": declared()}

    report["environment_consistent"] = all(
        value == str(threads) for value in report["environment"].values()
    )

    if "torch" in sys.modules:  # pragma: no cover - torch is an optional extra
        torch = sys.modules["torch"]
        torch.set_num_threads(threads)
        report["torch_threads"] = int(torch.get_num_threads())
        report["torch_consistent"] = report["torch_threads"] == threads

    return report


def apply_to_torch(threads: int) -> int | None:
    """Set torch's thread pool, importing torch only if the caller needs it.

    Returns the count torch reports, or ``None`` when torch is not installed.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is an optional extra
        return None
    torch.set_num_threads(threads)
    return int(torch.get_num_threads())
