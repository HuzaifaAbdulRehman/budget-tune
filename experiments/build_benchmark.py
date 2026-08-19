"""Build the enumerated benchmark: every canonical configuration, measured once per seed.

This is the artifact every later table derives from, so it is written to be interrupted and
resumed rather than to be lucky. Four rules govern it, all inherited and all load-bearing:

* **Seeds are the outermost loop.** An interrupted campaign then leaves one complete
  observation of every configuration rather than three of the first family and none of the
  rest -- which is the difference between a usable partial result and a wasted afternoon.
* **The search and reporting splits are written to separate files** from the moment they are
  produced. Nothing downstream has to remember to keep them apart.
* **Cheap stages are repeated until they clear the clock quantum.** A reading below the
  scheduler tick is a tick count, not a duration, and popularity's fit is four orders of
  magnitude below it.
* **The manifest is written before the analysis and updated as the run proceeds.** The
  fidelity study lost a completed twelve-minute run's conditions record to a bug in the
  arithmetic that followed it; that ordering is not repeated here.

Run::

    python -m experiments.build_benchmark --all --threads 1
    python -m experiments.build_benchmark --all --threads 1        # resumes where it stopped
"""

from __future__ import annotations

# Thread pinning must precede numpy -- see budget_tune.measure.threads.
from budget_tune.measure.threads import pin  # isort: skip

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Rows land here as they are measured, then are promoted to the schema's four files.
#:
#: JSON Lines, not CSV. The first version appended CSV with the header written once, and
#: families do not share hyperparameter columns -- so a flush containing ALS rows wrote four
#: more fields than the header declared, and 33 minutes of completed measurement ended as an
#: unparseable file. A row-oriented format has no header to disagree with.
SEARCH_PARTIAL = "search_runs.partial.jsonl"
REPORT_PARTIAL = "report_runs.partial.jsonl"


def source_fingerprint() -> dict:
    """A content hash of the code that produced a results directory.

    The design requires one code version per results directory. Git records a revision, but
    hashing the sources still catches uncommitted edits the same way a dirty working tree
    would -- by changing -- and does not depend on git being available to whoever reads the
    manifest.
    """
    digest = hashlib.sha256()
    files = sorted(
        p
        for d in ("budget_tune", "experiments")
        for p in (REPO_ROOT / d).rglob("*.py")
        if "__pycache__" not in p.parts
    )
    for path in files:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return {"source_sha256": digest.hexdigest()[:16], "files": len(files)}


def seeds_for(family: str, seeds: list[int]) -> list[int]:
    """Which seeds a family needs.

    Deterministic families are measured once. That is only legitimate because
    ``tests/test_splits.py`` and ``tests/test_markov.py`` assert their determinism rather
    than assuming it -- a stochastic family measured once would put a single draw into the
    benchmark with a spread column of zero, which is a claim about noise rather than a
    measurement of it.
    """
    from budget_tune.space.grids import FAMILY_BY_NAME

    return [seeds[0]] if FAMILY_BY_NAME[family].deterministic else seeds


def measure_configuration(dataset, config, seed: int, k: int, session, evaluate) -> tuple:
    """Measure one configuration and score it on both splits. Returns ``(search, report)``.

    Training, catalogue scoring and top-k selection are measured separately; metric
    computation happens after all three windows have closed. Both splits are graded from the
    *same* recommendation lists, so the reporting row costs no extra compute and cannot
    diverge from the search row by accident.
    """
    from budget_tune.families import build

    fold = dataset.fold(config.data_fraction)
    users = dataset.eval_users()

    kwargs = dict(config.kwargs)
    from budget_tune.space.grids import FAMILY_BY_NAME

    if not FAMILY_BY_NAME[config.family].deterministic:
        kwargs["seed"] = seed

    model = build(config.family, **kwargs)
    needs_sequences = getattr(model, "needs_sequences", False)

    fit_args = (fold.matrix, fold.sequences) if needs_sequences else (fold.matrix,)

    def fit():
        return model.fit(*fit_args)

    _, train = session.measure_repeated("train", config.family, fit)

    scores, scoring = session.measure_repeated(
        "score", config.family, evaluate.score_catalogue, model, fold.matrix, users
    )
    items, selection = session.measure_repeated(
        "select", config.family, evaluate.select_top, model, scores, users, k
    )

    # Outside every window. Metric computation is a Python loop over users and would be the
    # majority of the reading for a cheap family.
    validation = evaluate.score(dataset, items, users, "validation", k=k)
    test = evaluate.score(dataset, items, users, "test", k=k)

    identity = {
        "config_id": config.config_id,
        "dataset": dataset.name,
        "family": config.family,
        "data_fraction": config.data_fraction,
        "seed": seed,
    }
    hyperparameters = {f"{config.family}.{name}": value for name, value in config.params}

    search = {
        **identity,
        **hyperparameters,
        "val_ndcg_at_10": validation.ndcg,
        "val_recall_at_10": validation.recall,
        "val_exposure_parity": validation.exposure_parity,
        "train_cpu_seconds": train.cpu_seconds_each,
        "train_wall_seconds": train.wall_seconds_each,
        "score_cpu_seconds": scoring.cpu_seconds_each,
        "score_wall_seconds": scoring.wall_seconds_each,
        "select_cpu_seconds": selection.cpu_seconds_each,
        "serve_cpu_seconds_per_request": (
            scoring.cpu_seconds_each + selection.cpu_seconds_each
        )
        / max(len(users), 1),
        "peak_rss_bytes": max(r.peak_rss_bytes or 0 for r in (train, scoring, selection)),
        "model_bytes": model.model_bytes,
        "n_train_interactions": fold.n_interactions,
        "n_eval_users": int(len(users)),
        "train_repeats": train.repeats,
        "score_repeats": scoring.repeats,
        "train_below_quantum": bool(train.meta.get("below_quantum", False)),
        "score_below_quantum": bool(scoring.meta.get("below_quantum", False)),
        # Overwritten by the caller with the contention actually observed across this
        # configuration's window. Present here so the row this function returns is
        # schema-complete on its own rather than only after the loop has finished with it.
        "other_cores": 0.0,
    }
    report = {
        **identity,
        **hyperparameters,
        "test_ndcg_at_10": test.ndcg,
        "test_recall_at_10": test.recall,
    }
    return search, report


def read_partial(directory: Path, name: str):
    """Load a partial file into a frame, reconciling columns across families."""
    import pandas as pd

    path = directory / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def completed(directory: Path) -> set[tuple]:
    """Cells already measured, from the partial files. The basis of resumption."""
    frame = read_partial(directory, SEARCH_PARTIAL)
    if frame.empty:
        return set()
    return set(zip(frame.dataset, frame.config_id, frame.seed, strict=True))


def append(directory: Path, name: str, rows: list[dict]) -> None:
    """Flush rows to a partial file. Append-safe across differing column sets."""
    path = directory / name
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=float) + "\n")


def on_mains() -> bool:
    """Whether the machine is on AC right now.

    Called once per configuration, outside every measured window. The conditions monitor
    already records power state, but it only *reports* at the end -- and a campaign that
    discovers after six hours that the cable came out has lost six hours. This turns that
    into a stop after one configuration.
    """
    from green_rerank.measure.guards import power_source

    return power_source() == "ac"


class PowerLost(RuntimeError):
    """The machine left mains power mid-run."""


class Contended(RuntimeError):
    """Another process consumed CPU while a configuration was being measured."""


#: Non-idle fields of ``psutil.cpu_times`` that are *additive*.
#:
#: ``interrupt`` and ``dpc`` are deliberately absent. On Windows they are already counted
#: inside ``system``, so including them double-counts. Measured directly rather than taken
#: from the documentation: over a 3-second window on this machine,
#: ``idle + user + system = 24.125`` CPU-seconds against the 24.0 expected for 8 CPUs, while
#: adding ``interrupt + dpc`` overshoots to 24.172.
#:
#: The Linux-only fields stay. There ``nice``, ``irq``, ``softirq`` and ``steal`` really are
#: separate columns of ``/proc/stat`` rather than subsets of ``system``, and on Windows they
#: are simply absent and contribute nothing.
BUSY_CPU_FIELDS: tuple[str, ...] = ("user", "system", "nice", "irq", "softirq", "steal")

#: Cores a competing process must add *over this machine's own idle baseline* before its
#: contention is treated as real rather than as desktop noise.
#:
#: The single source of truth for the threshold policy. The threshold itself is never a
#: constant: it is this margin plus a baseline measured on the machine at hand. An earlier
#: version hard-coded 0.25 cores, which is *below* this laptop's idle floor of 0.15-0.45
#: cores, so the guard fired on an empty machine and the campaign made 19 attempts without
#: ever reaching its first flush.
#:
#: 0.75 sits above ordinary desktop bursts and below the ~1.0 core a competing
#: single-threaded measurement job holds continuously.
CONTENTION_MARGIN_CORES: float = 0.75


def cpu_busy_seconds(times) -> float:
    """Busy CPU-seconds from a ``psutil.cpu_times`` tuple, without double counting."""
    return sum(getattr(times, field, 0.0) for field in BUSY_CPU_FIELDS)


def system_and_own_cpu() -> tuple[float, float]:
    """``(system-wide busy CPU-seconds, this process's CPU-seconds)``.

    Kernel counters, read instantaneously. A sampling call such as
    ``psutil.cpu_percent(interval=0.1)`` would have added eight minutes of pure waiting
    across a 5,052-run campaign.
    """
    import psutil

    own = sum(psutil.Process().cpu_times()[:2])
    return cpu_busy_seconds(psutil.cpu_times()), own


def measure_baseline_cores(
    seconds: float = 5.0,
    sampler=system_and_own_cpu,
    clock=time.perf_counter,
    sleeper=time.sleep,
) -> float:
    """Cores other processes use on this machine while nothing of ours is running.

    Taken after preflight has established the machine is quiet, so it measures the floor the
    guard has to sit above: background services, the browser, whatever the operating system
    is doing. Recorded in the manifest, because a threshold derived from a number nobody
    wrote down is indistinguishable from an invented one.
    """
    busy0, own0 = sampler()
    start = clock()
    sleeper(seconds)
    busy1, own1 = sampler()
    elapsed = max(clock() - start, 1e-9)
    return max((busy1 - busy0) - (own1 - own0), 0.0) / elapsed


class ContentionGuard:
    """Detect a *sustained* second CPU-bound process during measurement.

    The existing guards do not cover this. ``preflight`` samples machine load once, at start,
    so a job launched afterwards is invisible; the conditions monitor watches power and
    frequency, never load. That gap is not hypothetical here -- a scheduler in a companion
    project polls every two minutes and launches its own measurement sweep whenever the
    machine looks idle, which is exactly what happens whenever this campaign pauses.

    **Sustained, not instantaneous.** The first version stopped on a single configuration
    exceeding the tolerance, and a test measured 1.48 cores of "contention" on a machine
    running nothing but the test -- a browser and a virus scanner are enough. Desktop noise
    is bursty; a competing measurement job is not. So a violation must persist across
    ``consecutive`` configurations before the run stops.

    **The tolerance is required, not defaulted.** It is derived by the caller from
    :func:`measure_baseline_cores` plus :data:`CONTENTION_MARGIN_CORES`. An earlier version
    carried a default here *and* a different one in the argument parser, so the two
    disagreed and the stricter one silently won -- at 0.25 cores, below the machine's own
    idle floor, which blocked every run.

    **The reading is recorded on every row regardless.** Guarding on a quantity and then
    discarding it would leave the benchmark unable to answer "was this row measured on a
    quiet machine?" -- which is precisely the question that could not be answered about the
    rows this guard was written for.

    The sampler is injectable so the arithmetic can be tested exactly rather than inferred
    from timings on a machine that may stall, following green-rerank's practice for its own
    measurement session.
    """

    def __init__(
        self,
        tolerance_cores: float,
        consecutive: int = 5,
        sampler=system_and_own_cpu,
        clock=time.perf_counter,
    ) -> None:
        self.tolerance = tolerance_cores
        self.consecutive = consecutive
        self._sampler = sampler
        self._clock = clock
        self.violations = 0
        self.reset()

    def reset(self) -> None:
        busy, own = self._sampler()
        self._before = (busy, own, self._clock())

    def check(self) -> float:
        """Cores used by *other* processes since the last call. Resets the window."""
        busy, own = self._sampler()
        wall = self._clock()
        elapsed = max(wall - self._before[2], 1e-9)
        others = (busy - self._before[0]) - (own - self._before[1])
        self._before = (busy, own, wall)
        return max(others, 0.0) / elapsed

    def observe(self, done: int) -> float:
        """Record one configuration's contention, and stop if it has become sustained."""
        cores = self.check()
        if cores > self.tolerance:
            self.violations += 1
            if self.violations >= self.consecutive:
                raise Contended(
                    f"another process has used more than {self.tolerance} cores for "
                    f"{self.violations} consecutive configurations (latest {cores:.2f}) "
                    f"around run {done + 1}. CPU contention is charged to whichever process "
                    "happens to be running. Stop the competing job and re-run to resume."
                )
        else:
            self.violations = 0
        return cores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument(
        "--contention-margin",
        type=float,
        default=CONTENTION_MARGIN_CORES,
        help="cores a competing process must add over this machine's measured idle baseline "
        "before the run stops. The threshold is baseline + margin; there is no absolute "
        "tolerance to set.",
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=5.0,
        help="how long to measure the idle baseline for, after preflight",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="measure at most N cells. For validating the pipeline end to end, never for "
        "producing a benchmark -- a truncated space is not the space.",
    )
    parser.add_argument(
        "--finalise-only",
        action="store_true",
        help="promote existing partial files to the schema's four files and stop",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="finalise even when the manifest records a conditions change. Only after "
        "establishing that the affected rows are not in this directory.",
    )
    args = parser.parse_args()

    if not args.finalise_only:
        pin(args.threads)

    from budget_tune.benchmark import evaluate, schema
    from budget_tune.companion import ensure_all_importable, revisions
    from budget_tune.data import catalogues
    from budget_tune.measure.threads import apply_to_torch, verify
    from budget_tune.space.grids import DATA_FRACTIONS, enumerate_configurations, space_size

    ensure_all_importable()
    from green_rerank.measure.guards import ConditionsMonitor, ExclusiveLock, preflight
    from green_rerank.measure.session import MeasurementSession, clock_quantum

    args.out.mkdir(parents=True, exist_ok=True)

    def finalise() -> int:
        """Promote the partials to the schema's four files, unless the run is untrustworthy.

        A results directory produced from measurements taken in two power states is worse
        than no results directory: every number in it looks normal and the cost column is
        wrong by a factor of nearly three. So this refuses rather than warns.
        """
        manifest_path = args.out / "manifest.json"
        if manifest_path.exists() and not args.force:
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
            conditions = recorded.get("conditions", {})
            if conditions.get("power_source_changed") or conditions.get("throttled"):
                print(
                    "REFUSING to finalise: the conditions monitor recorded "
                    f"{conditions.get('power_sources_seen')} and a frequency drop of "
                    f"{conditions.get('frequency_drop', 0):.1%}. Timings taken in different "
                    "power states are not comparable.\n"
                    "Discard the partials and re-measure, or pass --force if you have "
                    "established the affected rows are not in this directory."
                )
                return 1

        search = read_partial(args.out, SEARCH_PARTIAL)
        report = read_partial(args.out, REPORT_PARTIAL)
        if search.empty:
            print("nothing measured yet")
            return 1

        # A partial benchmark must not be promoted to something that looks like a benchmark.
        # The first version wrote its four schema files from 50 of 5,052 rows after a
        # power-loss stop, and reported success doing it -- a results directory indexed by
        # every later table, holding 1% of the space, with nothing in its filenames to say so.
        expected = expected_rows()
        if len(search) < expected and not args.force:
            print(
                f"REFUSING to finalise: {len(search)} of {expected} planned rows measured. "
                "Re-run to resume; the completed rows are kept. Pass --force only to "
                "deliberately publish a partial benchmark."
            )
            return 1

        schema.write(args.out, search, report)
        print(f"wrote {len(search)} rows to {args.out}")
        return 0

    def expected_rows() -> int:
        return len(names) * sum(
            len(seeds_for(config.family, args.seeds)) for config in enumerate_configurations()
        )

    if args.finalise_only:
        return finalise()

    names = list(catalogues.HEADLINE) + [catalogues.META] if args.all else args.catalogue
    if not names:
        parser.error("pass --catalogue NAME (repeatable) or --all")

    configurations = enumerate_configurations()
    already = completed(args.out)
    thread_report = verify(args.threads)
    thread_report["torch_threads"] = apply_to_torch(args.threads)

    planned = [
        (name, config, seed)
        for seed in args.seeds
        for name in names
        for config in configurations
        for s in [seeds_for(config.family, args.seeds)]
        if seed in s
    ]
    todo = [cell for cell in planned if (cell[0], cell[1].config_id, cell[2]) not in already]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"space: {space_size()}")
    print(f"catalogues: {names}")
    print(f"planned {len(planned)} runs, {len(already)} already done, {len(todo)} to go")
    if args.limit is not None:
        print(f"LIMITED to {len(todo)} cells -- this is a pipeline check, not a benchmark")
    if not todo:
        return finalise()

    with ExclusiveLock(args.out / ".measure.lock") as lock:
        checks = preflight(require_mains=True, max_busy_pct=25.0, lock=lock)
        print(f"preflight: power={checks.power_source} busy={checks.machine_busy_pct}")
        print(f"threads: {thread_report}")

        manifest = {
            "catalogues": names,
            "seeds": args.seeds,
            "k": args.k,
            "space": space_size(),
            "data_fractions": list(DATA_FRACTIONS),
            "threads": thread_report,
            "clock_quantum_seconds": clock_quantum(),
            "preflight": checks.as_meta(),
            "machine": {
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "python": sys.version,
            },
            "companions": revisions(),
            "source": source_fingerprint(),
            "planned_runs": len(planned),
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )

        loaded: dict = {}
        monitor = ConditionsMonitor()
        monitor.start()
        started = time.perf_counter()

        search_buffer: list[dict] = []
        report_buffer: list[dict] = []
        done = 0
        stopped_early = False

        # Measured after preflight has established the machine is quiet, so it is the floor
        # the guard must sit above rather than a number chosen in advance.
        baseline = measure_baseline_cores(args.baseline_seconds)
        tolerance = baseline + args.contention_margin
        print(
            f"contention baseline: {baseline:.3f} cores measured over "
            f"{args.baseline_seconds:.0f}s -> threshold {tolerance:.3f} cores"
        )
        manifest["contention"] = {
            "baseline_cores": baseline,
            "baseline_seconds": args.baseline_seconds,
            "margin_cores": args.contention_margin,
            "tolerance_cores": tolerance,
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )

        contention = ContentionGuard(tolerance_cores=tolerance)

        try:
            for name, config, seed in todo:
                # Checked per configuration, not per flush. On battery this laptop pins to
                # 1.297 GHz and every timing rises ~2.8x while every quality metric stays
                # byte-identical -- so nothing looks wrong except the column the whole
                # project rests on. Stopping here costs one configuration; not stopping cost
                # 33 minutes the first time.
                if not on_mains():
                    raise PowerLost(
                        f"machine left mains power after {done} runs. Rows measured on "
                        "battery are not comparable with rows measured on AC. Reconnect and "
                        "re-run; completed AC rows are kept and will be resumed."
                    )

                contention.reset()

                if name not in loaded:
                    print(f"\nloading {name}")
                    loaded[name] = catalogues.load(name, fractions=DATA_FRACTIONS)
                    session = MeasurementSession(
                        label=name,
                        meta={"catalogue": name, "threads": args.threads},
                        preflight=checks,
                    )
                    # Warm up once per catalogue, before any measured window: torch's first
                    # call would otherwise be charged to whichever configuration ran first.
                    from experiments.calibrate import warm_up

                    warm_up(loaded[name], evaluate, args.k)

                search, report = measure_configuration(
                    loaded[name], config, seed, args.k, session, evaluate
                )

                # Checked again *after* the measurement, not only before. A check on entry
                # alone leaves a hole: if the machine drops to battery during a
                # configuration and returns before the next check, that row is contaminated
                # and kept. Both ends must be on mains for the row to count -- and on this
                # machine the transition is not rare, it happens under sustained load.
                if not on_mains():
                    raise PowerLost(
                        f"machine left mains power during run {done + 1}. That row is "
                        "discarded along with the unflushed buffer; earlier flushed rows "
                        "were verified on mains at both ends."
                    )
                search["other_cores"] = contention.observe(done)

                search_buffer.append(search)
                report_buffer.append(report)
                done += 1

                if len(search_buffer) >= args.flush_every:
                    append(args.out, SEARCH_PARTIAL, search_buffer)
                    append(args.out, REPORT_PARTIAL, report_buffer)
                    search_buffer, report_buffer = [], []
                    elapsed = time.perf_counter() - started
                    rate = done / elapsed
                    print(
                        f"   {done}/{len(todo)}  {elapsed / 60:.1f} min elapsed, "
                        f"~{(len(todo) - done) / rate / 60:.0f} min remaining"
                    )
        except (PowerLost, Contended) as lost:
            # The unflushed buffer is discarded rather than kept. The power source is checked
            # between configurations, so a row in flight when the cable came out cannot be
            # distinguished from one measured before it.
            search_buffer, report_buffer = [], []
            stopped_early = True
            print(f"\nSTOPPED: {lost}")
        finally:
            if search_buffer:
                append(args.out, SEARCH_PARTIAL, search_buffer)
                append(args.out, REPORT_PARTIAL, report_buffer)
            conditions = monitor.stop()
            elapsed = time.perf_counter() - started

            manifest["conditions"] = conditions
            manifest["elapsed_seconds"] = elapsed
            manifest["completed_runs"] = done
            (args.out / "manifest.json").write_text(
                json.dumps(manifest, indent=2, default=str), encoding="utf-8"
            )

    print(f"\nmeasured {done} runs in {elapsed / 60:.1f} min")
    if conditions.get("power_source_changed") or conditions.get("throttled"):
        print("WARNING: conditions changed during the run; timings are not comparable")

    if stopped_early:
        # Non-zero, so a wrapper script cannot mistake an interrupted campaign for a
        # finished one. The first version returned finalise()'s status, which was 0 after a
        # stop at 60 of 5,052 runs.
        print("campaign did not complete; re-run to resume from the measured rows")
        return 2

    return finalise()


if __name__ == "__main__":
    raise SystemExit(main())
