"""The campaign runner: its planning, its resumption, and its schema conformance.

The campaign is a multi-hour unattended run whose output every later table derives from. The
failures worth testing are the ones that would survive it: a plan that silently omits cells,
a resume that re-measures work already done or skips work it has not, and a row that does not
conform to the schema -- discovered only at the end, after the machine conditions are gone.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from budget_tune.benchmark import schema
from experiments.build_benchmark import append, completed, seeds_for, source_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestSeedPlan:
    def test_deterministic_families_are_measured_once(self):
        for family in ("popularity", "itemknn", "markov"):
            assert seeds_for(family, [0, 1, 2]) == [0]

    def test_stochastic_families_get_every_seed(self):
        for family in ("als", "multvae"):
            assert seeds_for(family, [0, 1, 2]) == [0, 1, 2]

    def test_the_planned_run_count_matches_the_design(self):
        """1,263 runs per catalogue: 396 stochastic configurations x 3 seeds, 75 x 1."""
        from budget_tune.space.grids import enumerate_configurations

        total = sum(
            len(seeds_for(config.family, [0, 1, 2]))
            for config in enumerate_configurations()
        )
        assert total == 396 * 3 + 75 == 1263


class TestResumption:
    def test_nothing_completed_on_a_fresh_directory(self, tmp_path):
        assert completed(tmp_path) == set()

    def test_completed_cells_are_recognised(self, tmp_path):
        from experiments.build_benchmark import SEARCH_PARTIAL

        rows = [
            {"dataset": "ml100k", "config_id": "als|a=1|frac=1.0", "seed": 0, "x": 1},
            {"dataset": "ml100k", "config_id": "als|a=1|frac=1.0", "seed": 1, "x": 2},
        ]
        append(tmp_path, SEARCH_PARTIAL, rows)

        done = completed(tmp_path)
        assert ("ml100k", "als|a=1|frac=1.0", 0) in done
        assert ("ml100k", "als|a=1|frac=1.0", 1) in done
        assert ("ml100k", "als|a=1|frac=1.0", 2) not in done
        assert ("software", "als|a=1|frac=1.0", 0) not in done

    def test_appending_survives_families_with_different_columns(self, tmp_path):
        """The bug that destroyed 33 minutes of completed measurement.

        Families do not share hyperparameter columns. Appending CSV with the header written
        once meant a later flush containing ALS rows wrote four more fields than the header
        declared, and the file became unparseable *after* every measurement had succeeded.
        A row-oriented format has no header to disagree with.
        """
        from experiments.build_benchmark import read_partial

        append(tmp_path, "p.jsonl", [{"config_id": "a", "popularity_only": 1}])
        append(tmp_path, "p.jsonl", [{"config_id": "b", "als.factors": 64, "als.epochs": 5}])

        frame = read_partial(tmp_path, "p.jsonl")
        assert len(frame) == 2
        assert set(frame.columns) >= {"config_id", "popularity_only", "als.factors"}
        assert list(frame.config_id) == ["a", "b"]

    def test_a_partial_that_does_not_exist_reads_as_empty(self, tmp_path):
        from experiments.build_benchmark import read_partial

        assert read_partial(tmp_path, "absent.jsonl").empty

    def test_a_resumed_run_does_not_duplicate_a_cell(self, tmp_path):
        """Duplicate (config, dataset, seed) rows would make the median a lie."""
        from experiments.build_benchmark import SEARCH_PARTIAL

        rows = [{"dataset": "d", "config_id": "c", "seed": 0}]
        append(tmp_path, SEARCH_PARTIAL, rows)
        done = completed(tmp_path)

        planned = [("d", "c", 0), ("d", "c", 1)]
        todo = [cell for cell in planned if cell not in done]
        assert todo == [("d", "c", 1)]


class TestPowerGuard:
    """Fail fast on a power-state change rather than reporting it after the fact.

    The first campaign run measured 1,263 configurations across a transition from AC to
    battery. Every quality metric was byte-identical to what it would have been; the CPU
    frequency fell from 1,696 to 1,297 MHz and the cost column -- the axis every conclusion
    rests on -- was wrong for an unknown subset of the rows.
    """

    def test_mains_is_detected(self, monkeypatch):
        import experiments.build_benchmark as runner
        from budget_tune.companion import ensure_importable

        ensure_importable("green_rerank")
        import green_rerank.measure.guards as guards

        monkeypatch.setattr(guards, "power_source", lambda: "ac")
        assert runner.on_mains()

        monkeypatch.setattr(guards, "power_source", lambda: "battery")
        assert not runner.on_mains()

    def test_unknown_power_state_is_not_treated_as_mains(self, monkeypatch):
        """A check that cannot run must not be recorded as a check that passed."""
        import experiments.build_benchmark as runner
        from budget_tune.companion import ensure_importable

        ensure_importable("green_rerank")
        import green_rerank.measure.guards as guards

        monkeypatch.setattr(guards, "power_source", lambda: "unknown")
        assert not runner.on_mains()


class TestPartialBenchmarkIsRefused:
    """A results directory must not be produced from an interrupted campaign.

    The first version wrote its four schema files from 50 of 5,052 rows after a power-loss
    stop, and exited 0 doing it. Nothing in the filenames would have said the benchmark held
    1% of the space, and every later table indexes that directory.
    """

    def test_expected_row_count_scales_with_catalogues(self):
        from budget_tune.space.grids import enumerate_configurations
        from experiments.build_benchmark import seeds_for

        per_catalogue = sum(
            len(seeds_for(config.family, [0, 1, 2]))
            for config in enumerate_configurations()
        )
        assert per_catalogue == 1263
        assert 4 * per_catalogue == 5052

    def test_an_interrupted_run_reports_a_distinct_exit_code(self):
        """0 finished, 1 refused, 2 interrupted -- a wrapper script must be able to tell."""
        source = (
            REPO_ROOT / "experiments" / "build_benchmark.py"
        ).read_text(encoding="utf-8")
        assert "return 2" in source
        assert "stopped_early" in source


class TestContentionGuard:
    """The guard added after a companion project's scheduler ran a measurement sweep
    concurrently with the campaign, undetected.

    Two ways this can be useless, and both are silent. If it counted *our own* work it would
    fire on every configuration and nothing would ever be measured. If it counted nothing it
    would pass every configuration and manufacture confidence in contaminated numbers. Both
    are tested.
    """

    def _guard(self, script, **kwargs):
        """A guard driven by a scripted sequence of ``(system_busy, own_cpu, wall)``.

        Injected rather than measured. The first version of this test burned CPU in-process
        and asserted the guard read near zero; it read 1.48 cores, because a browser and a
        virus scanner are enough to swamp the signal. That failure was real and useful --
        it is why the guard now requires sustained violations -- but it could not
        distinguish a bug in the subtraction from noise on the machine, which is precisely
        what a test has to do.
        """
        from experiments.build_benchmark import ContentionGuard

        steps = iter(script)
        state = {"now": 0.0}

        def sampler():
            busy, own, wall = next(steps)
            state["now"] = wall
            return busy, own

        kwargs.setdefault("tolerance_cores", 0.5)
        return ContentionGuard(sampler=sampler, clock=lambda: state["now"], **kwargs)

    def test_our_own_work_is_not_contention(self):
        """The critical case: the guard subtracts this process's CPU from the system's.

        One wall-second in which the system burned 1.0 CPU-seconds and *all* of it was ours.
        """
        guard = self._guard([(100.0, 50.0, 0.0), (101.0, 51.0, 1.0)])
        assert guard.check() == pytest.approx(0.0)

    def test_a_competing_process_is_detected(self):
        """One wall-second, 1.5 system CPU-seconds, 1.0 of them ours -> 0.5 cores of other."""
        guard = self._guard([(100.0, 50.0, 0.0), (101.5, 51.0, 1.0)])
        assert guard.check() == pytest.approx(0.5)

    def test_a_full_competing_core_is_detected(self):
        guard = self._guard([(0.0, 0.0, 0.0), (2.0, 1.0, 1.0)])
        assert guard.check() == pytest.approx(1.0)

    def test_a_single_burst_does_not_stop_the_run(self):
        """Desktop noise is bursty; a competing measurement job is not."""
        from experiments.build_benchmark import Contended

        guard = self._guard(
            [(0.0, 0.0, 0.0), (3.0, 0.0, 1.0), (3.0, 0.0, 2.0), (3.0, 0.0, 3.0)],
            tolerance_cores=0.5, consecutive=3,
        )
        assert guard.observe(0) == pytest.approx(3.0)   # one violation
        assert guard.observe(1) == pytest.approx(0.0)   # quiet again, counter resets
        try:
            guard.observe(2)
        except Contended:
            pytest.fail("a single burst stopped the run")
        assert guard.violations == 0

    def test_sustained_contention_stops_the_run(self):
        from experiments.build_benchmark import Contended

        guard = self._guard(
            [(0.0, 0.0, 0.0), (2.0, 0.0, 1.0), (4.0, 0.0, 2.0), (6.0, 0.0, 3.0)],
            tolerance_cores=0.5, consecutive=3,
        )
        guard.observe(0)
        guard.observe(1)
        with pytest.raises(Contended, match="consecutive"):
            guard.observe(2)

    def test_check_resets_its_window(self):
        """Otherwise one busy period would condemn every later configuration."""
        guard = self._guard([(0.0, 0.0, 0.0), (2.0, 0.0, 1.0), (2.0, 0.0, 2.0)])
        assert guard.check() == pytest.approx(2.0)
        assert guard.check() == pytest.approx(0.0)

    def test_negative_readings_are_clamped(self):
        """Counter granularity can make 'others' come out slightly negative."""
        guard = self._guard([(100.0, 50.0, 0.0), (100.0, 50.5, 1.0)])
        assert guard.check() == 0.0

    def test_the_reading_is_recorded_on_every_row(self):
        """Guarding on a quantity while discarding it leaves the row unauditable."""
        from budget_tune.benchmark.schema import MEASUREMENT_COLUMNS

        assert "other_cores" in MEASUREMENT_COLUMNS

    def test_the_tolerance_must_be_supplied(self):
        """No default here, because a second default is what caused the failure.

        The guard once defaulted to 0.5 while the argument parser passed 0.25; the two
        disagreed, the stricter won silently, and it sat below the machine's idle floor.
        """
        import inspect

        from experiments.build_benchmark import ContentionGuard

        parameter = inspect.signature(ContentionGuard).parameters["tolerance_cores"]
        assert parameter.default is inspect.Parameter.empty


class TestBusyCpuAccounting:
    """Regression: ``interrupt`` and ``dpc`` are subsets of ``system`` on Windows."""

    def test_interrupt_and_dpc_are_excluded_from_the_busy_fields(self):
        from experiments.build_benchmark import BUSY_CPU_FIELDS

        assert "interrupt" not in BUSY_CPU_FIELDS
        assert "dpc" not in BUSY_CPU_FIELDS

    def test_linux_only_fields_are_kept(self):
        """There ``nice``/``irq``/``softirq``/``steal`` are separate ``/proc/stat`` columns.

        Dropping them to fix a Windows double-count would have under-reported busy time on
        Linux, which is the opposite error and just as silent.
        """
        from experiments.build_benchmark import BUSY_CPU_FIELDS

        assert {"nice", "irq", "softirq", "steal"} <= set(BUSY_CPU_FIELDS)

    def test_a_windows_style_tuple_sums_to_user_plus_system_only(self):
        from collections import namedtuple

        from experiments.build_benchmark import cpu_busy_seconds

        Times = namedtuple("Times", "user system idle interrupt dpc")
        times = Times(user=10.0, system=4.0, idle=900.0, interrupt=1.0, dpc=2.0)
        assert cpu_busy_seconds(times) == pytest.approx(14.0)

    def test_a_linux_style_tuple_sums_every_additive_field(self):
        from collections import namedtuple

        from experiments.build_benchmark import cpu_busy_seconds

        Times = namedtuple("Times", "user nice system idle iowait irq softirq steal")
        times = Times(
            user=10.0, nice=1.0, system=4.0, idle=900.0,
            iowait=5.0, irq=0.5, softirq=0.25, steal=0.125,
        )
        # iowait is excluded: the CPU is idle waiting on I/O, not busy.
        assert cpu_busy_seconds(times) == pytest.approx(15.875)

    @pytest.mark.timing
    def test_busy_plus_idle_accounts_for_every_cpu_second(self):
        """The invariant the double-count violated, checked against the real kernel.

        Over a window of ``w`` seconds on ``n`` CPUs the counters must advance by ``n*w``
        in total. Adding ``interrupt`` and ``dpc`` overshot it.
        """
        import time

        import psutil

        from experiments.build_benchmark import cpu_busy_seconds

        before, start = psutil.cpu_times(), time.perf_counter()
        time.sleep(2.0)
        after, elapsed = psutil.cpu_times(), time.perf_counter() - start

        busy = cpu_busy_seconds(after) - cpu_busy_seconds(before)
        idle = (after.idle - before.idle) + (
            getattr(after, "iowait", 0.0) - getattr(before, "iowait", 0.0)
        )
        expected = psutil.cpu_count() * elapsed
        assert (busy + idle) == pytest.approx(expected, rel=0.05)


class TestBaselineDerivedThreshold:
    """The threshold must come from a measurement of this machine, not from a constant."""

    def _baseline(self, busy_rate: float, own_rate: float, seconds: float = 5.0) -> float:
        from experiments.build_benchmark import measure_baseline_cores

        state = {"t": 0.0}
        readings = iter([(0.0, 0.0), (busy_rate * seconds, own_rate * seconds)])

        return measure_baseline_cores(
            seconds=seconds,
            sampler=lambda: next(readings),
            clock=lambda: state["t"],
            sleeper=lambda s: state.__setitem__("t", state["t"] + s),
        )

    def test_baseline_is_other_process_cores_only(self):
        """1.4 cores of system activity, 1.0 of it ours -> a baseline of 0.4."""
        assert self._baseline(busy_rate=1.4, own_rate=1.0) == pytest.approx(0.4)

    def test_an_idle_machine_reports_a_small_baseline(self):
        assert self._baseline(busy_rate=0.3, own_rate=0.0) == pytest.approx(0.3)

    def test_baseline_cannot_go_negative(self):
        assert self._baseline(busy_rate=0.5, own_rate=0.9) == 0.0

    def test_threshold_is_baseline_plus_the_margin(self):
        from experiments.build_benchmark import CONTENTION_MARGIN_CORES

        baseline = self._baseline(busy_rate=0.45, own_rate=0.0)
        assert baseline + CONTENTION_MARGIN_CORES == pytest.approx(1.20)

    def test_the_margin_clears_this_machines_idle_floor(self):
        """The bug in one assertion.

        The measured idle floor here is 0.15-0.45 cores. The old fixed tolerance of 0.25 sat
        inside that range, so the guard fired on an empty machine. Any policy whose threshold
        can fall below the floor is the same bug again.
        """
        from experiments.build_benchmark import CONTENTION_MARGIN_CORES

        observed_idle_floor = 0.45
        assert observed_idle_floor < CONTENTION_MARGIN_CORES

    def test_the_margin_stays_below_a_competing_single_threaded_job(self):
        """A competing measurement job holds ~1.0 core; the guard must still catch it."""
        from experiments.build_benchmark import CONTENTION_MARGIN_CORES

        assert CONTENTION_MARGIN_CORES < 1.0

    def test_there_is_one_source_of_truth_for_the_policy(self):
        """The parser must reference the constant rather than repeat a number."""
        source = (REPO_ROOT / "experiments" / "build_benchmark.py").read_text(encoding="utf-8")
        assert "default=CONTENTION_MARGIN_CORES" in source
        assert "contention-tolerance" not in source


class TestProvenance:
    def test_the_fingerprint_is_stable_and_short(self):
        first = source_fingerprint()
        assert first == source_fingerprint()
        assert len(first["source_sha256"]) == 16
        assert first["files"] > 10

    def test_the_fingerprint_changes_with_the_source(self, tmp_path, monkeypatch):
        import experiments.build_benchmark as runner

        package = tmp_path / "budget_tune"
        package.mkdir()
        (package / "a.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / "experiments").mkdir()

        monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
        before = runner.source_fingerprint()
        (package / "a.py").write_text("x = 2", encoding="utf-8")
        assert runner.source_fingerprint() != before


@pytest.fixture(scope="module")
def measured():
    """One row per family, measured end to end on the synthetic catalogue."""
    from budget_tune.benchmark import evaluate
    from budget_tune.companion import ensure_importable
    from budget_tune.data.catalogues import synthetic
    from budget_tune.space.grids import enumerate_configurations
    from experiments.build_benchmark import measure_configuration

    ensure_importable("green_rerank")
    from green_rerank.measure.session import MeasurementSession

    dataset = synthetic(fractions=(0.25, 0.5, 1.0))
    session = MeasurementSession(label="synthetic")

    rows = []
    for family in ("popularity", "itemknn", "markov", "als"):
        config = next(c for c in enumerate_configurations() if c.family == family)
        rows.append(measure_configuration(dataset, config, 0, 10, session, evaluate))
    return rows


@pytest.mark.companion
class TestSchemaConformance:
    """A row must satisfy the schema, discovered now rather than after a six-hour run."""

    def test_rows_validate_against_the_schema(self, measured):
        search = pd.DataFrame([row[0] for row in measured])
        report = pd.DataFrame([row[1] for row in measured])
        schema.validate_runs(search, "search")
        schema.validate_runs(report, "report")

    def test_the_search_row_carries_no_test_metric(self, measured):
        for search, _ in measured:
            for column in schema.REPORT_QUALITY_COLUMNS:
                assert column not in search

    def test_both_splits_are_graded_from_the_same_recommendations(self, measured):
        """The reporting row costs no extra compute and cannot drift from the search row."""
        for search, report in measured:
            assert search["config_id"] == report["config_id"]
            assert search["seed"] == report["seed"]

    def test_cheap_stages_are_repeated_until_they_mean_something(self, measured):
        """Popularity's fit is four orders of magnitude below the clock quantum."""
        popularity = next(s for s, _ in measured if s["family"] == "popularity")
        assert popularity["train_repeats"] > 1
        assert popularity["train_cpu_seconds"] >= 0

    def test_per_request_cost_is_divided_by_the_user_count(self, measured):
        for search, _ in measured:
            expected = (
                search["score_cpu_seconds"] + search["select_cpu_seconds"]
            ) / search["n_eval_users"]
            assert search["serve_cpu_seconds_per_request"] == pytest.approx(expected)

    def test_the_written_directory_round_trips(self, measured, tmp_path):
        search = pd.DataFrame([row[0] for row in measured])
        report = pd.DataFrame([row[1] for row in measured])
        schema.write(tmp_path, search, report)

        view = schema.load_search(tmp_path, "synthetic")
        assert len(view) == len(measured)
        with pytest.raises(schema.LeakageError):
            view.column("test_ndcg_at_10")
