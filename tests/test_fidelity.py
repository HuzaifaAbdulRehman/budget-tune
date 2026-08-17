"""The epoch ladder and the statistics that judge it.

These matter because the fidelity decision rests on them. A simulated-halving routine with an
off-by-one in its promotion count, or a top-k overlap that silently compared the wrong ends of
a ranking, would produce a number that looks like evidence and is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from budget_tune.fidelity import (
    LADDERS,
    NON_ITERATIVE,
    Ladder,
    rank_agreement,
    simulate_halving,
    top_k_overlap,
)


class TestDeclaredLadder:
    """The schedule is frozen. These tests are what makes 'frozen' mean something."""

    def test_rungs_come_from_the_existing_epoch_grid(self):
        """No rung may introduce an epoch value the campaign would not otherwise measure.

        A ladder needing `epochs=45` would mean inventing measurements to make a baseline
        tidier, and those measurements would exist only to serve the baseline.
        """
        from budget_tune.space.grids import FAMILY_BY_NAME

        for family, ladder in LADDERS.items():
            grid = next(
                h.values
                for h in FAMILY_BY_NAME[family].hyperparameters
                if h.name == "epochs"
            )
            assert set(ladder.rungs) <= set(grid), f"{family}: {ladder.rungs} not in {grid}"

    def test_the_top_rung_is_the_full_budget(self):
        from budget_tune.space.grids import FAMILY_BY_NAME

        for family, ladder in LADDERS.items():
            grid = next(
                h.values
                for h in FAMILY_BY_NAME[family].hyperparameters
                if h.name == "epochs"
            )
            assert ladder.rungs[-1] == max(grid)

    def test_every_iterative_family_has_a_ladder_and_no_other_does(self):
        """A family with no epoch parameter must not be given a fake fidelity."""
        from budget_tune.space.grids import FAMILIES

        iterative = {
            spec.name
            for spec in FAMILIES
            if any(h.name == "epochs" for h in spec.hyperparameters)
        }
        assert set(LADDERS) == iterative
        assert set(NON_ITERATIVE) == {s.name for s in FAMILIES} - iterative

    def test_the_declared_schedule_is_exactly_what_the_design_states(self):
        assert LADDERS["als"].rungs == (5, 15, 30)
        assert LADDERS["als"].keep == pytest.approx((1 / 3, 1 / 2))
        assert LADDERS["multvae"].rungs == (10, 20)
        assert LADDERS["multvae"].keep == pytest.approx((0.5,))

    def test_survivor_counts(self):
        assert LADDERS["als"].survivors(108) == [108, 36, 18]
        assert LADDERS["multvae"].survivors(36) == [36, 18]

    def test_at_least_one_configuration_always_survives(self):
        """Otherwise regret would be measured against something never evaluated."""
        assert LADDERS["als"].survivors(2) == [2, 1, 1]
        assert LADDERS["als"].survivors(1) == [1, 1, 1]

    def test_malformed_ladders_are_refused(self):
        with pytest.raises(ValueError, match="ascend"):
            Ladder("x", rungs=(30, 5), keep=(0.5,))
        with pytest.raises(ValueError, match="keep fractions"):
            Ladder("x", rungs=(5, 15, 30), keep=(0.5,))
        with pytest.raises(ValueError, match="in \\(0, 1\\)"):
            Ladder("x", rungs=(5, 15), keep=(1.5,))


class TestStatistics:
    def test_perfect_agreement(self):
        scores = np.array([0.1, 0.5, 0.3, 0.9])
        agreement = rank_agreement(scores, scores)
        assert agreement["spearman"] == pytest.approx(1.0)
        assert agreement["kendall"] == pytest.approx(1.0)

    def test_reversed_agreement(self):
        scores = np.array([0.1, 0.5, 0.3, 0.9])
        assert rank_agreement(scores, -scores)["spearman"] == pytest.approx(-1.0)

    def test_top_k_overlap_counts_the_right_end(self):
        """Overlap must be computed on the *best* k, not the first k or the worst k."""
        high = np.array([0.9, 0.8, 0.1, 0.2])
        same = np.array([0.7, 0.6, 0.0, 0.1])
        assert top_k_overlap(same, high, 2) == 1.0

        swapped = np.array([0.1, 0.2, 0.9, 0.8])
        assert top_k_overlap(swapped, high, 2) == 0.0

    def test_top_k_overlap_is_partial_when_the_ranking_is(self):
        high = np.array([0.9, 0.8, 0.7, 0.1])
        low = np.array([0.9, 0.1, 0.8, 0.2])
        assert top_k_overlap(low, high, 2) == 0.5

    def test_top_k_beyond_the_population_is_refused(self):
        with pytest.raises(ValueError, match="exceeds"):
            top_k_overlap(np.zeros(3), np.zeros(3), 5)


class TestSimulatedHalving:
    def _ladder(self) -> Ladder:
        return Ladder("test", rungs=(5, 15, 30), keep=(1 / 3, 1 / 2))

    def test_a_perfect_fidelity_finds_the_best_and_regrets_nothing(self):
        final = np.linspace(0, 1, 12)
        scores = {5: final.copy(), 15: final.copy(), 30: final}

        result = simulate_halving(scores, self._ladder())
        assert result["found_true_best"]
        assert result["regret"] == pytest.approx(0.0)
        assert result["discarded_then_strong"] == 0
        assert result["survivors_per_rung"] == [12, 4, 2]

    def test_discarded_then_strong_cannot_punish_a_perfect_schedule(self):
        """Regression: the comparison set must not exceed the number of promotion slots.

        Against a fixed top ten, a perfect fidelity promoting only four survivors would be
        charged six discarded-then-strong configurations -- a metric measuring the keep
        fraction rather than the fidelity, and one that would have read as evidence against
        epoch fidelity when the fidelity was flawless.
        """
        final = np.linspace(0, 1, 12)
        result = simulate_halving({5: final, 15: final, 30: final}, self._ladder())
        assert result["discarded_then_strong_k"] == 4
        assert result["discarded_then_strong"] == 0

        # With enough configurations the cap does not bind and the full top ten is used.
        big = np.linspace(0, 1, 120)
        result = simulate_halving({5: big, 15: big, 30: big}, self._ladder())
        assert result["discarded_then_strong_k"] == 10

    def test_an_inverted_fidelity_discards_the_winner(self):
        """The failure mode the whole experiment exists to detect."""
        final = np.linspace(0, 1, 12)
        scores = {5: -final, 15: -final, 30: final}

        result = simulate_halving(scores, self._ladder())
        assert not result["found_true_best"]
        assert result["regret"] > 0
        assert result["regret_normalised"] == pytest.approx(1.0, abs=0.35)
        assert result["discarded_then_strong"] > 0

    def test_regret_is_measured_against_the_full_budget_ranking(self):
        """Not against the rung the survivor was selected on.

        Selecting on rung 0 and then reporting rung 0's score would make every fidelity look
        perfect, since the survivor is by construction the best thing rung 0 saw.
        """
        final = np.array([0.0, 0.1, 0.9, 0.2, 0.3, 0.4])
        # Rung 0 ranks the eventual winner last, so it is cut immediately.
        low = np.array([0.9, 0.8, 0.0, 0.7, 0.6, 0.5])
        ladder = Ladder("test", rungs=(5, 30), keep=(0.5,))

        result = simulate_halving({5: low, 30: final}, ladder)
        assert result["true_best_score"] == pytest.approx(0.9)
        assert result["survivor_best_score"] < 0.9
        assert result["regret"] == pytest.approx(0.9 - result["survivor_best_score"])

    def test_normalised_regret_uses_the_observed_spread(self):
        final = np.array([0.20, 0.21, 0.22, 0.30])
        low = np.array([0.30, 0.22, 0.21, 0.20])
        ladder = Ladder("test", rungs=(5, 30), keep=(0.5,))

        result = simulate_halving({5: low, 30: final}, ladder)
        assert result["score_spread"] == pytest.approx(0.10)
        assert result["regret_normalised"] == pytest.approx(
            result["regret"] / 0.10, rel=1e-9
        )

    def test_a_flat_objective_does_not_divide_by_zero(self):
        flat = np.full(8, 0.25)
        result = simulate_halving({5: flat, 15: flat, 30: flat}, self._ladder())
        assert result["regret_normalised"] == 0.0


class TestAnalysisOnMixedFamilies:
    """Regression: a frame holding several families must not silently analyse to nothing.

    Pandas gives such a frame the union of both families' ``param.*`` columns, so ALS rows
    carry an all-NaN ``param.latent``. ``pivot_table`` drops every row with a NaN anywhere in
    its index, which emptied the table and raised a ``KeyError`` on the first rung -- after a
    thirteen-minute measurement run had already completed. The measurement was fine; only the
    arithmetic was wrong, and this test is what should have caught it beforehand.
    """

    def _frame(self):
        import itertools

        import pandas as pd

        rows = []
        for factors, alpha, fraction, epochs, seed in itertools.product(
            (16, 64), (1.0, 40.0), (0.25, 1.0), (5, 15, 30), (0, 1)
        ):
            rows.append(
                {
                    "family": "als", "epochs": epochs, "data_fraction": fraction,
                    "seed": seed, "param.factors": factors, "param.alpha": alpha,
                    "train_cpu_seconds": 0.1 * epochs,
                    # Quality rises with factors and epochs, so the fidelity is informative
                    # and the statistics have a known sign.
                    "val_ndcg_at_10": 0.001 * factors + 0.0001 * epochs + 0.01 * seed,
                }
            )
        for latent, fraction, epochs, seed in itertools.product(
            (32, 128), (0.25, 1.0), (10, 20), (0, 1)
        ):
            rows.append(
                {
                    "family": "multvae", "epochs": epochs, "data_fraction": fraction,
                    "seed": seed, "param.latent": latent,
                    "train_cpu_seconds": 0.2 * epochs,
                    "val_ndcg_at_10": 0.002 * latent + 0.0001 * epochs,
                }
            )
        return pd.DataFrame(rows)

    def test_both_families_are_analysed(self):
        from experiments.validate_fidelity import analyse

        report = analyse(self._frame())
        assert set(report) == {"als", "multvae"}

    def test_configuration_counts_survive_the_pivot(self):
        """The bug's signature was an empty table, so the count is what pins it."""
        from experiments.validate_fidelity import analyse

        report = analyse(self._frame())
        assert report["als"]["n_configurations"] == 2 * 2 * 2  # factors x alpha x fraction
        assert report["multvae"]["n_configurations"] == 2 * 2  # latent x fraction

    def test_an_informative_fidelity_reports_as_such(self):
        from experiments.validate_fidelity import analyse

        report = analyse(self._frame())
        assert report["als"]["low_vs_high"]["spearman"] == pytest.approx(1.0)
        assert report["als"]["top_k_overlap"]["5"] == 1.0
        assert report["als"]["halving"]["regret"] == pytest.approx(0.0)

    def test_the_seed_ceiling_is_computed(self):
        from experiments.validate_fidelity import analyse

        report = analyse(self._frame())
        assert report["als"]["seed_ceiling_at_max_rung"] is not None
        assert report["als"]["cost_ratio_low_to_high"] < 1.0


@pytest.mark.companion
class TestExperimentWiring:
    def test_the_parameter_grid_excludes_epochs_and_covers_everything_else(self):
        from experiments.validate_fidelity import parameter_grid

        als = parameter_grid("als")
        assert len(als) == 4 * 3 * 3  # factors x regularisation x alpha
        assert all("epochs" not in params for params in als)
        assert len({tuple(sorted(p.items())) for p in als}) == len(als)

        multvae = parameter_grid("multvae")
        assert len(multvae) == 3 * 2 * 2  # latent x hidden x dropout

    def test_the_experiment_is_pinned_to_the_meta_catalogue(self):
        """A headline catalogue must not be reachable by passing an argument."""
        from budget_tune.data import catalogues
        from experiments.validate_fidelity import META_CATALOGUE

        assert catalogues.role(META_CATALOGUE) == "meta"
        assert META_CATALOGUE not in catalogues.HEADLINE

    def test_the_experiment_never_names_the_reporting_split(self):
        from pathlib import Path

        from budget_tune.benchmark.schema import REPORT_QUALITY_COLUMNS

        source = (
            Path(__file__).resolve().parents[1] / "experiments" / "validate_fidelity.py"
        ).read_text(encoding="utf-8")
        for column in REPORT_QUALITY_COLUMNS:
            assert column not in source
        assert '"validation"' in source
