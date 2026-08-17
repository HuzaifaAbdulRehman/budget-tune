"""The search space: its size, its identifiers, and its agreement with the families.

The counts are asserted against numbers written out by hand rather than against a product
computed the same way the code computes it. That is the point of the test: it fails when
the grid changes, which forces the campaign estimate and the design document to be updated
together with it, instead of the space quietly growing and the reported configuration count
becoming wrong.
"""

from __future__ import annotations

import pytest

from budget_tune.space.grids import (
    DATA_FRACTIONS,
    FAMILIES,
    FAMILY_BY_NAME,
    Configuration,
    Hyperparameter,
    binary_width,
    enumerate_configurations,
    hyperparameter_columns,
    space_size,
)

#: Hand-computed, per family. Any change here must be reflected in docs/design.md and in
#: the campaign estimate.
EXPECTED_BASE = {
    "popularity": 1,           # no hyperparameters
    "itemknn": 4 * 3,          # topk x shrink
    "als": 4 * 3 * 3 * 3,      # factors x epochs x regularisation x alpha
    "multvae": 3 * 2 * 2 * 2,  # latent x hidden x epochs x dropout
    "markov": 2 * 3 * 2,       # order x smoothing x decay
}
EXPECTED_TOTAL = sum(EXPECTED_BASE.values()) * len(DATA_FRACTIONS)


class TestSize:
    def test_data_fractions_are_the_fidelity_ladder(self):
        """Three levels, ascending, ending at full data.

        The rungs of successive halving *are* these values, so a fourth level or a
        reordering would break the identity between a low-fidelity probe and an
        already-measured configuration.
        """
        assert DATA_FRACTIONS == (0.25, 0.5, 1.0)
        assert list(DATA_FRACTIONS) == sorted(DATA_FRACTIONS)
        assert DATA_FRACTIONS[-1] == 1.0

    @pytest.mark.parametrize("family,expected", sorted(EXPECTED_BASE.items()))
    def test_family_base_size(self, family, expected):
        assert FAMILY_BY_NAME[family].base_size == expected

    def test_total_is_471(self):
        assert space_size()["total"] == EXPECTED_TOTAL == 471

    def test_enumeration_matches_the_arithmetic(self):
        configs = enumerate_configurations()
        assert len(configs) == space_size()["total"]
        for family, base in EXPECTED_BASE.items():
            assert sum(c.family == family for c in configs) == base * len(DATA_FRACTIONS)

    def test_binary_width(self):
        """The identifiability number, pinned because an argument rests on it.

        ``d`` binary variables give a second-order surrogate ``1 + d + d(d-1)/2``
        parameters, and a realistic budget affords tens of observations. If the space
        changes, that ratio changes, and the hypothesis about the surrogate being the
        bottleneck has to be restated with the new number.
        """
        width = binary_width()
        assert width["blocks"] == 2 + sum(len(s.hyperparameters) for s in FAMILIES)
        assert width["variables"] == 44
        assert width["surrogate_parameters"] == 1 + 44 + 44 * 43 // 2 == 991


class TestIdentifiers:
    def test_config_ids_are_unique(self):
        configs = enumerate_configurations()
        assert len({c.config_id for c in configs}) == len(configs)

    def test_config_ids_are_stable_across_calls(self):
        first = [c.config_id for c in enumerate_configurations()]
        second = [c.config_id for c in enumerate_configurations()]
        assert first == second

    def test_float_formatting_cannot_split_one_configuration_into_two(self):
        """``0.1`` must render identically however it was arrived at.

        A configuration whose identifier differs between the campaign and the analysis
        does not fail -- it simply never joins, and the row silently disappears from a
        table that still looks complete.
        """
        a = Configuration("markov", (("smoothing", 0.1),), 0.25)
        b = Configuration("markov", (("smoothing", 0.3 - 0.2),), 0.25)
        assert a.config_id == b.config_id

    def test_booleans_render_as_words_not_numbers(self):
        config = Configuration("markov", (("decay", True),), 1.0)
        assert "decay=true" in config.config_id

    def test_enumeration_order_is_fixed(self):
        """The campaign resumes by position, so a reordering would resume wrong cells."""
        configs = enumerate_configurations()
        families_in_order = []
        for config in configs:
            if not families_in_order or families_in_order[-1] != config.family:
                families_in_order.append(config.family)
        assert families_in_order == [spec.name for spec in FAMILIES]


class TestRows:
    def test_row_carries_only_its_own_family_columns(self):
        row = Configuration("markov", (("order", 2), ("smoothing", 0.1)), 0.5).as_row()
        assert row["family"] == "markov"
        assert row["markov.order"] == 2
        assert not any(key.startswith("als.") for key in row)

    def test_hyperparameter_columns_cover_every_family(self):
        columns = hyperparameter_columns()
        assert len(columns) == len(set(columns))
        for spec in FAMILIES:
            for hyperparameter in spec.hyperparameters:
                assert f"{spec.name}.{hyperparameter.name}" in columns


class TestGridDeclarations:
    def test_duplicate_values_are_refused(self):
        with pytest.raises(ValueError, match="duplicate"):
            Hyperparameter("topk", (10, 10))

    def test_empty_grids_are_refused(self):
        with pytest.raises(ValueError, match="no values"):
            Hyperparameter("topk", ())

    def test_every_family_states_why_it_is_included(self):
        for spec in FAMILIES:
            assert spec.note.strip()


@pytest.mark.companion
class TestAgreementWithTheFamilies:
    """Every configuration must actually construct the model it names.

    This is the test that catches a hyperparameter renamed on one side only. Without it,
    a space declaring ``regularisation`` against a constructor taking ``reg`` would either
    raise deep inside the campaign, hours in, or -- if the constructor accepted keyword
    arguments loosely -- silently train the default model for every cell of that axis.
    """

    @pytest.mark.parametrize("spec", FAMILIES, ids=lambda s: s.name)
    def test_first_configuration_of_each_family_constructs(self, spec, strict):
        from tests.conftest import require_companion

        require_companion("green_rerank", strict)
        if spec.name == "multvae":
            pytest.importorskip("torch")

        from budget_tune.families import build

        config = next(c for c in enumerate_configurations() if c.family == spec.name)
        model = build(config.family, **config.kwargs)
        assert model.name == spec.name

    @pytest.mark.parametrize("spec", FAMILIES, ids=lambda s: s.name)
    def test_declared_hyperparameters_reach_the_model(self, spec, strict):
        from tests.conftest import require_companion

        require_companion("green_rerank", strict)
        if spec.name == "multvae":
            pytest.importorskip("torch")

        from budget_tune.families import build

        # The last configuration of a family differs from the first on every axis, so a
        # parameter that was accepted but ignored shows up as an attribute still holding
        # the constructor default.
        config = [c for c in enumerate_configurations() if c.family == spec.name][-1]
        model = build(config.family, **config.kwargs)
        for name, value in config.params:
            assert getattr(model, name) == value
