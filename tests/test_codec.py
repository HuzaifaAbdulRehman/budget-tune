"""One-hot codec: round-trips, gated vs feasible, E2 widths."""

from __future__ import annotations

import pytest

from budget_tune.space.codec import (
    BLOCKS,
    N_VARIABLES,
    decode,
    e2_width,
    encode,
    encode_family,
    is_onehot_feasible,
)
from budget_tune.space.grids import binary_width, coarse_grid, enumerate_configurations


class TestLayout:
    def test_width_matches_the_pinned_identifiability_number(self):
        assert N_VARIABLES == binary_width()["variables"] == 44
        assert sum(block.size for block in BLOCKS) == 44

    def test_every_canonical_config_round_trips_gated(self):
        for config in enumerate_configurations():
            bits = encode(config, mode="gated")
            assert bits.shape == (44,)
            assert decode(bits).config_id == config.config_id

    def test_feasible_encoding_is_onehot_on_every_block(self):
        for config in enumerate_configurations()[:20]:
            bits = encode(config, mode="feasible")
            assert is_onehot_feasible(bits)
            assert decode(bits).config_id == config.config_id

    def test_gated_encoding_is_not_onehot_when_other_families_exist(self):
        als = next(c for c in enumerate_configurations() if c.family == "als")
        bits = encode(als, mode="gated")
        assert not is_onehot_feasible(bits)
        assert int(bits.sum()) == 2 + len(als.params)  # family + fraction + own HPs

    def test_argmax_repair_does_not_raise_on_a_zero_block(self):
        bits = encode(next(c for c in enumerate_configurations() if c.family == "popularity"))
        # popularity gated encoding leaves every HP block at zero
        recovered = decode(bits, repair="argmax")
        assert recovered.family == "popularity"

    def test_raise_repair_catches_a_zero_vector(self):
        bits = encode(next(iter(enumerate_configurations())), mode="feasible")
        bits[:] = 0
        with pytest.raises(ValueError, match="not one-hot"):
            decode(bits, repair="raise")


class TestE2:
    def test_als_is_identifiable(self):
        width = e2_width("als")
        assert width["variables"] == 3 + 4 + 3 + 3 + 3  # fraction + four HP blocks
        assert width["surrogate_parameters"] == 1 + 16 + 16 * 15 // 2 == 137

    def test_family_encode_round_trips_through_gated_decode(self):
        config = next(c for c in enumerate_configurations() if c.family == "itemknn")
        bits = encode_family(config)
        assert bits.sum() == 1 + len(config.params)


class TestCoarseGrid:
    def test_is_a_strict_subset(self):
        full = {c.config_id for c in enumerate_configurations()}
        coarse = coarse_grid()
        ids = [c.config_id for c in coarse]
        assert len(ids) == len(set(ids))
        assert set(ids) <= full
        assert len(ids) < len(full)

    def test_includes_endpoints(self):
        families = {c.family for c in coarse_grid()}
        assert families == {"popularity", "itemknn", "als", "multvae", "markov"}
