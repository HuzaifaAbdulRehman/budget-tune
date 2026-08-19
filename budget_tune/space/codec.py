"""One-hot encoding of a canonical configuration.

Two encodings of the same 44-bit layout:

* **gated** — the family's block and the data-fraction block are one-hot; the active
  family's hyperparameter blocks are one-hot; every inactive family's bits are zero.
  This is the representation a quadratic surrogate actually sees when fitting observations.
  It is not one-hot-feasible on inactive blocks.
* **feasible** — every block is one-hot. Inactive hyperparameter blocks are pinned to
  their first value. This is the representation a penalty-encoded QUBO can legally return.

They are not the same optimisation problem (design §5.4). The codec converts; it does not
compare their energies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from budget_tune.space.grids import (
    FAMILIES,
    FAMILY_BY_NAME,
    Configuration,
    block_layout,
)


@dataclass(frozen=True)
class Block:
    """One one-hot block of the flat encoding."""

    name: str
    values: tuple
    start: int
    size: int

    @property
    def stop(self) -> int:
        return self.start + self.size


def _blocks() -> tuple[Block, ...]:
    blocks = []
    cursor = 0
    for name, values in block_layout():
        block = Block(name, values, cursor, len(values))
        blocks.append(block)
        cursor += block.size
    return tuple(blocks)


BLOCKS: tuple[Block, ...] = _blocks()
N_VARIABLES: int = BLOCKS[-1].stop if BLOCKS else 0
BLOCK_BY_NAME: dict[str, Block] = {block.name: block for block in BLOCKS}


def _index(block: Block, value) -> int:
    try:
        return block.values.index(value)
    except ValueError:
        raise KeyError(f"{value!r} is not a value of {block.name}") from None


def encode(config: Configuration, mode: str = "gated") -> np.ndarray:
    """Pack ``config`` into a ``{0,1}^d`` vector.

    Args:
        config: a canonical configuration.
        mode: ``"gated"`` or ``"feasible"``.
    """
    if mode not in {"gated", "feasible"}:
        raise ValueError(f"mode must be 'gated' or 'feasible'; got {mode!r}")

    bits = np.zeros(N_VARIABLES, dtype=np.int8)
    bits[BLOCK_BY_NAME["family"].start + _index(BLOCK_BY_NAME["family"], config.family)] = 1
    bits[
        BLOCK_BY_NAME["data_fraction"].start
        + _index(BLOCK_BY_NAME["data_fraction"], config.data_fraction)
    ] = 1

    params = dict(config.params)
    for spec in FAMILIES:
        for hyperparameter in spec.hyperparameters:
            block = BLOCK_BY_NAME[f"{spec.name}.{hyperparameter.name}"]
            if spec.name == config.family:
                bits[block.start + _index(block, params[hyperparameter.name])] = 1
            elif mode == "feasible":
                bits[block.start] = 1
    return bits


def decode(bits, repair: str = "raise") -> Configuration:
    """Invert :func:`encode`. Inactive-family bits are ignored.

    Args:
        bits: length-``d`` binary vector.
        repair: ``"raise"`` if a required block is not one-hot; ``"argmax"`` takes the
            first maximum in each block (the FMQA-line repair that hides infeasibility
            rather than reporting it — available because RQ3 has to measure that habit).
    """
    vector = np.asarray(bits, dtype=int).reshape(-1)
    if vector.size != N_VARIABLES:
        raise ValueError(f"expected {N_VARIABLES} bits; got {vector.size}")

    family = _read_block(BLOCK_BY_NAME["family"], vector, repair)
    fraction = _read_block(BLOCK_BY_NAME["data_fraction"], vector, repair)
    spec = FAMILY_BY_NAME[family]
    params = tuple(
        (
            hyperparameter.name,
            _read_block(
                BLOCK_BY_NAME[f"{spec.name}.{hyperparameter.name}"],
                vector,
                repair,
            ),
        )
        for hyperparameter in spec.hyperparameters
    )
    return Configuration(family=family, params=params, data_fraction=float(fraction))


def _read_block(block: Block, vector: np.ndarray, repair: str):
    slice_ = vector[block.start : block.stop]
    ones = np.flatnonzero(slice_ == 1)
    if repair == "argmax":
        return block.values[int(np.argmax(slice_))]
    if repair != "raise":
        raise ValueError(f"unknown repair {repair!r}")
    if len(ones) != 1:
        raise ValueError(f"block {block.name!r} is not one-hot: {int(slice_.sum())} ones")
    return block.values[int(ones[0])]


def is_onehot_feasible(bits) -> bool:
    """Whether every block has exactly one 1 — the QUBO penalty's feasible set."""
    vector = np.asarray(bits, dtype=int).reshape(-1)
    if vector.size != N_VARIABLES:
        return False
    if np.any((vector != 0) & (vector != 1)):
        return False
    return all(int(vector[block.start : block.stop].sum()) == 1 for block in BLOCKS)


def family_blocks(family: str) -> tuple[Block, ...]:
    """The E2 encoding: data-fraction plus this family's own hyperparameters.

    Family identity is an outer choice, not a bit, so the gated product disappears.
    """
    spec = FAMILY_BY_NAME[family]
    names = ["data_fraction", *[f"{family}.{h.name}" for h in spec.hyperparameters]]
    return tuple(BLOCK_BY_NAME[name] for name in names)


def encode_family(config: Configuration) -> np.ndarray:
    """E2 bit vector for ``config``, whose family is implied by the caller."""
    blocks = family_blocks(config.family)
    width = sum(block.size for block in blocks)
    bits = np.zeros(width, dtype=np.int8)
    params = dict(config.params)
    cursor = 0
    for block in blocks:
        if block.name == "data_fraction":
            value = config.data_fraction
        else:
            _, hp_name = block.name.split(".", 1)
            value = params[hp_name]
        bits[cursor + _index(block, value)] = 1
        cursor += block.size
    return bits


def e2_width(family: str) -> dict[str, int]:
    """Identifiability numbers for one family's E2 encoding."""
    d = sum(block.size for block in family_blocks(family))
    return {
        "family": family,
        "variables": d,
        "surrogate_parameters": 1 + d + d * (d - 1) // 2,
    }
