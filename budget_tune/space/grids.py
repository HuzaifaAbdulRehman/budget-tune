"""The search space, declared as data.

Everything downstream -- the campaign, the one-hot encoder, the optimisers, the report's
space table -- reads this module. Declaring the grid once is what stops the encoder and
the campaign from disagreeing about what a configuration *is*, which would show up as a
surrogate fitted to a space that was never measured.

**Configurations are canonical.** A configuration is a family, that family's own
hyperparameters, and a data fraction. Hyperparameters belonging to other families do not
appear, are not enumerated, and cannot vary. The flat one-hot encoding used by the QUBO
surrogates *does* carry every family's block simultaneously, so many binary vectors map to
one canonical configuration -- that degeneracy is a property of the encoding and is studied
as such. It is deliberately not a property of the benchmark, which measures each distinct
model exactly once.

**Data fraction has three levels, and that is a decision variable, not the fidelity ladder.**
The calibration pilot measured that a quarter of the data costs the same as all of it for
ALS and MultVAE, so a fraction rung would not be a cheap approximation of anything. The
successive-halving ladder is *epochs* (see ``budget_tune.fidelity``). Three fraction values
remain because they are part of the CASH space every method may return, and because nested
recency retention is still a well-defined data-reduction axis even when it is not a cost
lever. A fourth level would densify that axis without restoring a cheap fidelity.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

#: Training-data retention levels. A decision variable of every configuration, not the
#: successive-halving resource -- that ladder is epochs, in ``budget_tune.fidelity``.
DATA_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 1.0)


@dataclass(frozen=True)
class Hyperparameter:
    """One categorical axis of the space.

    Attributes:
        name: attribute name passed to the family's constructor.
        values: the grid, in the order the one-hot block will use. Order is fixed here so
            that a re-run cannot silently permute the encoding.
    """

    name: str
    values: tuple

    def __post_init__(self) -> None:
        if len(self.values) < 1:
            raise ValueError(f"hyperparameter {self.name!r} has no values")
        if len(set(map(repr, self.values))) != len(self.values):
            raise ValueError(f"hyperparameter {self.name!r} has duplicate values")

    @property
    def size(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class FamilySpec:
    """A model family and its grid.

    Attributes:
        name: family identifier, matching the family class's ``name``.
        hyperparameters: the family's own axes.
        deterministic: whether two fits with different seeds produce identical output.
            Declared rather than detected, and then **asserted by the test suite**: a
            family measured once because it was assumed deterministic, but which is not,
            would put a single noisy draw into the benchmark with no spread column to
            reveal it.
        note: why the family earns a place in the comparison.
    """

    name: str
    hyperparameters: tuple[Hyperparameter, ...]
    deterministic: bool
    note: str

    @property
    def base_size(self) -> int:
        """Configurations before the data-fraction axis is applied."""
        size = 1
        for hyperparameter in self.hyperparameters:
            size *= hyperparameter.size
        return size

    @property
    def size(self) -> int:
        return self.base_size * len(DATA_FRACTIONS)


FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        name="popularity",
        hyperparameters=(),
        deterministic=True,
        note="the accuracy floor and the zero point of the cost axis",
    ),
    FamilySpec(
        name="itemknn",
        hyperparameters=(
            Hyperparameter("topk", (10, 50, 100, 300)),
            Hyperparameter("shrink", (0.0, 10.0, 100.0)),
        ),
        deterministic=True,
        note="cheap to train, cost per request scales with catalogue size",
    ),
    FamilySpec(
        name="als",
        hyperparameters=(
            Hyperparameter("factors", (16, 32, 64, 128)),
            Hyperparameter("epochs", (5, 15, 30)),
            Hyperparameter("regularisation", (0.001, 0.01, 0.1)),
            Hyperparameter("alpha", (1.0, 10.0, 40.0)),
        ),
        deterministic=False,
        note="expensive iterative training, cheap serving; the four-axis workhorse",
    ),
    FamilySpec(
        name="multvae",
        hyperparameters=(
            Hyperparameter("latent", (32, 64, 128)),
            Hyperparameter("hidden", (200, 600)),
            Hyperparameter("epochs", (10, 20)),
            Hyperparameter("dropout", (0.0, 0.5)),
        ),
        deterministic=False,
        note="neural cost shape without a ruinous training cost",
    ),
    FamilySpec(
        name="markov",
        hyperparameters=(
            Hyperparameter("order", (1, 2)),
            Hyperparameter("smoothing", (0.0, 0.1, 0.5)),
            Hyperparameter("decay", (False, True)),
        ),
        deterministic=True,
        note="the only family that uses interaction order; new implementation",
    ),
)

FAMILY_BY_NAME: dict[str, FamilySpec] = {spec.name: spec for spec in FAMILIES}


def _format(value) -> str:
    """Render a hyperparameter value stably for identifiers.

    Floats go through ``repr`` of a rounded value rather than ``str`` of the raw one, so
    that ``0.1`` cannot become ``0.10000000000000001`` in one run's identifiers and not in
    another's -- which would silently split one configuration into two rows that never
    join.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(round(value, 10))
    return str(value)


def match_grid_value(value, grid: tuple):
    """Re-attach a CSV-round-tripped value to the declared grid cell.

    Pandas will read ``1`` as ``1.0`` and ``False`` as ``0``. Comparing to the declared
    tuple by equality would then miss, and a configuration would silently become a
    different one.
    """
    if value in grid:
        return value
    if isinstance(value, bool):
        for cell in grid:
            if isinstance(cell, bool) and cell is value:
                return cell
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for cell in grid:
            if isinstance(cell, bool):
                continue
            if isinstance(cell, (int, float)) and abs(float(value) - float(cell)) < 1e-12:
                return cell
    formatted = _format(value)
    for cell in grid:
        if _format(cell) == formatted:
            return cell
    raise KeyError(f"{value!r} does not match any of {grid}")


def configuration_from_row(row) -> Configuration:
    """Rebuild a :class:`Configuration` from a search-table row."""
    family = str(row["family"])
    spec = FAMILY_BY_NAME[family]
    params = tuple(
        (h.name, match_grid_value(row[f"{family}.{h.name}"], h.values))
        for h in spec.hyperparameters
    )
    fraction = match_grid_value(row["data_fraction"], DATA_FRACTIONS)
    return Configuration(family=family, params=params, data_fraction=float(fraction))


@dataclass(frozen=True)
class Configuration:
    """One canonical configuration: a family, its hyperparameters, and a data fraction."""

    family: str
    params: tuple[tuple[str, object], ...]
    data_fraction: float

    @property
    def spec(self) -> FamilySpec:
        return FAMILY_BY_NAME[self.family]

    @property
    def kwargs(self) -> dict:
        """Constructor arguments for the family class."""
        return dict(self.params)

    @property
    def config_id(self) -> str:
        """Stable identifier, readable in a CSV and safe as a join key."""
        body = ";".join(f"{name}={_format(value)}" for name, value in self.params)
        return f"{self.family}|{body}|frac={_format(self.data_fraction)}"

    def as_row(self) -> dict:
        """Identity columns for the benchmark table.

        Hyperparameters are written to ``<family>.<name>`` columns, blank for every other
        family. That keeps the table self-describing and matches the flat encoding the
        surrogates see, without pretending a configuration carries values it does not have.
        """
        row = {
            "config_id": self.config_id,
            "family": self.family,
            "data_fraction": self.data_fraction,
        }
        row.update({f"{self.family}.{name}": value for name, value in self.params})
        return row


def hyperparameter_columns() -> list[str]:
    """Every ``<family>.<hyperparameter>`` column, in declaration order."""
    return [
        f"{spec.name}.{hyperparameter.name}"
        for spec in FAMILIES
        for hyperparameter in spec.hyperparameters
    ]


def enumerate_family(spec: FamilySpec) -> Iterator[Configuration]:
    """Every canonical configuration of one family, data fractions included."""
    from itertools import product

    names = [hyperparameter.name for hyperparameter in spec.hyperparameters]
    grids = [hyperparameter.values for hyperparameter in spec.hyperparameters]
    for fraction in DATA_FRACTIONS:
        for combination in product(*grids) if grids else [()]:
            yield Configuration(
                family=spec.name,
                params=tuple(zip(names, combination, strict=True)),
                data_fraction=float(fraction),
            )


def enumerate_configurations() -> list[Configuration]:
    """The whole canonical space, in a fixed order.

    Fixed because the campaign is resumable: a re-run that enumerated in a different order
    would resume against the wrong cells.
    """
    return [config for spec in FAMILIES for config in enumerate_family(spec)]


def space_size() -> dict[str, int]:
    """Per-family and total canonical configuration counts, for the report's table."""
    counts = {spec.name: spec.size for spec in FAMILIES}
    counts["total"] = sum(counts.values())
    return counts


def block_layout() -> list[tuple[str, tuple]]:
    """One-hot blocks of the flat (E1) encoding, in the order bits are packed.

    Shared by the width calculator and the codec so the two cannot disagree about
    what bit 17 means. Family and data-fraction come first; each family's own
    hyperparameters follow in declaration order.
    """
    blocks: list[tuple[str, tuple]] = [
        ("family", tuple(spec.name for spec in FAMILIES)),
        ("data_fraction", DATA_FRACTIONS),
    ]
    for spec in FAMILIES:
        for hyperparameter in spec.hyperparameters:
            blocks.append((f"{spec.name}.{hyperparameter.name}", hyperparameter.values))
    return blocks


def binary_width() -> dict[str, int]:
    """One-hot width of the flat (E1) encoding, and the resulting surrogate size.

    Reported because it is the number behind the identifiability problem: a second-order
    surrogate over ``d`` binary variables has ``1 + d + d(d-1)/2`` parameters, and a
    realistic optimisation budget affords tens of observations, not hundreds.
    """
    blocks = block_layout()
    d = sum(len(values) for _, values in blocks)
    return {
        "blocks": len(blocks),
        "variables": d,
        "surrogate_parameters": 1 + d + d * (d - 1) // 2,
    }


def coarse_grid() -> list[Configuration]:
    """The grid-search baseline: endpoints and midpoint of every axis, not the table.

    Enumerating the whole 471-cell space would make "grid search" the benchmark itself.
    Each hyperparameter contributes its first value, its last value, and the value at
    index ``(n - 1) // 2``. Axes of length 2 or 3 therefore appear in full; longer axes
    drop the extra interior points. Data fraction is included as an axis. Order is the
    same as :func:`enumerate_configurations` so a resumed comparison cannot reshuffle.
    """
    from itertools import product

    selected: list[Configuration] = []
    for spec in FAMILIES:
        axes = []
        names = []
        for hyperparameter in spec.hyperparameters:
            values = hyperparameter.values
            keep = {0, (len(values) - 1) // 2, len(values) - 1}
            axes.append(tuple(values[i] for i in sorted(keep)))
            names.append(hyperparameter.name)
        for fraction in DATA_FRACTIONS:
            for combination in product(*axes) if axes else [()]:
                selected.append(
                    Configuration(
                        family=spec.name,
                        params=tuple(zip(names, combination, strict=True)),
                        data_fraction=float(fraction),
                    )
                )
    return selected
