"""Calibration pilot: fit the campaign's cost model instead of assuming it.

The design's campaign estimate was arithmetic on four numbers borrowed from green-rerank,
extrapolated through a cost model nobody had checked. The load-bearing guess was ALS's
dependence on ``factors``, which the linear algebra places somewhere between quadratic (the
``nnz * f^2`` update term) and cubic (the ``f^3`` solve per user) -- and which of those
dominates moves the ALS subtotal, and therefore the whole campaign, by a factor of two.

So this measures a designed subset and fits the exponents. The design is one-factor-at-a-time
around a base point, plus a corner far from it: the sweeps identify each exponent and the
corner tests whether the model is multiplicative at all. A model that predicts its own sweep
points and misses the corner is a model that has been fitted rather than validated.

Nothing here writes a benchmark row. The pilot's output is a cost model and a runtime
estimate; the campaign is a separate script and a separate decision.

Run::

    python -m experiments.calibrate --catalogue ml100k --threads 1
    python -m experiments.calibrate --all --threads 1 --out results/calibration
"""

from __future__ import annotations

# Thread pinning must happen before numpy loads -- see budget_tune.measure.threads. This
# import sits above the others deliberately, and ruff is told so rather than reordering it.
from budget_tune.measure.threads import pin  # isort: skip

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402


@dataclass(frozen=True)
class Point:
    """One configuration to measure, and why it is in the design."""

    family: str
    params: dict
    fraction: float
    role: str


def als_design() -> list[Point]:
    """One-factor-at-a-time around ``factors=64, epochs=15, fraction=1.0``, plus a corner."""
    base = {"factors": 64, "epochs": 15, "regularisation": 0.01, "alpha": 40.0}
    points = [Point("als", dict(base), 1.0, "base")]
    for factors in (16, 32, 128):
        points.append(Point("als", {**base, "factors": factors}, 1.0, "factors"))
    for epochs in (5, 30):
        points.append(Point("als", {**base, "epochs": epochs}, 1.0, "epochs"))
    for fraction in (0.25, 0.5):
        points.append(Point("als", dict(base), fraction, "fraction"))
    # Far from the base on every axis at once. If the multiplicative model is wrong, this is
    # where it shows.
    points.append(Point("als", {**base, "factors": 128, "epochs": 30}, 0.25, "corner"))
    return points


def multvae_design() -> list[Point]:
    base = {"latent": 64, "hidden": 600, "epochs": 20, "dropout": 0.5}
    points = [Point("multvae", dict(base), 1.0, "base")]
    points.append(Point("multvae", {**base, "hidden": 200}, 1.0, "hidden"))
    for latent in (32, 128):
        points.append(Point("multvae", {**base, "latent": latent}, 1.0, "latent"))
    points.append(Point("multvae", {**base, "epochs": 10}, 1.0, "epochs"))
    for fraction in (0.25, 0.5):
        points.append(Point("multvae", dict(base), fraction, "fraction"))
    points.append(
        Point("multvae", {"latent": 128, "hidden": 200, "epochs": 10, "dropout": 0.0},
              0.25, "corner")
    )
    return points


def cheap_design() -> list[Point]:
    """The families assumed negligible, at their most expensive grid point, across fractions.

    Assuming a family is free is how a campaign estimate goes wrong in the direction nobody
    checks. ItemKNN at ``topk=300`` and a second-order Markov chain are the worst cases in
    their grids.

    The fraction sweep is here for a reason beyond budgeting. The first pilot found that ALS
    and MultVAE are almost *insensitive* to training-set size -- both iterate over every user
    regardless of how many interactions each has -- which puts H1's premise in question. Both
    of these families do work proportional to the interaction count, so if the data lever
    moves cost anywhere in this space, it moves it here. Measuring only at full data would
    have left that unanswerable.
    """
    markov = {"order": 2, "smoothing": 0.1, "decay": True}
    points = []
    for fraction in (0.25, 0.5, 1.0):
        points.append(Point("popularity", {}, fraction, "cheap"))
        points.append(Point("itemknn", {"topk": 300, "shrink": 0.0}, fraction, "cheap"))
        points.append(Point("markov", dict(markov), fraction, "cheap"))
    return points


def warm_up(dataset, evaluate, k: int) -> None:
    """Pay every library's first-call cost before anything is measured.

    Found by the smoke run, which is exactly the sort of error this pilot exists to catch:
    MultVAE's base point read 1.938 CPU-seconds and the same amount of work read 0.484 later
    in the same process. The difference is torch's first call -- thread pool construction and
    kernel selection -- charged in full to whichever configuration happened to run first.

    Unfixed, the cost model would have been fitted to a base point inflated four-fold, and
    since the base point anchors the intercept, every campaign prediction would have been
    wrong by the same factor while each individual reading looked plausible.

    A cheap fit of every family, discarded, before the session opens.
    """
    from budget_tune.families import build

    fold = dataset.fold(min(dataset.folds))
    users = dataset.eval_users()[:8]

    for family, kwargs in (
        ("popularity", {}),
        ("itemknn", {"topk": 10, "shrink": 0.0}),
        ("markov", {"order": 2, "smoothing": 0.1, "decay": True}),
        ("als", {"factors": 8, "epochs": 1, "regularisation": 0.01, "alpha": 40.0}),
        ("multvae", {"latent": 8, "hidden": 16, "epochs": 1, "dropout": 0.0}),
    ):
        model = build(family, **kwargs)
        args = (
            (fold.matrix, fold.sequences)
            if getattr(model, "needs_sequences", False)
            else (fold.matrix,)
        )
        model.fit(*args)
        evaluate.recommend(model, fold.matrix, users, k)


def measure_point(dataset, point: Point, seed: int, k: int, session, evaluate):
    """Fit, serve and score one configuration, with only the right parts inside a window."""

    from budget_tune.families import build

    fold = dataset.fold(point.fraction)
    users = dataset.eval_users()

    kwargs = dict(point.params)
    if point.family in {"als", "multvae"}:
        kwargs["seed"] = seed
    model = build(point.family, **kwargs)

    needs_sequences = getattr(model, "needs_sequences", False)
    fit_args = (fold.matrix, fold.sequences) if needs_sequences else (fold.matrix,)

    with session.window("train", point.family) as out:
        model.fit(*fit_args)
    train = out[0]

    with session.window("serve", point.family) as out:
        items = evaluate.recommend(model, fold.matrix, users, k)
    serve = out[0]

    # Outside every window, always: metric computation is Python-loop work and would be the
    # majority of the reading for a cheap family.
    score_start = time.perf_counter()
    scores = evaluate.score(dataset, items, users, "validation", k=k)
    score_wall = time.perf_counter() - score_start

    return {
        "dataset": dataset.name,
        "family": point.family,
        "role": point.role,
        "seed": seed,
        "data_fraction": point.fraction,
        **{f"param.{name}": value for name, value in point.params.items()},
        "n_train_interactions": fold.n_interactions,
        "n_items": dataset.n_items,
        "n_users": dataset.n_users,
        "n_eval_users": int(len(users)),
        "train_cpu_seconds": train.cpu_seconds,
        "train_wall_seconds": train.wall_seconds,
        "train_cpu_utilisation": train.cpu_utilisation,
        # A window shorter than a few clock quanta is a tick count, not a duration, and
        # feeding one to a log-log fit would let quantisation set an exponent. Flagged here
        # and excluded from the fit rather than silently rounded into the model.
        "train_below_quantum": bool(train.cpu_seconds < _quantum() * 4),
        "serve_cpu_seconds": serve.cpu_seconds,
        "serve_wall_seconds": serve.wall_seconds,
        "score_wall_seconds": score_wall,
        "peak_rss_bytes": train.peak_rss_bytes,
        "model_bytes": model.model_bytes,
        "val_ndcg_at_10": scores.ndcg,
        "val_recall_at_10": scores.recall,
    }


def _quantum() -> float:
    from green_rerank.measure.session import clock_quantum

    return clock_quantum()


def fit_power_law(frame, family: str, axes: list[str]):
    """Least squares on ``log(cpu) ~ a + sum_i b_i log(axis_i)``.

    Log-log because the hypothesised model is multiplicative -- cost as a product of powers
    of the hyperparameters -- so the exponents are what a linear fit in log space returns
    directly. Fitted on the sweep rows only; the corner is held out so the reported error on
    it is a prediction rather than a residual.

    Rows flagged ``train_below_quantum`` are dropped: a reading that could not clear the
    scheduler tick is a tick count, and log-log regression would happily turn quantisation
    into an exponent.
    """
    import numpy as np

    rows = frame[(frame.family == family) & (frame.role != "corner")]
    if "train_below_quantum" in rows.columns:
        rows = rows[~rows.train_below_quantum.astype(bool)]
    if len(rows) < len(axes) + 1:
        return None

    design = np.column_stack(
        [np.ones(len(rows))]
        + [np.log(axis_values(rows, axis).to_numpy(dtype=float)) for axis in axes]
    )
    target = np.log(rows["train_cpu_seconds"].to_numpy(dtype=float))
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)

    residual = target - design @ coefficients
    ss_total = float(((target - target.mean()) ** 2).sum())
    ss_residual = float((residual**2).sum())
    r_squared = 1 - ss_residual / ss_total if ss_total > 0 else float("nan")

    return {
        "family": family,
        "axes": axes,
        "intercept": float(coefficients[0]),
        "exponents": {axis: float(c) for axis, c in zip(axes, coefficients[1:], strict=True)},
        "r_squared": float(r_squared),
        "n_points": int(len(rows)),
    }


def axis_values(source, axis: str):
    """Resolve an axis to a column, whether or not it carries the ``param.`` prefix.

    Hyperparameters are written to the pilot's CSV as ``param.factors`` so they cannot
    collide with measurement columns, while the search space names them bare. Rather than
    have the cost model know which is which, both spellings resolve here.
    """
    for key in (axis, f"param.{axis}"):
        if key in source:
            return source[key]
    raise KeyError(f"no column for axis {axis!r}; have {sorted(source)}")


def predict(model: dict, values: dict) -> float:
    """Cost predicted by a fitted power law."""
    import numpy as np

    total = model["intercept"]
    for axis, exponent in model["exponents"].items():
        total += exponent * np.log(float(axis_values(values, axis)))
    return float(np.exp(total))


#: Axes each fitted family's cost is modelled against. ``n_train_interactions`` carries the
#: data fraction; ``n_users`` and ``n_items`` are constant within a catalogue and so are
#: absorbed into the intercept by fitting per catalogue rather than pooling.
COST_AXES: dict[str, list[str]] = {
    "als": ["factors", "epochs", "n_train_interactions"],
    "multvae": ["latent", "hidden", "epochs", "n_train_interactions"],
}


def analyse(frame, datasets: dict) -> dict:
    """Fit the cost model per catalogue and predict the full campaign from it."""
    import numpy as np

    from budget_tune.space.grids import FAMILY_BY_NAME, enumerate_configurations

    report: dict = {"models": {}, "campaign": {}}

    for name, dataset in datasets.items():
        rows = frame[frame.dataset == name]
        models = {}
        for family, axes in COST_AXES.items():
            model = fit_power_law(rows, family, axes)
            if model is None:
                continue

            corner = rows[(rows.family == family) & (rows.role == "corner")]
            if len(corner):
                observed = float(corner.iloc[0]["train_cpu_seconds"])
                expected = predict(model, corner.iloc[0].to_dict())
                model["corner_observed"] = observed
                model["corner_predicted"] = expected
                # The only out-of-sample check the pilot has. A multiplicative model that
                # nails its own sweep and misses here is not a model, it is an
                # interpolation.
                model["corner_ratio"] = expected / observed if observed > 0 else float("nan")
            models[family] = model

        cheap = {
            family: float(rows[rows.family == family]["train_cpu_seconds"].median())
            for family in ("popularity", "itemknn", "markov")
            if (rows.family == family).any()
        }
        overhead = float((rows.serve_wall_seconds + rows.score_wall_seconds).median())

        per_family: dict[str, float] = {}
        runs = 0
        for config in enumerate_configurations():
            spec = FAMILY_BY_NAME[config.family]
            seeds = 1 if spec.deterministic else 3
            runs += seeds

            if config.family in models:
                values = dict(config.kwargs)
                values["n_train_interactions"] = dataset.fold(
                    config.data_fraction
                ).n_interactions
                cost = predict(models[config.family], values)
            else:
                cost = cheap.get(config.family, 0.0)

            per_family[config.family] = per_family.get(config.family, 0.0) + cost * seeds

        per_family["_serve_and_score"] = runs * overhead
        total = float(sum(per_family.values()))

        report["models"][name] = models
        report["campaign"][name] = {
            "per_family_seconds": {k: float(v) for k, v in per_family.items()},
            "total_seconds": total,
            "total_minutes": total / 60,
            "runs": runs,
            "serve_and_score_per_run": overhead,
            "cheap_family_seconds": cheap,
            "below_quantum_rows": int(rows.train_below_quantum.sum())
            if "train_below_quantum" in rows
            else 0,
        }

    grand = sum(c["total_seconds"] for c in report["campaign"].values())
    report["total_seconds"] = grand
    report["total_hours"] = grand / 3600
    report["cpu_to_wall"] = {
        name: float(np.median(frame[frame.dataset == name].train_cpu_utilisation))
        for name in datasets
    }
    return report


def print_report(report: dict) -> None:
    """Human-readable summary. The saved JSON is the artifact; this is for the console."""
    print("\n" + "=" * 78)
    for name, models in report["models"].items():
        print(f"\n{name}")
        for family, model in models.items():
            exponents = ", ".join(
                f"{axis}^{value:.2f}" for axis, value in model["exponents"].items()
            )
            print(f"  {family:9s} {exponents}   R2={model['r_squared']:.4f}")
            if "corner_ratio" in model:
                print(
                    f"  {'':9s} corner: predicted {model['corner_predicted']:.3f} vs "
                    f"observed {model['corner_observed']:.3f} "
                    f"(ratio {model['corner_ratio']:.2f})"
                )
        campaign = report["campaign"][name]
        print(f"  campaign: {campaign['total_minutes']:.1f} min over {campaign['runs']} runs")
        for family, seconds in sorted(
            campaign["per_family_seconds"].items(), key=lambda kv: -kv[1]
        ):
            print(f"      {family:18s} {seconds / 60:8.2f} min")

    print(f"\nTOTAL: {report['total_hours']:.2f} h")
    print(f"cpu/wall ratios: {report['cpu_to_wall']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", action="append", default=[])
    parser.add_argument(
        "--all", action="store_true", help="every headline catalogue plus the meta one"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/calibration"))
    parser.add_argument(
        "--allow-busy", action="store_true",
        help="proceed on a busy machine. Records the fact; does not make it trustworthy.",
    )
    parser.add_argument(
        "--analyse-only", action="store_true",
        help="re-fit the cost model from a saved pilot CSV without measuring again",
    )
    args = parser.parse_args()

    if not args.analyse_only:
        pin(args.threads)

    import pandas as pd

    from budget_tune.benchmark import evaluate
    from budget_tune.companion import ensure_all_importable, revisions
    from budget_tune.data import catalogues
    from budget_tune.measure.threads import apply_to_torch, verify
    from budget_tune.space.grids import DATA_FRACTIONS

    ensure_all_importable()
    from green_rerank.measure.guards import ConditionsMonitor, ExclusiveLock, preflight
    from green_rerank.measure.session import MeasurementSession, clock_quantum

    names = list(catalogues.HEADLINE) + [catalogues.META] if args.all else args.catalogue
    if not names:
        parser.error("pass --catalogue NAME (repeatable) or --all")

    if args.analyse_only:
        # Re-fitting must never require re-measuring: the measurements are the expensive,
        # condition-sensitive part and a bug in the arithmetic is not a reason to spend
        # them again.
        frame = pd.read_csv(args.out / f"pilot_threads{args.threads}.csv")
        loaded = {name: catalogues.load(name, fractions=DATA_FRACTIONS) for name in names}
        report = analyse(frame, loaded)
        (args.out / f"cost_model_threads{args.threads}.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print_report(report)
        return 0

    torch_threads = apply_to_torch(args.threads)
    thread_report = verify(args.threads)
    thread_report["torch_threads"] = torch_threads

    args.out.mkdir(parents=True, exist_ok=True)

    with ExclusiveLock(args.out / ".measure.lock") as lock:
        checks = preflight(
            require_mains=True,
            max_busy_pct=None if args.allow_busy else 25.0,
            lock=lock,
        )
        print(f"preflight: power={checks.power_source} busy={checks.machine_busy_pct}")
        print(f"threads: {thread_report}")
        print(f"clock quantum: {clock_quantum():.6f} s")

        rows = []
        loaded = {}
        monitor = ConditionsMonitor()
        monitor.start()
        started = time.perf_counter()

        for name in names:
            print(f"\n== {name}")
            dataset = catalogues.load(name, fractions=DATA_FRACTIONS)
            loaded[name] = dataset
            session = MeasurementSession(
                label=name,
                meta={"catalogue": name, "threads": args.threads, "pilot": True},
                preflight=checks,
            )

            # Before the first measured window, never inside one.
            warm_up(dataset, evaluate, args.k)

            for point in [*cheap_design(), *als_design(), *multvae_design()]:
                row = measure_point(dataset, point, args.seed, args.k, session, evaluate)
                row["threads"] = args.threads
                rows.append(row)
                print(
                    f"   {point.family:11s} {point.role:8s} f={point.fraction:<5} "
                    f"cpu={row['train_cpu_seconds']:8.3f} "
                    f"wall={row['train_wall_seconds']:8.3f} "
                    f"util={row['train_cpu_utilisation']:.2f} "
                    f"ndcg={row['val_ndcg_at_10']:.4f}"
                )
            session.close()

        conditions = monitor.stop()
        elapsed = time.perf_counter() - started

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / f"pilot_threads{args.threads}.csv", index=False)

    report = analyse(frame, loaded)
    (args.out / f"cost_model_threads{args.threads}.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print_report(report)

    manifest = {
        "threads": thread_report,
        "clock_quantum_seconds": clock_quantum(),
        "preflight": checks.as_meta(),
        "conditions": conditions,
        "elapsed_seconds": elapsed,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": sys.version,
        },
        "companions": revisions(),
        "catalogues": names,
    }
    (args.out / f"manifest_threads{args.threads}.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    print(f"\npilot finished in {elapsed / 60:.1f} min -> {args.out}")
    if conditions.get("power_source_changed"):
        print("WARNING: power source changed during the pilot; timings are not comparable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
