"""Test C2: is a low-epoch ranking informative about a full-epoch one?

The calibration pilot established that epochs are a real cost axis (C1). That makes
successive halving *runnable*; it does not make it *work*. This measures whether the ranking
it would discard on has anything to do with the ranking it is trying to find.

Protocol, fixed in ``docs/design.md`` §7.0 before this ran:

* **Gift Cards only.** It is the meta catalogue, reserved for freezing a method's own
  settings. A fidelity schedule is one of those settings. Headline catalogues are not touched
  and their enumerated optima are not consulted.
* **Every configuration, no sampling.** All 108 ALS parameter combinations and all 36 MultVAE
  combinations, each at every rung, at two seeds. Nothing is selected, so nothing can be
  cherry-picked.
* **Validation split only.** Test metrics are not computed, not written and not loaded.
* **Judged against a ceiling.** Cross-fidelity agreement is compared with same-fidelity
  agreement across seeds, because a correlation of 0.6 means one thing when the ranking is
  perfectly reproducible and another when it is not.

Run::

    python -m experiments.validate_fidelity --threads 1
"""

from __future__ import annotations

# Thread pinning must precede numpy -- see budget_tune.measure.threads.
from budget_tune.measure.threads import pin  # isort: skip

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

#: The catalogue this may run on. A constant rather than an argument: the whole point is that
#: the schedule is chosen without seeing a headline result, and an argument would make that a
#: matter of remembering.
META_CATALOGUE = "gift_cards"


def parameter_grid(family: str) -> list[dict]:
    """Every combination of a family's non-epoch hyperparameters, in grid order."""
    from budget_tune.space.grids import FAMILY_BY_NAME

    spec = FAMILY_BY_NAME[family]
    axes = [h for h in spec.hyperparameters if h.name != "epochs"]
    names = [h.name for h in axes]
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*[h.values for h in axes])
    ]


def measure(dataset, family: str, params: dict, epochs: int, fraction: float,
            seed: int, k: int, session, evaluate) -> dict:
    """Fit at one rung and score on validation. Never touches the reporting split."""
    from budget_tune.families import build

    fold = dataset.fold(fraction)
    users = dataset.eval_users()

    model = build(family, **params, epochs=epochs, seed=seed)
    with session.window("train", family) as out:
        model.fit(fold.matrix)
    train = out[0]

    items = evaluate.recommend(model, fold.matrix, users, k)
    scores = evaluate.score(dataset, items, users, "validation", k=k)

    return {
        "family": family,
        "epochs": epochs,
        "data_fraction": fraction,
        "seed": seed,
        **{f"param.{name}": value for name, value in params.items()},
        "train_cpu_seconds": train.cpu_seconds,
        "train_wall_seconds": train.wall_seconds,
        "val_ndcg_at_10": scores.ndcg,
        "val_recall_at_10": scores.recall,
    }


def analyse(frame) -> dict:
    """Rank agreement, top-k overlap and simulated halving, per family."""
    import numpy as np

    from budget_tune.fidelity import (
        LADDERS,
        rank_agreement,
        simulate_halving,
        top_k_overlap,
    )

    report: dict = {}
    for family, ladder in LADDERS.items():
        rows = frame[frame.family == family]
        if rows.empty:
            continue

        # Only this family's own parameter columns. A frame holding several families carries
        # the union of their columns, so ALS rows have an all-NaN ``param.latent`` -- and
        # ``pivot_table`` silently drops every row with a NaN anywhere in its index, leaving
        # an empty table and a KeyError three lines later. Selecting the fully-populated
        # columns keeps the index meaningful per family.
        keys = [
            c for c in rows.columns if c.startswith("param.") and rows[c].notna().all()
        ] + ["data_fraction"]
        # Averaged over seeds, so the ranking being compared is the one an optimiser would
        # actually see rather than one draw of it.
        table = rows.pivot_table(
            index=keys, columns="epochs", values="val_ndcg_at_10", aggfunc="mean"
        ).sort_index()

        scores = {int(rung): table[rung].to_numpy() for rung in ladder.rungs}
        low, high = scores[ladder.rungs[0]], scores[ladder.rungs[-1]]

        # The ceiling: how reproducible the ranking is at the top rung across seeds. Judging
        # cross-fidelity agreement against 1.0 instead of against this would call a fidelity
        # bad for noise it did not cause.
        top = ladder.rungs[-1]
        by_seed = rows[rows.epochs == top].pivot_table(
            index=keys, columns="seed", values="val_ndcg_at_10"
        ).sort_index()
        seed_columns = list(by_seed.columns)
        ceiling = None
        if len(seed_columns) >= 2:
            first = by_seed[seed_columns[0]].to_numpy()
            second = by_seed[seed_columns[1]].to_numpy()
            ceiling = rank_agreement(first, second)

        cost = rows.groupby("epochs")["train_cpu_seconds"].mean()
        report[family] = {
            "n_configurations": int(len(low)),
            "rungs": list(ladder.rungs),
            "keep": list(ladder.keep),
            "low_vs_high": rank_agreement(low, high),
            "seed_ceiling_at_max_rung": ceiling,
            "intermediate_vs_high": {
                str(rung): rank_agreement(scores[rung], high)
                for rung in ladder.rungs[1:-1]
            },
            # Only k values the population can support. Reporting an overlap at k=10 over
            # eight configurations would be a fraction of a set that does not exist.
            "top_k_overlap": {
                str(k): top_k_overlap(low, high, k) for k in (5, 10) if k <= len(low)
            },
            "halving": simulate_halving(scores, ladder),
            "mean_cpu_seconds_per_rung": {str(r): float(c) for r, c in cost.items()},
            "cost_ratio_low_to_high": float(cost[ladder.rungs[0]] / cost[ladder.rungs[-1]]),
            "score_range": [float(np.min(high)), float(np.max(high))],
        }
    return report


def print_report(report: dict) -> None:
    for family, result in report.items():
        print(f"\n=== {family}  ({result['n_configurations']} configurations)")
        print(f"    rungs {result['rungs']}  keep {[round(k, 3) for k in result['keep']]}")

        low = result["low_vs_high"]
        print(
            f"    rung0 vs max : spearman {low['spearman']:+.3f}  "
            f"kendall {low['kendall']:+.3f}"
        )
        ceiling = result["seed_ceiling_at_max_rung"]
        if ceiling:
            print(
                f"    seed ceiling : spearman {ceiling['spearman']:+.3f}  "
                f"kendall {ceiling['kendall']:+.3f}   <- the yardstick"
            )
        for rung, agreement in result["intermediate_vs_high"].items():
            print(f"    rung {rung:>3} vs max: spearman {agreement['spearman']:+.3f}")

        overlap = ", ".join(
            f"top-{k} {value:.2f}" for k, value in result["top_k_overlap"].items()
        )
        print(f"    overlap      : {overlap}")

        halving = result["halving"]
        print(
            f"    simulated SH : survivors {halving['survivors_per_rung']}  "
            f"regret {halving['regret']:+.4f} "
            f"({halving['regret_normalised']:.1%} of the spread)"
        )
        print(
            f"                   found the best: {halving['found_true_best']}   "
            f"discarded-then-strong: {halving['discarded_then_strong']}"
        )
        print(
            f"    cost ratio rung0/max = {result['cost_ratio_low_to_high']:.3f} "
            f"(C1: lower is cheaper)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/fidelity"))
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()

    if not args.analyse_only:
        pin(args.threads)

    import pandas as pd

    from budget_tune.benchmark import evaluate
    from budget_tune.companion import ensure_all_importable, revisions
    from budget_tune.data import catalogues
    from budget_tune.fidelity import LADDERS
    from budget_tune.measure.threads import apply_to_torch, verify
    from budget_tune.space.grids import DATA_FRACTIONS

    ensure_all_importable()
    from green_rerank.measure.guards import ConditionsMonitor, ExclusiveLock, preflight
    from green_rerank.measure.session import MeasurementSession, clock_quantum

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "fidelity.csv"

    if args.analyse_only:
        report = analyse(pd.read_csv(csv_path))
        (args.out / "fidelity_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print_report(report)
        return 0

    if catalogues.role(META_CATALOGUE) != "meta":
        raise SystemExit(
            f"{META_CATALOGUE} is not the meta catalogue. A fidelity schedule is a "
            "meta-parameter and must be frozen away from every headline result."
        )

    thread_report = verify(args.threads)
    thread_report["torch_threads"] = apply_to_torch(args.threads)

    with ExclusiveLock(args.out / ".measure.lock") as lock:
        checks = preflight(require_mains=True, max_busy_pct=25.0, lock=lock)
        print(f"preflight: power={checks.power_source} busy={checks.machine_busy_pct}")
        print(f"threads: {thread_report}")

        dataset = catalogues.load(META_CATALOGUE, fractions=DATA_FRACTIONS)
        session = MeasurementSession(
            label=META_CATALOGUE,
            meta={"catalogue": META_CATALOGUE, "threads": args.threads, "study": "fidelity"},
            preflight=checks,
        )

        # Warm up before the first measured window, for the reason the pilot found the hard
        # way: torch's first call would otherwise be charged to whichever rung ran first.
        from experiments.calibrate import warm_up

        warm_up(dataset, evaluate, args.k)

        monitor = ConditionsMonitor()
        monitor.start()
        started = time.perf_counter()

        rows = []
        for family, ladder in LADDERS.items():
            grid = parameter_grid(family)
            total = len(grid) * len(DATA_FRACTIONS) * len(ladder.rungs) * len(args.seeds)
            print(
                f"\n== {family}: {len(grid)} combinations x {len(DATA_FRACTIONS)} fractions "
                f"x {len(ladder.rungs)} rungs x {len(args.seeds)} seeds = {total} runs"
            )

            done = 0
            for params in grid:
                for fraction in DATA_FRACTIONS:
                    for rung in ladder.rungs:
                        for seed in args.seeds:
                            rows.append(
                                measure(dataset, family, params, rung, fraction,
                                        seed, args.k, session, evaluate)
                            )
                            done += 1
                    if done % 100 == 0:
                        print(f"   {done}/{total}")

        conditions = monitor.stop()
        elapsed = time.perf_counter() - started
        session.close()

    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)

    # The manifest is written before the analysis, deliberately. It documents the
    # *measurement* -- conditions, thread pinning, provenance -- and the first version of
    # this script wrote it afterwards, so a bug in the arithmetic destroyed the record of a
    # thirteen-minute run that had completed perfectly well. Analysis can be re-run from the
    # CSV; the conditions monitor cannot.
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "catalogue": META_CATALOGUE,
                "ladders": {f: {"rungs": lad.rungs, "keep": lad.keep}
                            for f, lad in LADDERS.items()},
                "seeds": args.seeds,
                "threads": thread_report,
                "clock_quantum_seconds": clock_quantum(),
                "preflight": checks.as_meta(),
                "conditions": conditions,
                "elapsed_seconds": elapsed,
                "rows": len(frame),
                "machine": {
                    "platform": platform.platform(),
                    "cpu_count": os.cpu_count(),
                    "python": sys.version,
                },
                "companions": revisions(),
                "split_used": "validation",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    report = analyse(frame)
    (args.out / "fidelity_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print_report(report)
    print(f"\nfinished in {elapsed / 60:.1f} min -> {args.out}")
    if conditions.get("power_source_changed") or conditions.get("throttled"):
        print("WARNING: conditions changed during the run; timings are not comparable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
