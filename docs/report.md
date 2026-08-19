# budget-tune — report

Status: **methods exist; the enumerated campaign has not finished.** Numbers below are
recomputed from committed artifacts, not from memory. HPO comparisons are **not yet
tested**. This file is the write-up the design promised; it will grow when
`results/benchmark/` exists.

## Question

Can quantum-inspired combinatorial optimisation provide a useful and defensible way to
perform multi-objective model selection and hyperparameter optimisation for recommender
systems under a constrained computational budget?

QUBO is the HPO optimiser, not a thing being tuned. The stronger claim is not "QUBO beats
grid search." A useful finding may be that QUBO is most valuable for *expressing
constraints*. That hypothesis is stated so it can fail.

## What is implemented

- Leave-two-out split with post-k-core deduplication. The loader refuses a leaking
  dataset. Absolute NDCG is not comparable to the companion leave-one-out tables.
- Canonical CASH space of **471** configurations (`d = 44` binary variables, `p = 991`
  quadratic parameters). Families: popularity, ItemKNN, ALS, MultVAE, sequential Markov.
- BOCS (horseshoe Gibbs, Thompson sampling) and FMQA (rank-`K=8` factorization machine,
  Adam; Kitai et al., *Phys. Rev. Research* 2, 013319, 2020). RQ1 acquisition is brute
  force over the 471 cells so the comparison measures surrogates, not solvers.
- Classical baselines: coarse grid, random, Optuna TPE, successive halving, Hyperband,
  SM²-style energy-aware SH.
- Equal-cost loop: cumulative CPU-seconds, duplicate proposals charged zero, optimiser
  overhead in its own counters.

## Supported findings (before the campaign)

**Repeat interactions would have biased the benchmark toward this project's own
hypothesis.** 18.2% of Luxury Beauty 5-core rows are duplicate `(user, item)` pairs. A
held-out item can also sit in training; serving masks seen items, so that user scores
zero under every configuration. The count moved with data fraction (503 users at
`f=0.25`, 802 at `f=1.0`), so less training data left more users scorable. Deduplicating
after k-core removes the leak. `assemble` raises rather than returning a leaking dataset.

**`data_fraction` is not a cost lever for ALS or MultVAE in these implementations.** The
calibration pilot (`results/calibration/`, 84 configurations, four catalogues, one thread,
AC, 1,696 MHz stable) measured training-cost ratios `f=1.0 / f=0.25` of 0.76–1.30 for
those families. Cost tracks user count, not interaction count. The energy lever is
`factors`, `epochs`, and family. H1 as originally written is already contradicted.

**Epochs are a real cost axis (C1). Rank correlation at low epochs is strong for ALS and
weak for MultVAE (C2).** Gift Cards fidelity study, 792 fits, validation split only
(`results/fidelity/fidelity_report.json`):

- ALS Spearman 0.922 against a same-fidelity seed ceiling of 0.991 (93%); simulated SH
  regret 0; top-5 overlap 0.80; cost ratio 0.165.
- MultVAE Spearman 0.587 against ceiling 0.868 (68%); SH regret 0; top-5 overlap **0.20**;
  cost ratio 0.494.

Hyperband stays in the study with the MultVAE weakness declared. Companions were dirty on
that run (`a08f0a6-dirty`, `2e2c938-dirty`).

**Thread pinning makes CPU-seconds and wall-seconds agree** to 0.990–0.999 in the pilot.
`codecarbon` is not an energy axis on this machine (hardcoded 10.000 W RAM in the
companion validity study; total reported power moves ~1.06× idle to load). An earlier
companion sentence that a loaded run used less energy than idle was withdrawn and is not
repeated here.

**Additive Markov smoothing would have been a dead axis.** Interpolation is required;
tests pin that smoothing changes the ranking and that scores match an independent oracle
to 1e-12.

## Not yet tested

RQ0 (quadratic ceiling on the enumerated table), RQ1 (equal-cost HPO), RQ2 (in-optimizer
constraints vs post-filter), RQ3 (does the companion one-hot / cardinality barrier appear
at `d=44`?). A 471-cell space makes a QUBO *solver* unnecessary for RQ1; that comparison
lives in RQ3.

## Campaign

`python -m experiments.build_benchmark --all --threads 1` measures 5,052 cells. Partial
directories are refused. See `docs/campaign-history.md` for the 23 failed attempts that
preceded the runner now in the tree.
