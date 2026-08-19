# budget-tune — report

Status: **enumerated table, RQ0, H1, RQ1, RQ2 and RQ3 are in the artifacts.** Numbers
below are recomputed from those files, not from memory. Absolute NDCG is not comparable
to the companion leave-one-out tables.

## Question

Can quantum-inspired combinatorial optimisation provide a useful and defensible way to
perform model selection and hyperparameter optimisation for recommender systems under a
constrained computational budget?

QUBO is the HPO optimiser. The stronger claim is not "QUBO beats grid search." A useful
finding may be that QUBO is most valuable for *expressing constraints*. That hypothesis
was tested; it can fail.

## What was measured

- Leave-two-out split with post-k-core deduplication. `assemble` refuses a leaking
  dataset.
- Canonical CASH space of **471** configurations (`d = 44`, `p = 991`). Families:
  popularity, ItemKNN, ALS, MultVAE, sequential Markov.
- Enumerated table: **5,052** per-seed rows, four catalogues, one thread, AC
  (`results/benchmark/`, source fingerprint `a0c52b4f4d5f77f4`).
- BOCS and FMQA with **brute-force acquisition** over the 471 cells (RQ1 measures
  surrogates, not QUBO solvers). Classical baselines: coarse grid, random, Optuna TPE,
  successive halving, Hyperband, SM². Equal CPU-second budgets; Gift Cards freeze
  (`n_init=20`, TPE startup 10, budget fraction 0.10), then 30 seeds on headline
  catalogues.

## Supported findings

**Repeat interactions would have biased the benchmark toward this project's own
hypothesis.** 18.2% of Luxury Beauty 5-core rows are duplicate `(user, item)` pairs.
The leak count moved with data fraction (503 users at `f=0.25`, 802 at `f=1.0`).
Deduplicating after k-core removes it.

**`data_fraction` is not a cost lever for ALS or MultVAE.** Calibration ratios
`f=1.0/f=0.25` were 0.76–1.30. On the full table (`results/h1/frontier.json`) ALS
ratios are 1.00–1.32 and MultVAE 0.94–1.02. Markov *does* scale (about 1.7–2.6×).
H1 as originally written remains contradicted for the expensive families.

**Epochs are a real cost axis (C1). C2 is strong for ALS and weak for MultVAE** on Gift
Cards (`results/fidelity/fidelity_report.json`): ALS Spearman **0.922** (93% of a 0.991
seed ceiling), top-5 overlap 0.80; MultVAE Spearman **0.587**, top-5 overlap **0.20**.

**A quadratic can point at a good CASH cell (RQ0).** Cross-validated ridge on the gated
encoding (`results/rq0/oracle_surrogate.json`) has E1 argmax regret **0.0** on all four
catalogues, with held-out R² about 0.90–0.96. The QUBO story is not dead of
misspecification at this width.

**RQ1, validation medians over 30 seeds** (`results/hpo/*_summary.csv`):

| method | Gift Cards | ML-100K | Luxury Beauty | Software |
|---|---|---|---|---|
| tpe | 0.1977 | 0.0790 | 0.3114 | 0.2354 |
| fmqa | 0.1975 | 0.0788 | 0.3104 | 0.2353 |
| grid | 0.1902 | 0.0738 | 0.3114 | 0.2273 |
| bocs | 0.1898 | 0.0776 | 0.3001 | 0.2270 |

TPE and FMQA are close. BOCS is not the winner. On Luxury Beauty the table optimum is
ItemKNN; grid and TPE both hit **0.3114**.

**The test split does not ratify every validation ranking** (`results/analyse/selected.csv`,
`report.csv` read four times). Software is the clearest inversion: grid median test
NDCG@10 **0.1342** against **0.044** for TPE/FMQA/BOCS. Validation winners can be
wrong; that is why the split exists.

**RQ2: post-filter is exact.** Slack-QUBO (linearised cost over bits, which is
misspecified) often returns a feasible cell but not the post-filter optimum
(`results/rq2/`). Scalarisation is feasible on Gift Cards and Software at the tested
τ; it is infeasible at Luxury Beauty τ = 10th percentile. When the space is enumerable,
post-filtering wins the constraint axis QUBO was supposed to own.

**RQ3: the companion cardinality barrier does not show up at d=44.** On Gift Cards,
neal and tabu samples were **one-hot feasible before repair**. Tabu recovered the same
surrogate argmax as brute force. Simulated bifurcation was unavailable under the
current companion import. Categorical SA was run in the *feasible* encoding and is
therefore not a like-for-like energy comparison with gated brute force.

**Thread pinning:** CPU/wall 0.990–0.999 in the pilot. `codecarbon` is not an energy
axis here.

## Not claimed

- BOCS or FMQA beats strong classical HPO at equal cost as a general result.
- A QUBO *solver* is necessary at 471 cells.
- Retraining on train+val. Hit-rate vs depth. Companion Phase 5/6 CI ratios.
- Transfer of the companion `P(Σx−k)²` disconnect to this encoding.

## How to regenerate

```text
python -m experiments.verify_claims
python -m experiments.check_report
python -m experiments.oracle_surrogate
python -m experiments.frontier
python -m experiments.run_hpo --dataset gift_cards
python -m experiments.run_hpo --dataset ml100k
python -m experiments.analyse
python -m experiments.barrier --dataset gift_cards
python -m experiments.constrained --dataset ml100k
```
