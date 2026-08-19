# budget-tune — report

Status: **enumerated table, RQ0, H1, RQ1, RQ2 and RQ3 are in the artifacts.** Numbers below
are recomputed from those files, not from memory. Absolute NDCG is not comparable to the
companion leave-one-out tables.

## Question

Can quantum-inspired combinatorial optimisation provide a useful and defensible way to
perform model selection and hyperparameter optimisation for recommender systems under a
constrained computational budget?

QUBO is the HPO optimiser. The stronger claim is not "QUBO beats grid search." A useful
finding may be that QUBO is most valuable for *expressing constraints*. That hypothesis was
tested; it failed on an enumerable space.

## What was measured

- Leave-two-out split with post-k-core deduplication. `assemble` refuses a leaking dataset.
- Canonical CASH space of **471** configurations (`d = 44`, `p = 991`). Families: popularity,
  ItemKNN, ALS, MultVAE, sequential Markov.
- Enumerated table: **5,052** per-seed rows, four catalogues, one thread, AC
  (`results/benchmark/`, source fingerprint `a0c52b4f4d5f77f4`, which is commit `1cb04f0`).
- BOCS and FMQA with **brute-force acquisition** over the 471 cells (RQ1 measures surrogates,
  not QUBO solvers). Classical baselines: coarse grid, random, Optuna TPE, successive halving,
  Hyperband, SM². Equal CPU-second budgets; Gift Cards freeze (`n_init=20`, TPE startup 10,
  budget fraction 0.10), then 30 seeds on the headline catalogues.

## Supported findings

**Repeat interactions would have biased the benchmark toward this project's own hypothesis.**
18.2% of Luxury Beauty 5-core rows are duplicate `(user, item)` pairs. The leak count moved
with data fraction (503 users at `f=0.25`, 802 at `f=1.0`). Deduplicating after k-core removes
it.

**`data_fraction` is not a cost lever for ALS or MultVAE.** On the full table
(`results/h1/frontier.json`) the `f=1.0 / f=0.25` cost ratios are ALS 1.00–1.32 and MultVAE
0.94–1.02. Markov does scale, 1.73–2.63. H1 as originally written remains contradicted for the
expensive families.

**Epochs are a real cost axis (C1). C2 is strong for ALS and weak for MultVAE** on Gift Cards
(`results/fidelity/fidelity_report.json`): ALS Spearman **0.922** (93% of a 0.991 seed
ceiling), top-5 overlap 0.80; MultVAE Spearman **0.587**, top-5 overlap **0.20**.

**A quadratic can represent the ranking; it is weaker at picking a cell it has not seen
(RQ0).** `results/rq0/oracle_surrogate.json` carries both numbers, and they answer different
questions:

| catalogue | held-out R² | in-sample argmax regret | held-out fold regret | share of spread | folds finding their own best |
|---|---|---|---|---|---|
| ML-100K | 0.9016 | 0.0 | 0.00168 | 2.6% | 0/5 |
| Luxury Beauty | 0.9023 | 0.0 | 0.01986 | 6.8% | 0/5 |
| Software | 0.9630 | 0.0 | 0.00017 | 0.08% | 4/5 |
| Gift Cards | 0.9603 | 0.0 | 0.00992 | 5.5% | 2/5 |

Cross-validation chooses the ridge penalty; the reported coefficients then come from a fit on
every row, so **the zero regret is in-sample** and says only that a quadratic of this width can
represent the ranking. Refitting per fold and picking among rows that fold never saw, the
surrogate finds the held-out fold's best in **6 of 20 folds** and gives up 0.08–6.8% of the
quality spread. Informative, not decisive, and materially weaker than the in-sample figure
alone suggests. The QUBO story is not dead of misspecification at this width, but nothing here
shows a surrogate can locate the optimum from tens of observations.

**RQ1, validation medians over 30 seeds** (`results/analyse/selected.csv`), shown with what
each method actually bought with its budget:

| method | Gift Cards | ML-100K | Luxury Beauty | Software | unique cells | own overhead |
|---|---|---|---|---|---|---|
| tpe | 0.1977 | 0.0790 | 0.3114 | 0.2354 | 77.0 | 3.4% |
| fmqa | 0.1975 | 0.0788 | 0.3104 | 0.2353 | 60.5 | 1.3% |
| grid | 0.1902 | 0.0738 | 0.3114 | 0.2273 | 98.5 | 1.5% |
| bocs | 0.1898 | 0.0776 | 0.3001 | 0.2270 | 40.0 | 13.1% |
| grid (interleaved) | 0.1977 | 0.0769 | 0.3114 | 0.2354 | 142.0 | 1.5% |
| successive halving | 0.1936 | 0.0776 | 0.3001 | 0.2329 | 46.5 | 0% |
| random | 0.1909 | 0.0779 | 0.3013 | 0.2337 | 46.0 | 0% |
| sm2 | 0.1913 | 0.0782 | 0.3013 | 0.2334 | 45.5 | 0% |
| hyperband | 0.1936 | 0.0770 | 0.3007 | 0.2329 | 45.0 | 0% |

TPE and FMQA are close. BOCS is not the winner. **Equal CPU-seconds does not mean equal
looks:** BOCS evaluated a median of 40 distinct cells against TPE's 77, because 13.1% of its
budget went on its own surrogate fitting against FMQA's 1.3%. That is a legitimate outcome of
a cost-based comparison — the overhead is real and is charged — but on this evidence BOCS's
deficit cannot be attributed to surrogate quality. TPE also proposed a median of 473 duplicate
configurations (charged zero), saturating a 471-cell space.

**The test split does not ratify every validation ranking, and the reason is the split rather
than the optimisers** (`results/split_bias/summary.json`, `results/analyse/selected.csv`;
`report.csv` read four times).

Software test medians: grid **0.1342**, and **every other method 0.0442–0.0445** — bocs, fmqa,
hyperband, random, sm2, successive halving and tpe alike. This is not model-based methods
overfitting a validation split: random search lands with them.

The two splits disagree about which *family* wins. On Software, validation's top five cells are
all `markov` and test's top five are all `itemknn`, while Spearman(val, test) across all 471
cells is 0.811 — the disagreement is concentrated exactly where selection happens. Gift Cards
shows the same swap (Markov on validation, ItemKNN on test); Luxury Beauty and ML-100K agree.

Markov keeps a far smaller share of its validation score than any other family on the
short-history Amazon catalogues, and loses nothing on ML-100K. Median `test/val`: Markov 0.168
on Software, 0.172 Luxury Beauty, 0.583 Gift Cards, **1.018 ML-100K**, against 0.52–0.85 for
the other families. Leave-two-out places validation one interaction after the training history
and test two after, so temporal adjacency would favour a first-order Markov model on
validation. **That is the reading these ratios fit; it is not established here** — confirming
it needs a split whose two targets are equidistant from training, which was not measured.

**Grid's apparent advantage is enumeration order, not judgement — and re-running it proves
that rather than inferring it.** The coarse grid *contains* the validation optimum on all four
catalogues and evaluates it on only one, Luxury Beauty, where it happens to sit at position 30
of 381. Grid enumerates in family-declaration order and its CPU budget expires after 96–99
candidates, always inside ALS: **it never evaluated a single MultVAE or Markov configuration on
any catalogue.** On Software the validation optimum sits at position 375 of 381.

`grid_interleaved` is the same candidate set walked round-robin across families instead, run
at the same 30 seeds and the same budgets. It is reported *beside* the original, not in place
of it — swapping a baseline after seeing its result would be choosing the number:

| | Gift Cards | ML-100K | Luxury Beauty | Software |
|---|---|---|---|---|
| grid (declaration order), validation | 0.1902 | 0.0738 | 0.3114 | 0.2273 |
| grid (interleaved), validation | **0.1977** | **0.0769** | 0.3114 | **0.2354** |
| grid (declaration order), **test** | **0.1502** | 0.0614 | 0.1685 | **0.1342** |
| grid (interleaved), **test** | 0.1239 | 0.0645 | 0.1685 | **0.0443** |

Sampling families evenly makes grid better on validation — it now ties TPE on Gift Cards and
Software — and **destroys its test advantage**, which falls to 0.0443 on Software, exactly
where every other method lands. The declaration-order grid's test result was an artifact of
never reaching the family that overfits validation. It is also the *cheaper* walk: interleaving
reaches a median of 142 distinct cells against 96–99, because the declaration order spends the
budget inside ALS. Coarse grid search is not a better method here; one particular ordering of
it accidentally avoided the trap.

**RQ2: post-filter is exact, and QUBO does not own the constraint axis.** Across 20
(catalogue, τ) cells (`results/rq2/`), post-filter is feasible and optimal in **20/20**.
Slack-QUBO — which linearises cost over bits and is therefore misspecified — matches the
post-filter optimum in **1/20** and returns an **infeasible** configuration in **3/20**.
Scalarisation ties the optimal quality in **14/20** (the same configuration in 11/20) and is
infeasible in 1/20 (Luxury Beauty, τ = 10th percentile). Where a QUBO variant appears to beat
post-filter, it did so by violating the budget; that is not a win. When the space is
enumerable, post-filtering wins the constraint axis QUBO was supposed to own.

**RQ3: the companion cardinality barrier does not show up at d=44.** On Gift Cards
(`results/rq3/gift_cards_barrier.json`), neal and tabu samples were one-hot feasible **before**
repair, and tabu recovered the same surrogate argmax as brute force. Simulated bifurcation was
unavailable under the current companion import. Categorical SA was run in the *feasible*
encoding and is therefore not a like-for-like energy comparison with gated brute force. The
penalty strength here is 0.513 against a surrogate range of order 0.1 — a ratio near 5, where
the companion's disconnect arose near 738 — so the absence may be a property of that ratio
rather than of the encoding width. One catalogue, one diagnostic.

**Thread pinning:** CPU/wall 0.990–0.999 in the pilot. `codecarbon` is not an energy axis here.

## Measurement caveats

The benchmark manifest describes the **last resume only** (`completed_runs: 827` of 5,052), so
83.6% of rows have no recorded baseline, preflight or conditions, and the contention threshold
it records (1.810 cores) governed only that segment. `frequency_sensor_responsive` is
**false**, so the constant 1,696 MHz reading is not evidence that nothing throttled. Per-row
`other_cores` has median 0.31 and maximum 6.93; within a configuration the seed measured under
the most competing load is usually the most expensive one (rank Spearman 0.804), and cells with
a contended seed show 51% cost spread against 4.2% for quiet cells. The median-of-three
aggregation absorbs this for ALS and MultVAE — only 4.0% of cells had two or more of three
seeds contended — but the **300 single-seed deterministic rows are unprotected**, and 66% of
them were measured above 0.5 cores. Quality columns are unaffected; the cost axis for
popularity, ItemKNN and Markov carries this noise.

## Not claimed

- BOCS or FMQA beats strong classical HPO at equal cost as a general result.
- A QUBO *solver* is necessary at 471 cells.
- That a surrogate can find the optimum from tens of observations.
- That coarse grid search is a better method than TPE. Its test advantage disappears once
  the same candidate set is walked in a different order.
- That the leave-two-out family bias is established rather than the best-fitting reading.
- Retraining on train+val. Hit-rate vs depth. Companion Phase 5/6 CI ratios.
- Transfer of the companion `P(Σx−k)²` disconnect to this encoding.

## How to regenerate

```text
python -m experiments.oracle_surrogate     # RQ0, both regrets
python -m experiments.frontier             # H1
python -m experiments.split_bias           # val/test family disagreement, grid reach
python -m experiments.analyse              # RQ1 selection, reads the reporting split
python -m experiments.constrained          # RQ2
python -m experiments.barrier              # RQ3
python -m experiments.verify_claims
python -m experiments.check_report
```
