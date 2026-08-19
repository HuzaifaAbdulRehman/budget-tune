# budget-tune — experimental design

Status: **measured.** The 5,052-row enumerated table, RQ0, H1, RQ1, RQ2 and RQ3 are in
`results/`; eight optimisers exist under `budget_tune/optimizers/`. This document remains the
*protocol*, including the hypotheses it is capable of falsifying and a log of every revision;
`docs/report.md` carries what the artifacts actually say, and wins wherever the two differ.

This document has been through one hostile-reviewer audit (§13). Where the audit changed a
decision, the change is marked **[AUDIT]** and the superseded version is stated, because a
design document that hides its own revisions is the same failure as a results table that
hides its retractions.

---

## 1. Decisions — final, not open

| | choice |
|---|---|
| benchmark | **exhaustive / tabular.** Enumerate the whole search space once, measure it, run every HPO method over the resulting table |
| families | **popularity · itemknn · als · multvae · sequential markov** |
| GRU4Rec | **out.** Not in the search space, not in the campaign, not in the risk register. Recorded once in §12 as an optional future ablation and nowhere else |
| datasets | **MovieLens 100K · Amazon Luxury Beauty · Amazon Software** |
| surrogates | **FMQA and BOCS**, both |

Everything below is written to those five decisions. No alternative to them is discussed.

---

## 2. What already exists

| source | what is reused |
|---|---|
| `qubo-rerank` | `metrics/` (NDCG, recall, parity, Gini, DPFR), `solvers/` (neal SA, tabu, swap annealer, Simulated Bifurcation), `formulations/` (BQM assembly, penalty encodings), `experiments/paired.py` (Wilcoxon + Holm + bootstrap), Amazon/MovieLens loaders, ALS |
| `rerank-green` | `measure/` (CPU-second windows, clock quantum, preflight/battery/load guards, manifests), `families/` (Popularity, ItemKNN, ImplicitALS, MultVAE — hyperparameters already constructor arguments), `data.py`, `catalogues.py`, held-out-item NDCG |
| machine | i5-8350U, 4c/8t, 16 GB, Windows 10, mains, Python 3.10, no GPU |

`optuna` and `scikit-learn` are pinned in `pyproject.toml` so TPE is a real TPE when that
baseline is built. They are unused until the optimiser layer exists.

New code, kept to a minimum: leave-two-out splits, the benchmark builder, the sequential
Markov family, the space encoder, the two surrogates, the acquisition→QUBO layer, the
optimizers, the report.

Measured training costs on ML-100K, from `rerank-green/results/main/tables/ml100k.cost.csv`
(CPU-seconds): popularity 0.00017 · itemknn 0.195 · multvae 4.375 · als 5.781. These four
numbers set the campaign budget in §7.1.

---

## 3. Literature position

### 3.1 What was read

**BOCS** (Baptista & Poloczek, ICML 2018, arXiv 1806.08838) — read, including Appendix A:

- surrogate: second-order polynomial `f_α(x) = α₀ + Σⱼ αⱼxⱼ + Σ_{i<j} α_{ij} x_i x_j`,
  linear in `α ∈ ℝᵖ`, `p = 1 + d + C(d,2)`;
- prior: **horseshoe**, `α_k | β_k²,τ²,σ² ~ N(0, β_k²τ²σ²)`, `τ, β_k ~ C⁺(0,1)`;
- posterior: **Gibbs sampler**, `α` sampled in `O(N²p)` via Bhattacharya et al. (2016);
- acquisition: **Thompson sampling** — draw `α_t ~ P(α|X,y)`, then
  `argmax_x f_{α_t}(x) − λ𝒫(x)` with `𝒫(x) = ‖x‖₁` or `‖x‖₂²`, `λ ∈ {0, 10⁻⁴, 10⁻²}`;
- acquisition optimization: **BOCS-SDP** (semidefinite relaxation + randomized rounding,
  `O(log d)` guarantee) and **BOCS-SA** (simulated annealing, `O(d²)` per iteration);
- benchmarks: `d ∈ {10, 21, 24, 25}`, `N₀ = 20`, 100–250 iterations.

**The passage that changes this project**, Appendix A, on categorical variables:

> "We introduce `m_i` new binary variables `x_ij` with `x_ij=1` if `x_i=e_j^i` […] Note
> that `∑_j x_ij=1` for all `i∈I` since the variable takes exactly one value"
>
> "Instead, we undo the above expansion: SA operates on `d`-tuples `x` where each `x_i`
> with `i∈I` takes values in its original domain `D_i`. Then the neighborhood `N(x)` of
> any tuple `x` is given by all vectors where at most one variable differs in its
> assignment."

**BOCS already solves the one-hot problem by never penalty-encoding it.** Its acquisition SA
moves in the categorical domain, changing which value one hyperparameter takes — exactly the
"constraint-preserving move set" this project's first draft proposed to contribute. That
contribution does not exist. See §5.4 for what survives.

**FMQA** (Kitai et al., *Phys. Rev. Research* 2, 013319, 2020; arXiv 1902.06573): the full
text could not be retrieved (ar5iv redirects, the PDF did not parse). What is established
from secondary sources: a factorization machine is fitted as the surrogate, converted
directly to QUBO, minimised on an Ising machine, and the winner evaluated and appended.
**The FM model equation, rank, and initialization must be read from the paper before
implementation** — §14 step 5 is blocked on it. The reason FMQA penalty-encodes one-hot
while BOCS does not is structural: an Ising machine has a fixed single-spin move set and
cannot be given a custom neighborhood.

One-hot handling in the FMQA line (arXiv 2605.04825, read): penalty
`λ_pen = 8·max(1, ⌊max|f_BB|+0.5⌋)`, blockwise decoding that takes the first active bit or
picks one at random if none are active, and the report that no violation was observed. That
is feasibility being used as evidence of optimisation, which feasible-rerank measured to be
invalid.

### 3.2 Prior art that narrows the contribution

Found by search, and it narrows things considerably:

- **The carbon cost of recommender hyperparameter tuning is already published.**
  Wegmeth/Vente/Said/Beel (arXiv 2509.13001) contribute "an empirical quantification of the
  carbon cost of hyperparameter tuning" and find that tuning greatly increases emissions for
  small accuracy gains. Vente et al., *Towards Sustainability-aware Recommender Systems*
  (RecSys 2023). A 2025 benchmark, *Balancing carbon footprint and algorithm performance in
  recommender systems* (ScienceDirect S2210537925002070).
- **Data reduction as a green lever is already published**: arXiv 2410.09359 reports 30%
  downsampling cutting runtime 52%.
- **AutoML for recommender systems exists**: arXiv 2402.04453, plus a WSDM'23 tutorial.
- **Quantum-inspired HPO exists** in other domains: QIBONN (arXiv 2511.08940), quantum-
  inspired high-dimensional HPO (IEEE 2017), formulation-level auto-tuning for QUBO-based ML
  (arXiv 2607.18774).
- **QUBO in recommenders is feature/carousel selection**, not HPO: CQFS (arXiv 2110.05089),
  Ferrari Dacrema et al. RecSys 2021, PDQUBO (ACM TORS 2025), and FM-recommender *inference*
  on an annealer (arXiv 2210.12953).
- **FMQA has already been applied to HPO** in another domain (Xiao et al., *IJNAMG* 2024).

### 3.3 What may be said, and what may not

**H1 is a replication, not a contribution.** That greener configurations exist and that data
reduction is the lever is published. This project re-measures it on three catalogues with a
kernel-measured cost unit; that is a validity check on our own benchmark, and it is
described as such.

**Not claimable:** any use of "first", "novel", "no one has done this", "quantum advantage".
Not the constraint-preserving move set (BOCS Appendix A). Not FMQA-for-HPO (Xiao 2024). Not
the carbon cost of recsys tuning (arXiv 2509.13001). Not data reduction (arXiv 2410.09359).

**Bounded claim that survives**, phrased as a negative search result rather than a fact about
the world: *searching for FMQA/BOCS/QUBO applied to recommender-system hyperparameter
optimisation, and for quantum-inspired or energy-aware HPO in recommendation, did not return
a study comparing a QUBO-surrogate optimiser against strong classical HPO on recommender
model selection at equal measured cost.* The searches run are listed in the report so a
reader can check the bound. Absence of a found paper is not absence of a paper.

---

## 4. Research questions and hypotheses

### 4.1 Questions

- **RQ0 (prerequisite).** Is the recommender HPO objective even approximately a quadratic
  function of the one-hot encoding? If the best possible quadratic surrogate cannot locate
  good configurations, every QUBO method is capped regardless of solver, and the rest of the
  project is measuring the cap.
- **RQ1.** At equal cumulative CPU-seconds, how do grid, random, TPE, successive halving,
  Hyperband, SM²-style energy-aware SH, FMQA and BOCS compare on held-out NDCG@10?
- **RQ2.** Under "maximise validation NDCG@10 subject to training cost ≤ τ and peak memory
  ≤ M", does a constraint *inside* the optimiser beat classical post-filtering, and at what
  feasible-region fraction does the answer change?
- **RQ3.** The FMQA line must penalty-encode one-hot because an Ising machine's move set is
  fixed. BOCS need not, and does not. **Does that penalty encoding cost anything measurable
  on the acquisition problems this project generates?**

### 4.2 Hypotheses

Stated as directions with mechanisms, without numeric thresholds. **[AUDIT: the previous
version attached thresholds — "≥5× cost, <15% NDCG", "≥95% of optima", "<2× spread". Those
were arbitrary and are removed. Every quantity they referenced is reported with a confidence
interval regardless of which side of any line it falls.]**

| | hypothesis | mechanism | what would falsify it |
|---|---|---|---|
| **H1** | The cost–accuracy trade-off has a knee: configurations exist that cost substantially less with statistically indistinguishable quality, and `data_fraction` moves cost more than the ordinary hyperparameters do | training cost is roughly multiplicative in data × factors × epochs | a flat or strictly monotone frontier, or ordinary hyperparameters moving cost as much as fraction does |
| **H2** | The best-possible quadratic surrogate — least squares over the *entire* enumerated table — recovers a near-optimal configuration | HPO landscapes are dominated by main effects and low-order interactions | a large regret for the oracle quadratic's argmin; then every QUBO method here is capped and RQ1's outcome is decided by misspecification, not search |
| **H3** | Classical multi-fidelity HPO (Hyperband, SM²) wins RQ1 | it exploits cheap low-fidelity signal that single-fidelity methods ignore | QUBO methods matching it at equal cost. **This is the expected outcome and reporting it plainly is the deliverable** |
| **H4** | In-optimizer constraints beat post-filtering only when the feasible region is small **and the space cannot be enumerated** | with an enumerable space, post-filtering is exact and free | no crossover in the τ sweep; then the expressiveness argument fails and the project says so |
| **H5** | The QUBO pipeline's own overhead (Gibbs sampling, FM fitting, acquisition solving) exceeds the training cost it saves | `p = 1 + d + C(d,2) = 991` parameters at `d = 44`, fitted from ~50 observations, per iteration | overhead being a small fraction of trial cost |
| **H6** | Penalty-encoded one-hot degrades single-flip samplers on the acquisition QUBOs relative to BOCS's categorical-domain SA and to brute force | the Project-1 barrier: two one-hot-feasible states are never adjacent under single flips | no measurable degradation; **a genuine negative result about how far the Project-1 finding generalises**, and entirely plausible here since `d≈44` with blocks of 2–5 is far smaller than the `n=200, k=10` instance where the barrier bit |

H2 and H5 are the two most likely ways this project ends with the QUBO losing, and both are
measured before any headline comparison is run.

---

## 5. Search space

### 5.1 Grids

`data_fraction ∈ {0.25, 0.50, 1.00}` applies to every family. **[AUDIT: was
{0.25,0.5,0.75,1.0}. Reduced to three values so that the multi-fidelity rungs are exactly
the fraction levels already in the table — see §6.4. This removes the need for any extra
measurement and removes an ambiguity, at the cost of a coarser fraction axis.]**

| family | hyperparameters | base | × fractions | deterministic? |
|---|---|---|---|---|
| `popularity` | — | 1 | 3 | yes |
| `itemknn` | topk {10,50,100,300} × shrink {0,10,100} | 12 | 36 | yes |
| `als` | factors {16,32,64,128} × epochs {5,15,30} × reg {1e-3,1e-2,1e-1} × alpha {1,10,40} | 108 | 324 | **no** (random init) |
| `multvae` | latent {32,64,128} × hidden {200,600} × epochs {10,20} × dropout {0,0.5} | 24 | 72 | **no** (init + batching) |
| `markov` | order {1,2} × smoothing {0,0.1,0.5} × recency-decay {off,on} | 12 | 36 | yes |

**Canonical configurations: 471 per dataset**, identical for all three because the space is
a property of the search, not of the catalogue.

**[AUDIT: this was 459 in the previous version, from a Markov grid recorded as "8 base
configurations, one cell dropped as degenerate". That was wrong twice over. The product of
the three axes is 12, not 8, and no cell is degenerate: the implementation backs off from
an unseen bigram to the first-order distribution and from an unseen unigram to popularity,
so every cell returns a full ranking. Implementing the family is what exposed it.]**

**[AUDIT: the same implementation work found that the originally specified *additive*
smoothing would have been a dead axis. Scoring one user uses exactly one context, so adding
a constant to every count and renormalising is a monotone transform: all three smoothing
values would have produced byte-identical rankings, and 24 of the 36 Markov rows would have
been duplicates wearing different identifiers. The family now uses Jelinek-Mercer style
interpolation between orders, which genuinely moves the ranking, and
``test_smoothing_changes_the_ranking`` pins it.]**

Binary encoding width: blocks are `family(5) + fraction(3) + topk(4) + shrink(3) +
factors(4) + als_epochs(3) + reg(3) + alpha(3) + latent(3) + hidden(2) + vae_epochs(2) +
dropout(2) + order(2) + smoothing(3) + decay(2)` → **d = 44 binary variables**, and hence
`p = 1 + 44 + C(44,2) = 991` surrogate parameters. BOCS's own benchmarks ran at `d ≤ 25`.
At the ~50 observations a realistic budget affords, that is **≈20 parameters per
observation** — the number behind H5 and the most likely reason this approach fails.

### 5.2 The sequential Markov family — new code, independently validated

The only family not inherited. A first/second-order item-transition model over the training
sequences: `P(next = j | current = i) ∝ count(i→j) + smoothing`, with optional recency
decay weighting recent transitions more heavily. It is here because it is cheap and because
it is the model class the target supervisor actually publishes on (SEMSRec, semantic-enhanced
Markov sequential e-commerce recommendation).

**It must be validated against an independent oracle before it enters the campaign**: a
brute-force transition-count implementation written directly from the definition, in the test
file, not sharing code with the model. The test asserts exact agreement of the transition
matrix and of the resulting scores on a small fixture, for both orders, both decay settings,
and every smoothing value. Additional invariants: rows sum to 1 where defined; a user whose
history is empty scores fallback (popularity) rather than NaN; the held-out and validation
items never appear in any transition count. Until those pass, the family is not in the space.

### 5.3 The CASH structure is not quadratic — RQ0, and it is a mathematical claim

`als_factors` is meaningless when `family = itemknn`. Under a flat one-hot encoding the true
objective has the gated form

```
f(x) = Σ_families x_family · g_family(hyperparameters of that family)
```

which contains products of **three or more** binary variables (`x_family · x_hp1 · x_hp2`).
A second-order polynomial cannot represent it. **The CASH objective is therefore structurally
outside the surrogate class both BOCS and FMQA use.** The best a quadratic can do is capture
family main effects and pairwise family×hyperparameter interactions.

This is checkable exactly, cheaply, and before any optimiser exists, because the table is
enumerated: fit the best least-squares quadratic to all 471 configurations, and report its
R², its argmin, and the regret of that argmin against the true optimum. Do the same
per-family (where the gating vanishes and the objective may well be near-quadratic). That is
RQ0 and it is the first experiment run after the campaign.

**[AUDIT: an unregularised least-squares fit cannot answer RQ0 under E1, and the arithmetic
says so before any data exists.]** The flat encoding has `d = 44` variables and therefore
`p = 991` quadratic parameters, against `n = 471` canonical configurations. The system is
underdetermined: the fit interpolates exactly, `R² = 1` by construction, and infinitely many
solutions achieve it with different argmins. "The best possible quadratic" would be a
vacuous ceiling — and one that flatters the surrogate approach, because it would report that
a quadratic can represent the objective perfectly when what it has done is memorise it.

RQ0 is therefore specified as **cross-validated ridge regression**, reporting held-out R² and
the regret of the argmin of a fit that never saw the configuration it selects. Three
consequences:

- The ceiling becomes a statement about generalisation rather than interpolation, which is
  what a surrogate fitted from tens of observations actually needs to do.
- The regularisation strength is itself a meta-parameter and is frozen on the meta catalogue,
  like every other.
- Per-family (E2) the problem partly disappears: ALS has `d = 16`, `p = 137` against `n = 324`
  rows, so ordinary least squares is determined there. Reporting E1 and E2 side by side
  therefore separates *misspecification* (gating is not quadratic) from *identifiability*
  (there are more parameters than points), which are different objections with different
  remedies and would otherwise be confounded.

The `p > n` observation is not merely a fitting nuisance. At a realistic optimisation budget
of 20–50 evaluations the ratio is far worse — ~20 parameters per observation — which is H5's
mechanism stated exactly.

If the oracle quadratic's argmin has large regret, then no surrogate-fitting procedure and no
QUBO solver can do better, and the honest conclusion is that the approach is capped by
misspecification — reported as the result, not worked around.

### 5.4 Encodings — two, not three

**[AUDIT: E3 (canonical-index encoding) is dropped. One-hot over 471 canonical configurations
is a single block with one parameter per configuration; a quadratic surrogate over it cannot
generalise between configurations at all and reduces to a bandit. A log/binary index encoding
is worse — the surrogate would be quadratic in meaningless bit patterns. E3 was not a third
encoding, it was a category error.]**

- **E1 flat-gated**: all 44 variables, inactive blocks ignored by the objective. `d=44`,
  `p=991`. Structurally misspecified per §5.3, and every real configuration appears many
  times over settings of inactive blocks.
- **E2 per-family**: one surrogate and one QUBO per family (`d = 16` and `p = 137` for ALS),
  with family choice made by an explicit outer mechanism (round-robin, or a bandit over
  families, declared and identical across methods). Gating disappears, so the quadratic is
  no longer structurally wrong, and identifiability improves ~7×.

**E1 and E2 are not the same optimisation problem** — different variables, different feasible
sets, different degeneracy. They are therefore compared **only end-to-end**, on validation
NDCG at equal cumulative CPU-seconds. Comparing their QUBO energies would be meaningless and
is forbidden in the analysis code, not merely discouraged.

E2 is predicted to win, with a mechanism (§5.3 and identifiability). If it does not, that
prediction was wrong and gets reported.

---

## 6. Protocol

### 6.1 Splits — leave-two-out, and the test column is locked

```
per user, time-ordered: [ ... training interactions ... ] [ v ] [ t ]
                                     train                 val   test
```

- **train** — everything but the last two interactions. All fitting uses this only.
- **val (`v`)** — second-most-recent item. Every HPO method selects on this and nothing else.
- **test (`t`)** — most recent item. Read once, by the report script.

**No retraining on train+val after selection.** The model scored on test is the model scored
on val, so the cost of the selected configuration is exactly the cost that was spent.

**[AUDIT: leakage through the benchmark artifact.]** The table holds test NDCG for all 471
configurations. That is a loaded gun. Mitigations, enforced in code rather than by discipline:

1. The artifact is written as **two files**: `search.csv` (validation NDCG, all costs, all
   resources) and `report.csv` (test NDCG). The optimizer API accepts only a `SearchView`
   object, which has no attribute exposing test columns.
2. A test asserts that no module under `budget_tune/optimizers/` or `budget_tune/surrogate/`
   imports or reads `report.csv`.
3. The number of times the test split is read is recorded in the manifest.
4. Every test-set comparison in the report is Holm-corrected across the whole family of
   comparisons made (methods × budgets × datasets), because the same test split is reused.

**A consequence to state, not hide:** leave-two-out removes one more interaction per user
than the companions' leave-one-out, so absolute NDCG here is **not comparable** to numbers in
`feasible-rerank` or `green-rerank`. Their tables and ours never appear in the same column.

### 6.1a Repeat interactions, and why the split deduplicates

**Found during step 1 verification. Resolved by Option B — deduplicate after k-core,
keeping the most recent interaction. Measured impact at the end of this section.**

The Amazon exports contain repeat interactions: the same ``(user, item)`` pair appears
more than once with different timestamps. 18.2% of Luxury Beauty's 5-core rows and 6.9% of
Software's are repeats; MovieLens and Gift Cards have essentially none. So a user's
held-out item can *also* appear earlier in that user's training history, and since
``Family.score_users`` masks seen items with ``-inf``, the target is then unreachable and
that user scores zero under every configuration.

The damage is not the constant handicap. It is that **the count depends on the data
fraction**, because retention drops the earlier repeat:

| Luxury Beauty, validation targets also present in training | f=0.25 | f=0.5 | f=1.0 |
|---|---|---|---|
| users | 503 | 723 | **802** |
| share of 3,589 | 14.0% | 20.1% | **22.3%** |

Less training data leaves *more* users scorable, so a low-fraction configuration is
measured on an easier population than a full-data one. That is a systematic accuracy bonus
for exactly the configurations H1 predicts should be competitive — the benchmark would have
manufactured evidence for this project's own preferred conclusion, and every individual
number would have looked entirely normal.

It also falsifies a promise this design makes in §6.4: that a fidelity rung is the same
measurement as the configuration above it. It is not, if the evaluation population moves
between rungs.

**This is latent in both companion projects too.** green-rerank's leakage test
(``test_held_out_item_is_not_in_the_training_matrix``) is written against its *synthetic*
dataset, where every item is unique per user, so the Amazon catalogues were never checked.
Under leave-one-out roughly 10% of Luxury Beauty users have an unreachable target, which
depresses the reported recall ceiling for a reason that is not retrieval quality.

Note the inconsistency that hid it: the binary interaction matrix **already** collapses
repeats — Luxury Beauty retains 25,554 interactions but the matrix has 21,073 non-zeros.
Repeats were visible only to the split and to the sequences, never to the matrix-based
models.

Candidate resolutions, measured rather than argued (see §12 of the step-1 report):

| | effect on Luxury Beauty | leakage after |
|---|---|---|
| **A** dedupe before k-core | 32,732 → 20,050 interactions, 3,589 → 2,028 users, 1,366 → 936 items | zero |
| **B** dedupe after k-core, keep most recent | 32,732 → 26,784 interactions, users and items **unchanged** | zero |
| **C** keep repeats, stop masking seen items | population unchanged; changes the task to "predict the next interaction including repeats" | n/a |
| **D** restrict evaluation to users unaffected at f=1.0 | drops 22% of users, non-randomly (heavy repeat buyers) | zero |

**Chosen: B.** It removes the leakage by construction, keeps the 5-core user and item
population, and merely finishes the deduplication the interaction matrix was already doing.

#### Measured impact

Deduplication runs after k-core and before the split, in `budget_tune.data.splits.deduplicate`.
Survivors are chosen on a sorted copy and then taken from the original frame **in its
original order**, so the stable tie-break is untouched and a catalogue with no repeats
passes through byte-identical — MovieLens and Gift Cards are unchanged to the row.

| | interactions removed | share | users 5-core → matrix | items 5-core → matrix | train interactions |
|---|---|---|---|---|---|
| ML-100K | 0 | 0.0% | 943 → 943 | 1,349 → 1,349 | 97,401 (unchanged) |
| Luxury Beauty | **5,948** | **18.2%** | 3,589 → **3,582** | 1,366 → **1,347** | 25,554 → **19,789** |
| Software | **856** | **6.9%** | 1,779 → **1,777** | 729 → **717** | 8,896 → **8,044** |
| Gift Cards | 7 | 0.2% | 456 → 456 | 147 → **146** | 2,048 → 2,041 |

**Held-out targets appearing in training: zero, on every catalogue, at every data
fraction.** `assemble` raises rather than returning a dataset that leaks, so this is a
property of the loader and not merely of the catalogues that happened to be tested.

Three kinds of attrition, kept as separate numbers because merging them would hide which
step caused what:

- **Deduplication removes no user and no item.** At least one row of every pair survives,
  so `users_removed` and `items_removed` are zero by construction, and asserted.
- **The split removes a few.** Seven Luxury Beauty users bought a *single* distinct item
  repeatedly; once the repeats collapse, their whole history is one interaction, which
  leave-two-out claims as the test target, leaving nothing to train on. Nineteen Luxury
  Beauty items and twelve Software items appear only as someone's held-out interaction.
  Recorded as `users_without_training_rows` / `items_without_training_rows`.
- **The validation threshold removes more.** 183 Luxury Beauty users and 4 Software users
  now hold fewer than three distinct items and so donate a test target but no validation
  target — the existing `MIN_HISTORY_FOR_VALIDATION` rule, doing what it was written for.

Evaluation populations after all three: **943 / 3,281 / 1,674 / 449** users with both
targets, against 943 / 3,494 / 1,678 / 449 before. Luxury Beauty loses 6% of its evaluation
users — and it is worth being clear that this is a *smaller* loss than the bug caused, since
802 of its users previously had an unreachable validation target and contributed nothing but
zeros while still being counted.

**The k-core property is re-checked, not re-imposed.** After deduplication 1,110 Luxury
Beauty users and 346 Software users hold fewer than five distinct items. Re-running k-core to
convergence would cascade — dropping a user changes item counts and vice versa — and would
land back at Option A's population (2,028 users, 936 items). The violation is therefore
counted and reported, and the catalogue keeps its users. Anyone who prefers the stricter
reading has the numbers to make that call.

**Comparable task definition across the fidelity ladder**, which was the point. At every
data fraction the same users are scorable against the same targets, with none masked, and
`fold.matrix.nnz == fold.n_interactions` now holds exactly — the gap between those two
counts *was* the bug, and equality detects any surviving duplicate directly.

**What it costs.** The sequential Markov family loses self-transitions, and a repeat purchase
is genuine signal in a repeat-purchase catalogue. Option C — keep repeats and stop masking
seen items — would have preserved that signal, at the price of changing the task to
"predict the next interaction including repeats" and diverging from both companion projects.
That trade is recorded here rather than settled silently.

**Consequences for the companion projects stand.** They do not deduplicate, so under
leave-one-out roughly 10% of Luxury Beauty users have an unreachable target, and their
reported recall ceiling is depressed for a reason that is not retrieval quality.

### 6.2 Cost unit — CPU-seconds, with the caveat stated on the same page

Primary unit: process CPU-seconds from `time.process_time`, via the inherited measurement
session (probe outside the window, scoring outside the window, repeats when below the
15.625 ms clock quantum, sequential runs, preflight refusing to start on battery).

`codecarbon` is recorded as a secondary channel and **reported as invalid on this machine** —
green-rerank measured a hardcoded 10.000 W RAM figure supplying 81–87% of reported total,
utilisation that does not track load, and total reported power moving only ~1.06× from idle
to a saturated CPU. An earlier sentence that a fully-loaded run used *less* energy than idle
was withdrawn there; do not repeat it. codecarbon appears in this project's report as
evidence that the backend is invalid, never as an energy measurement. The dependency is
listed so that claim stays reproducible; this repository does not yet call it.

**[AUDIT: three problems with CPU-seconds, all now handled explicitly.]**

1. **Threading.** `process_time` sums over threads, so a BLAS matmul on four cores bills four
   CPU-seconds per wall-second. MultVAE (torch) and ALS (numpy/BLAS) multi-thread; ItemKNN
   (scipy sparse) largely does not. Comparing them on CPU-seconds without control measures
   thread counts as much as work. **Fix: pin `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
   `OPENBLAS_NUM_THREADS` and `torch.set_num_threads` to a single declared value for the
   whole campaign, record it in the manifest, and assert it at run start.**
2. **Static power.** True energy is `P_static × wall + P_dynamic × cpu`. A single
   watts-per-CPU-second constant assumes `P_static = 0`, which is false. **Fix: wall-seconds
   are recorded for every window and the entire RQ1 analysis is repeated on the wall-second
   axis as a robustness check. If a conclusion flips between the two axes, it is reported as
   axis-dependent rather than as a conclusion.**
3. **Joules.** Only ever `cpu_seconds × watts_per_cpu_second` with the constant passed
   explicitly at the call site, as green-rerank already enforces. Never called a measurement.

Memory: `psutil` peak working set for the window, plus `model_bytes` (the trained model's
actual arrays). `tracemalloc` is not used anywhere, so the two can never be mixed.

### 6.3 The comparison axis, and what is charged to whom

Methods are compared at **equal cumulative CPU-seconds**, never at equal trial count.

```
spend(method, t) = Σ training cost of configurations evaluated so far   [table lookup]
                 + optimiser overhead measured live                     [this machine]
```

**[AUDIT: three fairness problems with that formula.]**

1. **Mixing a table lookup with a live measurement** is only valid if both were taken under
   the same conditions. Fix: overhead is measured under the identical preflight/guard regime,
   and the analysis asserts that the machine fingerprint and thread pinning in the HPO run's
   manifest match the benchmark's. Mismatch is an error, not a footnote.
2. **Overhead comparisons measure implementation quality.** Optuna's TPE is optimised C-backed
   Python; our Gibbs sampler will not be. Fix: overhead is reported **in its own panel**, never
   silently folded into a single curve; and two implementation-independent counters are
   reported beside it — number of surrogate fits and number of acquisition solves. The claim
   "the QUBO's overhead exceeds its saving" is made about *this implementation* and says so.
3. **Duplicate proposals.** A method re-proposing an evaluated configuration would in reality
   read its own history rather than retrain. Fix: results are cached and re-proposals are
   charged **zero** training cost — identically for every method — while still counting as a
   trial. Both counts are reported, so a method that stalls by re-proposing cannot hide it.

Primary figure: **anytime curves** — best validation NDCG so far against cumulative
CPU-seconds, median over seeds with bootstrap bands. Secondary: normalised regret against the
enumerated optimum at pre-registered budget checkpoints.

**Grid search is a coarse sub-grid**, not the whole table — otherwise "grid search" is the
benchmark itself and beats everything by construction. Definition: the factorial product of
the endpoints and midpoint of each hyperparameter, enumerated in a fixed order, declared in
the config file before the runs.

### 6.4 Multi-fidelity — the ladder is `epochs`, and that is two claims, not one

**[AUDIT: this section previously made `data_fraction` the ladder. The calibration pilot
(§7.1) measured that a quarter of the data costs the same as all of it for ALS and MultVAE,
so a fraction rung is not a cheap approximation of anything. The ladder is now `epochs`,
which the same pilot measured to be a real cost axis — exponent ≈1.0 for ALS and 0.8–1.1 for
MultVAE.]**

Successive halving allocates an increasing *resource* to surviving configurations, and
training iterations are the resource the method was designed around. But "epochs are cheaper"
justifies only half of what Hyperband needs, and the two halves must be kept apart:

| claim | status |
|---|---|
| **C1.** Epoch count is a genuine cost axis, so a low-epoch rung is cheaper than a full run | **measured true** in the pilot: ALS `epochs^1.0`, MultVAE `epochs^0.8–1.1` |
| **C2.** Validation performance at a low epoch budget is predictive enough of full-budget performance that discarding on it is better than not | **measured in §7.0: true for ALS (93% of the reproducibility ceiling), weak for MultVAE (68%, top-5 overlap 0.20).** Cheapness was never sufficient; the literature on early discarding reports the benefit is not automatic, and here it is not uniform across families either |

C1 makes Hyperband *runnable*. Only C2 makes it *work*. This design therefore treats C2 as an
empirical question to be answered before the campaign, on the meta catalogue, under the
protocol in §7.0 — and commits in advance to reporting the answer whichever way it falls.

#### The ladder, declared before it is validated

Rungs are taken from the **existing** epoch grids. No new epoch values, no new measurements:

| family | rungs (epochs) | keep fraction at each promotion | rungs |
|---|---|---|---|
| `als` | 5 → 15 → 30 | 1/3, then 1/2 | 3 |
| `multvae` | 10 → 20 | 1/2 | 2 |
| `popularity`, `itemknn`, `markov` | — | — | 1 (see below) |

The keep fractions are not a single `eta` because the grids are not geometric: ALS steps ×3
then ×2, MultVAE ×2. Forcing a uniform `eta=3` would require an epoch value of 45 that is not
in the search space and that nothing else would ever evaluate, which would mean inventing
measurements to make a baseline tidier. The ladder is therefore reported as what it is — a
heterogeneous, grid-derived schedule — and the departure from textbook Hyperband is stated
rather than smoothed over.

**The schedule is fixed here, before the validation runs.** Choosing rungs or keep fractions
after seeing which schedule performs best would be post-hoc selection of the baseline's own
hyperparameters, which is the asymmetry this project exists to avoid.

#### Promotion continues training rather than restarting

A configuration promoted from 5 to 15 epochs should not pay for 20 epochs of work. Training
for 15 epochs passes through the state at epoch 5, and both families are deterministic given
a seed, so **the quality at rung `b` reached by continuation is exactly the table row for
`epochs=b`** — no new measurement is needed and no approximation is involved.

The cost of continuation is the part that needs care. The pilot's epoch sweep gives, per
family and catalogue, a fit `cost(b) = setup + per_epoch · b`, and the incremental cost of
promoting from `b0` to `b1` is `per_epoch · (b1 − b0)`. The setup term is paid once. On
ML-100K's ALS the intercept is roughly 7% of a 15-epoch run, so charging continuation as
`cost(b1) − cost(b0)` instead would under-charge slightly; the explicit two-term fit avoids
that. A checkpoint-free implementation that restarted would be charged `cost(b0) + cost(b1)`,
and the design records that this is *not* what is being modelled.

#### Non-iterative families have no fidelity, and are not given a fake one

Popularity, ItemKNN and the Markov family have no epoch parameter. They are evaluated once,
at their only fidelity, which is full fidelity. Three consequences, and the second is the
uncomfortable one:

1. **They cost almost nothing.** All 75 non-iterative configurations together are ~0.2
   minutes of the campaign, against ~5 hours for ALS and MultVAE. Whatever is done with them
   barely moves the cost axis.
2. **At rung 0 they are systematically flattered.** A non-iterative configuration enters the
   first rung at its *best* quality while every ALS configuration is at 5 epochs and every
   MultVAE at 10. Successive halving will therefore over-promote them relative to what it
   would do on a homogeneous ladder. This is a real distortion of the method, not of the
   comparison — and on these catalogues it may not even be an error, since green-rerank found
   that almost nothing beats popularity on ML-100K.
3. **It is a known-hard case.** Multi-fidelity over a CASH space where only some algorithms
   expose a resource has no canonical answer in the literature.

Hyperband is therefore run in two declared variants, and both are reported:

- **(a) single space** — the natural form. Rung 0 samples the whole space, non-iterative
  configurations sit at their only fidelity, and the promotion statistics are reported so the
  distortion in (2) is visible rather than inferred: what fraction of each rung's survivors
  are non-iterative.
- **(b) per-family** — successive halving runs inside the ALS and MultVAE subspaces, with the
  non-iterative families evaluated separately, matching the E2 encoding. This removes the
  rung-0 distortion at the price of no longer being one search.

#### Is Hyperband still a fair baseline?

Fair, and narrower than it looks. It receives the same search space, the same budget in
CPU-seconds, the same seeds and the same validation split as every other method, and it is
allowed to exploit fidelity wherever fidelity exists — denying it that would be building the
strawman this project is supposed to be above. But its advantage is confined to the ALS and
MultVAE subspace. That subspace is ~99% of the campaign's cost, so the confinement is not
crippling; it does mean a headline of the form "multi-fidelity wins" would really be
"multi-fidelity wins on the iterative families", and it will be written that way.

#### What happened to H3

H3 predicted that classical multi-fidelity HPO wins RQ1. Its status was made conditional on
C2, and §7.0 has now measured C2:

- **C1 true, C2 true** → H3 remains testable exactly as written. **This is where ALS lands.**
- **C1 true, C2 partial** → the ladder works where the family's own ranking is stable and not
  otherwise. **This is where MultVAE lands**, and it is declared now so that a Hyperband
  result driven by ALS is not written up as a result about multi-fidelity generally.
- **C1 true, C2 false** → Hyperband is runnable but its discarding is uninformative, and it
  should degenerate toward random search. Not reached, but the branch stays documented
  because the finding may not transfer from Gift Cards to the headline catalogues.
- **Degenerate through heterogeneity** — if variant (a)'s promotions are dominated by
  non-iterative families and variant (b) is the only one that functions, that is reported as
  a limitation of multi-fidelity over CASH, not repaired by picking the flattering variant.
  Still open: it cannot be tested until the campaign exists.

No version of this permits "Hyperband underperformed, therefore QUBO is good". A baseline
that fails for a structural reason is a weaker comparison, and a weaker comparison weakens
every claim built on it.

#### The decision-variable asymmetry, restated

`data_fraction` remains a decision variable — any method may return a low-fraction
configuration — but it is no longer a fidelity. Since the pilot found it moves accuracy
substantially while barely moving cost, low fractions are expected to be Pareto-dominated
and the axis will mostly dilute the space. That expectation is recorded now so that a flat
fraction result in RQ1 is not written up as a discovery.

### 6.4a Measurement integrity during a long unattended run

Four failures on the first campaign attempts changed the runner. They are recorded because
each produced output that looked entirely normal, which is this project's characteristic
failure mode.

**Power state is verified per configuration, at both ends.** The development laptop has a
loose charger. Windows' Balanced plan caps the processor at 70% on battery — 0x46 against
0x64 — so an unnoticed disconnection runs the CPU at ~1,297 MHz instead of 1,696 and inflates
every timing by ~1.3×, while every quality metric stays byte-identical. The first attempt
discovered this from the conditions monitor *after* 33 minutes and 1,263 completed
measurements. The runner now checks mains before **and** after each configuration: a row
counts only if both checks pass, so a dip that recovers before the next configuration
discards its row rather than keeping it. The unflushed buffer is discarded too, since a row
in flight cannot be distinguished from one measured before the drop.

Consequence, and it is a good one: **an arbitrary number of interruptions cannot contaminate
the benchmark**, they only cost wall-clock. The campaign resumes from the rows already
measured.

**The power-plan cap was deliberately not raised.** Setting the battery processor cap to
100% would have removed the frequency cliff and made interrupted rows *look* comparable
without making them comparable — a machine on battery is a different power-delivery regime
whatever the cap says. Waiting for real mains power is the honest version, and it is what the
runner does.

**A partial benchmark is refused, not published.** An early version wrote the four schema
files from 50 of 5,052 rows after a power stop, and exited zero doing it: a results directory
holding 1% of the space, indexed by every later table, with nothing in its filenames to say
so. `finalise` now requires the measured row count to match the plan, and the exit code
distinguishes finished (0) from refused (1) from interrupted (2).

**Partial files are row-oriented, not CSV.** Families do not share hyperparameter columns, so
appending CSV with one header meant an ALS flush wrote four more fields than the header
declared — 1,263 successful measurements ending in an unparseable file. JSON Lines has no
header to disagree with.

**The campaign's cost figures are not directly comparable with the pilot's, and are better.**
The pilot measured each stage with a single window; the campaign uses `measure_repeated`,
repeating cheap work until it clears ~20 clock quanta and reporting the per-execution figure.
For anything above the quantum — all of ALS and MultVAE, which is where the cost model was
fitted — the two are identical and the pilot's exponents carry over unchanged. Below the
quantum the campaign's numbers are quantisation-averaged and the pilot's were tick counts.

The price is wall-clock: a ~0.9 s floor per run wherever all three stages sit below the
quantum. On Gift Cards, where nearly everything does, that turned a predicted 13.8 minutes
into 33. The floor is kept — it is what makes the cost column a measurement rather than a
tick count — and the campaign estimate is revised from the pilot's 5.2 h to **6.5–8 h of
measurement**, plus whatever the charger costs in restarts.

### 6.5 Seeds

**Benchmark table.** **[AUDIT: was one training seed. Changed.]** Stochastic families (ALS,
MultVAE) are measured at **3 training seeds**, and the objective is the median. Deterministic
families (popularity, ItemKNN, Markov) are measured once, with a test asserting that two runs
at different seeds produce byte-identical output — which is what makes measuring them once
legitimate rather than convenient. Cost columns get 3 measurement repeats for every family,
stochastic or not, because timing varies run to run even when output does not.

Rationale: with one seed the table's argmax may be a lucky draw, every method then chases
noise, and "fraction of the available improvement recovered" is anchored on a noisy optimum.
Cost of the fix is ~2 h of extra campaign (§7.1), which is affordable, and it is the
difference between a benchmark and an anecdote.

**HPO runs.** **[AUDIT: was ≥10. Changed to 30.]** At n=10 a two-sided paired Wilcoxon cannot
go below p≈0.002 and has poor power for the small effects expected between good HPO methods.
Trial costs are table lookups, so the only real cost of more seeds is optimiser overhead;
30 seeds is affordable. The pilot's observed variance is used to state achieved power in the
report rather than assumed.

---

## 7. Experiments

### 7.0 Fidelity validation — does a low-epoch ranking predict a full-epoch one?

Answers C2 from §6.4, before the campaign, and decides whether Hyperband stays in the study.

**Where.** Gift Cards, the meta catalogue, which exists precisely so that a method's own
settings can be chosen without touching a headline result. The fidelity schedule is a
meta-parameter of the baseline, so it is frozen here like every other. The cost of that
discipline is external validity — Gift Cards is 456 users and 146 items, and epoch fidelity
could behave differently on a larger catalogue. That limitation is stated rather than solved,
because solving it would mean looking at a headline catalogue first.

**What.** Every ALS configuration in the grid crossed with every rung, and the same for
MultVAE. No sampling and no selection: 108 ALS parameter combinations × {5, 15, 30} epochs
and 36 MultVAE combinations × {10, 20}, at 2 seeds. Cherry-picking is impossible because
nothing is picked.

**Read only the validation split.** Test columns are not written, not loaded and not
computed. This is a fidelity experiment on the search side of the benchmark and nothing else.

**Reported, all four decided in advance:**

1. **Spearman and Kendall rank correlation** between validation NDCG@10 at the lowest rung
   and at the highest, over the configuration set.
2. **Top-k overlap** at k = 5 and 10: how much of the full-budget top-k the low-budget
   ranking already contains.
3. **Simulated successive halving regret** — apply the declared keep fractions to the
   low-rung ranking, and report the gap between the best survivor at full budget and the true
   best at full budget. This is the decision-relevant number: rank correlation can be
   mediocre while the top of the ranking is preserved, and the converse.
4. **Discarded-then-strong count** — configurations cut at the first rung that would have
   finished in the full-budget top 10.

**Against the right yardstick.** A cross-fidelity correlation of 0.6 means nothing in
isolation. The same statistics are therefore computed *within* the maximum rung across the
two seeds, which measures how much of the ranking is reproducible at all. Cross-fidelity
agreement is judged against that ceiling, not against 1.0.

**Decision rule, fixed before the run.** If cross-fidelity agreement is close to the
same-fidelity ceiling and simulated-SH regret is small, C2 holds and Hyperband proceeds as
designed. If agreement is near zero, or if SH regret is comparable to the spread of the whole
configuration set, C2 fails and the design is revised — by reporting Hyperband as degenerate
on this space, not by searching for a schedule that rescues it.

#### Result: C2 holds for ALS, and is weak for MultVAE

792 measured fits on Gift Cards, 12.2 minutes, thread-pinned to one, AC power, 1,696 MHz
across 723 condition samples with no frequency drop and no power-source change, exclusive
lock held, validation split only. Artifacts in `results/fidelity/`. The run was executed
twice and the quality columns are **byte-identical** between them (max |ΔNDCG| = 0.0), so the
seeds are doing what they claim.

| | ALS | MultVAE |
|---|---|---|
| configurations | 108 | 36 |
| rungs | 5 → 15 → 30 | 10 → 20 |
| **C1** cost of rung 0 ÷ cost of full | **0.165** | **0.494** |
| Spearman, rung 0 vs full | **+0.922** | **+0.587** |
| *same-fidelity seed ceiling* | *+0.991* | *+0.868* |
| agreement as a share of the ceiling | **93%** | **68%** |
| Kendall, rung 0 vs full (ceiling) | +0.774 (0.923) | +0.394 (0.673) |
| top-5 overlap | 0.80 | **0.20** |
| top-10 overlap | 0.60 | 0.80 |
| simulated SH regret | **0.0000** | **0.0000** |
| found the true best | yes | yes |
| discarded-then-strong (of top 10) | 0 | 0 |

**C1 is confirmed and the continuation model with it.** ALS costs 0.289 / 0.870 / 1.753
CPU-seconds at 5 / 15 / 30 epochs — a ratio of 1 : 3.01 : 6.06 against the ideal 1 : 3 : 6.
MultVAE is 0.253 / 0.511, exactly ×2. Cost is linear in epochs to within measurement noise,
which is what `cost(b) = setup + per_epoch · b` assumed.

**C2 holds for ALS.** Agreement recovers 93% of what is reproducible at all, the intermediate
rung is nearly perfect (+0.989), and the declared schedule finds the true best configuration
having discarded nothing that would have finished in the top ten — at a sixth of the cost per
early evaluation.

**C2 is weak for MultVAE and is recorded as such.** Agreement reaches only 68% of its
ceiling, and **top-5 overlap is 0.20** — the low-budget ranking shares one of the true top
five. Top-10 overlap of 0.80 says it locates the right neighbourhood; the fine ordering at
the very top is not preserved. Its rungs also buy less: two rungs at ×2 against ALS's three
at ×6.

**Four caveats, none of which the zero-regret figure should be allowed to hide.**

1. **The schedule is generous, and zero regret partly reflects that.** Keeping 1/3 then 1/2
   of 108 ALS configurations leaves 18 survivors; keeping 1/2 of 36 MultVAE configurations
   leaves 18. A harsher `eta` would be a sharper test. The schedule was frozen before the run
   and is **not** being re-tuned now that the result is known — but the top-k overlap is the
   harsher signal available in the same data, and MultVAE fails it at k=5.
2. **Gift Cards is 456 users and 146 items.** External validity to the headline catalogues is
   assumed, not demonstrated. That is the price of freezing meta-parameters away from
   headline data, and it was the right price.
3. **ALS's seed ceiling of 0.991 makes this an unusually reproducible catalogue**, which
   flatters every agreement statistic computed on it.
4. **No ties inflate the rank statistics** — 215 distinct scores among 216 ALS rows and 72
   among 72 for MultVAE — so the correlations are not an artefact of a coarse objective.

**Verdict.** The ladder is defensible for ALS and marginal for MultVAE. Hyperband stays in
the study with the MultVAE weakness declared in advance, so that a strong Hyperband result on
the iterative families is not read as a general one. Under the §6.4 taxonomy this is
**C1 true, C2 true for ALS and partial for MultVAE**, so H3 remains testable — but its test
is now explicitly a test on the ALS subspace with MultVAE as a known-weaker case, not a claim
about multi-fidelity in general.

### 7.1 Build the benchmark — once per dataset

Exhaustive measurement of all 471 canonical configurations: validation NDCG@10, test
NDCG@10, recall@10, train CPU-seconds, serve CPU-seconds per request, peak RSS, model bytes,
exposure parity. Sequential, mains power, preflight guard, load monitor, manifest recording
both companion revisions, package versions, thread pinning and measured clock quantum.

**Estimated cost, and why it is only an estimate.** Extrapolated from green-rerank's
measured ML-100K figures under a stated cost model: ALS scales as
`epochs × factors² × interactions`, MultVAE as `epochs × hidden × interactions`, ItemKNN and
Markov as `interactions`. Stochastic families run 3 seeds, deterministic families 1, so the
campaign is 972 ALS runs, 216 MultVAE runs, and 111 cheap runs per catalogue.

| | runs | mean CPU-s | subtotal |
|---|---|---|---|
| ALS | 972 | 4.97 | ≈ 81 min |
| MultVAE | 216 | 1.3 | ≈ 5 min |
| ItemKNN · Markov · popularity | 111 | < 0.2 | < 1 min |
| scoring, outside every measured window | 1,263 | ≈ 0.7 | ≈ 15 min |
| **ML-100K total** | | | **≈ 1.7 h** |

Three headline catalogues plus the meta catalogue: **≈ 4–6 h**.

Deduplication (§6.1a) moved this down slightly on the Amazon catalogues and not at all on
ML-100K: training interactions fell 23% on Luxury Beauty and 10% on Software, and ALS and
MultVAE both scale with interaction count. ML-100K, which dominates the estimate, is
unchanged.

#### Calibration pilot — measured, and the estimate above was wrong in three ways

Run: `python -m experiments.calibrate --all --threads 1`, 6.4 minutes, 84 measured
configurations, thread-pinned to one, AC power, CPU pinned at 1,696 MHz with no throttling
and no power-source change, exclusive lock held. Artifacts in `results/calibration/`.

**1. ALS is linear in `factors`, not quadratic-to-cubic.** Measured exponents 0.75–0.95
across the four catalogues (R² 0.92–0.97). The `O(f³)` solve and `O(nnz·f²)` update are both
swamped at these sizes by the per-user Python loop in `ImplicitALS._solve`, which calls
`np.linalg.solve` once per user per half-iteration. The single largest uncertainty in the
estimate resolves in the cheap direction.

**2. Cost tracks user count, not interaction count.** Luxury Beauty is the most expensive
catalogue at 157 minutes despite holding a fifth of ML-100K's interactions, because it has
3,582 users against 943 and both expensive families iterate over users. MultVAE reads ~15
CPU-seconds there against ~4.8 on ML-100K.

**3. The multiplicative model under-predicts the extremes.** The held-out corner
(`factors=128, epochs=30, fraction=0.25`) is under-predicted for ALS on every catalogue —
ratios 0.50, 0.64, 0.80, 0.85 — and over-predicted for MultVAE. So the fitted model is an
interpolation, not a law, and the campaign figure below is a **lower bound**.

| | ML-100K | Luxury Beauty | Software | Gift Cards |
|---|---|---|---|---|
| ALS | 67.6 min | 114.1 min | 45.1 min | 11.4 min |
| MultVAE | 10.0 min | 31.5 min | 11.5 min | 1.3 min |
| serve + score | 3.9 min | 11.2 min | 4.3 min | 1.1 min |
| ItemKNN · Markov · popularity | 0.14 min | 0.05 min | 0.02 min | 0.01 min |
| **total** | **81.6 min** | **156.9 min** | **60.9 min** | **13.8 min** |

**Campaign: 5.2 h predicted, 6–8 h realistic** once the corner under-prediction is carried.
The 4–6 h band holds at its top end.

**Thread pinning works.** CPU-to-wall ratios are 0.990–0.999 on every catalogue, so the two
cost axes now measure the same thing. That validates the pinning; it also means the
wall-second robustness check promised in §6.2 can no longer independently corroborate a
conclusion, and it will be reported as a check on measurement rather than on inference.

**Below-quantum readings are real and must be handled by the campaign runner.** 28 of 84
readings — every popularity row, most ItemKNN and Markov rows — fell under four clock
quanta. The campaign must use `measure_repeated` for the cheap families; the pilot flags
such rows and excludes them from the fit rather than letting quantisation set an exponent.

#### The pilot's substantive finding: `data_fraction` is not a cost lever

Training CPU-seconds at `f=1.0` divided by the same at `f=0.25`:

| family | ML-100K | Luxury Beauty | Software | Gift Cards |
|---|---|---|---|---|
| als | 0.90 | 0.89 | 1.06 | 1.30 |
| multvae | 0.96 | 0.88 | 0.76 | 0.95 |
| itemknn | 1.86 | 1.67 | 0.50 | — |
| markov | 3.33 | 3.00 | — | 1.00 |

**Four times less data buys nothing for the two expensive families**, and several ratios sit
below one — less data costing more, within noise. The mechanism is not subtle: ALS loops
over every user and every item each half-iteration regardless of how many interactions each
has, and MultVAE batches over every user each epoch. Interaction count only affects the work
*inside* each row, which is not what dominates. It is a genuine lever for ItemKNN and Markov,
whose cost is proportional to interactions — and those are the families whose cost is already
negligible.

**This is implementation-dependent and must be reported as such.** A vectorised or compiled
ALS would be interaction-dominated. The published result this project intended to replicate
(arXiv 2410.09359: 30% downsampling, 52% runtime saved) used different implementations, and
nothing here contradicts it — what is measured is that *these* implementations, at *these*
catalogue sizes, do not behave that way.

Two consequences follow, and the second is a design problem rather than a finding.

**H1 relocates.** The energy lever in this space is `factors`, `epochs` and family choice,
not data fraction. H1 as written — "`data_fraction` moves cost more than the ordinary
hyperparameters do" — is contradicted by the pilot, before the campaign has run. It is
restated as a question about where the lever actually is, and the pilot's answer is
pre-registered here so the campaign cannot be read as having discovered it.

**The multi-fidelity ladder has no cheap rung.** §6.4 made the fidelity rungs the data
fractions, on the reasoning that a rung would then be an already-measured row. It still would
be — but it would cost the same as the configuration it is meant to cheaply approximate, so
successive halving and Hyperband would save nothing and degenerate to random search. H3 would
be falsified by construction, for an implementation reason rather than an interesting one,
and the strongest classical baseline would be a strawman this project built. **This needs a
decision before the campaign** — see §15.

Resumable: an interrupted campaign picks up the cells it has not done, with repeat as the
outermost loop so an interruption leaves one complete pass rather than three of the first
family and none of the rest — the pattern green-rerank already uses.

### 7.2 RQ0 — the oracle-surrogate ceiling (run first, before any optimiser)

Fit the best least-squares quadratic to the *entire* enumerated table under E1 and, per
family, under E2. Report R², the argmin of each fit, and the regret of that argmin against
the true optimum. This bounds every surrogate-based method here from above and costs
minutes. If the ceiling is low, §7.4's outcome is already explained and the project's
headline becomes a misspecification result.

### 7.3 RQ1 — unconstrained HPO at equal cost

grid (coarse sub-grid) · random · TPE (Optuna) · successive halving · Hyperband (both
variants) · SM²-style energy-aware SH · FMQA · BOCS. 30 seeds, three datasets, E1 and E2.

**Meta-parameters are tuned on a catalogue that carries no headline result.** Amazon Gift
Cards (147 items, already downloaded) is the meta-tuning catalogue. TPE's `n_startup_trials`,
Hyperband's `η` and minimum rung, the surrogates' priors and penalty weights, and the
initial-design size are all chosen there and then **frozen** before the three headline
catalogues are touched. This is the only defence against the designers' knowledge of the
enumerated optimum leaking into method configuration, and it applies to the classical
baselines exactly as much as to the QUBO methods.

**Acquisition is solved by brute force in RQ1.** With 471 configurations, enumeration is
exact and instant, so RQ1 measures *surrogates* at their best rather than confounding
surrogate quality with solver quality. Heuristic solvers are studied separately in §7.5.

### 7.4 RQ2 — constrained selection, τ swept

`maximise validation NDCG@10 s.t. train CPU-seconds ≤ τ and peak RSS ≤ M`, τ swept from
loose to tight. The feasible-region fraction is known exactly at every τ because the space is
enumerated. Reported per τ: each method's returned configuration, its test NDCG, and whether
it was feasible on every seed — the same shape as feasible-rerank's fairness-budget table,
including the `--` for "no configuration met the budget on every seed".

Four mechanisms: classical post-filtering · QUBO soft scalarisation (`quality − ρ·cost`) ·
QUBO hard inequality with binary slack variables (§8.3) · a budget-preserving solver that
visits only feasible states.

**Stated in advance, because it is the likeliest outcome and it runs against the project's
own hoped-for conclusion:** when the space is enumerable, post-filtering is exact and free,
while the QUBO's inequality needs slack variables, a second penalty weight, and a cost
*prediction* that can be wrong. The expressiveness argument for QUBO depends on the space
being too large to enumerate — which is precisely the regime §7.6 isolates, and precisely the
regime this benchmark is not in. If QUBO loses here it loses on the axis it was supposed to
win, and that is the finding.

### 7.5 RQ3 — does the penalty encoding cost anything?

Acquisition problems are harvested from the RQ1 runs and solved by: **brute force** (exact
reference), **BOCS-SA in the categorical domain** (the published penalty-free move set),
**penalty-encoded `neal`**, **penalty-encoded tabu** (with `timeout` pinned — its 20 ms
default makes quality hardware-dependent), **penalty-encoded Simulated Bifurcation**, and a
**block-preserving annealer** (attributed to BOCS Appendix A, not claimed).

Reported: fraction of available improvement recovered against the exact reference,
one-hot feasibility rate *before* any repair, the penalty-to-objective coefficient ratio that
predicts a barrier, and energy-versus-budget curves. Penalty weight is swept, including the
FMQA rule `λ=8·max(1,⌊max|f|+0.5⌋)`.

**Framing that must survive into the write-up:** this is a diagnostic, not a competition. The
useful question is whether a hardware-targeted penalty formulation loses something a
penalty-free classical formulation keeps — which matters because a physical annealer has no
choice. It is entirely plausible that at `d=44` with blocks of 2–5 the barrier does not
appear at all; the Project-1 instance was `n=200, k=10`, an incomparably harder cardinality
constraint. **A null result here is a real result about the scope of the earlier finding.**

### 7.6 The scaling study — where the solver question is actually open

**[AUDIT: the previous version generated synthetic surrogates with random coefficients and
had no exact reference at scale, so it would have compared heuristics to each other on
landscapes of unknown realism.]** Revised:

- Instances are built by **fitting an FM to the real benchmark** and then extending the space
  — more blocks, larger blocks — with coefficients drawn from the fitted distribution, so the
  landscape has the statistics of a real HPO surrogate rather than of white noise.
- Each instance carries a **planted optimum** by construction, so an exact reference exists
  where enumeration does not.
- Sizes span roughly 10³ to 10¹² feasible configurations (about 15 to 80 binary variables at
  block size 4).
- No model training is involved, so this is cheap.

What it tests: whether penalty-encoded single-flip samplers degrade with block count on
FM-shaped landscapes with known optima. What it does not test: HPO performance. That
distinction is stated in the figure caption, not just the text.

### 7.7 H1 — is there anything green to find?

Independent of QUBO entirely: the cost/accuracy frontier over the enumerated table, and
whether `data_fraction` moves cost more than the ordinary hyperparameters. Reported as a
**replication** of arXiv 2410.09359 and arXiv 2509.13001 on three catalogues with a
kernel-measured cost unit — and, equally importantly, as a validity check on our own
benchmark. If our numbers disagree with theirs, the first suspect is our measurement, not
their result.

---

## 8. Mathematics to derive explicitly (and test)

Everything here is checkable by an independent implementation on small instances, and each
gets a test that would fail if it were wrong.

**8.1 Sign convention.** HPO maximises NDCG; QUBO minimises. The objective handed to any
solver is `−quality` (or `+cost` where cost is minimised). One conversion point in the code,
asserted by a test that a solver's returned energy equals the recomputed objective of its
returned configuration.

**8.2 One-hot penalty and the offset.** For block `j` with values `a`:
`P·(Σ_a x_{ja} − 1)² = P·(Σ_a x_{ja} + 2Σ_{a<b} x_{ja}x_{jb} − 2Σ_a x_{ja} + 1)` using
`x² = x`, i.e. linear `−P` per variable, quadratic `+2P` per pair, and **a constant `+P` per
block**. With `J` blocks the total offset is `P·J`, and it must appear in the BQM's offset
field. Tests: energy of a feasible assignment equals the objective exactly (offset restores
it); dropping the offset is caught; the composed BQM's energy agrees with an independent
dense recomputation to floating-point tolerance.

**8.3 Inequality constraints need slack, and slack is not free.** `Σ_i c_i x_i ≤ τ` becomes
`Σ_i c_i x_i + s = τ` with `s ≥ 0` binary-expanded as `s = Σ_b 2^b y_b · δ`, penalised by
`Q·(Σ_i c_i x_i + s − τ)²`. Four costs to report rather than gloss:
`δ` introduces discretisation error in the constraint; the `y_b` add variables; changing `s`
by one unit can require flipping several bits, which is **a second barrier of the same
species as the one under study**; and `Q` is a second penalty weight interacting with `P`.
Tests on tiny instances against brute-force enumeration of the constrained optimum.

**8.4 Penalty scaling is per-iteration.** `P` must exceed the largest objective gain
available from a single flip, `max_v (|h_v| + Σ_u |J_uv|)` — the companion's
`suggest_strength`. Under Thompson sampling the surrogate coefficients change every
iteration, so `P` is recomputed every iteration, never fixed once. Tests: after scaling, no
infeasible assignment has energy below any feasible one on small instances (brute-force
checked).

**8.5 QUBO → Ising.** `x = (1+s)/2`, with the standard `h`/`J` derivation. Checked
numerically against a dense reference — the companion project had a `J/8` vs `J/4` error here
that was invisible in the output because the result was still a perfectly plausible Ising
model, merely a different one.

**8.6 Normalisation.** If a BQM is normalised, the scale factor is recorded and every
reported energy is de-normalised before comparison. Objective value, BQM energy, offset,
normalised objective and constraint penalty are five distinct quantities carried in separate
fields, never one column called "energy".

---

## 9. Statistics

**[AUDIT: HPO trajectories are sequential and correlated; testing at every point on the
anytime curve would be a multiple-comparison disaster with dependent tests.]**

- **Two units of analysis, never conflated.** Comparing *HPO methods*: the unit is the seed
  (n=30), paired because all methods see the same table and the same budget. Comparing the
  *selected configurations'* recommendation quality: the unit is the user (n≈900), paired
  because every configuration is scored on the same users.
- **Pre-registered checkpoints.** Three budget checkpoints (25%, 50%, 100% of the budget),
  declared in the config before the runs. Paired Wilcoxon at each; **Holm correction across
  the entire family** — methods × checkpoints × datasets — computed once, not per table.
- **Effect size before p-value**, with bootstrap confidence intervals on the median paired
  difference, and win/loss/tie counts. The companion's `experiments/paired.py` already does
  this and is imported rather than rewritten.
- **No cross-dataset significance test.** Three datasets cannot support Friedman/Nemenyi.
  Consistency across catalogues is described and shown, never tested.
- **Anytime curves carry bootstrap bands, not stars.** The curves are descriptive; inference
  happens only at the declared checkpoints.

---

## 10. Ablations

- **surrogate × solver.** Cross surrogate quality (10/25/50/100 observations) with solver
  quality (brute force / block-preserving / penalty `neal`). If brute-force acquisition does
  not beat `neal` acquisition at low observation counts, the surrogate is the bottleneck and
  solver work is worthless — the most likely failure mode, and directly measurable because
  the oracle exists.
- **E1 vs E2**, end-to-end only (§5.4).
- **FMQA vs BOCS** — low-rank pairwise against full pairwise with a horseshoe prior, at
  identical budgets. Their difference is the surrogate-parameterisation question, which
  H2 and H5 both bear on.
- **penalty weight sweep**, feasibility rate and recovered-optimum fraction reported
  separately, never merged.
- **initial-design ablation**: random versus one-hot marginal-coverage-guaranteeing designs
  (the problem raised in arXiv 2605.04825).
- **objective noise**: the 3-seed spread on every stochastic configuration, reported against
  the between-configuration differences methods are choosing among. If seed noise is of the
  same order, that caps what any method can distinguish and must be shown next to the RQ1
  results, not in an appendix.

---

## 11. Reporting thresholds — not failure criteria

**[AUDIT: the previous version listed numeric "failure criteria" — ≥95%, <2×, 10 trials,
100% overhead. Those numbers were invented. Pre-declaring an arbitrary line and then reporting
which side of it a result fell on manufactures a verdict.]**

Replaced by a pre-registered **reporting commitment**: each quantity below is reported with a
confidence interval whatever its value, and each is given equal prominence whether it supports
or undermines the approach.

| quantity | reported as |
|---|---|
| cost and accuracy spread across the enumerated benchmark | frontier plot + range, per dataset |
| trials for random search to reach the optimum | distribution over 30 seeds, not a threshold |
| oracle quadratic's regret (RQ0) | value + CI, E1 and E2 |
| recovered-optimum fraction per solver (RQ3) | value + CI, per penalty weight |
| optimiser overhead as a share of trial cost (H5) | its own panel, with implementation caveat |
| τ at which any crossover occurs (RQ2) | the whole sweep, including "none" |
| seed noise vs between-configuration differences | side by side with RQ1 |

The write-up commits in advance to leading with whichever of these is largest in magnitude,
including when that is a result against the QUBO.

---

## 12. Risks

**[AUDIT: GRU4Rec removed from this register per the decision in §1. Recorded once here as an
optional future ablation: if the RQ1 anytime curves come out flat between multi-fidelity and
single-fidelity methods, one candidate explanation is that no configuration in the space is
expensive enough for early stopping to matter, and adding an expensive family would test that.
It is future work and is not in this project's plan, campaign estimate, or budget.]**

| risk | severity | handling |
|---|---|---|
| `p=991` parameters from ~50 observations (§5.1) | **high** | it is H5 and RQ0, measured, not mitigated |
| CASH gating is not quadratic (§5.3) | **high** | RQ0 quantifies the ceiling before optimisers are built |
| Enumerable space makes the QUBO solver unnecessary | **high** for claim strength | stated in the abstract, not buried; the solver question lives only in §7.6 |
| The space discretises hyperparameters, which QUBO requires but continuous BO does not | medium | **this handicaps classical BO and therefore favours the QUBO.** Stated as a limitation; conclusions are scoped to discrete spaces |
| Test-split reuse across methods, datasets and analyses | medium | split artifact, import test, read counter, Holm across the family (§6.1) |
| Overhead comparison measures implementation quality | medium | separate panel + implementation-independent counters (§6.3) |
| Sequential Markov is new code | medium | brute-force transition-count oracle in tests before it enters the campaign (§5.2) |
| Benchmark noise from 3 seeds may still understate variance | low–medium | reported against between-configuration differences (§10) |
| FMQA's published details not yet read | **blocking** | §14 step 5 does not start until the paper is read |

---

## 13. What the audit changed

Fourteen substantive changes, listed so the diff is reviewable rather than buried:

1. BOCS already uses a feasibility-preserving categorical move set — the "constraint-preserving
   solver" contribution is withdrawn and attributed (§3.1, §7.5).
2. RQ3 is reframed from "does the barrier transfer" (a competition) to "does the penalty
   encoding a physical annealer *requires* cost anything" (a diagnostic), with a null result
   declared valuable in advance.
3. **RQ0 added**: the CASH objective is not a quadratic function of the one-hot encoding, and
   the oracle-quadratic ceiling is measured before any optimiser is built (§5.3, §7.2).
4. E3 dropped as a category error (§5.4).
5. `data_fraction` reduced to three levels so fidelity rungs are existing table rows (§5.1, §6.4).
6. Benchmark seeds: 1 → 3 for stochastic families, with determinism tests for the rest (§6.5).
7. HPO seeds: 10 → 30 (§6.5).
8. Test column split into a separate artifact with an import test and a read counter (§6.1).
9. Thread pinning mandated; wall-second axis added as a robustness check; static power named
   as the reason a single joule constant is wrong (§6.2).
10. Optimiser overhead put in its own panel with implementation-independent counters, and a
    duplicate-proposal caching policy declared identically for all methods (§6.3).
11. Meta-parameters for **all** methods frozen on a separate catalogue (§7.3).
12. Grid search defined as a coarse sub-grid rather than the table (§6.3).
13. Numeric failure criteria replaced by reporting commitments (§11).
14. Scaling study rebuilt around FM-fitted landscapes with planted optima (§7.6).

Two acknowledged limitations that have no fix within this project, and are therefore stated
rather than solved: the benchmark's discretisation favours the QUBO over continuous BO, and a
tabular benchmark with a fixed seed set is a smoother, less noisy problem than live HPO, which
flatters every model-based method including ours.

---

## 14. Implementation order

1. **Splits and benchmark schema.** Leave-two-out on the companion primitives. Tests: val and
   test items never in the training matrix; agreement with the companions' leave-one-out on
   shared interactions; users with too little history excluded explicitly, not silently.
2. **Sequential Markov family + its brute-force oracle tests** (§5.2). Blocked from the
   campaign until they pass.
3. **Campaign** — 471 configurations × 3 datasets, 3 seeds for stochastic families, thread-pinned,
   guarded, manifested, resumable. ≈4 h.
4. **RQ0 oracle-surrogate ceiling** and **H1 frontier**. Both are cheap, both are independent
   of QUBO, and either can invalidate the rest — so they run before any optimiser exists.
5. **Read the FMQA paper**, then build the space encoder, both surrogates, and the
   acquisition→QUBO layer with brute-force acquisition and the §8 tests wired in from the
   first commit.
6. **Classical optimizers** (grid, random, TPE, SH, Hyperband ×2, SM²) and the cumulative-cost
   axis. Built before the QUBO comparison so the bar cannot be set to flatter it.
7. **RQ1** at 30 seeds, meta-parameters frozen on Gift Cards first.
8. **RQ3** barrier diagnostic; **RQ2** constrained sweep including the slack derivation.
9. **Scaling study**, remaining ablations, report, and a `verify_claims`-style script that
   regenerates every table and figure from saved artifacts.

## 15. The fidelity decision, and what it turned on

**Resolved in form: the ladder is `epochs` (§6.4). Conditional in substance: whether that
ladder is any use is C2, answered by the §7.0 validation.**

The pilot removed `data_fraction` as a candidate — a quarter of the data costs the same as
all of it for ALS and MultVAE, so a rung would have been a full-price approximation of a full
run. Two alternatives were considered and rejected:

- **Redefine `data_fraction` to subsample training users.** It would restore the cost lever,
  since both expensive families loop over users. Rejected because it changes the split
  semantics for a third time, discards the sequence-adjacency argument that motivated recency
  retention, and needs care that evaluation users keep a profile — a lot of moving parts to
  rescue an axis that `epochs` already provides.
- **Accept that this space has no cheap fidelity** and report SH and Hyperband as degenerate.
  Rejected as premature: it would discard the strongest classical baseline on the strength of
  one axis failing, when the axis the method was actually designed around had not been tried.

**What was explicitly *not* accepted as sufficient: that epochs are cheaper.** That is C1, it
is measured, and on its own it justifies nothing. A resource that is cheap but uninformative
makes early discarding worse than not discarding, and the recent literature on early
discarding is clear that the benefit is not automatic. The design therefore separates C1 from
C2 (§6.4), declares the schedule before testing it (§6.4), and tests it on the meta catalogue
with a decision rule fixed in advance (§7.0).

**No outcome here is allowed to help the QUBO.** If Hyperband degenerates, the comparison is
weaker and every claim resting on it is weaker with it. A baseline that fails structurally is
not evidence for the alternative.

## 16. Repository structure

```
budget_tune/
├── space/        one-hot codec, canonicalisation, E1/E2 encodings
├── surrogate/    BOCS (horseshoe + Gibbs + Thompson), FMQA (factorization machine)
├── qubo/         acquisition -> BQM, one-hot penalty, slack encoding, offsets, Ising checks
├── solvers/      brute force, companion SA/tabu/SB wrappers, block-preserving annealer
├── optimizers/   grid, random, tpe, successive_halving, hyperband, sm2, fmqa, bocs
├── benchmark/    leave-two-out split, campaign builder, SearchView / report split
└── report/       anytime curves, regret, frontier, tables
experiments/      build_benchmark · oracle_surrogate · run_hpo · constrained · barrier
                  · scaling · ablations · analyse · figures · tables · verify_claims
configs/          YAML per dataset and per study
results/<study>/  search.csv · report.csv · manifest.json · tables/ · figures/
tests/            invariants + a mutation suite in green-rerank's style
docs/             design.md · report.md · findings.md
```
