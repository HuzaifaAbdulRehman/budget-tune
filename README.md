# budget-tune

**Does quantum-inspired optimisation find better recommender configurations per unit of
compute spent than strong classical hyperparameter optimisation?**

The question is open, and this repository is set up to answer it in either direction. The
design is written so that "no", "only under constraints", "the surrogate is the bottleneck",
and "a brute-force enumeration makes the whole apparatus unnecessary" are all reachable
conclusions, and it pre-registers which of them the evidence would support.

> **Status: table, RQ0, H1 and RQ1 are measured.** The 5,052-cell benchmark is in
> `results/benchmark/`. Equal-cost HPO does not show a BOCS/FMQA win over TPE; see
> [`docs/report.md`](docs/report.md). [`docs/design.md`](docs/design.md) is the protocol.

## What it is

`feasible-rerank` optimises which items go in a list. `green-rerank` measures what the
pipeline around it costs. This project optimises **which model and settings produced the list
in the first place**, under a measured resource budget.

The technical core is the established bridge from black-box hyperparameter search to a QUBO:
encode a configuration as one-hot blocks, fit a quadratic surrogate over observed
(configuration, objective) pairs, and minimise the resulting binary quadratic acquisition
function. Two published surrogates are the intended methods — **BOCS** (Baptista & Poloczek, ICML
2018: horseshoe prior, Gibbs sampling, Thompson sampling) and **FMQA** (Kitai et al., *Phys.
Rev. Research* 2020: a factorization machine minimised on an Ising machine). Both are
implemented in this tree. The encoder, the QUBO layer and the equal-cost HPO loop exist;
they have been run on the enumerated table. The comparison is in [`docs/report.md`](docs/report.md).

Neither is novel, and the repository says so. What has not been done, as far as a documented
search could establish, is comparing them against strong classical HPO on recommender model
selection **at equal measured cost**.

## Two findings before the campaign has run

**Repeat interactions were silently biasing the benchmark toward the project's own
hypothesis.** The Amazon exports record the same `(user, item)` pair more than once — 18.2%
of Luxury Beauty's 5-core rows. A user's held-out item could therefore also sit in their
training history, and since serving masks seen items, that user scored zero under every
configuration. The count moved with the data fraction (503 affected users at `f=0.25`, 802 at
`f=1.0`), so **less training data left more users scorable** — an accuracy bonus for exactly
the configurations the project hypothesises should be competitive. Fixed by deduplicating
after k-core; `assemble` now refuses to return a dataset that leaks. This is latent in both
companion projects, whose leakage test runs against a synthetic catalogue where every item is
unique per user. See [`docs/design.md`](docs/design.md) §6.1a.

**The data fraction is not an energy lever for the expensive families.** Four times less
training data changes ALS and MultVAE training cost by a factor of 0.76–1.30 — that is,
not at all, and sometimes the wrong way. Both iterate over every user regardless of how many
interactions each has. It is a real lever for ItemKNN and the Markov family, whose cost is
already negligible. This is implementation-dependent and does not contradict the published
result it was meant to replicate, but it relocates the project's energy lever to `factors`,
`epochs` and family choice. The fidelity ladder is therefore epochs, not data fraction;
that decision is in [`docs/design.md`](docs/design.md) §6.4 and was checked on Gift Cards
(§7.0) before Hyperband is allowed to use it.

## Design in one page

| | |
|---|---|
| **benchmark** | exhaustive: all **471** canonical configurations measured once per catalogue, then every HPO method runs over the resulting table |
| **families** | popularity · ItemKNN · ALS · MultVAE · sequential Markov |
| **catalogues** | MovieLens 100K · Amazon Luxury Beauty · Amazon Software (headline) · Amazon Gift Cards (meta-parameter freezing only) |
| **split** | leave-two-out — validation for selection, test read once by the report |
| **cost axis** | CPU-seconds, kernel-measured, thread-pinned; wall-seconds as a robustness axis |
| **comparison** | equal cumulative CPU-seconds, never equal trial count; anytime curves over 30 seeds |
| **baselines** | grid · random · TPE (Optuna) · successive halving · Hyperband · SM²-style energy-aware SH |

Enumerating the space buys the exact optimum, the exact Pareto front, the exact feasible set
at every budget, and a brute-force oracle for the acquisition problems. It also means **a
QUBO solver is unnecessary at this scale**, which the report states rather than implies; the
solver question survives only in a scaling study on spaces too large to enumerate.

## Measurement rules

Inherited from the companion projects, each learned the hard way:

- **`codecarbon` is reported as invalid on this machine.** green-rerank measured a hardcoded
  10.000 W RAM figure supplying most of the total, and utilisation that does not track load.
  The cost unit is CPU-seconds; joules appear only as an explicitly-passed conversion, never
  as a measurement.
- **The clock starts after the probe and stops before teardown**, and scoring happens outside
  every measured window.
- **Thread pinning is mandatory.** `process_time` sums over threads, and the families do not
  parallelise alike. Pinned to one thread, CPU and wall now agree to within 1%.
- **Runs are sequential, on mains power**, under an exclusive lock, with power source and CPU
  frequency sampled throughout.
- **Readings below the 15.6 ms clock quantum are flagged, not reported as durations.**

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

pytest tests/ --strict-companion
ruff check .
python -m experiments.verify_claims

python -m experiments.calibrate --all --threads 1     # ~6 min, fits the cost model
python -m experiments.build_benchmark --all --threads 1   # ~6.5–8 h, resumable
python -m experiments.oracle_surrogate                  # RQ0, minutes
python -m experiments.frontier                          # H1 frontier
python -m experiments.run_hpo --dataset gift_cards      # freeze meta-parameters
python -m experiments.run_hpo --dataset ml100k          # then headline catalogues
python -m experiments.barrier --dataset gift_cards      # RQ3
python -m experiments.constrained --dataset ml100k      # RQ2
python -m experiments.analyse
python -m experiments.check_report
```

Both companion checkouts must be findable — `../qubo-rerank` and `../rerank-green`, or set
`BUDGET_TUNE_FEASIBLE_RERANK` / `BUDGET_TUNE_GREEN_RERANK`. They supply the metrics, solvers,
measurement session and four of the five families, and are imported rather than vendored: two
implementations of "the same" metric that disagree is the failure this family of projects is
meant to be above. Datasets are located, never copied.

## Testing

Tests assert invariants, not coverage — the target is anything that could fail *silently*.
The ones that have already earned their place:

- held-out items never appear in any training fold or any sequence, at any data fraction,
  checked on the catalogue that actually has repeat interactions rather than only on the
  synthetic one
- the test split is byte-identical to the companions' held-out item, so the three
  repositories are provably splitting the same catalogues the same way
- deduplication preserves original row order, so a catalogue with no repeats passes through
  unchanged and the tie-break does not move
- multi-fidelity rungs are nested subsets of the data above them
- smoothing changes the Markov ranking — the design originally specified additive smoothing,
  which cannot change a ranking at all and would have made 24 grid cells silent duplicates
- the Markov family matches an independent brute-force oracle to 1e-12
- no module outside `benchmark/` and `report/` names a reporting column, so search-side code
  cannot reach the test split even by accident
- thread pinning refuses to run after numpy is imported, when it would change the manifest
  and nothing else

## Licence

MIT — see [LICENSE](LICENSE).
