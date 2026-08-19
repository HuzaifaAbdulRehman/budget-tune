# Campaign history: what failed before the benchmark was measured

The enumerated benchmark took several attempts. This file records them, because the
alternative is a results directory that looks as though it was produced on the first try.

**A caveat about these counts.** The per-attempt logs were written with `Tee-Object`, which
truncates its target on every launch, so `results/campaign.log` only ever holds the most
recent run. The counts and causes below come from observations recorded while the session
was running, not from a preserved log. That the operational log destroyed its own history is
itself part of why this summary exists in a committed file.

## Summary

**Twenty-three attempts failed before the run that produced the first kept rows.** They were
not one cause; they were four, and only the last was the machine's fault rather than this
repository's.

| # | attempts | cause | rows lost | what changed |
|---|---|---|---|---|
| 1 | 1 | Ragged CSV on finalisation, plus an undetected power-state change | 1,263 | JSONL partials; per-configuration mains check |
| 2 | ~2 | Power loss detected, but a partial benchmark was published and reported success | 50 | `finalise` refuses an incomplete benchmark; exit code 2 for interruption |
| 3 | ~2 | Competing jobs from companion projects, undetected | 200 | `ContentionGuard` added |
| 4 | **19** | **Contention threshold hard-coded below the machine's own idle floor** | 0 | threshold derived from a measured idle baseline |
| 5 | 1 | Genuine contention from a competing process | 24 | none needed — the guard worked |

## The failures in detail

### 1. Ragged CSV, and a power-state change nobody was watching for

The first full attempt measured all 1,263 configurations of one catalogue in 33 minutes and
then died writing them out. Partial rows were appended to CSV with the header written once,
but families do not share hyperparameter columns, so the first flush containing ALS rows
wrote four more fields than the header declared. Every measurement had succeeded; the file
was unparseable.

Separately, the conditions monitor reported that the machine had moved from AC to battery
mid-run, with the CPU dropping from 1,696 to 1,297 MHz — Windows' Balanced plan caps the
processor at 70% on battery. Every timing in that window was inflated by ~1.3× while every
quality metric stayed byte-identical.

**Changed:** partials became JSON Lines, which has no header to disagree with. Mains power
became a per-configuration check rather than an end-of-run report.

### 2. A partial benchmark that reported success

With the mains check in place, the next attempt stopped correctly after 60 runs. It then
wrote the four schema files from the 50 flushed rows and exited zero. A results directory
holding 1% of the space, indexed by every later table, with nothing in its filenames to say
so.

**Changed:** `finalise` requires the measured row count to match the plan. Exit codes now
distinguish finished (0) from refused (1) from interrupted (2), so a wrapper cannot mistake
an interrupted campaign for a complete one.

### 3. Competing measurement jobs

Two runs were contaminated by jobs from the companion projects — `feasible-rerank`'s
`optimality.py`, and a `green-rerank` sweep launched automatically by an `experiments.when_idle`
watcher that starts work whenever the machine looks idle. A paused campaign is exactly what
looks idle.

Neither existing guard covered this: `preflight` samples machine load once at startup, so a
job launched afterwards is invisible, and the conditions monitor watches power and frequency
but never load.

**Changed:** `ContentionGuard` differences system-wide busy CPU against this process's own,
per configuration, and records the result on every row as `other_cores`.

### 4. Nineteen attempts blocked by the guard meant to protect them

The largest group, and entirely self-inflicted. The new guard's tolerance was hard-coded at
0.25 cores. This machine's idle floor is 0.15–0.87 cores — a browser and a virus scanner are
enough — so the guard fired on an empty machine. Nineteen consecutive attempts each stopped
after four configurations, none reaching its first 25-row flush, so no data was produced and
none was lost.

Two defects underneath it:

* the threshold was an invented constant rather than a property of the machine, and a second
  contradictory default in the argument parser silently overrode the one in the guard;
* `interrupt` and `dpc` were added to the busy-CPU sum, and on Windows both are already
  counted inside `system`. Verified by measurement: over three seconds on eight CPUs,
  `idle + user + system` = 24.125 CPU-seconds against the 24.0 expected, while adding
  `interrupt + dpc` overshoots to 24.172.

**Changed:** the threshold is now a five-second idle baseline measured after preflight, plus
a 0.75-core margin, both written to the manifest before any measurement. There is one source
of truth for the policy and no absolute tolerance to set. On this machine that yields a
baseline of ~0.65–0.87 cores and a threshold of ~1.4–1.62.

### 5. The guard working as intended

The run that produced the first kept rows measured 249 configurations and stopped when a
competing process reached 2.01 cores against a 1.622 threshold, sustained across five
configurations. 225 rows were flushed and kept; the 24 in the unflushed buffer were
discarded, because a row in flight when contention began cannot be distinguished from one
measured before it.

This is the intended behaviour: interruptions cost wall-clock, never data.

### 6. The 225-row remnant was discarded, not resumed

Those 225 rows sat on disk under `results/benchmark/`, gitignored so they could not be
mistaken for a finished table. They were never committed. On 18 August 2026 a one-cell
pipeline check appended a 226th row from a later source fingerprint and overwrote the
manifest, so the remnant was no longer a single-code-version prefix. The directory was
deleted. The campaign that follows starts from zero.

## What the runner now guarantees

Every kept row is measured on mains power, verified before and after its own measurement,
on a machine whose competing load is recorded in the row itself. The `other_cores` column
means the question "was this measured on a quiet machine?" is answerable from the data
rather than from a promise.

## How it ended

The campaign completed: **5,052 rows**, 471 configurations across four catalogues, three
seeds for the stochastic families and one for the deterministic ones, every row measured on
mains power and verified before and after its own measurement. No training reading fell below
the clock quantum. `results/benchmark/` holds the four schema files and the manifest.

Two things this history does not let you check, and they are stated in `docs/report.md`
rather than left for a reader to discover:

* **The manifest describes the last resume only** -- `completed_runs: 827`, 2,934 seconds.
  It is rewritten by each segment, so 83.6% of the rows have no recorded baseline, preflight
  or conditions, and the contention threshold it names governed only the final stretch.
* **Contention was recorded per row but not prevented.** `other_cores` has median 0.31 and
  maximum 6.93. The guard stops only on five consecutive violations, so isolated spikes pass.
  The median-of-three aggregation absorbs this for ALS and MultVAE; the 300 single-seed
  deterministic rows have no such protection.

Neither affects the quality columns. Both bear on the cost axis, which is the axis the
comparison is drawn on, which is why they are in the report and not only here.
