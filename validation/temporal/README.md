# Temporal audit: no training required

Run from the repository root with the existing reproduction dependencies:

```bash
python validation/temporal/run_audit.py
```

This inspects the recorded 251-column feature schema, independently reconstructs 154
historical columns on synthetic data, tests the actual notebook feature functions,
checks all available modeling metadata and measures the V1.1 meta-CV date correction.
It does not execute notebook workflows, load model checkpoints, fit models, tune,
calibrate, perform inference, or alter saved data/predictions. The default output is
`validation/temporal/results/`. All cell indices in this audit are **zero-based**.

Read [inspection.md](inspection.md) for the pre-edit inspection and
[report.md](report.md) for findings, evidence limits and the exact change list.

## Independent reconstruction against real feature exports

The local metadata has only the 259,407 target-eligible rows. It cannot substitute
for the complete historical input, which includes many matches without serve targets.
Supply the original cleaned A/B history and the pre-imputation feature export:

```bash
python validation/temporal/run_audit.py --history data/created_data/matches_1991_to_2026_cleaned.csv --saved-features data/created_data/matches_1991_2026_features_added.csv --confirm-full-history --sample-size 120
```

Use files from the same frozen run, not a fresh upstream download. `match_id` must
match across exports. All historical input rows must be included; `--confirm-full-history`
is an explicit attestation, not an automatic proof. The slow reference uses direct
historical filtering and a separate date-frozen Elo implementation. Sampling spreads
across year/surface/level strata, taking stratum boundaries before thinning. Reported
counts disclose the actual rows/values compared. Set a large sample size to include
every first/last stratum example; this is not automatically a full-dataset value audit.

Discrepancies go to an ignored row-level CSV. Agreement tests the *legacy reference-date
policy*, not whether the historical input was actually public at that time. No models
or feature matrices are replaced and no original row is dropped.

## Timestamp evidence and conservative exclusion

Keep four distinct concepts: event completion, public availability, provider recording,
and the pipeline's use cutoff. Original `tourney_date` is preserved in the evidence
ledger and never overwritten by a reconstructed timestamp.

Create private copies of the three header-only templates. They contain no invented
real-world dates. Use a single frozen-run `match_id` namespace for `observation_id` and
`query_id`. `event_key` must identify a canonical edition, joining qualifying/main-draw
aliases when appropriate. Do not include reference date in this key: inconsistent
dates must not let an edition exclude itself from the same-event check.

Evidence is per **observation and information family**: `result`, `statistics`,
`ranking`, `profile`, or `schedule`. Result-date proof is not automatically proof of
statistic or ranking availability. Each item keeps completion bounds, a defensible
public-availability upper bound, evidence kind/source/rationale and optional provider
recording/retrieval times. Accepted kinds are `verified_match`, `verified_event_end`,
and `documented_upper_bound`. These labels are human evidence assertions; the tool
does not authenticate the cited source. A scheduled event end is not automatically an
actual completion bound: delays and postponements need support.

All timestamps require a timezone. For date-only evidence, supply an explicit,
defensible end-of-day upper bound converted from the source's local timezone; do not
invent midnight completion. Provider recording/retrieval dates never stand in for
contemporaneous public availability. A documented source/reporting delay may inform an
upper bound only with its rationale recorded. No universal 7/10/14-day duration is used.

Queries require a sourced prediction timestamp and a strictly earlier cutoff. The
legacy cutoff is reference-date minus ten calendar days; the historical notebooks
do not record actual prediction times. Do not claim that later-round participants or
entry statuses were already known at the tournament's reference start. Evidence
auditing does not silently reinterpret reference dates as actual prediction times.

Contributions must enumerate actual observation/feature dependencies, including
rank/profile dependencies. Full historical lineage is currently unavailable; an empty
or partial contribution table cannot prove complete feature validity. Run:

```bash
python validation/temporal/run_audit.py --evidence private/evidence.csv --queries private/queries.csv --contributions private/contributions.csv
```

The gate excludes the match itself, its entire canonical event edition and missing
event identities. Results/statistics need both completion and public-availability
upper bounds at/before the cutoff. Other information families need their own public
availability evidence. Known completion lower bounds after prediction demonstrate a
future-result violation; a late/loose upper bound only means safety is unestablished.
Unresolved observations are denied in the audit overlay, with reasons logged. They
are **not removed from the original datasets**. This overlay has not been wired into
production feature construction: doing so would change model inputs and needs an
explicit impact analysis first.

## Preprocessing, split and OOF limitations

Outer train/validation/test reference-date groups and event IDs can be checked with
metadata. Actual training-label availability requires additional completion evidence;
date-grouped CV alone cannot establish it. Outer-training fitted imputation/encoding
is reused by internal folds, and base hyperparameters were selected before OOF fits.
The linear scalers are fold-local, but that does not make the full pipeline nested
and fold-local. Existing inner CV/stacking scores should be described accordingly.

Future correction, if a fully isolated CV estimate is needed: fit imputation/encoding
inside each fold, nest model selection where necessary, and purge training observations
whose label-availability bounds exceed the evaluation origin. This is not needed to
run the present audit, and no such fitting has been performed here.

## Outputs and rights

`results/audit.json` includes counts, limitations and code/notebook hashes;
`feature_inventory.csv` has one entry for each recorded predictor; `tests.txt` records
test outcomes. Private row-level comparison/lineage outputs are ignored by Git.
Templates and fixtures are original synthetic data; no additional source records are
redistributed. Existing historical outputs and their artefact hashes remain intact.
