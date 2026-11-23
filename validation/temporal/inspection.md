# Phase 1: repository inspection (before code edits)

Baseline: `7cfac7dceae6bbfb8f47184c6eb64af71d451156`. Inspected 23 November 2026.
Remote refresh failed because network/DNS access was unavailable; this is the local baseline, not a claim about the latest GitHub head.

## Architecture

Notebook 01 downloads three Sackmann CSV families, preserves source-file/row identities,
deduplicates, sorts tournament reference dates, assigns randomized A/B orientation, and
constructs matchup, Elo, recent-form, serve/return, workload and experience features.
The full pre-surface-filter history feeds most families. Notebook 02 removes same-match
outcome/statistic columns, splits complete date groups, fits imputation/encoding on the
outer training partition and exports regular/log matrices plus metadata. Notebooks 03
and 05 tune models and construct expanding-date OOF predictions; final models use train
plus validation. Test predictions are evaluated separately.

## Findings grounded in source

| ID | Finding | Evidence | Consequence |
|---|---|---|---|
| T01 | Source availability is unresolved | Notebook 01 cells 65, 74, 85, 94, 105, 127 release on `tourney_date <= current_date - 10 days`; no completion/publication fields | Ten days is a reference-date filter, not proof that results/statistics were available at the historical cutoff |
| T02 | No event-ID exclusion in most history functions | Their required fields omit event identity, or release without comparing it with the query | Inconsistent dates for the same edition can admit its own results; demonstrate with synthetic data before changing production inputs |
| T03 | Outer-training preprocessing is reused in inner CV | Notebook 02 cells 49, 53, 58, 62, 63, 81 fit before notebook 03/05 folds | Inner CV/OOF estimates are not fully fold-local; this does not establish final-test contamination |
| T04 | V1.1 meta-CV uses wrong date positions | Notebook 03 cell 164 resets OOF indices; cell 163 uses those indices into the full `train_dates` | Meta folds can split the actual OOF date group; explicit OOF dates are needed |
| T05 | Base hyperparameters were selected before OOF generation | Notebook 03 cell 164 and notebook 05 cell 156 clone tuned templates | Base fits exclude each OOF row, but hyperparameter selection is not nested OOF |
| T06 | Label availability at split/fold boundaries is untested | Date-grouped splits separate reference dates only | Overlapping events can leave training outcomes unresolved at evaluation origins |
| T07 | Workload windows use reference dates | Notebook 01 cell 105 windows end at the ten-day cutoff | These are lagged tournament-reference workload proxies, not actual preceding-30-day match counts |

Existing safeguards: positive lag; read-before-update; whole date groups; order-invariant
Elo/date-block rolling handling; train-only outer imputation/encoding; fold-local linear
model scalers; unused initial OOF rows removed; explicit forbidden same-match predictors.
These are code observations, not proof of contemporaneous source availability.

## Evidence available and missing

Available locally: five executed notebooks, final checkpoints, 259,407-row modeling
metadata and both compact test-prediction artifacts. Missing: original 855,130-row
deduplicated A/B history, pre-imputation feature export, six feature matrices,
preprocessing manifest/encoder, exact completion dates, public result/statistic release
times, ranking vintage dates, provider publication snapshots and event-end evidence.
Metadata is a target-eligible subset and MUST NOT stand in for complete historical input.

## Minimum changes proposed

Add a separate audit runner, an independently written reference reconstruction,
timestamp-evidence contract, deterministic tests and machine-readable results. Correct
only the V1.1 OOF date-identity bug in notebook 03; retain previous displayed outputs as
historical evidence with a warning. Document remaining defects instead of silently
changing production features or the ten-day policy. Preserve saved models, matrices,
predictions and source timestamps. No training or tuning is needed to run this audit.
