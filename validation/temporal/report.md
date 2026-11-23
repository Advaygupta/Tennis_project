# Temporal audit and changes — 23 November 2026

**No model retraining, tuning, calibration or inference was performed.** Saved models,
prediction files, datasets and reported test metrics have been preserved. This is an
audit of the professor's **first verification gate: temporal leakage**. It does not
close the separate historical-scoring, paired-uncertainty or calibration gates.

The source inspected was local commit `7cfac7dceae6bbfb8f47184c6eb64af71d451156`.
GitHub refresh failed because of network/DNS access, so the latest remote state has
not been established. Changes are local and have not been pushed.

## What the pipeline does

Notebook 01 downloads ATP, qualifying/Challenger and Futures records, deduplicates
them, assigns A/B player identities, and builds histories from the cleaned full
match pool. For a query with tournament reference date **T**, its history cutoff is
**T minus ten calendar days**, inclusive. Most history functions queue observations
until their recorded tournament reference date satisfies that condition. Same-date
queries read frozen history. Elo uses date-frozen summed updates rather than an
invented sequence of individual rounds.

Notebook 02 selects valid serve targets, removes same-match outcomes/statistics from
predictors, splits complete reference-date groups, and fits imputation/encoding on
the outer training partition. Notebooks 03/05 then use date-grouped CV and OOF fits.
Final models use training plus validation, with separate held-out test predictions.

## Results and their limits

| Check | Result | What it establishes |
|---|---|---|
| Feature inventory | 251 recorded predictor columns documented, including 154 historical columns | Coverage of the schema displayed in notebook 02; saved feature matrices/manifest were not available |
| Independent reconstruction | 154 historical columns × 64 synthetic rows = 9,856 values agree within numerical tolerance | Arithmetic and implementation of the legacy reference-date policy on this fixture |
| Deterministic tests | 15 passed | Listed safeguards, audit gate behavior and defect reproduction; not real-world source availability |
| Real modeling metadata | All 259,407 rows checked | Unique row/source identities, ordered partitions, disjoint reference-date and event-ID groups |
| Real feature values independently reconstructed | 0 | Blocked: complete cleaned history and pre-imputation feature export were not available locally |
| Historical public availability verified | 0 real rows | Blocked: no completion/publication evidence and full dependency lineage supplied |
| Original rows removed / original feature values changed | 0 / 0 | This audit does not replace the original feature pipeline |

The metadata contains 181,632 training, 26,109 validation and 51,666 test matches.
Their reference-date ranges are 1990-12-31–2022-07-25, 2022-08-01–2024-07-01, and
2024-07-08–2026-06-01 respectively. These are reference dates, not verified match
completion dates. The target-eligible metadata is not the full history used to build
features, and has not been substituted for that history.

## Confirmed correction: V1.1 stacking dates

**Original behavior.** The stacking cell removes rows without OOF predictions and
resets their indices. The Ridge meta-model helper then uses the reset indices to
look dates up in the *full* training-date Series. Consequently it retrieves dates
from the beginning of training instead of dates attached to the retained OOF rows.

**Evidence.** Reconstructing the five OOF fold memberships from the saved metadata
gives 160,707 OOF rows. All 160,707 receive incorrect dates under the old lookup. The
three subsequent meta-CV boundaries each share one actual reference-date group:

| Meta fold | Old actual train end / validation start | Corrected actual train end / validation start | Shared actual dates, before → after |
|---|---|---|---|
| 1 | 2006-08-28 / 2006-08-28 | 2005-02-14 / 2005-02-21 | 1 → 0 |
| 2 | 2011-09-26 / 2011-09-26 | 2011-09-05 / 2011-09-12 | 1 → 0 |
| 3 | 2016-06-06 / 2016-06-06 | 2016-08-22 / 2016-08-29 | 1 → 0 |

**Change.** Notebook 03 now derives `stack_dates` with the same `oof_seen` mask as
the retained predictions, passes those dates explicitly to both meta-model helpers,
and checks lengths, missing dates and date ordering. This fixes date identity for
future meta-CV execution. The existing displayed outputs remain historical and are
explicitly marked as predating the source fix. No meta-model was fitted here.

**Effect on artifacts.** This correction concerns the optional V1.1 stacking
experiment, not the primary inverse-validation-MAE ensemble. The primary saved test
metrics are unchanged. If a corrected optional stacking result is needed later,
its meta-model selection and dependent outputs must be recomputed with the correct
dates; that does not by itself require retraining all base models. Further inner-CV
limitations below remain even after this date correction.

## Remaining issues: what is proved and what is uncertain

1. **Availability is unresolved, not verified by the lag.** The code observes
   tournament reference dates but has no per-result completion/publication evidence.
   Synthetic tests show that it accepts a result whose true completion is later than
   prediction when that result's tournament reference date is old enough. That proves
   the filter is insufficient in general; it does **not** prove any particular saved
   real match leaked. The ten-day cutoff has not been changed.
2. **Event exclusion is incomplete.** A synthetic event with reference dates fourteen
   days apart can feed its earlier-dated results into its own later-dated query. Most
   history functions do not check canonical event identity. No event ID has multiple
   dates in the available 259,407-row metadata, but the much larger history pool was
   unavailable, so this is not a full-history clearance. The audit evidence gate
   excludes canonical same-event contributors; the original feature generators
   remain unchanged pending a measured input-impact assessment.
3. **Inner CV/OOF has preprocessing and selection limitations.** Outer-training
   imputations/encoding are reused by internal folds. The linear model's scaler is
   fitted inside folds, but the whole preprocessing pipeline is not. Tuned base
   hyperparameters are also reused for OOF, so these are not fully nested estimates.
   This is evaluation contamination within outer training; final-test contamination
   has not been demonstrated by these observations.
4. **Training-label availability at fold/split origins is unresolved.** Disjoint
   reference-date/event-ID groups are useful, but adjacent/overlapping events can
   finish after the nominal evaluation origin. Completion/availability evidence is
   needed before claiming these boundaries are fully temporally valid.
5. **Ranking, profile and current-match metadata need separate evidence.** The
   notebook treats recorded ranks/points as pre-match values but does not retain
   ranking-release timestamps or historical publication snapshots. Later-round
   pairings, entries and profile data should not be assumed known at tournament
   reference start. An actual prediction-time convention is needed.
6. **Workload is a lagged proxy.** The 30/60-day and 5/10-week windows use tournament
   reference dates and end at the eligibility cutoff. They are not independently
   verified counts of matches played immediately before the actual match.

No real-match temporal leakage is claimed as confirmed. Confirmed items are the
meta-CV date-identity error, the code-level gaps above, and their synthetic counterexamples.

## Timestamp solution added for the audit

`availability.py` keeps the original reference date, completion lower/upper bounds,
public-availability upper bound and provider-recording timestamp separately. Evidence
is per result/statistics/ranking/profile/schedule family. The prediction time and
cutoff are separate, sourced query fields. Header-only templates document the input
contract without inventing real dates.

The gate uses verified match evidence, verified event-end evidence or a documented
conservative upper bound. It never adds a universal tournament duration. Missing
completion/publication evidence, unknown event identity and same-match/event
contributions are denied in this audit overlay, with reasons reported. A late upper
bound means safety is unestablished; a completion lower bound after prediction proves
a future result. A retrospective provider timestamp alone verifies neither case.

The gate does not automatically authenticate sources or reconstruct missing lineage.
It is separate from production features. Applying it to regenerate model inputs
would first require counting exclusions/changes and assessing affected saved outputs.
Tournament/date batching and the ten-day lag remain useful additional safeguards,
but neither replaces source evidence or event exclusion.

## Exact file changes

| Existing file | Change | Why |
|---|---|---|
| `notebooks/03_v1_1_model_training.ipynb` | Explicit original OOF dates; alignment/date checks; historical-output warning | Prevent wrong dates after OOF index reset; avoid presenting old results as rerun |
| `reproduction/notebook_manifest.json` | Updated notebook 03 hash/byte count only | Integrity checks must track the edited source; all other notebook hashes retained |
| `README.md` | Audit status and link; qualified date/scoring safeguards | Separate implementation claims from independent verification |
| `reproducibility_notes.md` | Audit command, evidence limits and stacking warning | Explain exactly what can be checked without training |
| `reproduction/README.md` | Link to the no-training temporal audit | Give reviewers a runnable verification entry point |
| `.gitignore` | Ignore private row-level temporal discrepancies/lineage decisions | Keep source-derived row-level audit records out of commits |

New files under `validation/temporal/`:

- `inspection.md`: pre-edit source inspection and phase-2 design rationale.
- `notebook_adapter.py`: extracts original feature/CV definitions without running the notebooks.
- `reference.py`: independent direct-filter/group reference implementation.
- `availability.py`: conservative per-information-family evidence gate.
- `test_temporal.py`: deterministic synthetic, invariance, gate and CV checks.
- `run_audit.py`: reproducible inventory, metadata/CV audit and optional real-data comparison/lineage checks.
- `README.md` and three CSV templates: instructions and explicit input contracts.
- `results/audit.json`, `results/feature_inventory.csv`, `results/tests.txt`: measured audit outputs.
- This report: reviewer summary and change log.

The 15 tests cover all 154 history columns; own-result/statistic exclusion;
future-result/statistic mutation; equal-date permutation; lag/window boundaries;
same-event vulnerability; multiweek future-result admission; ambiguous availability;
provider-vs-public timestamps; separate statistic/ranking evidence; aware timestamps;
same-match/event exclusion; original CV grouping; and corrected OOF date plumbing.
Some tests deliberately reproduce known legacy weaknesses. A green suite therefore
does not mean that all historical inputs have been verified.

## Evidence still needed to close this gate

Use the original frozen cleaned full A/B match history and pre-imputation feature
export from the existing run. The optional audit command can compare sampled real
feature values against the independent reconstruction without training. For actual
availability, obtain sourced completion/publication bounds, ranking vintages and
prediction-time conventions for representative records first, then expand coverage.
Preserve unresolved cases and report their number rather than filling convenient dates.

If later evidence confirms changed model inputs or unsafe original evaluation rows,
identify those rows, columns and affected predictions first. No full retraining is
being requested on the evidence currently available. Previously reported performance
remains a saved-artifact result with unresolved temporal assumptions, not a newly
independently verified predictive claim.
