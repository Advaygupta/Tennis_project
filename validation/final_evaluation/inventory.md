# Point 4 evidence inventory and minimal implementation

Inspected 24 November 2026. GitHub baseline:
`981fa3d7b23187f8b7073b1d6eadf9490023f8c8` (Point 3 publication).
The existing local checkout contains earlier staged work, which this publication
does not include or replace. Only this new audit directory is published.

| Requirement | Existing evidence | Point 4 action |
|---|---|---|
| Targets, chronology and source limitations | Root README, notebooks, `validation/temporal/real_data_results/report.md` | Refer to existing evidence; do not repeat feature reconstruction |
| Scoring and historical-rule checks | `validation/scoring/historical/report.md` and saved private sidecar | Reuse flags and 48 already-computed corrections; never execute scoring code |
| Exact primary metrics | `reproduction/expected_metrics.json`, Point 3 report | Require agreement to 1e-12 before analysis |
| Alignment and orientation | Point 3 `alignment.py` | Import unchanged; retain conditional V1 order and 35 unique-context rows |
| Input/code hashes | Point 3 `results/manifest.json` | Hash actual bytes before/after; reject mismatches before joblib loading |
| Bootstrap methodology | Point 3 `paired_bootstrap.py` | Reuse cluster labels and paired multinomial draws; aggregate cluster sums for calibration |
| Discrimination and paired score uncertainty | Point 3, 5,000 draws under three designs | Reuse existing findings; do not repeat full AUC bootstraps |
| Calibration curves, metrics and sensitivity | Incomplete | Add width/quantile bins at 5/10/20, diagnostic intercept/slope, mean bias, winner confidence |
| Calibration uncertainty | Incomplete | 5,000 event and calendar draws for primary calibration; event draws for three exploratory subsets |
| Log-loss explanation | Point 3 identified extreme errors and top-1% concentration | Reproduce that finding; add complete loss quantiles, distinct tail rankings and attribution |
| Validation probabilities for recalibration | Notebook references, no authenticated supplied match-probability pair | Inventory source text only; no calibrator fitting or inference |
| Rights and reviewer handoff | Private artifact convention exists; consolidated inventory missing | Add provenance/rights inventory, commands, explicit compact-package allowlist |

## Reused artifacts

The three original inputs are `v1_1_results.joblib`,
`v2_1_test_match_probabilities.parquet`, and `matches_v1_1_metadata.csv`.
The optional Point 2 `saved_probability_comparison.csv` supplies completion and
historical-policy flags and the separately labelled scoring counterfactual.
The recorded run additionally reuses the exact private Point 3
`aligned_comparison.csv`. The latter's SHA-256 is first recorded in Point 4;
its existence is preserved-export evidence, not independent historical lineage
authentication. See the execution-route qualification in the report.

## Remaining gaps

No saved ID-bearing V1 prediction output or authenticated export-lineage record
was found in the supplied evidence. Notebook source cell 232 saves the V1
DataFrame without match/player IDs. Rerunning metrics cannot resolve this.
No suitable authenticated, out-of-sample validation match-probability pair for
both final models was supplied. Source inspection shows final train+validation
refits; a final-model prediction on those same validation rows would be in-sample.
Earlier-stage validation predictions cannot automatically be treated as outputs
of the final refitted probability model.

The ten-day feature cutoff remains a conservative assumption: exact match
completion/publication times are incompletely verified. More conservative
cutoffs have not been evaluated because rebuilding their features and predictions
is costly and outside the no-training/no-inference scope. Inferred scoring
policies, four best-of feature conflicts and upstream rights/provenance gaps
also remain. This package reproduces downstream frozen-output diagnostics,
not every upstream construction decision.

## Rerun decision

No existing notebook, script entry point, model inference or training pipeline
needs rerunning for this audit. Existing numerical helper functions are imported;
their original CLI runners and test suites are not executed. New isolated tests
and the new evaluation runner are the only computations. Notebook JSON is read
as text, never executed. No deployable calibrator is fitted.
