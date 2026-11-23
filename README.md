# Tennis Match Prediction

This repository contains the corrected V1.1 regression-plus-Markov model and the V2.1 direct classifier used for a paired comparison on the same chronological test set.

V1.1 predicts each player's serve-point win probability and passes those estimates through an analytical tennis scoring model. V2.1 predicts the probability that Player A wins the match directly. Both approaches use the same eligible matches, features, labels and train/validation/test partitions.

## Main corrections

- Duplicate records are resolved before feature construction; ambiguous groups are retained in audit files.
- Historical features use a 10-day reference-date eligibility lag and tournament-date batches. This avoids unsupported within-day ordering but does not establish actual result/publication availability.
- Elo and other player histories are read before the current batch enters history.
- Imputation and categorical encoding are fitted on training data only.
- V1.1 uses an inverse-validation-MAE ensemble of Elastic Net, LightGBM, XGBoost and CatBoost.
- The analytical model implements standard sets and historical Grand Slam deciding-set formats; complete historical-rule verification remains a separate gate.
- V2.1 reuses the V1.1 modeling sample and exact chronological partitions.

## Data flow

```text
855,194 downloaded rows
        |
deduplication and invalid-match quarantine
        |
855,130 structurally retained matches
        |
historical feature construction
        |
259,407 matches with valid V1.1 serve targets
        |
181,632 train | 26,109 validation | 51,666 test
        |
V1.1 regression + scoring     V2.1 direct classification
        \________________ paired test evaluation ________________/
```

Rows with a missing surface are excluded before the final feature table, which is why some intermediate displays show 855,077 rows.

## Saved test outputs

These values were recomputed from the saved V1.1 and V2.1 prediction artifacts. V2.1 identifiers, oriented players and outcomes agree with the test metadata. V1.1 lacks match/player identifiers: its saved tournament/date/best-of/outcome context agrees in order, but most rows are ambiguous within that context. Full-population pairing is therefore conditional on preserved V1.1 order. The exact aggregate output is committed as `reproduction/paired_metrics.csv`; row-level artifacts remain private.

| Approach | Accuracy | ROC-AUC | Brier score | Log loss |
|---|---:|---:|---:|---:|
| V1.1 regression plus Markov | 0.68904115 | 0.75988840 | 0.20670643 | 0.63115991 |
| V2.1 equal classifier ensemble | 0.69047343 | 0.76048069 | 0.19887455 | 0.58104872 |

Both rows cover the 51,666-record chronological test population. The [independent paired evaluation](validation/paired_evaluation/report.md) reproduces all four metrics and reports match, event and calendar-block uncertainty. The small accuracy and ROC-AUC differences do not establish a material predictive advantage. Brier/log-loss differences and scoring-only corrections are interpreted separately, subject to the alignment limitation.

## Repository contents

- `notebooks/` - the five executed project notebooks in workflow order.
- `models/` - the final checkpoints currently used by the notebooks.
- `reproduction/` - lightweight notebook checks and an optional paired metric rerun.
- `validation/scoring/` - an earlier standard-set formula validation with narrower scope than the final historical-rule implementation.
- `docs/` - the correction package for Professor Neelesh S. Upadhye.
- `notebook_guide.md` - notebook inputs, outputs and reading order.
- `reproducibility_notes.md` - environment, data, runtime and evidence limitations.

## Quick verification

The [paired evaluation package](validation/paired_evaluation/README.md) uses frozen
predictions only, with 5,000 paired bootstrap replicates per design. It preserves
the original full population and reports completed-match, supported-policy and
unique-shared-context sensitivities. No existing notebook or model needs rerunning
for this audit; definitive authentication of V1.1 identity requires preserved
ID-bearing evidence rather than assigning new IDs by array position.

The [independent historical scoring audit](validation/scoring/historical/report.md)
compares all 51,666 saved Markov predictions with a server-state reference engine.
All agree under their original assigned rules. The audit separately identifies
30 Next Gen and 18 Laver Cup format mismatches, four test best-of metadata conflicts,
and the remaining historical-evidence limitations. Scoring-only counterfactuals
are reported separately; the original saved metrics above have not been replaced.
See its [reproduction instructions](validation/scoring/historical/README.md).

The [real-data reconstruction and full-history report](validation/temporal/real_data_results/report.md)
records the completed two-check run: all 36,960 values across 154 historical columns
and 240 sampled matches matched, and the scan of 855,130 historical rows found no
flagged event/identity inconsistencies. These results do not verify actual source
availability. The runnable package is in `validation/temporal/two_audit_runner/`.

The [temporal audit report](validation/temporal/report.md) documents the independent
checks, a corrected V1.1 meta-CV date-identity bug, and unresolved source-availability
assumptions. Run it without training:

```bash
python validation/temporal/run_audit.py
```

The initial audit covers 251 recorded predictor columns, 154 historical columns on
synthetic data, and the supplied modeling metadata. The later report linked above
adds real-feature reconstruction and full-history consistency checks. Real
completion/publication evidence remains unavailable, so source availability is **not
independently verified**. The edited optional stacking cells have not been rerun;
their saved displayed outputs predate the fix. Primary models/predictions are preserved.

The notebook-level check uses only the Python standard library:

```bash
python reproduction/check_package.py
```

To recompute the paired metrics, place the three files listed in `reproduction/artifacts/README.md` in that directory and run:

```bash
pip install -r reproduction/requirements.txt
python reproduction/evaluate_predictions.py
```

## Data source and licence

The notebooks download historical men's tennis records from Jeff Sackmann's [`tennis_atp`](https://github.com/JeffSackmann/tennis_atp) repository. This repository does not redistribute the raw CSVs. Review the upstream terms before redistributing raw or processed records.

## Reproducibility claim

The repository supports inspection of the methodology, verification of notebook outputs, and a completed paired metric recomputation from the saved prediction artifacts. A full from-scratch rerun is possible through the notebooks but is not one-command or bit-for-bit reproducible because the workflow uses Google Colab, Google Drive paths, a moving upstream dataset, GPU-dependent libraries and long tuning runs.

