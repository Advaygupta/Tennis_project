# Tennis Match Prediction

This repository contains the corrected V1.1 regression-plus-Markov model and the V2.1 direct classifier used for a paired comparison on the same chronological test set.

V1.1 predicts each player's serve-point win probability and passes those estimates through an analytical tennis scoring model. V2.1 predicts the probability that Player A wins the match directly. Both approaches use the same eligible matches, features, labels and train/validation/test partitions.

## Main corrections

- Duplicate records are resolved before feature construction; ambiguous groups are retained in audit files.
- Historical features use a 10-day eligibility lag and tournament-date batches, avoiding unsupported within-day match ordering.
- Elo and other player histories are read before the current batch enters history.
- Imputation and categorical encoding are fitted on training data only.
- V1.1 uses an inverse-validation-MAE ensemble of Elastic Net, LightGBM, XGBoost and CatBoost.
- The analytical model supports standard sets and historical Grand Slam deciding-set formats.
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

These values are recorded in the executed notebooks. They have not yet been independently regenerated from the private Drive artifacts in this repository revision.

| Approach | Accuracy | ROC-AUC | Brier score | Log loss |
|---|---:|---:|---:|---:|
| V1.1 regression plus Markov | 0.68904 | 0.75989 | 0.20671 | 0.63116 |
| V2.1 equal classifier ensemble | 0.69047 | 0.76048 | 0.19887 | 0.58105 |

Both rows cover 51,666 held-out matches. V2.1 is slightly better on the saved metrics, but the accuracy and ROC-AUC differences are small.

## Repository contents

- `notebooks/` - the five executed project notebooks in workflow order.
- `models/` - the final checkpoints currently used by the notebooks.
- `reproduction/` - lightweight notebook checks and an optional paired metric rerun.
- `validation/scoring/` - an earlier standard-set formula validation with narrower scope than the final historical-rule implementation.
- `docs/` - the correction package for Professor Neelesh S. Upadhye.
- `notebook_guide.md` - notebook inputs, outputs and reading order.
- `reproducibility_notes.md` - environment, data, runtime and evidence limitations.

## Quick verification

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

The repository supports inspection of the full methodology, verification of the saved notebook outputs, and metric recomputation when the compact saved prediction artifacts are supplied. A full from-scratch rerun is possible through the notebooks but is not one-command or bit-for-bit reproducible because the workflow uses Google Colab, Google Drive paths, a moving upstream dataset, GPU-dependent libraries and long tuning runs.

