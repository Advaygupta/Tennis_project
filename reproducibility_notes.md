# Reproducibility notes

## Supported levels

1. **Notebook verification.** `reproduction/check_package.py` checks notebook integrity, hashes, executed outputs, split sizes and headline metrics.
2. **Paired metric reproduction.** `reproduction/evaluate_predictions.py` recomputes four metrics from three compact Drive artifacts.
3. **Full rerun.** The notebooks can regenerate data, models and predictions, but this is a long Colab-oriented workflow rather than a one-command pipeline.

## Source data

The first notebook requests annual files from Jeff Sackmann's `tennis_atp` repository for 1991-2026. The original work used a moving upstream branch rather than a pinned commit, so later downloads can differ.

Raw source files and large feature tables are not redistributed. Confirm the upstream usage terms before publishing derived row-level tables.

## Original environment

The notebooks were run in Google Colab and refer to `/content/drive/MyDrive/Projects/Tennis/`. Major packages include pandas, NumPy, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP, Matplotlib, Seaborn, joblib and PyArrow.

The training environment was not locked before the experiments. CPU/GPU implementation differences, parallel training, package changes and source-data updates can produce numerical differences.

## Execution order

```text
01_deduplication_feature_engineering.ipynb
    -> feature-engineered CSV and duplicate audit
02_v1_1_preprocessing.ipynb
    -> six Parquet matrices, metadata and preprocessing manifest
03_v1_1_model_training.ipynb
    -> V1.1 checkpoints, validation predictions and v1_1_results.joblib
04_v1_1_model_analysis.ipynb
    -> subgroup, calibration and interpretation analyses
05_v2_1_model_training.ipynb
    -> V2.1 checkpoints and v2_1_test_match_probabilities.parquet
```

## Files needed for a full rerun with existing checkpoints

- `v1_1_train.parquet`, `v1_1_val.parquet`, `v1_1_test.parquet`
- `v1_1_train_log.parquet`, `v1_1_val_log.parquet`, `v1_1_test_log.parquet`
- `matches_v1_1_metadata.csv`
- `preprocessing_manifest.json`
- preprocessing artifacts referenced by notebook 02, unless the matrices are regenerated
- `v1_1_individual_model_prediction_dataframe.joblib` for the frozen V1.1 validation weights

For lightweight paired evaluation, only the three files in `reproduction/artifacts/README.md` are needed.

## Evidence distinctions

- Notebook hashes and displayed outputs can be verified immediately.
- Headline numbers remain saved execution outputs until the compact predictions are added and evaluated.
- `validation/scoring/` validates the earlier standard 6-6 seven-point-tiebreak formula, not the complete historical deciding-set mapping.
- Exact bit-for-bit retraining is not claimed.

## Sensitive and large files

Do not commit credentials, tokens, service-account files, `.env`, private Drive links, authentication output, caches, raw data or multi-gigabyte generated datasets. Use Git LFS or release assets for unusually large checkpoints and publish SHA-256 hashes.

