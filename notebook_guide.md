# Notebook guide

Run the notebooks in numerical order. They were developed in Google Colab and retain executed outputs, exploratory comparisons and Drive paths.

## 1 `01_deduplication_feature_engineering.ipynb`

Downloads and combines ATP, Challenger/qualifying and Futures records, records source provenance, resolves duplicate candidates, quarantines ambiguous and invalid matches, assigns Player A/B orientation, and creates pre-match historical features.

Important corrections include tournament-batch Elo, a 10-day historical cutoff, delayed history updates, flexible recent-form windows, revised workload features and removal of current-tournament duration variables.

The notebook writes a feature-engineered CSV and deduplication audit files in the original Drive environment.

## 2 `02_v1_1_preprocessing.ipynb`

Creates the two realized serve-point targets, filters to 259,407 eligible matches, and splits complete tournament-date groups chronologically:

| Split | Matches | Tournament-date range |
|---|---:|---|
| Train | 181,632 | 1990-12-31 to 2022-07-25 |
| Validation | 26,109 | 2022-08-01 to 2024-07-01 |
| Test | 51,666 | 2024-07-08 to 2026-06-01 |

It fits imputation and encoding on training data, saves ordinary and log-feature Parquet files, and writes `matches_v1_1_metadata.csv` plus `preprocessing_manifest.json`.

## 3 `03_v1_1_model_training.ipynb`

Develops the serve-probability regressors, compares validation performance, refits selected models on train plus validation, and evaluates V1.1.

The final ensemble uses Elastic Net, LightGBM, XGBoost and CatBoost with inverse-validation-MAE weights. Historical deciding-set rules are assigned from test metadata. The executed test output is accuracy 0.689041, ROC-AUC 0.759888, Brier score 0.206706 and log loss 0.631160.

## 4 `04_v1_1_model_analysis.ipynb`

Loads V1.1 results and performs calibration, subgroup, Elo-gap, ranking-gap, tournament-level, round, surface, experience, ablation and interpretability analyses. Small subgroups should be treated as exploratory.

## 5 `05_v2_1_model_training.ipynb`

Loads the V1.1 features and metadata, removes the current-match serve targets from predictors, and trains direct match-winner classifiers on the same splits.

The primary V2.1 model is an equal-weight CatBoost, LightGBM, XGBoost and SGD ensemble. Its executed test output on 51,666 matches is accuracy 0.69047, ROC-AUC 0.76048, Brier score 0.19887 and log loss 0.58105. It writes `v2_1_test_match_probabilities.parquet` with aligned identifiers.

## Execution notes

- Tuning can take hours and use substantial memory.
- Several cells assume a mounted Google Drive.
- Keep the test split untouched until model selection and refitting are complete.
- Preserve `dataset_row` and metadata row order; they link V1.1, V2.1 and the paired evaluation.
- Notebook outputs are evidence of recorded runs, not a substitute for a clean rerun.

