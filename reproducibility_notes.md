# Reproducibility Notes

## Scope

This repository preserves the original Version 1 research workflow and its executed notebook outputs. It supports inspection of the data-processing logic, feature construction, targets, model comparisons, analytical formulas, and reported analyses.

It is not currently a one-command reproduction package. The original work was developed in Google Colab and used intermediate datasets, trained models, and prediction tables stored in the author's Google Drive.

## What is reproducible from the preserved code

Subject to availability of the upstream data and compatible package versions, the notebooks document how to:

1. Download the source CSV files.
2. Combine and chronologically organize the matches.
3. Convert the records from winner/loser to Player A/Player B orientation.
4. build the historical feature table;
5. construct the two serve-proportion targets;
6. create chronological train, validation, and test partitions;
7. preprocess categorical and numerical variables;
8. train the constituent regressors;
9. construct and compare ensembles;
10. convert serve estimates into match probabilities; and
11. calculate the principal evaluation metrics and analyses.

The standalone analytical formulas can be inspected without the machine-learning artifacts. Full numerical reproduction of the reported model results requires regenerating the data and models or recovering the original intermediate artifacts.

## Original execution environment

The main modelling notebooks were run in Google Colab. They contain paths resembling:

```text
/content/drive/MyDrive/Projects/Tennis/...
```

These are references to the original author's Google Drive and will not resolve in another account. A new user must replace them with local or mounted paths.

The workflow was developed with Python and uses packages including:

- pandas;
- NumPy;
- scikit-learn;
- XGBoost;
- LightGBM;
- CatBoost;
- Optuna;
- SHAP;
- Matplotlib;
- Seaborn;
- joblib; and
- PyArrow or another Parquet engine.



## Source data

The download notebook requests annual files from:

```text
https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/
```

for these file families:

```text
atp_matches_<year>.csv
atp_matches_futures_<year>.csv
atp_matches_qual_chall_<year>.csv
```

and years 1991 through 2026.

The source was downloaded from the moving `master` branch rather than a pinned commit. Upstream corrections or additions can therefore cause future reconstructions to differ from the preserved experiment. The repository does not redistribute Jeff Sackmann's complete dataset; users should retrieve it from the upstream project and follow its usage terms.

## Expected execution order

The intended order is:

```text
01_data_collection_feature_engineering.ipynb
    -> matches_1991_2026_combined.csv
    -> matches_1991_2026_features_added.csv

02_v1_data_preparation.ipynb
    -> v1_train.parquet
    -> v1_val.parquet
    -> v1_test.parquet
    -> corresponding log-transformed tables

03_v1_model_training.ipynb
    -> fitted constituent models
    -> validation prediction table
    -> ensemble artifacts
    -> test predictions and headline metrics

04_v1_model_analysis.ipynb
    -> ablation and subgroup analyses
```

The notebooks should not be executed in arbitrary order because later stages expect files produced by earlier stages.

## Data sizes and storage

The raw combined table contains approximately 855,000 rows. Feature engineering expands this to roughly 200 pre-encoding variables. After categorical encoding, the Version 1 matrices contain approximately 247 predictor columns.

These artifacts can require several gigabytes across CSV, Parquet, model, prediction, and interpretation files. They should not be committed directly to Git unless deliberately managed with Git LFS. A `data/README.md` or artifact manifest is preferable for documenting them.

## Runtime expectations

Runtime depends substantially on hardware, package versions, and whether GPU acceleration is available.

- Downloading and combining data is comparatively inexpensive.
- Sequential feature engineering over the complete history can be time-consuming.
- Broad hyperparameter searches and Optuna tuning may take several hours.
- Refitting boosted-tree models and running ablation experiments can also take several hours.
- SHAP and permutation analyses can be computationally and memory intensive.

The repository retains executed outputs so that a reviewer can inspect the original experiment without repeating every expensive stage.

## Randomness

The Player A/Player B assignment uses NumPy seed `42`. Several models and samples also use `random_state=42` where supported.

Complete determinism is not guaranteed because:

- not every model call records an explicit seed;
- parallel tree training can vary across library versions and hardware;
- GPU and CPU implementations may not produce identical floating-point results;
- hyperparameter-search behaviour can vary if package versions change; and
- the upstream source files can change.

Small numerical differences should therefore be expected even after the workflow has been reconstructed successfully.

## Preserved versus missing artifacts

The notebooks contain many original outputs, including validation scores, test metrics, figures, and tables. Some external artifacts may not be present in the repository, including:

- intermediate CSV and Parquet datasets;
- fitted model files stored in Google Drive;
- ensemble coefficient files;
- large SHAP arrays;
- permutation-importance tables; and
- selected exported figures.

The absence of an external artifact does not remove the corresponding preserved notebook output, but it may prevent that section from being rerun independently.

The repository should include an artifact manifest before release, identifying which files are included, which can be regenerated, and which are unavailable.

## Interpretation of reported results

The preserved notebooks report an average serve-target test MAE of approximately `0.06715` and final match-level metrics of:

```text
Accuracy:    0.6958
Log loss:   0.6156
Brier score: 0.2032
ROC-AUC:    0.7666
```

These should be treated as preliminary out-of-sample results from the original workflow. They remain subject to minor changes after the final review of preprocessing consistency, chronological handling, analytical scoring, and repository reconstruction.

## Early Markov prototypes

The repository may include two early standalone files:

- an analytical formula prototype; and
- an exploratory Monte Carlo simulator.

They document how the scoring component developed but are not substitutes for a formal test suite. The Monte Carlo prototype was not used to produce the reported results. A future verification should compare a corrected, seeded, high-trial simulator against the analytical output over a grid of serve-probability pairs.

## Recommended verification checks

A future reproducibility pass should include lightweight tests that do not require retraining every model:

- service-game boundary values at probabilities 0, 0.5, and 1;
- set- and match-probability symmetry after swapping players;
- monotonicity as either player's serve probability changes;
- agreement between analytical and high-trial simulated probabilities;
- valid probability bounds;
- chronological separation of the three partitions;
- confirmation that the current match does not update its own historical features;
- fit preprocessing transformations on training data and reuse them unchanged on later data; and
- recomputation of metrics from a saved table of test labels and probabilities.

These checks would materially improve portability and confidence without repeating the full exploratory hyperparameter search.

## Reproduction standard claimed by this repository

The present repository claims **methodological transparency and partial computational reproducibility**, not turnkey or bit-for-bit reproduction.

Specifically:

- the data source, target definitions, feature logic, chronological design, models, formulas, and metrics are documented;
- original executed outputs are preserved for inspection;
- the order required to regenerate the major artifacts is described; but
- exact reproduction may require unavailable intermediate files, environment reconstruction, path changes, and several hours of computation.

This distinction should also be stated briefly when the repository is shared.

