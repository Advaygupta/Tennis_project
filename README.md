# Tennis Match Prediction with Machine Learning and an Analytical Scoring Model

This repository documents an ongoing tennis-modelling project that combines statistical learning with an analytical representation of tennis scoring. Machine-learning regressors estimate each player's probability of winning a point on serve, and those estimates are propagated through game, set, and match calculations to obtain a pre-match win probability.

The project uses historical men's tennis data maintained by [Jeff Sackmann](https://github.com/JeffSackmann/tennis_atp). His public `tennis_atp` repository is the sole match-data source used in the principal Version 1 experiment.

## Project status

This is an archival research repository for an ongoing project. The code was developed iteratively in Google Colab, with intermediate datasets and trained models stored in Google Drive. The executed notebooks preserve the original workflow and experimental outputs, but the project was not initially designed as a portable software package.

The reported figures should therefore be read as preliminary out-of-sample results. They may change slightly after the final reproducibility and consistency review.

## Version 1 framework

Version 1 separates the prediction problem into two stages:

```text
Historical match data
        |
Pre-match feature engineering
        |
Serve-point probability models
        |  p: Player A wins a point on serve
        |  q: Player B wins a point on serve
        |
Analytical tennis-scoring model
        |
Match-winning probability
```

The statistical-learning stage uses ranking, Elo, recent form, matchup, fatigue, serve, return, surface, experience, physical, seeding, and tournament-context features. The analytical stage converts the estimated serve-point probabilities into game-, set-, and match-winning probabilities using the rules of tennis scoring.

## Data and prediction targets

The source-data collection covers annual ATP, Challenger/qualifying, and Futures files from 1991 through 2026. The combined collection contained 855,194 records at the time of the experiment.

Version 1 requires detailed serve statistics for both players. Its regression targets are the realized proportions of service points won in the target match:

```text
player serve-point proportion
    = (first-serve points won + second-serve points won)
      / total service points played
```

After requiring valid targets for both players, 259,423 matches remained:

| Partition | Matches | Approximate period |
|---|---:|---|
| Training | 181,596 | Through July 2022 |
| Validation | 25,942 | July 2022 to July 2024 |
| Test | 51,885 | July 2024 to June 2026 |

The partitions were kept in chronological order. Model and ensemble selection used the training and validation periods, while the final period was reserved for out-of-sample evaluation.

## Preliminary results

The selected serve-probability model was an equal-weight ensemble of Ridge, LightGBM, XGBoost, and CatBoost regressors. Its average test MAE for the two serve targets was approximately `0.06715`.

After propagation through the analytical scoring model, the preliminary test results were:

| Metric | Result |
|---|---:|
| Accuracy | 0.6958 |
| Log loss | 0.6156 |
| Brier score | 0.2032 |
| ROC-AUC | 0.7666 |

These results indicate predictive information beyond a constant 50-50 forecast. They do not establish state-of-the-art performance or superiority to every possible tennis-specific baseline.

## Repository guide

The principal materials are intended to be read in this order:

1. `README.md` - project purpose, status, data provenance, and headline results.
2. `docs/methodology.md` - targets, feature construction, modelling, analytical formulation, evaluation, assumptions, and future work.
3. `docs/notebook_guide.md` - the purpose and expected reading order of the original notebooks.
4. `docs/reproducibility_notes.md` - environment-specific dependencies, expected inputs and outputs, runtime considerations, and the level of reproducibility currently supported.
5. `notebooks/` - original executed notebooks for the Version 1 workflow.
6. `markov/` - early standalone analytical and Monte Carlo prototypes retained for methodological history.

Version 2, which predicts the match winner directly as a classification target, is not part of the principal submission. It represents a separate modelling direction and is not required to understand the Version 1 ML-plus-scoring framework.

## Interpreting the notebooks

The notebooks contain exploratory cells, saved Google Drive paths, model-development experiments, and preserved outputs. They are research records rather than polished library modules. Readers interested primarily in the final experiment should follow the file map in `docs/notebook_guide.md` instead of executing isolated cells out of order.

The original notebook outputs have intentionally been retained. Full retraining can take several hours, and those outputs provide a record of the validation comparisons, test metrics, calibration analysis, interpretability work, ablation experiments, and error analyses produced during development.

## Main assumptions

The analytical model assumes that:

- each player's serve-point probability remains constant during a match;
- point outcomes are conditionally independent given those probabilities;
- the same set-winning probability can be propagated to the match level;
- in-match injuries, weather changes, momentum, tactical adaptation, and changing fatigue are not modelled dynamically; and
- the prediction uses only information intended to be available before the match.

These assumptions make the framework interpretable and computationally tractable, but they also define the limits of the conclusions that can be drawn from it.

## Planned extensions

The next stages of the project are expected to include:

- sensitivity and robustness analysis of how errors in the estimated serve probabilities affect match probabilities;
- uncertainty-aware treatment of close matches, particularly predictions in the 40-60% range;
- rolling or periodic retraining to measure how frequently the model benefits from new results;
- additional calibration diagnostics and recalibration if justified;
- comparison against practical ATP-ranking and Elo baselines; and
- state-dependent serve probabilities and simulation-based extensions that relax the constant-probability assumption.

A high-trial Monte Carlo implementation may also be used to check the analytical calculation under matching assumptions. Simulation noise itself is not expected to improve prediction quality; its value is in numerical validation, uncertainty propagation, and supporting models for which a convenient closed form is unavailable.

## Attribution

The underlying tennis records were compiled and maintained by [Jeff Sackmann](https://github.com/JeffSackmann). This project does not redistribute or claim ownership of that source data. Users should consult the upstream repository for its current files and usage terms.

