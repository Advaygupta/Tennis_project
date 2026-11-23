"""Read-only inventory of the frozen comparison and notebook evidence."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(repo, v1, v2, metadata):
    expected = json.loads((repo/'reproduction/expected_metrics.json').read_text(encoding='utf8'))
    candidates = [dict(name='V1.1 regression plus Markov', version='V1.1',
                       artifact='v1_1_results.joblib', probability='match_win_probA',
                       composition='Inverse-validation-MAE Elastic Net, LightGBM, XGBoost, CatBoost serve regressors; analytical scoring',
                       rows=len(v1), originally_reported=expected['v1_1'], primary=True)]
    columns = [('match_win_probA', 'V2.1 equal classifier ensemble'),
               ('stacked_probA', 'V2.1 stacking'), ('catboost_probA', 'V2.1 CatBoost'),
               ('lightgbm_probA', 'V2.1 LightGBM'), ('xgboost_probA', 'V2.1 XGBoost'),
               ('sgd_probA', 'V2.1 SGD')]
    for column, name in columns:
        if column in v2:
            candidates.append(dict(name=name, version='V2.1',
                artifact='v2_1_test_match_probabilities.parquet', probability=column,
                rows=len(v2), primary=column=='match_win_probA',
                composition='Equal CatBoost, LightGBM, XGBoost, SGD' if column=='match_win_probA' else name,
                originally_reported=expected['v2_1'] if column=='match_win_probA' else 'Rounded output in notebook 05 cell 170 (zero-based)'))
    evidence = []
    for name, cell_ids in [('03_v1_1_model_training.ipynb', [154, 170, 193, 201, 209, 213]),
                           ('05_v2_1_model_training.ipynb', [158, 169, 170, 172, 173])]:
        path = repo/'notebooks'/name
        notebook = json.loads(path.read_text(encoding='utf8'))
        evidence.append(dict(file='notebooks/'+name, sha256=digest(path),
            cells=[dict(zero_based=i, execution_count=notebook['cells'][i].get('execution_count'),
                        source_sha256=hashlib.sha256(''.join(notebook['cells'][i]['source']).encode()).hexdigest())
                   for i in cell_ids]))
    ranges = {}
    for split, group in metadata.groupby('split', sort=True):
        ranges[split] = dict(n=len(group), first=str(group.tourney_date.min()), last=str(group.tourney_date.max()))
    return dict(primary_comparison='V2.1 equal classifier ensemble (A) minus V1.1 regression plus Markov (B)',
        outcome='playerA_wins: 1 if the oriented Player A is the recorded winner, including saved retirements',
        direction='P(Player A wins match)', threshold='probability >= 0.5; tie predicts A',
        candidates=candidates, splits=ranges, notebook_evidence=evidence,
        out_of_sample_evidence='Notebook source and recorded outputs describe train+validation refits and a chronological untouched test partition; this audit does not independently authenticate prediction-generation history or prove no leakage.',
        unavailable='No frozen per-match V1.1 individual-regressor match probabilities or standalone Elo/rank baseline prediction artifact was supplied. No new baseline predictions are generated.',
        selection='This reproduces the existing README comparison, not a choice made by this audit. Earlier notebook exploration includes test displays; global pre-registration and absence of all test-guided choices are not established. Additional V2 methods and all subsets are descriptive/exploratory; no multiple-model winner search or confirmatory significance claim.')
