"""Reproduce metrics from saved v1 predictions; Python standard library only."""
import csv
import json
import math
from pathlib import Path


def evaluate(path):
    with path.open(newline='', encoding='utf-8') as f:
        records=list(csv.DictReader(f))
    y=[int(r['playerA_wins']) for r in records]
    probabilities=[float(r['match_win_probA']) for r in records]
    if not y or any(v not in (0,1) for v in y):
        raise ValueError('Expected nonempty binary labels')
    if any(not math.isfinite(p) or not 0<=p<=1 for p in probabilities):
        raise ValueError('Invalid probabilities')
    positives=sum(y); negatives=len(y)-positives
    if not positives or not negatives:
        raise ValueError('ROC-AUC needs both classes')
    # Average ranks explicitly handle tied prediction scores.
    ordered=sorted(zip(probabilities,y))
    positive_rank_sum=0.
    i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:
            j+=1
        average_rank=(i+1+j)/2
        positive_rank_sum+=average_rank*sum(label for _,label in ordered[i:j])
        i=j
    epsilon=2.220446049250313e-16
    clipped=[min(1-epsilon,max(epsilon,p)) for p in probabilities]
    return {
        'n':len(y),
        'accuracy':sum((p>=.5)==label for p,label in zip(probabilities,y))/len(y),
        'log_loss':-math.fsum(math.log(p) if label else math.log1p(-p) for p,label in zip(clipped,y))/len(y),
        'brier_score':math.fsum((p-label)**2 for p,label in zip(probabilities,y))/len(y),
        'roc_auc':(positive_rank_sum-positives*(positives+1)/2)/(positives*negatives),
    }


if __name__=='__main__':
    folder=Path(__file__).resolve().parent
    split_manifest=json.loads((folder/'split_manifest.json').read_text())
    with (folder/'split_membership.csv').open(newline='',encoding='utf-8') as f:
        membership=list(csv.DictReader(f))
    for split in split_manifest['splits']:
        rows=[r for r in membership if r['split']==split['split']]
        assert len(rows)==split['rows'], 'Split count differs from manifest'
        assert min(r['tourney_date'] for r in rows)==split['minimum_tournament_date']
        assert max(r['tourney_date'] for r in rows)==split['maximum_tournament_date']
        assert [int(r['split_row']) for r in rows]==list(range(len(rows)))
    assert len({r['match_id'] for r in membership})==len(membership)
    with (folder/'v1_results.csv').open(newline='',encoding='utf-8') as f:
        predictions=list(csv.DictReader(f))
    test_rows=[r for r in membership if r['split']=='test']
    assert [(r['match_id'],r['tourney_date']) for r in predictions]==[(r['match_id'],r['tourney_date']) for r in test_rows]
    actual=evaluate(folder/'v1_results.csv')
    expected=json.loads((folder/'expected_metrics.json').read_text())
    for key,value in actual.items():
        if not math.isclose(value,expected[key],rel_tol=0,abs_tol=1e-12):
            raise AssertionError(f'{key}: got {value}, expected {expected[key]}')
    print(json.dumps(actual,indent=2))
    print('All metrics agree with expected values within 1e-12.')
    print('Split counts, date ranges and test prediction membership agree with manifests.')
