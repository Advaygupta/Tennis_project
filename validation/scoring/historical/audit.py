"""Scoring-only audit; never imports or fits ML estimators.

Run from any directory: python validation/scoring/historical/audit.py
Private artifacts are optional; synthetic tests and registry code are public.
New outputs go into this audit's results/ and ignored audit_private/ directories.
"""
import argparse
import csv
import hashlib
import gzip
import io
import json
import math
import platform
from pathlib import Path
import sys
import unittest
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
import engine as ref
import registry
from production_adapter import extract

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
# Optional local dependencies, also used by earlier reproduction work.
sys.path.insert(0,str(REPO/'.repro_deps'))
TOLERANCE=2e-12


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8')


def write_csv(path,rows,fields=None):
    rows=list(rows)
    if fields is None:
        fields=list(rows[0]) if rows else ['no_rows']
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def metrics(y,probability):
    y=np.asarray(y,dtype=int)
    p=np.asarray(probability,dtype=float)
    if not len(y):return None
    if not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid metric inputs')
    # Match sklearn's double-precision log-loss clipping, no model import.
    eps=np.finfo(float).eps
    z=np.clip(p,eps,1-eps)
    order=np.argsort(p,kind='stable')
    positive_rank_sum=0.
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and p[order[j]]==p[order[i]]:j+=1
        positive_rank_sum+=(i+1+j)/2*int(y[order[i:j]].sum())
        i=j
    positives=int(y.sum())
    negatives=len(y)-positives
    auc=(positive_rank_sum-positives*(positives+1)/2)/(positives*negatives) if positives and negatives else None
    return dict(n=len(y),accuracy=float(np.mean((p>=.5)==y)),roc_auc=auc,
                brier_score=float(np.mean((p-y)**2)),
                log_loss=float(-np.mean(y*np.log(z)+(1-y)*np.log1p(-z))))


def original_vector(functions,p,q,best,rule):
    s=np.asarray(functions['set_win_probab'](p,q),float)
    d=s.copy()
    for name in sorted(set(rule)):
        mask=rule==name
        d[mask]=functions['deciding_set_win_probab'](p[mask],q[mask],name,standard_set=s[mask])
    return np.where(best==3,s*s+2*s*(1-s)*d,
                    s**3+3*s**3*(1-s)+6*s*s*(1-s)**2*d)


def synthetic(functions,out):
    rows=[]
    for p in (.05,.2,.35,.5,.65,.8,.95):
        for q in (.05,.2,.35,.5,.65,.8,.95):
            for name in ('standard','advantage','tb10','wimbledon_12'):
                for best in (3,5):
                    original=float(original_vector(functions,np.array([p]),np.array([q]),
                        np.array([best]),np.array([name]))[0])
                    a=ref.match_probability(p,q,best,0,deciding=ref.RULES[name])
                    b=ref.match_probability(p,q,best,1,deciding=ref.RULES[name])
                    rows.append(dict(p=p,q=q,deciding_rule=name,best_of=best,
                        original_probability=original,reference_A_first=a,reference_B_first=b,
                        absolute_error=max(abs(original-a),abs(original-b)),first_server_difference=abs(a-b)))
    write_csv(out/'synthetic_grid.csv',rows)
    return dict(probability_pairs=49,rule_categories=4,match_formats=2,
        cases=len(rows),initial_server_comparisons=2*len(rows),
        maximum_absolute_error=max(r['absolute_error'] for r in rows),
        maximum_first_server_difference=max(r['first_server_difference'] for r in rows),
        discrepancies=sum(r['absolute_error']>TOLERANCE for r in rows))


def numerical_stress(functions,out):
    rows=[]
    for p,q in [(1e-8,2e-8),(1-1e-8,1-2e-8),(1e-12,2e-12),(1-1e-12,1-2e-12),
                (0.,0.),(1.,1.),(1.,0.),(0.,1.)]:
        for name in ('standard','advantage','tb10','wimbledon_12'):
            with np.errstate(all='ignore'):
                old=float(original_vector(functions,np.array([p]),np.array([q]),np.array([3]),np.array([name]))[0])
            try:
                new=ref.match_probability(p,q,3,deciding=ref.RULES[name])
                status='defined'
            except ref.NonTerminating:
                new=None
                status='nonterminating'
            rows.append(dict(p=p,q=q,rule=name,reference_status=status,
                original_probability=old if math.isfinite(old) else None,
                reference_probability=new,
                absolute_error=abs(old-new) if math.isfinite(old) and new is not None else None))
    write_json(out/'numerical_stress.json',rows)
    return dict(cases=len(rows),nonterminating_reference_cases=sum(r['reference_status']=='nonterminating' for r in rows),
                nonfinite_original_cases=sum(r['original_probability'] is None for r in rows),
                defined_discrepancies=sum(r['absolute_error'] is not None and r['absolute_error']>TOLERANCE for r in rows))


def metadata_audit(path,out,private):
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len({r['dataset_row'] for r in rows})!=len(rows):raise ValueError('Nonunique dataset_row')
    assignments=[registry.resolve(r) for r in rows]
    coverage=Counter()
    events=Counter()
    flagged=[]
    for row,a in zip(rows,assignments):
        sources=';'.join(a['sources'])
        coverage[row['split'],a['status'],a['ordinary'],a['deciding'],a['best_of']]+=1
        events[row['split'],row['tourney_name'],row['tourney_date'][:4],row['tourney_level'],
               a['draw'],row['round'],row['best_of'],a['ordinary'],a['deciding'],a['best_of'],
               a['status'],sources,a['reason']]+=1
        flags=registry.observed_flags(row,a)
        if flags:flagged.append(dict(dataset_row=row['dataset_row'],flags=';'.join(flags)))
    keys=['split','status','ordinary_rule','deciding_rule','best_of','matches']
    write_csv(out/'rule_coverage.csv',[dict(zip(keys,(*k,v))) for k,v in sorted(coverage.items(),key=str)])
    ekeys=['split','tournament','year','recorded_level','draw','round','recorded_best_of',
           'ordinary_rule','deciding_rule','registry_best_of','status','source_ids','reason','matches']
    inventory=[dict(zip(ekeys,(*k,v))) for k,v in sorted(events.items(),key=str)]
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=ekeys,lineterminator='\n')
    writer.writeheader();writer.writerows(inventory)
    (out/'event_rule_inventory.csv.gz').write_bytes(gzip.compress(stream.getvalue().encode(),compresslevel=9,mtime=0))
    write_csv(private/'metadata_flags.csv',flagged,['dataset_row','flags'])
    status_by_split={split:dict(Counter(a['status'] for r,a in zip(rows,assignments) if r['split']==split))
                     for split in ('train','val','test')}
    return rows,assignments,dict(rows=len(rows),status_by_split=status_by_split,
        best_of_conflicts=sum(a['format_override'] for a in assignments),
        unknown_first_server=True,metadata_flags=len(flagged),
        scope='Target-eligible modeling metadata, not all 855130 historical feature-source rows')


def saved_comparison(path,rows,assignments,functions,out,private):
    import joblib
    data=joblib.load(path)
    selection=[i for i,r in enumerate(rows) if r['split']=='test']
    test=[rows[i] for i in selection]
    mapped=[assignments[i] for i in selection]
    required={'p','q','best_of','playerA_wins','set_win_probA','deciding_set_rule',
              'tourney_date','tourney_name','deciding_set_win_probA','match_win_probA'}
    if not required.issubset(data.columns):raise ValueError('Saved serve probabilities or context missing')
    if len(data)!=len(test):raise ValueError('Saved prediction count differs from metadata')
    # V1 artifact has no stable match IDs. Preserve and explicitly report that limit.
    for column in ('tourney_name','best_of','playerA_wins'):
        left=[str(x) for x in data[column]]
        right=[str(r[column]) for r in test]
        if left!=right:raise ValueError('Ordered artifact/metadata mismatch: '+column)
    if list(pd.to_datetime(data.tourney_date).dt.strftime('%Y-%m-%d'))!=[r['tourney_date'] for r in test]:
        raise ValueError('Ordered artifact/metadata dates mismatch')
    p=data.p.to_numpy(float)
    q=data.q.to_numpy(float)
    if not np.isfinite(p).all() or not np.isfinite(q).all() or ((p<0)|(p>1)|(q<0)|(q>1)).any():
        raise ValueError('Invalid saved serve probabilities')
    best=data.best_of.to_numpy(int)
    names=data.deciding_set_rule.to_numpy(str)
    saved=data.match_win_probA.to_numpy(float)
    regenerated=original_vector(functions,p,q,best,names)
    original_reference=[]
    rule_scenario=saved.copy()
    bestof_scenario=saved.copy()
    comparisons=[]
    group_differences=defaultdict(list)
    for i,(r,a) in enumerate(zip(test,mapped)):
        baseline=ref.match_probability(p[i],q[i],int(best[i]),None,deciding=ref.RULES[names[i]])
        original_reference.append(baseline)
        corrected=None
        if a['status']=='verified':
            corrected=ref.match_probability(p[i],q[i],a['best_of'],None,
                ordinary=ref.RULES[a['ordinary']],deciding=ref.RULES[a['deciding']])
            bestof_scenario[i]=corrected
            if not a['format_override']:rule_scenario[i]=corrected
        flags=registry.observed_flags(r,a)
        category='nextgen' if a['ordinary']=='nextgen' else ('laver_match_tb' if 'laver' in r['tourney_name'].lower()
                     else ('best_of_conflict' if a['format_override'] else 'other'))
        if corrected is not None and (a['ordinary']!='standard' or a['deciding']!=names[i] or a['format_override']):
            group_differences[category].append(abs(corrected-saved[i]))
        comparisons.append(dict(dataset_row=r['dataset_row'],original_rule=names[i],registry_status=a['status'],
            registry_ordinary=a['ordinary'],registry_deciding=a['deciding'],
            saved_probability=saved[i],original_function_probability=regenerated[i],
            reference_original_rule=baseline,reference_registry_probability=corrected,
            absolute_implementation_error=abs(saved[i]-baseline),
            absolute_rule_or_format_difference=None if corrected is None else abs(saved[i]-corrected),
            flags=';'.join(flags)))
        if (i+1)%5000==0:print(f'Scoring-only comparison: {i+1}/{len(test)}',flush=True)
    independent=np.asarray(original_reference)
    errors=np.abs(independent-saved)
    verified=np.array([a['status']=='verified' and not a['format_override'] for a in mapped])
    completed=np.array([not registry.observed_flags(r,a) for r,a in zip(test,mapped)])
    write_csv(private/'saved_probability_comparison.csv',comparisons)
    write_csv(private/'largest_rule_differences.csv',sorted(
        (r for r in comparisons if r['absolute_rule_or_format_difference'] is not None),
        key=lambda r:r['absolute_rule_or_format_difference'],reverse=True)[:50])
    y=data.playerA_wins.to_numpy(int)
    return dict(rows=len(test),required_columns=sorted(required),p_range=[float(p.min()),float(p.max())],
        q_range=[float(q.min()),float(q.max())],
        alignment='Ordered tournament name, date, best_of and outcome agree; saved V1 lacks row/match IDs, so identity cannot be independently established from unique keys',
        original_rule_counts=dict(Counter(names)),
        original_function_vs_saved=dict(matching=int(np.sum(abs(regenerated-saved)<=TOLERANCE)),
            maximum_absolute_error=float(np.max(abs(regenerated-saved)))),
        independent_vs_saved_under_original_rules=dict(matching=int(np.sum(errors<=TOLERANCE)),
            discrepancies=int(np.sum(errors>TOLERANCE)),maximum_absolute_error=float(errors.max())),
        rule_format_changes={name:dict(matches=len(values),changed_beyond_tolerance=sum(v>TOLERANCE for v in values),
            maximum_absolute_difference=max(values),mean_absolute_difference=float(np.mean(values)))
            for name,values in group_differences.items()},
        registry_status_counts=dict(Counter(a['status'] for a in mapped)),
        registry_best_of_conflicts=sum(a['format_override'] for a in mapped),
        retirement_or_incomplete_rows=sum('retirement_or_incomplete_outside_point_model' in registry.observed_flags(r,a) for r,a in zip(test,mapped)),
        metrics_saved=metrics(y,saved),
        metrics_rule_only_scenario=metrics(y,rule_scenario),
        metrics_rule_and_bestof_scenario=metrics(y,bestof_scenario),
        metrics_verified_policy_completed_saved=metrics(y[verified&completed],saved[verified&completed]),
        metrics_verified_policy_completed_scenario=metrics(y[verified&completed],rule_scenario[verified&completed]),
        scenario_limit='Frozen p/q scoring-only sensitivity, not a newly trained or fully historically verified model. Inferred/unresolved policies retain original probabilities; all original artifacts remain unchanged.',
        environment_joblib=joblib.__version__)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,default=REPO/'reproduction/artifacts/matches_v1_1_metadata.csv')
    parser.add_argument('--predictions',type=Path,default=REPO/'reproduction/artifacts/v1_1_results.joblib')
    parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--private-output',type=Path,default=HERE/'audit_private')
    parser.add_argument('--synthetic-only',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    private=args.private_output.resolve();private.mkdir(parents=True,exist_ok=True)
    functions,extraction=extract(REPO)
    protected=[REPO/'notebooks/03_v1_1_model_training.ipynb']
    for p in (args.metadata,args.predictions):
        if p.exists():protected.append(p)
    before={str(p.relative_to(REPO)):digest(p) for p in protected}
    buffer=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_reference.py')
    result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(suite)
    (out/'tests.txt').write_text(buffer.getvalue(),encoding='utf-8')
    if not result.wasSuccessful():raise RuntimeError('Independent tests failed; see tests.txt')
    summary=dict(scope='Independent Markov scoring and historical rules audit',baseline_commit='1d02053',
        model_fits=0,model_inference_calls=0,notebooks_executed=0,existing_scripts_rerun=0,
        absolute_tolerance=TOLERANCE,tail_truncation_error=0,
        float_error_bound='No universal IEEE-754 bound claimed; tolerance is an empirical comparison threshold. Rational cyclic-chain tests cover selected extremes; production stress limitations are recorded separately.',
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        tests=dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors)),
        extraction=extraction,synthetic=synthetic(functions,out),numerical_stress=numerical_stress(functions,out),
        input_sha256=before)
    if not args.synthetic_only and args.metadata.exists():
        rows,assignments,summary['metadata']=metadata_audit(args.metadata,out,private)
        if args.predictions.exists():
            summary['saved_predictions']=saved_comparison(args.predictions,rows,assignments,functions,out,private)
        else:summary['saved_predictions']={'status':'blocked','missing':str(args.predictions.name)}
    else:summary['metadata']={'status':'not_run','reason':'synthetic-only requested or private metadata absent'}
    after={str(p.relative_to(REPO)):digest(p) for p in protected}
    if before!=after:raise RuntimeError('Protected input changed during audit')
    summary['original_inputs_preserved']=True
    summary['code_sha256']={p.name:digest(p) for p in sorted(HERE.glob('*')) if p.suffix in ('.py','.json')}
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),flush=True)


if __name__=='__main__':main()
