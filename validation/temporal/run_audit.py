"""Run the temporal audit without fitting/loading models or overwriting predictions."""
import argparse
import csv
import hashlib
import html
import io
import json
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from availability import Evidence, decision, timestamp
from notebook_adapter import ROOT, FAMILIES, cv_definition, definitions
from reference import reconstruct
from test_temporal import synthetic

HERE = Path(__file__).resolve().parent


def formula(col, family, default):
    name = re.sub(r'^player[AB]_', '', col)
    if family == 'current_match_numeric':
        if name in ('draw_size', 'best_of'):
            return 'recorded current-match '+name+'; no fitted transformation'
        if name == 'seed':
            return 'recorded seed; missing -> pooled outer-train maximum seed + 1'
        if name in ('ht', 'age', 'rank_points', 'rank_filled'):
            return ('recorded '+name+'; invalid heights/ranks masked; missing -> '
                    + ('pooled outer-train maximum rank + 1' if name == 'rank_filled' else 'pooled outer-train median'))
        if name == 'is_unranked':
            return '1 if raw current-row rank is missing, else 0; does not establish true unranked status'
        if name == 'rank_points_missing':
            return '1 if raw current-row ranking points are missing, else 0'
        if re.fullmatch(r'top\d+', name):
            return f'1 if observed current-row rank <= {name[3:]}, else 0'
        if name == 'seeded':
            return '1 if raw seed is nonmissing, else 0'
        if name == 'seed_diff_missing':
            return '1 if either raw seed was missing, else 0'
        if name == 'rank_ratio':
            return 'imputed playerA_rank_filled / imputed playerB_rank_filled'
        return {'rank_diff': 'imputed rank A minus B', 'rank_points_diff': 'imputed points A minus B',
                'age_diff': 'imputed age A minus B', 'height_diff': 'imputed height A minus B',
                'seed_diff': 'imputed seed A minus B'}.get(name, default)
    if family in ('career_serve', 'matchup_serve', 'surface_serve', 'return'):
        return {
            'ace_pct': 'sum(valid aces)/sum(valid served points)',
            'df_pct': 'sum(valid double faults)/sum(valid served points)',
            '1stIn_pct': 'sum(valid first serves in)/sum(valid served points)',
            '1stWon_pct': 'sum(valid first-serve points won)/sum(valid first serves in)',
            '2ndWon_pct': 'sum(valid second-serve points won)/sum(valid served points-first serves in)',
            'career_p': 'sum(valid first+second-serve points won)/sum(valid served points)',
            'serve_matches': 'number of historical matches with a valid serve tuple for this player',
            'matchup_p': 'sum(valid points won serving against this opponent)/sum(valid points served against this opponent)',
            'matchup_service_points': 'sum(valid points served against this opponent)',
            'surface_p': 'sum(valid serve points won on current surface)/sum(valid serve points on current surface)',
            'surface_serve_matches': 'number of valid historical serve tuples on current surface',
            'return_pct': 'sum(valid opponent points served - won)/sum(valid opponent points served); no history -> .5',
            'bp_convert_pct': 'sum(valid opponent BP faced-saved)/sum(valid opponent BP faced); no opportunities -> .5',
            'return_matches': 'number of historical matches with valid opponent serve tuple',
        }[name]
    if family == 'elo':
        if name.endswith('momentum'):
            return 'sum last 5 released date-batch overall Elo changes; fewer than 5 -> 0'
        if name.endswith('volatility'):
            return 'population standard deviation of last 10 eligible post-update Elo snapshots; fewer than 10 -> 0'
        if name.endswith('_diff'):
            return name.removesuffix('_diff')+' A minus B'
        if name.endswith('_ratio'):
            return name.removesuffix('_ratio')+' A divided by B'
        if name == 'blended_elo':
            return '.7 * surface_elo + .3 * overall_elo'
        return 'eligible '+name+'; date-frozen zero-sum K-weighted match deltas; initial 1500, surface seeded from eligible overall'
    if family == 'recent':
        group, measure = name.rsplit('_', 1)
        return f'{group}: '+{'wins': 'eligible wins', 'losses': 'eligible matches minus wins',
                             'matches': 'eligible matches', 'pct': 'eligible wins/matches; empty -> .5'}[measure]+'; '+default
    return default


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def feature_inventory(ns):
    nb = json.loads((ROOT/'notebooks/02_v1_1_preprocessing.ipynb').read_text(encoding='utf-8'))
    table = ''.join(nb['cells'][88]['outputs'][0]['data']['text/html'])
    cols = [html.unescape(re.sub('<.*?>', '', v)).strip() for v in
            re.findall(r'<th[^>]*>(.*?)</th>', table.split('</thead>')[0], re.S)][1:]
    if len(cols) != 251 or len(set(cols)) != len(cols):
        raise ValueError('Recorded feature schema changed; review inventory extraction')
    family_of = {col: family for family, (_, names) in FAMILIES.items() for col in ns[names]}
    specs = {
        'matchup': ('outcomes; historical opponent ranks/hand; surface',
                    'H2H counts/differences; last-five tie-aware counts; top20/hand records', '01:65'),
        'elo': ('outcomes; surface; level; best_of',
                'date-frozen 1500-base K=24 batch Elo; blend .7 surface/.3 overall; 5/10-date summaries', '01:72-74'),
        'recent': ('outcomes; surface',
                   'career counts; whole-date-block last10/25; 364-day reference window ending at cutoff', '01:85'),
        'career_serve': ('historical serve totals', 'valid point-weighted cumulative serve ratios/counts', '01:94'),
        'matchup_serve': ('historical serve totals; opponent ID', 'directional matchup won/served totals', '01:96'),
        'surface_serve': ('historical serve totals; surface', 'surface-specific won/served totals and match count', '01:98'),
        'return': ('historical opponent serve and break-point totals',
                   'return won/played; (BP faced-saved)/faced; valid-return match count', '01:101'),
        'fatigue': ('outcomes; event identity; reference date',
                    '(cutoff-window,cutoff] workload: 30/60 days and 5/10 weeks', '01:105'),
        'experience': ('outcomes; tournament name/edition; level',
                       'tournament matches/editions; level wins/losses/matches/rate', '01:127'),
    }
    result = []
    for col in cols:
        family = family_of.get(col)
        if family:
            source, calculation, location = specs[family]
            used = 'historical tournament reference date <= query reference date - 10 calendar days'
            earliest = 'after contributing results/statistics are public; rankings/profile require their own vintages'
            risk = 'unknown public availability; event overlap; ranking/profile vintages where used'
            verified = 'synthetic arithmetic/reference-cutoff only; real source availability unresolved'
        elif col.endswith('_specialization'):
            family = 'derived_history'
            source = {'win_rate': 'surface_career_pct and career_pct', 'elo': 'surface_elo and elo',
                      'serve': 'surface_p and career_p'}[next(x for x in ('win_rate', 'elo', 'serve') if x in col)]
            calculation, location = 'surface-specific value minus overall/career value', '01:123-125;02:70'
            used, earliest = 'inherited from historical dependencies', 'after all dependency inputs are public'
            risk, verified = 'inherits dependency uncertainty', 'formula inspected; real values not reconstructed'
        elif col.endswith('_missing') and ('last' in col or 'surface' in col):
            family, source = 'ambiguity_indicator', 'date-tied rolling/H2H history'
            calculation, location = 'flag ambiguous history; impute using outer-train rate or neutral H2H split', '02:63'
            used, earliest = 'inherited reference-date policy', 'after historical dependencies are public'
            risk, verified = 'inherits availability; inner-fold imputation sees later outer-training rows', 'source inspection only'
        elif (col.startswith(('surface_', 'tourney_level_', 'round_', 'playerA_entry_', 'playerB_entry_',
                              'playerA_hand_', 'playerB_hand_'))):
            family = 'categorical'
            source = 'current-match '+ next(x for x in ('surface','tourney_level','round','entry','hand') if x in col)
            calculation, location = 'fixed category recodes; outer-training fitted one-hot encoder', '02:41-45,77-84'
            used, earliest = 'no explicit field availability timestamp', 'draw/order/schedule/profile publicly known before prediction'
            risk = 'reference-start predictions cannot assume later-round pairings, LL/entry updates, or profile vintages'
            verified = 'encoder train-only for outer split; current-match availability unresolved'
        else:
            family = 'current_match_numeric'
            source = ('rank/rank_points' if any(x in col for x in ('rank','top')) else
                      'seed' if 'seed' in col else 'age' if 'age' in col else
                      'height' if 'ht' in col or 'height' in col else col)
            calculation = ('A minus B; rank_ratio=A/B; flags use raw missingness/top-K; '
                           'base values use outer-training pooled imputations where needed')
            location = '01:109-121;02:49,53,67'
            used = 'recorded current-row value; no explicit publication timestamp'
            earliest = 'rank release / draw / profile snapshot known before prediction'
            risk = 'contemporaneous snapshot unresolved; outer-training imputation reused by inner CV'
            verified = 'source inspection only; no rank/profile vintage evidence'
        calculation = formula(col, family, calculation)
        result.append(dict(feature=col, family=family, source_data=source, calculation=calculation,
                           notebook_cells_zero_based=location, implemented_availability=used,
                           earliest_defensible_availability=earliest, risk=risk,
                           verification=verified, schema_origin='recorded notebook 02 cell 88 output; manifest unavailable'))
    return pd.DataFrame(result)


def metadata_audit(metadata):
    required = {'dataset_row', 'match_id', 'split', 'tourney_id', 'tourney_date',
                'playerA_wins', '_source_file', '_source_row'}
    if not required.issubset(metadata):
        raise ValueError(f'Missing metadata columns: {sorted(required-set(metadata))}')
    dates = pd.to_datetime(metadata.tourney_date, errors='raise')
    if dates.isna().any() or not dates.is_monotonic_increasing:
        raise ValueError('Metadata dates must be sorted and nonmissing')
    if not metadata.split.isin(['train', 'val', 'test']).all():
        raise ValueError('Unknown partition')
    if not metadata.dataset_row.is_unique or not metadata.match_id.is_unique:
        raise ValueError('Duplicate identities')
    if not metadata.playerA_wins.isin([0, 1]).all():
        raise ValueError('Invalid outcomes')
    if metadata[['_source_file', '_source_row']].isna().any().any():
        raise ValueError('Missing source provenance')
    if metadata[['_source_file', '_source_row']].duplicated().any():
        raise ValueError('Duplicate source identities')
    groups = metadata.groupby('tourney_id', dropna=True)
    split_summary = {}
    for split in ('train', 'val', 'test'):
        d = dates[metadata.split.eq(split)]
        if d.empty:
            raise ValueError(f'Missing partition {split}')
        split_summary[split] = dict(rows=len(d), first=str(d.min().date()), last=str(d.max().date()))
    checks = dict(
        reference_dates_exclusive=bool(metadata.groupby('tourney_date').split.nunique().le(1).all()),
        event_ids_exclusive=bool(groups.split.nunique().le(1).all()),
        ordered_split_boundaries=bool(dates[metadata.split.eq('train')].max() < dates[metadata.split.eq('val')].min()
                                      and dates[metadata.split.eq('val')].max() < dates[metadata.split.eq('test')].min()))
    if not all(checks.values()):
        raise ValueError(f'Partition integrity failure: {checks}')
    # The actual legacy reset-index date bug, measured without training.
    train_dates = dates[metadata.split.eq('train')].reset_index(drop=True)
    cv = cv_definition('03_v1_1_model_training.ipynb', 'date_grouped_cv', train_dates)
    seen = np.zeros(len(train_dates), dtype=bool)
    for tr, va in cv(n_splits=5):
        if not train_dates.iloc[tr].max() < train_dates.iloc[va].min():
            raise ValueError('OOF fold date overlap')
        seen[va] = True
    true = train_dates[seen].reset_index(drop=True)
    wrong = train_dates.iloc[:len(true)].reset_index(drop=True)
    old_folds = cv(n_splits=3, dates=wrong)
    corrected_folds = cv(n_splits=3, dates=true)
    def fold_record(pair):
        tr, va = pair
        shared = set(true.iloc[tr]).intersection(set(true.iloc[va]))
        return dict(train_rows=len(tr), validation_rows=len(va),
                    actual_train_end=str(true.iloc[tr].max().date()),
                    actual_validation_start=str(true.iloc[va].min().date()),
                    shared_actual_reference_dates=len(shared))
    return dict(rows=len(metadata), columns=len(metadata.columns), splits=split_summary, partition_checks=checks,
                event_ids_with_multiple_reference_dates=int(groups.tourney_date.nunique().gt(1).sum()),
                source_availability_rows_verified=0, source_availability_rows_unresolved=len(metadata),
                source_availability_reason='No per-observation completion/publication evidence supplied',
                dropped_rows=0, oof_rows=int(seen.sum()),
                legacy_meta_rows_with_wrong_dates=int(true.ne(wrong).sum()),
                legacy_meta_folds=[fold_record(f) for f in old_folds],
                corrected_meta_folds=[fold_record(f) for f in corrected_folds],
                caveat='Metadata is a target-eligible subset, not complete history; date checks do not prove label availability')


def check_contributions(evidence_path, queries_path, contributions_path, output):
    """Audit externally supplied lineage. Never infer contributions or timestamp evidence."""
    with evidence_path.open(encoding='utf-8', newline='') as f:
        items = [Evidence.from_row(row) for row in csv.DictReader(f)]
    evidence = {(e.observation_id, e.family): e for e in items}
    if len(evidence) != len(items):
        raise ValueError('Duplicate observation/family evidence')
    with queries_path.open(encoding='utf-8', newline='') as f:
        items = list(csv.DictReader(f))
    queries = {q['query_id']: q for q in items}
    if len(queries) != len(items):
        raise ValueError('Duplicate prediction query ID')
    decisions = []
    with contributions_path.open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            q = queries[row['query_id']]
            prediction, cutoff = timestamp(q['prediction_at']), timestamp(q['cutoff'])
            if prediction is None or cutoff is None or cutoff >= prediction:
                raise ValueError('Query requires explicit prediction and earlier cutoff')
            if not q.get('prediction_source') or not q.get('prediction_basis'):
                raise ValueError('Query prediction time requires source and basis')
            e = evidence.get((row['observation_id'], row['family']))
            if e is None:
                status, allowed = 'unresolved_missing_evidence', False
            else:
                status, allowed = decision(e, query_id=q['query_id'], query_event=q['event_key'],
                                           prediction_at=prediction, cutoff=cutoff)
            decisions.append({**row, 'status': status, 'allowed': allowed})
    result = pd.DataFrame(decisions, columns=['query_id','observation_id','family','feature','status','allowed'])
    result.to_csv(output/'contribution_decisions.csv', index=False)
    return dict(contributions_checked=len(result), decisions=result.status.value_counts().to_dict(),
                caveat='Only supplied lineage checked; ledger contents are asserted evidence, not automatically authenticated sources')


def real_comparison(history_path, saved_path, sample_size, output):
    history = pd.read_csv(history_path, parse_dates=['tourney_date'])
    saved = pd.read_csv(saved_path)
    if not saved.match_id.is_unique or saved.match_id.isna().any():
        raise ValueError('Saved feature export requires unique nonmissing match_id')
    ns = definitions()
    columns = [col for _, names in FAMILIES.values() for col in ns[names]]
    absent = set(columns)-set(saved)
    if absent:
        raise ValueError(f'Need pre-imputation feature export; missing: {sorted(absent)}')
    if not set(saved.match_id).issubset(set(history.match_id)):
        raise ValueError('Saved rows missing from full history')
    eligible = history.loc[history.match_id.isin(saved.match_id)].copy()
    eligible['_year'] = eligible.tourney_date.dt.year
    # First and last in every year/surface/level stratum, then deterministic spread.
    grouped = eligible.groupby(['_year','surface','tourney_level'], dropna=False, sort=True)
    chosen = pd.concat([grouped.head(1), grouped.tail(1)]).drop_duplicates('match_id')
    if len(chosen) > sample_size:
        chosen = chosen.iloc[np.linspace(0, len(chosen)-1, sample_size, dtype=int)]
    ids = chosen.match_id.tolist()
    rebuilt = reconstruct(history, ids)
    original = saved.set_index('match_id').loc[rebuilt.index, columns]
    rebuilt = rebuilt[columns]
    ok = np.isclose(original.to_numpy(dtype=float), rebuilt.to_numpy(dtype=float), atol=1e-6, rtol=1e-7, equal_nan=True)
    records = [dict(match_id=str(rebuilt.index[i]), feature=columns[j], before=str(original.iloc[i,j]),
                    independently_reconstructed=str(rebuilt.iloc[i,j])) for i,j in zip(*np.where(~ok))]
    pd.DataFrame(records, columns=['match_id','feature','before','independently_reconstructed']).to_csv(output/'real_feature_discrepancies.csv', index=False)
    # Source identities stay private; only aggregates are returned in the report.
    return dict(status='completed_reference_policy_only', full_history_rows=len(history),
                saved_rows=len(saved), sampled_rows=len(ids), features=len(columns),
                compared_values=int(ok.size), discrepancies=int((~ok).sum()),
                history_sha256=sha256(history_path), saved_features_sha256=sha256(saved_path),
                caveat='Arithmetic/reference-date reconstruction is not proof of actual completion/publication availability')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=ROOT/'reproduction/artifacts/matches_v1_1_metadata.csv')
    parser.add_argument('--output', type=Path, default=HERE/'results')
    parser.add_argument('--history', type=Path)
    parser.add_argument('--saved-features', type=Path)
    parser.add_argument('--confirm-full-history', action='store_true', help='Attest history is deduplicated full pre-target-filter A/B history')
    parser.add_argument('--sample-size', type=int, default=120)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--queries', type=Path)
    parser.add_argument('--contributions', type=Path)
    args = parser.parse_args()
    if bool(args.history) != bool(args.saved_features) or (args.history and not args.confirm_full_history):
        parser.error('Supply --history and --saved-features together, with --confirm-full-history')
    if args.sample_size < 1:
        parser.error('--sample-size must be positive')
    if any((args.evidence,args.queries,args.contributions)) and not all((args.evidence,args.queries,args.contributions)):
        parser.error('Evidence auditing requires --evidence, --queries and --contributions together')
    output = args.output.resolve()
    if output == ROOT/'reproduction/artifacts' or ROOT/'models' in output.parents:
        parser.error('Use a separate audit output directory')
    output.mkdir(parents=True, exist_ok=True)
    ns = definitions()
    inventory = feature_inventory(ns)
    inventory.to_csv(output/'feature_inventory.csv', index=False)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.discover(str(HERE), pattern='test_temporal.py'))
    (output/'tests.txt').write_text(stream.getvalue(), encoding='utf-8')
    report = dict(scope='Temporal audit only; no training, tuning, calibration or model inference',
                  baseline_commit='7cfac7dceae6bbfb8f47184c6eb64af71d451156',
                  remote_head_verified=False,
                  feature_inventory=dict(predictor_columns=len(inventory), historical_columns=154,
                                         origin='recorded notebook schema, not loaded saved feature matrix'),
                  synthetic=dict(rows=len(synthetic()), independently_reconstructed_features=154,
                                 compared_values=len(synthetic())*154, real_source_availability_verified=False),
                  tests=dict(run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                             successful=result.wasSuccessful(), known_defects_reproduced=[
                                 'T02 inconsistent-event dates', 'T01 synthetic multiweek future-result admission']),
                  real_feature_comparison=dict(status='blocked_missing_full_history_and_saved_features', compared_rows=0),
                  provenance_contribution_audit=dict(status='blocked_missing_timestamp_evidence_and_lineage', contributions_checked=0),
                  saved_models_or_predictions_changed=[], feature_values_changed=0, model_fits=0,
                  unresolved=['T01 public availability', 'T02 event exclusion in legacy feature generators',
                              'T03 inner-CV preprocessing', 'T05 non-nested base hyperparameter selection',
                              'T06 label availability at evaluation origins', 'T07 actual-date workload interpretation'])
    if args.metadata.exists():
        report['metadata'] = metadata_audit(pd.read_csv(args.metadata))
        report['metadata']['sha256'] = sha256(args.metadata)
    else:
        report['metadata'] = dict(status='not_supplied', rows_checked=0)
    if args.history:
        report['real_feature_comparison'] = real_comparison(args.history, args.saved_features, args.sample_size, output)
    if args.evidence:
        report['provenance_contribution_audit'] = check_contributions(args.evidence,args.queries,args.contributions,output)
    paths = sorted((ROOT/'notebooks').glob('*.ipynb'))+sorted(HERE.glob('*.py'))
    report['source_hashes'] = {str(p.relative_to(ROOT)).replace('\\','/'): sha256(p) for p in paths}
    (output/'audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('tests','feature_inventory','synthetic','real_feature_comparison')}, indent=2))
    print('Audit complete; source availability remains unresolved. No training was run.')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
