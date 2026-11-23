"""Two bounded audits: sampled feature reconstruction and full-history consistency.

No model libraries, model fitting, network access or original-file writes.
"""
import argparse
import hashlib
import json
import platform
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from independent_features import reconstruct

HERE = Path(__file__).resolve().parent
CORE = ['match_id', 'tourney_date', 'tourney_id', 'tourney_name', 'tourney_level',
        'surface', 'best_of', 'round', 'match_num', 'playerA_id', 'playerB_id',
        'playerA_wins', 'playerA_rank', 'playerB_rank', 'playerA_hand', 'playerB_hand',
        '_source_file', '_source_row']
STATS = [f'player{side}_{name}' for side in ('A','B')
         for name in ('ace','df','svpt','1stIn','1stWon','2ndWon','bpFaced','bpSaved')]
IDENTITY = ['match_id','tourney_date','tourney_id','playerA_id','playerB_id']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def clean_text(series):
    return series.astype('string').str.strip().str.replace(r'\s+', ' ', regex=True).str.casefold().replace('', pd.NA)


def consistency_audit(frame, private):
    """Check the COMPLETE supplied history; never repair/drop source rows."""
    issue_masks = {
        'missing_match_id': frame.match_id.isna(),
        'duplicate_match_id': frame.match_id.duplicated(keep=False),
        'invalid_reference_date': frame.tourney_date.isna(),
        'missing_player_id': frame[['playerA_id','playerB_id']].isna().any(axis=1),
        'same_player_both_sides': frame.playerA_id.eq(frame.playerB_id),
        'invalid_outcome': ~frame.playerA_wins.isin([0,1]),
        'missing_source_identity': frame[['_source_file','_source_row']].isna().any(axis=1),
        'duplicate_source_identity': frame[['_source_file','_source_row']].duplicated(keep=False),
    }
    issues = []
    for name, mask in issue_masks.items():
        part = frame.loc[mask, ['match_id','_source_file','_source_row']].copy()
        part.insert(0, 'issue', name)
        issues.append(part)
    pd.concat(issues, ignore_index=True).to_csv(private/'identity_issues.csv', index=False)
    keys = clean_text(frame.tourney_id)
    working = frame.copy(deep=False).assign(_event=keys)
    event_rows = []
    multi = working.loc[keys.notna()].groupby('_event', sort=True, dropna=False)
    # Aggregate first, then inspect only events with a potentially eligible date span.
    events = multi.agg(rows=('match_id','size'), first_reference_date=('tourney_date','min'),
                       last_reference_date=('tourney_date','max'), reference_dates=('tourney_date','nunique'),
                       names=('tourney_name','nunique'), surfaces=('surface','nunique'),
                       levels=('tourney_level','nunique')).reset_index()
    events['reference_span_days'] = (events.last_reference_date-events.first_reference_date).dt.days
    events.to_csv(private/'event_inventory.csv', index=False)
    risk_events = set(events.loc[events.reference_span_days.ge(10), '_event'])
    total_potential_queries = 0
    for key, event in working.loc[working._event.isin(risk_events)].groupby('_event', sort=True):
        # A same-event query can only read outcome/stat records involving one of its
        # own players. Count such pairs using the actual ten-day reference filter.
        people = pd.concat([event[['match_id','tourney_date','playerA_id']].rename(columns={'playerA_id':'player'}),
                            event[['match_id','tourney_date','playerB_id']].rename(columns={'playerB_id':'player'})])
        affected_queries = set()
        for _, appearances in people.dropna(subset=['player']).groupby('player', sort=False):
            earliest = appearances.tourney_date.min()
            affected_queries.update(appearances.loc[appearances.tourney_date.ge(earliest+pd.Timedelta(days=10)), 'match_id'])
        total_potential_queries += len(affected_queries)
        event_rows.append(dict(event_key=key, rows=len(event), reference_dates=event.tourney_date.nunique(),
                               potentially_affected_query_rows=len(affected_queries),
                               query_match_ids=';'.join(map(str, sorted(affected_queries)))))
    pd.DataFrame(event_rows, columns=['event_key','rows','reference_dates','potentially_affected_query_rows','query_match_ids']).to_csv(private/'same_event_candidates.csv', index=False)
    # Same edition/round/unordered pair collisions, including different match numbers.
    pair = pd.DataFrame({'event':keys, 'round':clean_text(frame['round']),
                         'p1':frame[['playerA_id','playerB_id']].min(axis=1),
                         'p2':frame[['playerA_id','playerB_id']].max(axis=1)}, index=frame.index)
    complete = pair.notna().all(axis=1)
    collision_mask = pd.Series(False, index=frame.index)
    collision_mask.loc[complete] = pair.loc[complete].duplicated(keep=False)
    frame.loc[collision_mask, CORE].to_csv(private/'same_event_identity_collisions.csv', index=False)
    weak = pair.assign(date=frame.tourney_date, name=clean_text(frame.tourney_name))
    weak_cols = ['date','name','round','p1','p2']
    good = weak[weak_cols].notna().all(axis=1)
    alias_mask = pd.Series(False, index=frame.index)
    subset = weak.loc[good]
    event_variants = subset.groupby(weak_cols, sort=False, dropna=False).event.transform('nunique')
    alias_mask.loc[subset.index] = event_variants.gt(1)
    frame.loc[alias_mask, CORE].to_csv(private/'cross_event_alias_candidates.csv', index=False)
    missing_events = int(keys.isna().sum())
    missing_key_rows = frame.loc[keys.isna(), CORE]
    missing_key_rows.to_csv(private/'missing_event_identity.csv', index=False)
    summary = dict(full_history_rows=len(frame), rows_dropped=0,
                   identity_issue_rows={key:int(mask.sum()) for key,mask in issue_masks.items()},
                   reference_dates_sorted=bool(frame.tourney_date.is_monotonic_increasing),
                   distinct_recorded_event_ids=len(events), missing_event_id_rows=missing_events,
                   event_ids_with_multiple_reference_dates=int(events.reference_dates.gt(1).sum()),
                   event_ids_with_reference_span_at_least_10_days=len(risk_events),
                   same_event_potentially_affected_query_rows=total_potential_queries,
                   same_event_identity_collision_rows=int(collision_mask.sum()),
                   cross_event_alias_candidate_rows=int(alias_mask.sum()),
                   interpretation='Flags are candidates requiring inspection, not automatic proof of duplicate identity or actual temporal leakage.')
    blocking = ['missing_match_id','duplicate_match_id','invalid_reference_date','missing_player_id',
                'same_player_both_sides','invalid_outcome']
    summary['reconstruction_input_valid'] = bool(not any(summary['identity_issue_rows'][k] for k in blocking)
                                                 and summary['reference_dates_sorted'])
    return summary


def select_queries(frame, size):
    eligible = frame.loc[frame.surface.notna()].copy()
    if eligible.empty:
        raise ValueError('No nonmissing-surface rows to sample')
    eligible['_year'] = eligible.tourney_date.dt.year
    by_stratum = eligible.groupby(['_year','surface','tourney_level'], dropna=False, sort=True)
    candidates = pd.concat([by_stratum.head(1), by_stratum.tail(1)]).drop_duplicates('match_id')
    # Prioritize year coverage plus surface/level coverage; fill by an even spread
    # over first/last stratum rows. Report actual coverage instead of assuming it.
    chosen = pd.concat([eligible.groupby('_year',sort=True).head(1),
                        eligible.groupby('_year',sort=True).tail(1),
                        eligible.groupby(['surface','tourney_level'],dropna=False,sort=True).head(1),
                        eligible.groupby(['surface','tourney_level'],dropna=False,sort=True).tail(1)]).drop_duplicates('match_id')
    if len(chosen) > size:
        chosen = chosen.iloc[np.linspace(0,len(chosen)-1,size,dtype=int)]
    remaining = candidates.loc[~candidates.match_id.isin(chosen.match_id)]
    n = min(max(size-len(chosen),0),len(remaining))
    if n:
        chosen = pd.concat([chosen,remaining.iloc[np.linspace(0,len(remaining)-1,n,dtype=int)]])
    remaining = eligible.loc[~eligible.match_id.isin(chosen.match_id)]
    n = min(max(size-len(chosen),0),len(remaining))
    if n:
        chosen = pd.concat([chosen,remaining.iloc[np.linspace(0,len(remaining)-1,n,dtype=int)]])
    return chosen.sort_values(['tourney_date','match_id']).drop(columns='_year')


def compare_saved(history, saved_path, sample_size, private, public, log):
    columns = json.loads((HERE/'feature_columns.json').read_text())
    if len(columns) != 154 or len(set(columns)) != 154:
        raise ValueError('The expected 154-column historical schema has changed')
    chosen = select_queries(history, sample_size)
    ids = set(chosen.match_id)
    wanted = IDENTITY+columns
    saved_ids = []
    selected = []
    saved_rows = 0
    log('Reading the saved feature CSV in bounded chunks; original file remains unchanged.')
    for chunk in pd.read_csv(saved_path, usecols=wanted, chunksize=20000):
        saved_rows += len(chunk)
        saved_ids.extend(chunk.match_id.tolist())
        selected.append(chunk.loc[chunk.match_id.isin(ids)])
    all_saved_ids = pd.Index(saved_ids)
    if all_saved_ids.has_duplicates or all_saved_ids.isna().any():
        raise ValueError('Saved feature export has duplicate/missing match_id')
    if not all_saved_ids.isin(history.match_id).all():
        raise ValueError('Saved feature IDs are not a subset of full-history IDs')
    saved = pd.concat(selected,ignore_index=True)
    if set(saved.match_id) != ids:
        raise ValueError('Selected matches are missing from the saved feature export; check file versions')
    keys = saved[IDENTITY].merge(chosen[IDENTITY],on='match_id',validate='one_to_one',suffixes=('_saved','_history'))
    for name in IDENTITY[1:]:
        left, right = keys[name+'_saved'], keys[name+'_history']
        if name == 'tourney_date':
            same = pd.to_datetime(left).eq(pd.to_datetime(right))
        else:
            same = left.eq(right) | (left.isna() & right.isna())
        if not same.all():
            raise ValueError(f'Sampled row identities differ between files: {name}')
    chosen[CORE].to_csv(private/'sample_manifest.csv',index=False)
    log(f'Reconstructing {len(chosen)} sampled rows across {len(columns)} historical columns; Elo uses the complete preceding history.')
    rebuilt = reconstruct(history, ids, progress=log)[columns]
    original = saved.set_index('match_id').loc[rebuilt.index,columns]
    before, after = original.to_numpy(dtype=float), rebuilt.to_numpy(dtype=float)
    matches = np.isclose(before,after,atol=1e-6,rtol=1e-7,equal_nan=True)
    stats = pd.DataFrame({'feature':columns,'compared_values':len(chosen),
                          'matching_values':matches.sum(axis=0),'mismatches':(~matches).sum(axis=0)})
    finite = np.isfinite(before) & np.isfinite(after)
    errors = np.where(finite, np.abs(before-after), np.nan)
    stats['max_finite_absolute_error'] = [float(np.nanmax(col)) if np.isfinite(col).any() else None for col in errors.T]
    stats.to_csv(public/'feature_comparison_summary.csv',index=False)
    differences = []
    for i,j in zip(*np.where(~matches)):
        differences.append(dict(match_id=rebuilt.index[i],feature=columns[j],saved_value=before[i,j],
                                 independently_reconstructed_value=after[i,j]))
    pd.DataFrame(differences,columns=['match_id','feature','saved_value','independently_reconstructed_value']).to_csv(private/'feature_discrepancies.csv',index=False)
    return dict(status='completed', full_history_rows=len(history), saved_feature_rows=saved_rows,
                sampled_rows=len(chosen), historical_feature_columns=len(columns), compared_values=int(matches.size),
                matching_values=int(matches.sum()), mismatching_values=int((~matches).sum()),
                sampled_rows_with_mismatches=int((~matches).any(axis=1).sum()),
                years=sorted(map(int,chosen.tourney_date.dt.year.unique())),
                surfaces=sorted(chosen.surface.dropna().astype(str).unique().tolist()),
                tournament_levels=sorted(chosen.tourney_level.dropna().astype(str).unique().tolist()),
                absolute_tolerance=1e-6,relative_tolerance=1e-7,
                interpretation='Sampled arithmetic agreement under the existing ten-day reference-date policy; actual public availability is not established.')


def report_text(summary):
    a,b = summary.get('feature_reconstruction',{}), summary.get('full_history_consistency',{})
    return f'''# Two focused feature-construction audits

Scope: independent reconstruction of saved historical feature values, and consistency
checks across the complete supplied history. This report does not claim to be a
complete review of every modeling procedure. No model was trained or modified.

## Independent feature reconstruction

Status: {a.get('status','not completed')}.
Historical input rows: {b.get('full_history_rows','not available')}.
Sampled saved rows compared: {a.get('sampled_rows',0)}.
Historical feature columns: {a.get('historical_feature_columns',0)}.
Compared values: {a.get('compared_values',0)}.
Mismatching values: {a.get('mismatching_values','not measured')}.
Rows with a mismatch: {a.get('sampled_rows_with_mismatches','not measured')}.

The reference implementation uses separately written calculations instead of calling
the original feature-generation functions. Elo is replayed over the full preceding
history; other values use direct filtering of each sampled player's history.
This is a stratified sample audit, not an exhaustive comparison of every feature row.
Mismatches require investigation; they are not automatically evidence of leakage.

## Full-history event and identity consistency

Recorded event IDs: {b.get('distinct_recorded_event_ids','not measured')}.
Event IDs with multiple reference dates: {b.get('event_ids_with_multiple_reference_dates','not measured')}.
Event IDs spanning at least ten reference days: {b.get('event_ids_with_reference_span_at_least_10_days','not measured')}.
Potential same-event query rows: {b.get('same_event_potentially_affected_query_rows','not measured')}.
Same-event identity collision rows: {b.get('same_event_identity_collision_rows','not measured')}.
Cross-event alias candidate rows: {b.get('cross_event_alias_candidate_rows','not measured')}.
Missing event-ID rows: {b.get('missing_event_id_rows','not measured')}.
Original rows dropped: 0.

Candidates are reported for investigation. Event aliases cannot be certified solely
from similar names, and consistent recorded IDs cannot rule out all unrecognized aliases.
Private diagnostic CSVs retain row identities; the public output contains aggregates only.

## Timestamp limitation

These checks use recorded tournament reference dates, not verified individual match
completion/publication timestamps. Without those timestamps or supported availability
bounds, conservative rules are a practical fallback. The existing ten-calendar-day
eligibility lag and date batching reduce risk but do not guarantee that leakage is absent.
Larger gaps can be audited for their effect on eligible history without training;
their effect on model accuracy remains unmeasured and would generally require new
features and model fits. No such fitting is performed by this runner.

Input and code hashes, environment versions, coverage and any run error are recorded
in summary.json. No conclusions should be drawn from a failed or incomplete stage.
'''


def zip_contents(folder,path):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(folder.glob('*')):
            if item.is_file():
                archive.write(item,item.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,default=Path(r'D:\Advay\Additional Learning\Machine Learning\projects\tennis\data'))
    parser.add_argument('--history',type=Path)
    parser.add_argument('--saved-features',type=Path)
    parser.add_argument('--sample-size',type=int,default=240)
    parser.add_argument('--output',type=Path,default=HERE/'audit_outputs')
    args = parser.parse_args(argv)
    if args.sample_size < 1:
        parser.error('Sample size must be positive')
    history_path = args.history or args.data_root/'created_data/matches_1991_to_2026_cleaned.csv'
    saved_path = args.saved_features or args.data_root/'created_data/matches_1991_2026_features_added.csv'
    for path in (history_path,saved_path):
        if not path.is_file():
            parser.error(f'Input file not found: {path}')
    output = args.output.resolve()
    if output in (history_path.parent.resolve(),saved_path.parent.resolve()):
        parser.error('Choose a separate output directory')
    public, private = output/'public',output/'private'
    for folder in (public,private):
        folder.mkdir(parents=True,exist_ok=True)
        if any(folder.iterdir()):
            parser.error('Output already contains results; choose a new --output directory to preserve the previous run')
    def log(message):
        print(message,flush=True)
        with (private/'run_log.txt').open('a',encoding='utf-8') as stream:
            stream.write(message+'\n')
    summary = dict(scope='Two focused feature-construction audits',started_at=datetime.now(timezone.utc).isoformat(),
                   environment=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__),
                   model_fits=0,original_files_modified=False,
                   input_hashes={},code_hashes={p.name:sha256(p) for p in sorted([*HERE.glob('*.py'), HERE/'feature_columns.json'])},
                   feature_reconstruction=dict(status='not_started'), full_history_consistency=dict(status='not_started'))
    exit_code = 0
    try:
        log('Hashing the two existing inputs; no downloads or training.')
        summary['input_hashes'] = {p.name:dict(sha256=sha256(p),bytes=p.stat().st_size) for p in (history_path,saved_path)}
        log('Loading the complete cleaned history.')
        history = pd.read_csv(history_path,usecols=CORE+STATS)
        history['tourney_date'] = pd.to_datetime(history.tourney_date,errors='coerce')
        log(f'Checking identities and events across {len(history):,} historical rows.')
        summary['full_history_consistency'] = consistency_audit(history,private)
        summary['full_history_consistency']['status'] = 'completed'
        if not summary['full_history_consistency']['reconstruction_input_valid']:
            raise ValueError('History has invalid reconstruction inputs. Diagnostics were retained; no rows were silently removed.')
        # These are the notebook's deterministic field normalizations. They do not
        # invent dates, change event identities or alter outcomes.
        history['surface'] = history.surface.str.capitalize()
        for side in ('A','B'):
            history[f'player{side}_hand'] = history[f'player{side}_hand'].fillna('U')
        summary['feature_reconstruction'] = compare_saved(history,saved_path,args.sample_size,private,public,log)
    except Exception as error:
        summary['error'] = dict(type=type(error).__name__,message=str(error))
        if summary['feature_reconstruction']['status'] == 'not_started':
            summary['feature_reconstruction']['status'] = 'not_completed'
        (private/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        log('Audit stopped with an error; send the results and diagnostics rather than editing the original data.')
        exit_code = 1
    summary['finished_at'] = datetime.now(timezone.utc).isoformat()
    save_json(public/'summary.json',summary)
    (public/'reviewer_report.md').write_text(report_text(summary),encoding='utf-8')
    zip_contents(public,output/'audit_results.zip')
    zip_contents(private,output/'private_diagnostics.zip')
    print(json.dumps({k:summary[k] for k in ('feature_reconstruction','full_history_consistency')},indent=2),flush=True)
    print(f'RETURN: {output / "audit_results.zip"}',flush=True)
    print(f'If a stage failed or any discrepancy/candidate needs investigation, also return: {output / "private_diagnostics.zip"}',flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
