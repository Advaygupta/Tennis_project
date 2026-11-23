"""Fail-closed alignment; positional corroboration is never promoted to unique ID."""
import numpy as np
import pandas as pd
from metrics import validate

KEY = ['tourney_name', 'tourney_date', 'best_of', 'playerA_wins']


def unique(frame, columns, label):
    if frame[columns].isna().any().any() or frame.duplicated(columns).any():
        raise ValueError(f'{label}: missing or duplicate identifiers')


def canonical(frame):
    frame = frame.copy().reset_index(drop=True)
    frame['tourney_date'] = pd.to_datetime(frame.tourney_date, errors='raise').dt.strftime('%Y-%m-%d')
    frame['best_of'] = pd.to_numeric(frame.best_of, errors='raise')
    return frame


def align(v1, v2, metadata):
    original_counts = dict(V1=len(v1), V2=len(v2), metadata=len(metadata))
    unique(metadata, ['dataset_row'], 'metadata dataset_row')
    t = canonical(metadata.loc[metadata.split.eq('test')])
    a = canonical(v1)
    b = v2.copy().reset_index(drop=True)
    unique(t, ['match_id'], 'test match_id')
    unique(t, ['_source_file', '_source_row'], 'source provenance')
    unique(b, ['dataset_row'], 'V2 dataset_row')
    unique(b, ['match_id'], 'V2 match_id')
    validate(a.playerA_wins, a.match_win_probA)
    validate(b.playerA_wins, b.match_win_probA)
    if not len(a) == len(b) == len(t):
        raise ValueError('Unexpected row counts; refusing to silently intersect populations')
    if set(b.dataset_row) != set(t.dataset_row):
        raise ValueError('Unmatched V2/metadata identifiers')
    b = t[['dataset_row']].merge(b, on='dataset_row', how='left', validate='one_to_one', sort=False)
    for column in ('match_id', 'playerA_id', 'playerB_id', 'playerA_wins'):
        if not np.array_equal(t[column].to_numpy(), b[column].to_numpy()):
            raise ValueError('V2 match identity/orientation/outcome mismatch: '+column)
    for column in KEY:
        if a[column].isna().any() or not np.array_equal(a[column].to_numpy(), t[column].to_numpy()):
            raise ValueError('V1 ordered context mismatch: '+column)
    # Reconcile independently by all shared saved context fields. A key is
    # unique only if it occurs exactly once in BOTH the predictions and metadata.
    single_a = ~a.duplicated(KEY, keep=False)
    single_t = ~t.duplicated(KEY, keep=False)
    reconciled = a.loc[single_a, KEY].merge(t.loc[single_t, KEY+['dataset_row']],
                                          on=KEY, validate='one_to_one')
    strict = t.dataset_row.isin(reconciled.dataset_row).to_numpy()
    t['v1_probability'] = a.match_win_probA.to_numpy(float)
    t['v2_probability'] = b.match_win_probA.to_numpy(float)
    t['original_scoring_rule'] = a.deciding_set_rule.to_numpy()
    t['alignment_category'] = np.where(strict, 'unique_shared_context', 'ambiguous_positional')
    # Event IDs include their year; date is deliberately NOT part of event key.
    # Main/qualifying and dates of the same edition are grouped together.
    unique(t, ['tourney_id', 'match_id'], 'event/match')
    if t.tourney_id.isna().any() or t[['playerA_id', 'playerB_id']].isna().any().any():
        raise ValueError('Missing event or player identity')
    repeated_pair = t.duplicated(['tourney_id', 'round', 'playerA_id', 'playerB_id'], keep=False)
    # Round-robin rematches across different rounds are allowed; same-round
    # collisions need review and are never silently deduplicated.
    if repeated_pair.any():
        raise ValueError('Conflicting same-event, same-round oriented player pair')
    info = dict(original_counts=original_counts, test_metadata=len(t), matched_pairs=len(t),
                unmatched=0, V2_exact_row_and_match_id=len(t), V1_exact_match_id=0,
                V1_unique_shared_context=int(strict.sum()),
                V1_ambiguous_positional=int((~strict).sum()),
                V1_ambiguous_context_groups=int(t.loc[~strict].groupby(KEY).ngroups),
                context_key=KEY, full_population_excluded=0,
                unique_context_excluded=int((~strict).sum()),
                limitation='V1 has no player IDs or stable match IDs. Unique shared context is corroboration within this frozen population, not independently authenticated player identity. Full-population pairing is conditional on preserved V1 ordering.')
    return t, b, strict, info


def add_scoring(frame, sidecar):
    unique(sidecar, ['dataset_row'], 'scoring sidecar')
    if set(frame.dataset_row) != set(sidecar.dataset_row):
        raise ValueError('Scoring sidecar population mismatch')
    extra = sidecar.rename(columns={'saved_probability': 'scoring_saved_probability'})
    t = frame.merge(extra, on='dataset_row', how='left', validate='one_to_one', sort=False)
    if not np.allclose(t.v1_probability, t.scoring_saved_probability, rtol=0, atol=2e-12):
        raise ValueError('Scoring sidecar does not match frozen V1 probabilities')
    if not np.array_equal(t.original_scoring_rule, t.original_rule):
        raise ValueError('Scoring sidecar rule mismatch')
    supported = t.registry_status.eq('verified')
    # The earlier audit intentionally leaves inferred-policy counterfactuals
    # missing. Require valid values only where that audit actually computed them.
    validate(t.loc[supported, 'playerA_wins'], t.loc[supported, 'reference_registry_probability'])
    event_fix = t.registry_ordinary.eq('nextgen') | t.registry_deciding.eq('match_tb10') & t.tourney_name.str.contains('Laver Cup', case=False)
    # The best-of/input-feature conflicts are deliberately excluded from this correction.
    event_fix &= ~t['flags'].fillna('').str.contains('best_of_conflict')
    t['event_corrected_probability'] = np.where(event_fix, t.reference_registry_probability, t.v1_probability)
    t['event_rule_correction'] = event_fix
    t['completed'] = ~t.score.fillna('').str.upper().str.contains(r'RET|W/O|DEF|ABD', regex=True)
    t['supported_policy'] = t.registry_status.eq('verified')
    return t
