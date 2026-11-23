"""Independent, deliberately slow reference reconstruction of the legacy policy.

Direct filtering/grouping per query; no production queues, history generators,
validators, window selectors or feature functions are called. This tests arithmetic
and reference-date eligibility, NOT historical public availability. Use full A/B
history before target/surface filtering when comparing with saved feature exports.
"""
from collections import defaultdict
from math import fsum

import numpy as np
import pandas as pd


def ratio(numerator, denominator, empty=0.5):
    return numerator / denominator if denominator else empty


def edition(r):
    if pd.notna(r['tourney_id']) and str(r['tourney_id']).strip():
        return ('id', str(r['tourney_id']), r['tourney_date'])
    if pd.notna(r['tourney_name']) and str(r['tourney_name']).strip():
        level = r['tourney_level'] if pd.notna(r['tourney_level']) else None
        return ('fallback', str(r['tourney_name']).strip(), level, r['tourney_date'])
    return None


def valid_serve(r, side):
    v = [r[f'player{side}_{s}'] for s in ('ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon')]
    if not all(pd.notna(x) and np.isfinite(x) for x in v):
        return False
    ace, df, total, first, won1, won2 = v
    return (total > 0 and 0 <= first <= total and 0 <= won1 <= first
            and 0 <= won2 <= total-first and 0 <= ace <= total and 0 <= df <= total-first)


def block_window(records, target):
    """Enumerate admissible whole-date suffixes independently of the queue code."""
    if not records:
        return 0, 0
    groups = defaultdict(list)
    for r in records:
        groups[r['date']].append(r['won'])
    dates = sorted(groups, reverse=True)
    total = len(records)
    if total < target:
        return sum(r['won'] for r in records), total
    lo, hi = {10: (8, 12), 25: (22, 28)}[target]
    choices = []
    for k in range(1, len(dates)+1):
        outcomes = [x for d in dates[:k] for x in groups[d]]
        if lo <= len(outcomes) <= hi:
            choices.append((abs(len(outcomes)-target), -len(outcomes), sum(outcomes), len(outcomes)))
        if len(outcomes) > hi:
            break
    if choices:
        _, _, wins, count = min(choices)
        return wins, count
    boundary = sorted((r['date'] for r in records), reverse=True)[target-1]
    after = [r['won'] for r in records if r['date'] > boundary]
    tied = groups[boundary]
    if len(set(tied)) != 1:
        return np.nan, np.nan
    return sum(after) + (target-len(after))*tied[0], target


def elo_reference(frame, query_ids=None, progress=None):
    """Date-frozen Elo via recomputed eligible-date sums, not production queues."""
    overall, courts = {}, {}
    changes, snapshots = defaultdict(list), defaultdict(list)
    deltas = {}
    released = set()
    results = {}
    multipliers = {'G': 1.25, 'M': 1.15, 'A': 1., 'C': .9, 'F': .8,
                   'S': .8, 'D': 1., 'O': 1., '15': .8, '25': .8}
    for date_index, (date, day) in enumerate(frame.groupby('tourney_date', sort=True)):
        date = pd.Timestamp(date)
        if progress is not None and date_index % 200 == 0:
            progress(f'Elo reference: processed date group {date_index}, now {date.date()}')
        for past in sorted(d for d in deltas if d <= date-pd.Timedelta(days=10) and d not in released):
            od, sd, people = deltas.pop(past)
            for person in people:
                delta = fsum(od[person])
                overall[person] = overall.get(person, 1500.) + delta
                changes[person].append(delta)
                snapshots[person].append(overall[person])
            for key, terms in sd.items():
                courts[key] += fsum(terms)
            released.add(past)
        od, sd, people = defaultdict(list), defaultdict(list), set()
        for r in day.to_dict('records'):
            a, b, court = r['playerA_id'], r['playerB_id'], r['surface']
            ra, rb = overall.get(a, 1500.), overall.get(b, 1500.)
            sa = sb = np.nan
            if pd.notna(court):
                sa = courts.setdefault((a, court), ra)
                sb = courts.setdefault((b, court), rb)
            if query_ids is None or r['match_id'] in query_ids:
                blended_a, blended_b = .7*sa+.3*ra, .7*sb+.3*rb
                f = dict(playerA_elo=ra, playerB_elo=rb, elo_diff=ra-rb,
                         playerA_surface_elo=sa, playerB_surface_elo=sb, surface_elo_diff=sa-sb,
                         playerA_blended_elo=blended_a, playerB_blended_elo=blended_b,
                         blended_elo_diff=blended_a-blended_b, elo_ratio=ra/rb,
                         surface_elo_ratio=sa/sb, blended_elo_ratio=blended_a/blended_b)
                for side, person in (('A', a), ('B', b)):
                    f[f'player{side}_elo_momentum'] = fsum(changes[person][-5:]) if len(changes[person]) >= 5 else 0.
                    f[f'player{side}_elo_volatility'] = float(np.std(snapshots[person][-10:])) if len(snapshots[person]) >= 10 else 0.
                results[r['match_id']] = f
            k = 24*multipliers.get(str(r['tourney_level']), 1.)*(1.1 if r['best_of'] == 5 else 1.)
            delta = k*(r['playerA_wins']-1/(1+10**((rb-ra)/400)))
            od[a].append(delta)
            od[b].append(-delta)
            if pd.notna(court):
                delta = k*(r['playerA_wins']-1/(1+10**((sb-sa)/400)))
                sd[(a, court)].append(delta)
                sd[(b, court)].append(-delta)
            people.update((a, b))
        deltas[date] = (od, sd, people)
    return results


def reconstruct(frame, query_ids=None, progress=None):
    if frame.match_id.isna().any() or not frame.match_id.is_unique:
        raise ValueError('Full history requires stable unique match_id')
    if not frame.playerA_wins.isin([0, 1]).all():
        raise ValueError('Binary outcomes required')
    if not frame.tourney_date.is_monotonic_increasing or frame.tourney_date.isna().any():
        raise ValueError('Sorted nonmissing reference dates required')
    query_ids = set(frame.match_id if query_ids is None else query_ids)
    if not query_ids.issubset(set(frame.match_id)):
        raise ValueError('Queries missing from full history')
    elo = elo_reference(frame, query_ids, progress)
    output = {}
    for q in frame.loc[frame.match_id.isin(query_ids)].to_dict('records'):
        cutoff = q['tourney_date']-pd.Timedelta(days=10)
        a, b, court = q['playerA_id'], q['playerB_id'], q['surface']
        relevant = frame.playerA_id.isin([a, b]) | frame.playerB_id.isin([a, b])
        history = frame.loc[frame.tourney_date.le(cutoff) & relevant].to_dict('records')
        f = dict(elo[q['match_id']])
        meetings = [r for r in history if {r['playerA_id'], r['playerB_id']} == {a, b}]
        for suffix, subset in (('h2h', meetings), ('surface_h2h', [r for r in meetings if pd.notna(court) and r['surface'] == court])):
            wa = sum((r['playerA_wins'] if r['playerA_id'] == a else 1-r['playerA_wins']) for r in subset)
            wb = len(subset)-wa
            f.update({f'playerA_{suffix}_wins': wa, f'playerB_{suffix}_wins': wb,
                      f'{suffix}_total': len(subset), f'{suffix}_diff': wa-wb})
        chosen = sorted(meetings, key=lambda r: r['tourney_date'], reverse=True)[:5]
        ambiguous = False
        if len(meetings) > 5:
            boundary = chosen[-1]['tourney_date']
            tied = [r for r in meetings if r['tourney_date'] == boundary]
            outcomes = {r['playerA_id'] if r['playerA_wins'] else r['playerB_id'] for r in tied}
            ambiguous = sum(r['tourney_date'] == boundary for r in chosen) < len(tied) and len(outcomes) > 1
        wa = np.nan if ambiguous else sum(r['playerA_wins'] if r['playerA_id'] == a else 1-r['playerA_wins'] for r in chosen)
        wb = len(chosen)-wa
        f.update(playerA_last5_h2h_wins=wa, playerB_last5_h2h_wins=wb,
                 last5_h2h_total=len(chosen), last5_h2h_diff=wa-wb)

        for side, person, opponent in (('A', a, b), ('B', b, a)):
            prefix = f'player{side}_'
            rows = []
            for r in history:
                if person not in (r['playerA_id'], r['playerB_id']):
                    continue
                own = 'A' if r['playerA_id'] == person else 'B'
                other = 'B' if own == 'A' else 'A'
                rows.append(dict(date=r['tourney_date'], won=int(r['playerA_wins']) if own == 'A' else 1-int(r['playerA_wins']),
                                 court=r['surface'], own=own, other=other, row=r,
                                 opponent=r[f'player{other}_id'], rank=r[f'player{other}_rank'],
                                 hand=r[f'player{other}_hand'], edition=edition(r)))
            surface_rows = [r for r in rows if pd.notna(court) and r['court'] == court]
            for group, subset in (
                ('top20', [r for r in rows if pd.notna(r['rank']) and r['rank'] <= 20]),
                ('hand', [r for r in rows if r['hand'] == q[f'player{"B" if side == "A" else "A"}_hand']])):
                wins, n = sum(r['won'] for r in subset), len(subset)
                f.update({prefix+group+'_wins': wins, prefix+group+'_losses': n-wins,
                          prefix+group+'_total': n, prefix+group+'_win_pct': ratio(wins, n)})
            groups = {'career': rows, 'surface_career': surface_rows,
                      '52w': [r for r in rows if r['date'] > cutoff-pd.Timedelta(days=364)]}
            for group in ('career', 'surface_career', 'last10', 'last25', 'surface10', 'surface25', '52w'):
                if group in ('last10', 'last25', 'surface10', 'surface25'):
                    wins, n = block_window(surface_rows if group.startswith('surface') else rows,
                                           10 if group.endswith('10') else 25)
                else:
                    wins, n = sum(r['won'] for r in groups[group]), len(groups[group])
                if group.startswith('surface') and pd.isna(court):
                    wins = n = np.nan
                f.update({prefix+group+'_wins': wins, prefix+group+'_losses': n-wins,
                          prefix+group+'_matches': n, prefix+group+'_pct': ratio(wins, n)})

            serve = [r for r in rows if valid_serve(r['row'], r['own'])]
            def total(records, field, serving='own'):
                return sum(r['row'][f'player{r[serving]}_{field}'] for r in records)
            points, first = total(serve, 'svpt'), total(serve, '1stIn')
            won1, won2 = total(serve, '1stWon'), total(serve, '2ndWon')
            for name, numerator, denominator in (
                ('ace_pct', total(serve, 'ace'), points), ('df_pct', total(serve, 'df'), points),
                ('1stIn_pct', first, points), ('1stWon_pct', won1, first),
                ('2ndWon_pct', won2, points-first), ('career_p', won1+won2, points)):
                f[prefix+name] = ratio(numerator, denominator, np.nan)
            f[prefix+'serve_matches'] = len(serve)
            for label, subset in (
                ('matchup', [r for r in serve if r['opponent'] == opponent]),
                ('surface', [r for r in serve if pd.notna(court) and r['court'] == court])):
                n = total(subset, 'svpt')
                f[prefix+label+'_p'] = ratio(total(subset, '1stWon')+total(subset, '2ndWon'), n, np.nan)
                f[prefix+('matchup_service_points' if label == 'matchup' else 'surface_serve_matches')] = n if label == 'matchup' else len(subset)
            returning = [r for r in rows if valid_serve(r['row'], r['other'])]
            rp = total(returning, 'svpt', 'other')
            rw = rp-total(returning, '1stWon', 'other')-total(returning, '2ndWon', 'other')
            bp = []
            for r in rows:
                faced, saved = (r['row'][f'player{r["other"]}_{s}'] for s in ('bpFaced', 'bpSaved'))
                if pd.notna(faced) and pd.notna(saved) and np.isfinite(faced) and np.isfinite(saved) and 0 <= saved <= faced:
                    bp.append(r)
            f[prefix+'return_pct'] = ratio(rw, rp)
            f[prefix+'return_matches'] = len(returning)
            f[prefix+'bp_convert_pct'] = ratio(total(bp, 'bpFaced', 'other')-total(bp, 'bpSaved', 'other'), total(bp, 'bpFaced', 'other'))
            for days in (30, 60):
                subset = [r for r in rows if r['date'] > cutoff-pd.Timedelta(days=days)]
                f[prefix+f'matches_{days}d'] = len(subset)
                if days == 30:
                    wins = sum(r['won'] for r in subset)
                    f.update({prefix+'wins_30d': wins, prefix+'losses_30d': len(subset)-wins,
                              prefix+'winpct_30d': ratio(wins, len(subset))})
            for weeks in (5, 10):
                events = {r['edition'] for r in rows if r['edition'] is not None and r['date'] > cutoff-pd.Timedelta(weeks=weeks)}
                f[prefix+f'tournaments_{weeks}w'] = len(events)
            for label, key in (('tourney', 'tourney_name'), ('level', 'tourney_level')):
                value = q[key]
                subset = [r for r in rows if pd.notna(value) and str(value).strip() and pd.notna(r['row'][key]) and str(r['row'][key]).strip() == str(value).strip()]
                f[prefix+label+'_matches'] = len(subset)
                if label == 'tourney':
                    f[prefix+'tourney_appearances'] = len({r['edition'] for r in subset if r['edition'] is not None})
                else:
                    wins = sum(r['won'] for r in subset)
                    f.update({prefix+'level_wins': wins, prefix+'level_losses': len(subset)-wins,
                              prefix+'level_win_pct': ratio(wins, len(subset))})
        output[q['match_id']] = f
    return pd.DataFrame.from_dict(output, orient='index').rename_axis('match_id').sort_index()
