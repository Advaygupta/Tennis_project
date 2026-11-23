"""Descriptive diagnostics; never tune a model or choose a threshold."""
import numpy as np
from metrics import losses, discordance, metrics


def quantiles(values):
    return dict(zip(('min', 'p01', 'p05', 'p25', 'median', 'p75', 'p95', 'p99', 'max'),
                    np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist()))


def diagnostics(frame, a, b):
    y = frame.playerA_wins.to_numpy()
    la, lb = losses(y, a), losses(y, b)
    ll = la[:, 2]-lb[:, 2]
    n_top = max(1, int(np.ceil(.01*len(y))))
    largest = np.argsort(abs(ll))[-n_top:]
    true_a, true_b = np.where(y, a, 1-a), np.where(y, b, 1-b)
    events = frame[['tourney_id']].copy()
    events['loss_difference'] = ll
    event_sums = events.groupby('tourney_id').loss_difference.sum().to_numpy()
    return dict(discordance=discordance(y, a, b),
        absolute_probability_difference=quantiles(abs(a-b)),
        per_match_log_loss_A_minus_B=quantiles(ll),
        overconfidence=dict(A_true_outcome_probability_below_005=int((true_a < .05).sum()),
                            B_true_outcome_probability_below_005=int((true_b < .05).sum()),
                            A_worse_log_loss_by_more_than_1=int((ll > 1).sum()),
                            B_worse_log_loss_by_more_than_1=int((ll < -1).sum())),
        concentration=dict(largest_absolute_loss_difference_rows=n_top,
            top_1pct_share_of_absolute_loss_difference=float(abs(ll[largest]).sum()/abs(ll).sum()) if abs(ll).sum() else 0.,
            top_1pct_signed_contribution_to_mean=float(ll[largest].sum()/len(y)),
            largest_10_events_share_of_absolute_net_event_loss_difference=float(np.sort(abs(event_sums))[-10:].sum()/abs(event_sums).sum()) if abs(event_sums).sum() else 0.),
        interpretation='Descriptive concentration and overconfidence counts; no event or observation is removed using these diagnostics.')


def subgroups(frame, a, b):
    work = frame.copy()
    work['year'] = work.tourney_date.str[:4]
    rows = []
    for column in ('year', 'surface', 'tourney_level'):
        for value, positions in work.groupby(column, dropna=False).indices.items():
            y = work.iloc[positions].playerA_wins.to_numpy()
            ma, mb = metrics(y, a[positions]), metrics(y, b[positions])
            rows.append(dict(category=column, value=str(value), n=len(positions),
                             A=ma, B=mb, difference={k: ma[k]-mb[k] if ma[k] is not None and mb[k] is not None else None for k in ma}))
    return dict(status='Exploratory, no subgroup hypothesis tests or selection', rows=rows)
