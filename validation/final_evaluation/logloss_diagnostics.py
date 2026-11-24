"""Explicit tail ranking and additive attribution; no observations are discarded."""
import numpy as np
from shared import validate, losses, EPS


def distribution(x):
    x = np.asarray(x, dtype=float)
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=0)),
        **dict(zip(['min', 'p01', 'p05', 'p25', 'median', 'p75', 'p90', 'p95', 'p99', 'p999', 'max'],
                   np.quantile(x, [0, .01, .05, .25, .5, .75, .9, .95, .99, .999, 1]).tolist())))


def tail_attribution(a_loss, b_loss, fractions=(.01, .05, .10)):
    a, b = np.asarray(a_loss, float), np.asarray(b_loss, float)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or not np.isfinite(a+b).all():
        raise ValueError('Invalid paired losses')
    delta = a-b
    rankings = dict(absolute_paired_difference=np.abs(delta), signed_A_minus_B=delta,
                    signed_B_minus_A=-delta, individual_A_loss=a, individual_B_loss=b)
    result = []
    for name, values in rankings.items():
        order = np.argsort(-values, kind='stable')
        for fraction in fractions:
            if not 0 < fraction <= 1:
                raise ValueError('Invalid tail fraction')
            k = max(1, int(np.ceil(fraction*len(a))))
            selected = order[:k]
            contribution = float(delta[selected].sum()/len(a))
            rest = float((delta.sum()-delta[selected].sum())/len(a))
            result.append(dict(ranking=name, nominal_fraction=fraction, selected_n=k,
                selected_A_loss_share=float(a[selected].sum()/a.sum()) if a.sum() else None,
                selected_B_loss_share=float(b[selected].sum()/b.sum()) if b.sum() else None,
                signed_contribution_to_full_mean=contribution,
                remaining_contribution_to_full_mean=rest,
                share_of_net_mean_difference=contribution/float(delta.mean()) if delta.mean() != 0 else None,
                selected_mean_difference=float(delta[selected].mean()),
                remaining_mean_difference=float((delta.sum()-delta[selected].sum())/(len(a)-k)) if k < len(a) else None))
    return result


def diagnose_logloss(y, a, b):
    y, a = validate(y, a)
    _, b = validate(y, b)
    la, lb = losses(y, a)[:, 2], losses(y, b)[:, 2]
    delta = la-lb
    true_a, true_b = np.where(y, a, 1-a), np.where(y, b, 1-b)
    adverse = []
    for t in [.01, .05, .10, .20]:
        for name, true_p, ll in [('V2.1', true_a, la), ('V1.1', true_b, lb)]:
            m = true_p < t
            adverse.append(dict(model=name, observed_outcome_probability_below=t, n=int(m.sum()),
                fraction=float(m.mean()), contribution_to_model_mean_loss=float(ll[m].sum()/len(y))))
    # Paired regions use B's observed-outcome probability; groups partition N.
    regions = []
    for lo, hi in zip([0, .01, .05, .1, .2], [.01, .05, .1, .2, 1.]):
        m = (true_b >= lo) & ((true_b < hi) if hi < 1 else (true_b <= hi))
        regions.append(dict(B_true_probability_lower=lo, upper=hi, n=int(m.sum()),
            signed_contribution_to_full_mean=float(delta[m].sum()/len(y))))
    clipping = []
    for epsilon in [EPS, 1e-15, 1e-12, 1e-6]:
        values = []
        for p in [a, b]:
            z = np.clip(p, epsilon, 1-epsilon)
            values.append(float(np.mean(-(y*np.log(z)+(1-y)*np.log1p(-z)))))
        clipping.append(dict(epsilon=float(epsilon), A=values[0], B=values[1], delta=values[0]-values[1]))
    def corr(x, z):
        return float(np.corrcoef(x, z)[0, 1]) if np.std(x)>0 and np.std(z)>0 else None
    return dict(A=distribution(la), B=distribution(lb), A_minus_B=distribution(delta),
        A_lower_loss_n=int((delta < 0).sum()), A_higher_loss_n=int((delta > 0).sum()),
        equal_loss_n=int((delta == 0).sum()), adverse_predictions=adverse,
        tail_attribution=tail_attribution(la, lb), B_true_outcome_regions=regions,
        clipping_sensitivity=clipping,
        correlations=dict(delta_loss_with_absolute_probability_difference=corr(delta, abs(a-b)),
            delta_loss_with_B_winner_confidence=corr(delta, np.maximum(b, 1-b))),
        interpretation='A=V2.1, B=V1.1; all paired quantities conditional on V1 saved order. '
          'Descending stable ranking; ties resolved by preserved row position. '
          'Contributions divide by full N, not selected N. Net shares can exceed 100% or be negative.')
