"""Exact paired multinomial bootstrap, including weighted AUC in each replicate."""
import numpy as np
import pandas as pd
from metrics import NAMES, WeightedAUC, losses, metrics, validate


def cluster_codes(frame, scheme):
    dates = pd.to_datetime(frame.tourney_date)
    if scheme == 'match':
        labels = np.arange(len(frame))
    elif scheme == 'event':
        labels = frame.tourney_id.astype(str).to_numpy()
    elif scheme == 'calendar_28d':
        # Fixed origin, never restarted per subset. Whole events are assigned
        # using their earliest recorded reference date in the complete test set.
        labels = ((pd.to_datetime(frame.event_reference_date) - pd.Timestamp('1970-01-05')).dt.days // 28).to_numpy()
    else:
        raise ValueError('Unknown scheme')
    if pd.isna(labels).any() or dates.isna().any():
        raise ValueError('Missing clustering metadata')
    codes, values = pd.factorize(labels, sort=True)
    return codes, len(values)


def replicate_weights(rng, codes, n_clusters, batch):
    # Sample K entire units with replacement, retaining every member of a unit.
    counts = rng.multinomial(n_clusters, np.full(n_clusters, 1/n_clusters), size=batch)
    return counts[:, codes]


def paired_bootstrap(y, a, b, codes, replicates=5000, seed=20261123, batch=16):
    y, a = validate(y, a)
    _, b = validate(y, b)
    codes = np.asarray(codes)
    if len(codes) != len(y) or (codes < 0).any():
        raise ValueError('Bad cluster codes')
    unique = np.unique(codes)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError('Cluster codes must be contiguous')
    if len(unique) < 2 or replicates < 20:
        raise ValueError('At least two resampling units and 20 replicates required')
    rng = np.random.default_rng(seed)
    au, bu = WeightedAUC(y, a), WeightedAUC(y, b)
    difference = losses(y, a) - losses(y, b)
    samples = np.empty((replicates, 4))
    for start in range(0, replicates, batch):
        size = min(batch, replicates-start)
        weights = replicate_weights(rng, codes, len(unique), size)
        total = weights.sum(axis=1)
        means = (weights @ difference)/total[:, None]
        samples[start:start+size, 0] = means[:, 0]
        samples[start:start+size, 1] = au(weights)-bu(weights)
        samples[start:start+size, 2:] = means[:, 1:]
    ma, mb = metrics(y, a), metrics(y, b)
    result = dict(replicates=replicates, seed=seed, n=len(y), resampling_units=len(unique),
                  method='paired multinomial pairs/whole-cluster bootstrap; percentile interval',
                  interval_level=.95, metrics={})
    for j, name in enumerate(NAMES):
        finite = samples[:, j][np.isfinite(samples[:, j])]
        entry = dict(observed=ma[name]-mb[name] if ma[name] is not None and mb[name] is not None else None,
                     ci95=np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                     valid_replicates=len(finite), undefined_replicates=replicates-len(finite))
        # An inexpensive Monte Carlo diagnostic, not an extra confidence interval.
        entry['half_run_ci95'] = [np.quantile(z[np.isfinite(z)], [.025, .975]).tolist()
                                   if np.isfinite(z).any() else None
                                   for z in np.array_split(samples[:, j], 2)]
        result['metrics'][name] = entry
    return result
