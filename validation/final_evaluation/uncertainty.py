"""Point 3 whole-event/block multinomial resampling, aggregated before drawing."""
import numpy as np
from calibration import reliability, bin_assignment
from shared import cluster_codes, replicate_weights, losses


def grouped_bootstrap(frame, probabilities, replicates=5000, seed=20261124, scheme='event'):
    if replicates < 20:
        raise ValueError('At least 20 replicates required')
    codes, k = cluster_codes(frame, scheme)
    if k < 2:
        raise ValueError('At least two clusters required')
    y = frame.playerA_wins.to_numpy()
    specs, columns = [], []

    def add(values):
        columns.append(np.bincount(codes, weights=values, minlength=k))
        return len(columns)-1

    total_col = add(np.ones(len(y)))
    y_col = add(y)
    model_specs = {}
    for model, p in probabilities.items():
        ll = losses(y, p)
        model_specs[model] = dict(mean_probability=add(p), brier_score=add(ll[:, 1]),
                                  log_loss=add(ll[:, 2]))
        for method in ['width', 'frequency']:
            for bins in [5, 10, 20]:
                labels, edges = bin_assignment(p, bins, method)
                indices = []
                for j in range(len(edges)-1):
                    m = labels == j
                    indices.append([add(m), add(m*p), add(m*y)])
                specs.append(dict(model=model, scheme=method, bins=bins,
                    indices=indices, point=reliability(y, p, bins, method)))
    design = np.asarray(columns).T
    samples = np.empty((replicates, len(columns)))
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, 16):
        count = min(16, replicates-start)
        weights = replicate_weights(rng, np.arange(k), k, count)
        samples[start:start+count] = weights@design

    def ci(values):
        valid = np.isfinite(values)
        return dict(ci95=np.quantile(values[valid], [.025, .975]).tolist() if valid.any() else None,
                    valid_replicates=int(valid.sum()), undefined_replicates=int((~valid).sum()))

    def ratio(num, den):
        return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)

    output = dict(scheme=scheme, resampling_units=k, n=len(y), replicates=replicates,
        seed=seed, models={}, binning=[], differences={},
        convention='Whole-cluster paired multinomial percentile intervals, match-weighted. '
          'Cutpoints fixed at original-sample values, including quantile bins; memberships do not split ties. '
          'Empty resampled bins are undefined and counted. ECE excludes empty bins; MCE considers occupied bins. '
          'Pointwise descriptive intervals conditional on fixed models and bin boundaries; ECE has finite-sample bias. '
          'Repeated players across clusters and longer serial dependence remain.')
    draws = {}
    total = samples[:, total_col]
    for model, indices in model_specs.items():
        draws[model] = {name: samples[:, idx]/total for name, idx in indices.items()}
        draws[model]['mean_bias'] = draws[model]['mean_probability']-samples[:, y_col]/total
        output['models'][model] = {name: ci(v) for name, v in draws[model].items()}
    for name in draws['V2.1']:
        output['differences'][name] = ci(draws['V2.1'][name]-draws['V1.1'][name])
    paired_bins = {}
    for spec in specs:
        count, pred, obs = samples[:, np.asarray(spec['indices'])].transpose(2, 0, 1)
        mean_p, frequency = ratio(pred, count), ratio(obs, count)
        gap = frequency-mean_p
        ece = np.nansum(count*np.abs(gap), axis=1)/total
        mce = np.nanmax(np.abs(gap), axis=1)
        rows = []
        for j, point in enumerate(spec['point']['rows']):
            rows.append(dict(**point, mean_probability_uncertainty=ci(mean_p[:, j]),
                observed_frequency_uncertainty=ci(frequency[:, j]),
                observed_minus_predicted_uncertainty=ci(gap[:, j])))
        output['binning'].append(dict(model=spec['model'], binning=spec['scheme'],
            requested_bins=spec['bins'], actual_bins=len(rows), ECE=ci(ece), MCE=ci(mce), rows=rows))
        paired_bins.setdefault((spec['scheme'], spec['bins']), {})[spec['model']] = (ece, mce)
    output['binning_differences_A_minus_B'] = [dict(binning=key[0], bins=key[1],
        ECE=ci(value['V2.1'][0]-value['V1.1'][0]),
        MCE=ci(value['V2.1'][1]-value['V1.1'][1])) for key, value in paired_bins.items()]
    return output
