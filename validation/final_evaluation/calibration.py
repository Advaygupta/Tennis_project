"""Descriptive test-set diagnostics only; never return adjusted predictions."""
import numpy as np
from shared import validate, EPS


def bin_assignment(p, bins=10, scheme='width'):
    _, p = validate(np.zeros(len(p)), p)
    if not isinstance(bins, int) or bins < 2:
        raise ValueError('At least two bins required')
    if scheme == 'width':
        edges = np.linspace(0., 1., bins+1)
    elif scheme == 'frequency':
        # Ties stay together. Duplicate quantile cuts collapse explicitly.
        edges = np.unique(np.r_[0., np.quantile(p, np.arange(1, bins)/bins), 1.])
    else:
        raise ValueError('Unknown binning scheme')
    codes = np.searchsorted(edges[1:-1], p, side='right')
    return codes, edges


def reliability(y, p, bins=10, scheme='width'):
    y, p = validate(y, p)
    codes, edges = bin_assignment(p, bins, scheme)
    rows = []
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = codes == j
        count = int(mask.sum())
        pred = float(p[mask].mean()) if count else None
        obs = float(y[mask].mean()) if count else None
        rows.append(dict(bin=j, lower=float(lo), upper=float(hi), n=count,
                         mean_probability=pred, observed_frequency=obs,
                         observed_minus_predicted=obs-pred if count else None,
                         sparse=count < 100))
    ece = sum(r['n']*abs(r['observed_minus_predicted']) for r in rows if r['n'])/len(y)
    mce = max(abs(r['observed_minus_predicted']) for r in rows if r['n'])
    return dict(scheme=scheme, requested_bins=bins, actual_bins=len(rows),
                n=len(y), edges=edges.tolist(), ECE=float(ece), MCE=float(mce), rows=rows)


def sigmoid(z):
    return np.exp(-np.logaddexp(0., -np.asarray(z)))


def logistic_diagnostic(y, p, offset_only=False, weights=None):
    """MLE of diagnostic logit(E[Y])=a+b*logit(p); a only fixes b=1.

    This estimates a statistical diagnostic using test outcomes. It never applies
    the coefficients to the predictions or evaluates recalibrated performance.
    Nonexistence/nonconvergence is returned explicitly, without regularization.
    """
    y, p = validate(y, p)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != y.shape or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Invalid diagnostic weights')
    clipped = np.clip(p, EPS, 1-EPS)
    x = np.log(clipped)-np.log1p(-clipped)
    if not 0 < np.dot(w, y) < w.sum() or (not offset_only and np.ptp(x[w > 0]) < 1e-12):
        return dict(status='undefined: one outcome or constant logit', intercept=None, slope=None)
    X = np.ones((len(y), 1)) if offset_only else np.column_stack((np.ones(len(y)), x))
    offset = x if offset_only else np.zeros(len(y))
    # Neutral start avoids saturated information at exact probability boundaries.
    beta = np.zeros(X.shape[1])

    def objective(b):
        z = X@b+offset
        return float(np.dot(w, np.logaddexp(0., z)-y*z)/w.sum())

    status = 'nonconverged'
    for iteration in range(100):
        z = X@beta+offset
        mu = sigmoid(z)
        score = X.T@(w*(y-mu))/w.sum()
        information = X.T@((w*mu*(1-mu))[:, None]*X)/w.sum()
        if not np.isfinite(information).all() or np.linalg.cond(information) > 1e14:
            break
        step = np.linalg.solve(information, score)
        scale = 1.
        old = objective(beta)
        while scale > 2**-30 and objective(beta+scale*step) > old+1e-14:
            scale *= .5
        beta += scale*step
        if np.max(np.abs(beta)) > 100:
            break
        if np.max(np.abs(score)) < 1e-9 and np.max(np.abs(scale*step)) < 1e-7:
            status = 'converged'
            break
    return dict(status=status, intercept=float(beta[0]) if status == 'converged' else None,
                slope=(1. if offset_only else float(beta[1])) if status == 'converged' else None,
                iterations=iteration+1, clipped_logits=int((p != clipped).sum()))


def confidence_groups(y, p):
    y, p = validate(y, p)
    confidence = np.maximum(p, 1-p)
    correct = ((p >= .5) == y).astype(int)  # .5 predicts A, exactly once per match
    rows = []
    for lo, hi in zip([.5, .6, .8, .95, .99], [.6, .8, .95, .99, 1.]):
        mask = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
        n = int(mask.sum())
        rows.append(dict(lower=lo, upper=hi, n=n,
            mean_confidence=float(confidence[mask].mean()) if n else None,
            accuracy=float(correct[mask].mean()) if n else None,
            confidence_minus_accuracy=float((confidence[mask]-correct[mask]).mean()) if n else None))
    return rows


def calibration_summary(y, p):
    y, p = validate(y, p)
    return dict(n=len(y), brier_score=float(np.mean((p-y)**2)),
        mean_probability=float(p.mean()), observed_positive_frequency=float(y.mean()),
        mean_bias_predicted_minus_observed=float((p-y).mean()),
        probability_sd=float(p.std()), mean_winner_confidence=float(np.maximum(p, 1-p).mean()),
        probability_at_boundary=int(((p == 0) | (p == 1)).sum()),
        near_boundary_001=int(((p < .01) | (p > .99)).sum()),
        near_boundary_005=int(((p < .05) | (p > .95)).sum()),
        joint_logistic_diagnostic=logistic_diagnostic(y, p),
        calibration_in_the_large=logistic_diagnostic(y, p, offset_only=True),
        binning=[reliability(y, p, k, method) for method in ['width', 'frequency'] for k in [5, 10, 20]],
        winner_confidence_groups=confidence_groups(y, p))
