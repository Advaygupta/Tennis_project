"""Independent binary metrics. No sklearn or original evaluation imports."""
import math
import numpy as np

NAMES = ('accuracy', 'roc_auc', 'brier_score', 'log_loss')
EPS = np.finfo(np.float64).eps


def validate(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError('Outcomes and probabilities must be nonempty aligned vectors')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Outcomes must be exactly zero or one')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probability')
    return y.astype(np.int8), p


def losses(y, p):
    y, p = validate(y, p)
    z = np.clip(p, EPS, 1 - EPS)
    return np.column_stack(((p >= .5) == y, (p - y)**2,
                            -(y*np.log(z) + (1-y)*np.log1p(-z))))


class WeightedAUC:
    """Exact pairwise ranking AUC with half credit for ties; sort only once."""
    def __init__(self, y, p):
        y, p = validate(y, p)
        self.order = np.argsort(p, kind='stable')
        self.y = y[self.order]
        scores = p[self.order]
        self.starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]

    def __call__(self, weights):
        w = np.atleast_2d(np.asarray(weights, dtype=float))[:, self.order]
        pos = np.add.reduceat(w*self.y, self.starts, axis=1)
        neg = np.add.reduceat(w*(1-self.y), self.starts, axis=1)
        below = np.cumsum(neg, axis=1) - neg
        denominator = pos.sum(axis=1)*neg.sum(axis=1)
        numerator = (pos*(below + .5*neg)).sum(axis=1)
        return np.divide(numerator, denominator, out=np.full(len(w), np.nan),
                         where=denominator > 0)


def metrics(y, p):
    y, p = validate(y, p)
    means = losses(y, p).mean(axis=0)
    auc = WeightedAUC(y, p)(np.ones(len(y)))[0]
    return dict(accuracy=float(means[0]), roc_auc=float(auc) if np.isfinite(auc) else None,
                brier_score=float(means[1]), log_loss=float(means[2]))


def discordance(y, a, b):
    y, a = validate(y, a)
    _, b = validate(y, b)
    ca, cb = (a >= .5) == y, (b >= .5) == y
    x, z = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n, k = x+z, min(x, z)
    if n == 0:
        pvalue = 1.
    else:
        # Exact two-sided Binomial(n, .5) tail, evaluated stably in log space.
        logs = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2)
                for i in range(k+1)]
        peak = max(logs)
        pvalue = min(1., 2*math.exp(peak)*sum(math.exp(t-peak) for t in logs))
    return dict(both_correct=int((ca & cb).sum()), both_wrong=int((~ca & ~cb).sum()),
                A_only_correct=x, B_only_correct=z, discordant=n,
                mcnemar_exact_two_sided_p=pvalue,
                null='Equal marginal accuracy; discordant directions equiprobable',
                assumption='Independent match pairs; descriptive reference only under tennis dependence')
