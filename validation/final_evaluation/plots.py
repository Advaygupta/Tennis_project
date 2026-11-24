"""Aggregate-only reviewer figures. No identifiers or individual predictions."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shared import losses

COLORS = {'V2.1': '#167b79', 'V1.1': '#b04e35'}


def make_figures(output, y, probabilities, summary):
    output = Path(output)
    output.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'tennis-final-evaluation-20261124'})
    foot = 'Frozen test N=51,666; P(A wins). Intervals where shown: pointwise event bootstrap 95%. Pairing conditional on V1 order.'

    def finish(fig, name):
        fig.text(.03, .015, foot, fontsize=8, color='#444444')
        fig.tight_layout(rect=(0, .05, 1, .95))
        fig.savefig(output/(name+'.svg'), metadata={'Date': None}, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Reliability and bin populations', fontsize=15)
    event = summary['uncertainty']['event']['binning']
    for j, method in enumerate(['width', 'frequency']):
        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], '--', color='#888888', lw=1, label='Perfect calibration')
        for model in probabilities:
            entry = next(e for e in event if e['model'] == model and e['binning'] == method and e['requested_bins'] == 10)
            rows = [r for r in entry['rows'] if r['n']]
            x = np.array([r['mean_probability'] for r in rows])
            obs = np.array([r['observed_frequency'] for r in rows])
            interval = np.array([r['observed_frequency_uncertainty']['ci95'] for r in rows])
            ax.plot(x, obs, 'o-', color=COLORS[model], ms=4, label=model)
            ax.vlines(x, interval[:, 0], interval[:, 1], color=COLORS[model], alpha=.65)
            axes[1, j].plot(x, [r['n'] for r in rows], 'o-', color=COLORS[model], label=model)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted probability', ylabel='Observed win frequency',
               title='10 equal-width bins' if method == 'width' else '10 quantile bins (ties together)')
        ax.legend(fontsize=8)
        axes[1, j].set(xlim=(0, 1), ylim=(0, None), xlabel='Mean predicted probability', ylabel='Bin count')
    finish(fig, 'reliability')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    fig.suptitle('Calibration deviation: observed minus predicted', fontsize=15)
    for ax, method in zip(axes, ['width', 'frequency']):
        ax.axhline(0, color='#888888', ls='--', lw=1)
        for model in probabilities:
            entry = next(e for e in event if e['model'] == model and e['binning'] == method and e['requested_bins'] == 10)
            rows = [r for r in entry['rows'] if r['n']]
            x = np.array([r['mean_probability'] for r in rows])
            gap = np.array([r['observed_minus_predicted'] for r in rows])
            interval = np.array([r['observed_minus_predicted_uncertainty']['ci95'] for r in rows])
            ax.plot(x, gap, 'o-', color=COLORS[model], label=model)
            ax.vlines(x, interval[:, 0], interval[:, 1], color=COLORS[model], alpha=.65)
        ax.set(xlim=(0, 1), ylim=(-.22, .22), xlabel='Mean predicted probability', ylabel='Observed - predicted', title=method)
        ax.legend()
    finish(fig, 'calibration_deviation')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    fig.suptitle('Probability dispersion and predicted-winner confidence', fontsize=15)
    for model, p in probabilities.items():
        axes[0].hist(p, bins=np.linspace(0, 1, 41), histtype='step', density=False, color=COLORS[model], label=model)
        rows = summary['calibration'][model]['winner_confidence_groups']
        occupied = [r for r in rows if r['n']]
        x = [r['mean_confidence'] for r in occupied]
        axes[1].plot(x, [r['accuracy'] for r in occupied], 'o-', color=COLORS[model], label=model)
        for r in occupied:
            axes[1].annotate(f"n={r['n']:,}", (r['mean_confidence'], r['accuracy']),
                xytext=(0, 7 if model == 'V2.1' else -13), textcoords='offset points', ha='center', fontsize=7)
    axes[0].set(xlim=(0, 1), xlabel='P(A wins)', ylabel='Matches per 0.025-wide bin')
    axes[1].plot([.5, 1], [.5, 1], '--', color='#888888')
    axes[1].set(xlim=(.49, 1.01), ylim=(.45, 1.02), xlabel='Mean predicted-winner confidence', ylabel='Observed winner accuracy')
    for ax in axes:
        ax.legend()
    finish(fig, 'confidence')

    a, b = probabilities['V2.1'], probabilities['V1.1']
    delta = losses(y, a)[:, 2]-losses(y, b)[:, 2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    fig.suptitle('Paired log-loss difference and concentration', fontsize=15)
    axes[0].hist(delta, bins=60, color='#475b78')
    axes[0].set(yscale='log', xlabel='V2.1 loss - V1.1 loss (all observations)', ylabel='Matches (log count scale)')
    axes[0].axvline(0, color='#888888', ls='--')
    order = np.argsort(-abs(delta), kind='stable')
    fractions = np.r_[0, np.linspace(.001, 1, 300)]
    cumulative = np.r_[0., np.cumsum(delta[order])/len(delta)]
    axes[1].plot(fractions*100, cumulative[np.ceil(fractions*len(delta)).astype(int)], color='#475b78')
    axes[1].axhline(delta.mean(), color='#888888', ls='--', label='Full mean difference')
    axes[1].set(xlabel='Top % by absolute paired difference (descending)', ylabel='Contribution to full mean loss difference', xlim=(0, 100))
    axes[1].legend(fontsize=8)
    finish(fig, 'loss_concentration')
