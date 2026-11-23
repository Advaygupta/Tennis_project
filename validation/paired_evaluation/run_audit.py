"""New, isolated evaluation only. Never execute notebooks or ML inference."""
import argparse
import json
import os
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow
from alignment import align, add_scoring
from diagnostics import diagnostics, subgroups
from inventory import inventory, digest
from metrics import metrics, NAMES
from paired_bootstrap import paired_bootstrap, cluster_codes

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n', encoding='utf8')


def evaluate(frame, a, b, replicates, schemes, seed):
    y = frame.playerA_wins.to_numpy()
    result = dict(n=len(y), A=metrics(y, a), B=metrics(y, b), bootstrap={})
    for offset, scheme in enumerate(schemes):
        codes, count = cluster_codes(frame, scheme)
        print(f'  {scheme}: n={len(y)}, units={count}, replicates={replicates}', flush=True)
        result['bootstrap'][scheme] = paired_bootstrap(y, a, b, codes, replicates, seed+offset)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=REPO/'reproduction/artifacts')
    parser.add_argument('--scoring-sidecar', type=Path, default=REPO/'validation/scoring/historical/audit_private/saved_probability_comparison.csv')
    parser.add_argument('--output', type=Path, required=True, help='NEW public output directory; existing directories rejected')
    parser.add_argument('--private-output', type=Path, required=True, help='NEW ignored/private directory; never publish its rows')
    parser.add_argument('--replicates', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261123)
    args = parser.parse_args()
    started = time.perf_counter()
    output, private = args.output.resolve(), args.private_output.resolve()
    if output.exists() or private.exists() or output == private or output in private.parents or private in output.parents:
        raise ValueError('Choose distinct NEW output directories; no existing files are overwritten')
    forbidden = [args.artifacts.resolve(), (REPO/'notebooks').resolve(), (REPO/'models').resolve(),
                 (REPO/'validation/scoring').resolve(), (REPO/'validation/temporal').resolve()]
    if any(p == root or root in p.parents or p in root.parents for p in (output, private) for root in forbidden):
        raise ValueError('Output overlaps preserved project artifacts/audits')
    paths = [args.artifacts/name for name in ('v1_1_results.joblib', 'v2_1_test_match_probabilities.parquet', 'matches_v1_1_metadata.csv')]
    if args.scoring_sidecar.exists():
        paths.append(args.scoring_sidecar)
    checksums = {str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name: digest(p) for p in paths}
    v1 = joblib.load(paths[0])  # trusted user-supplied DataFrame, never a model checkpoint
    v2 = pd.read_parquet(paths[1])
    metadata = pd.read_csv(paths[2])
    inv = inventory(REPO, v1, v2, metadata)
    paired, ordered_v2, strict, alignment = align(v1, v2, metadata)
    paired['event_reference_date'] = paired.groupby('tourney_id').tourney_date.transform('min')
    # Calendar blocks anchor each full edition using its earliest recorded date;
    # subsets retain the full-population anchor. No exact match dates are claimed.
    event_dates = paired.groupby('tourney_id').tourney_date.nunique()
    alignment['events_spanning_multiple_reference_dates'] = int((event_dates > 1).sum())
    alignment['events'] = int(paired.tourney_id.nunique())
    alignment['test_date_range'] = [paired.tourney_date.min(), paired.tourney_date.max()]
    has_scoring = args.scoring_sidecar.exists()
    if has_scoring:
        paired = add_scoring(paired, pd.read_csv(args.scoring_sidecar))
    # All inputs and alignment must validate before any output directory exists.
    output.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)
    write_json(output/'inventory.json', inv)
    a, b = paired.v2_probability.to_numpy(), paired.v1_probability.to_numpy()
    expected = json.loads((REPO/'reproduction/expected_metrics.json').read_text(encoding='utf8'))
    full_metrics = {'v1_1': metrics(paired.playerA_wins, b), 'v2_1': metrics(paired.playerA_wins, a)}
    reproduction_errors = {model: {key: values[key]-expected[model][key] for key in NAMES}
                           for model, values in full_metrics.items()}
    if len(paired) != expected['test_matches'] or any(abs(error)>1e-12 for m in reproduction_errors.values() for error in m.values()):
        raise ValueError('Original metric reproduction discrepancy; investigate before interpreting results')
    print('Primary, conditional on V1 preserved ordering', flush=True)
    results = {'full_original_conditional': evaluate(paired, a, b, args.replicates,
                ['match', 'event', 'calendar_28d'], args.seed)}
    populations = [('unique_shared_context', strict)]
    if has_scoring:
        populations += [('completed', paired.completed.to_numpy()),
                        ('supported_policy', paired.supported_policy.to_numpy()),
                        ('completed_supported_policy', (paired.completed & paired.supported_policy).to_numpy())]
    for offset, (name, mask) in enumerate(populations, 1):
        print(name, flush=True)
        # Both dependence-aware schemes for each pre-specified sensitivity;
        # also use iid reference on the tiny unique-context subset.
        schemes = ['match', 'event', 'calendar_28d'] if name=='unique_shared_context' else ['event', 'calendar_28d']
        results[name] = evaluate(paired.loc[mask].copy(), a[mask], b[mask], args.replicates,
                                 schemes, args.seed+10*offset)
        results[name]['excluded_from_full'] = int((~mask).sum())
    scoring = None
    if has_scoring:
        print('Separate scoring-only correction sensitivity (A=corrected V1, B=original V1)', flush=True)
        correction = paired.event_corrected_probability.to_numpy()
        scoring = evaluate(paired, correction, b, args.replicates,
                           ['match', 'event', 'calendar_28d'], args.seed+100)
        scoring.update(A_identity='V1 frozen p/q with 48 confirmed event rule corrections',
                       B_identity='Original saved V1', changed_rows=int(paired.event_rule_correction.sum()),
                       best_of_conflicts_excluded_from_correction=int(paired['flags'].fillna('').str.contains('best_of_conflict').sum()))
    ancillary = {column: metrics(paired.playerA_wins, ordered_v2[column].to_numpy())
                 for column in ordered_v2 if column.endswith('_probA')}
    public = dict(status='Completed frozen-artifact audit; full pairing conditional on V1 ordering',
        baseline_commit='65e1b58811ebeb44e1be513590a7009a1dd37b91',
        A='V2.1 equal classifier ensemble', B='V1.1 regression plus Markov',
        delta='A minus B; positive favours A for accuracy/AUC, negative favours A for Brier/log loss',
        alignment=alignment, reproduction_errors=reproduction_errors, populations=results,
        scoring_only_sensitivity=scoring, ancillary_V2_metrics_exploratory=ancillary,
        diagnostics=diagnostics(paired, a, b), subgroups=subgroups(paired, a, b),
        clustering=dict(event='Exact tourney_id edition, including year and all draws; any date/year span stays in the same recorded event ID',
            calendar_28d='Non-overlapping fixed 28-day bins since Monday 1970-01-05; group whole events by earliest full-test reference date, retain anchor for all subsets',
            limitations='Resampling units are assumed independent within each design, not empirically proven independent. Repeated players across events/blocks and longer dependence remain. Tournament reference dates are not exact match dates. No bootstrap resolves unidentified V1 row order or source availability.'),
        inferential_limits='Pointwise percentile intervals, no family-wise coverage claim. All subset/ancillary/scoring analyses exploratory. McNemar is an iid-match reference. No DeLong iid p-value, equivalence test, or practical superiority claim. Uncertainty conditions on fixed fitted models and frozen predictions; excludes training/model-selection uncertainty.',
        operations=dict(model_fits=0, inference_calls=0, notebooks_executed=0, existing_scripts_rerun=0),
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                         pyarrow=pyarrow.__version__, joblib=joblib.__version__),
        elapsed_seconds=time.perf_counter()-started)
    paired.to_csv(private/'aligned_comparison.csv', index=False)
    # Raw rows and probabilities remain only in the explicitly private output.
    write_json(output/'summary.json', public)
    table = []
    for name in NAMES:
        pop = results['full_original_conditional']
        table.append(dict(metric=name, A=pop['A'][name], B=pop['B'][name],
            delta=pop['bootstrap']['event']['metrics'][name]['observed'],
            **{f'{scheme}_ci95': pop['bootstrap'][scheme]['metrics'][name]['ci95'] for scheme in pop['bootstrap']}))
    pd.DataFrame(table).to_csv(output/'comparison.csv', index=False)
    preserved = all(digest(p)==checksums[str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name] for p in paths)
    if not preserved:
        raise RuntimeError('Input hashes changed during audit')
    code_files = sorted(HERE.glob('*.py'))
    write_json(output/'manifest.json', dict(algorithm='sha256', baseline_commit=public['baseline_commit'],
        inputs=checksums, preserved_inputs=preserved,
        code={p.name: digest(p) for p in code_files},
        output={p.name: digest(p) for p in output.iterdir() if p.is_file()},
        invocation=dict(replicates=args.replicates, seed=args.seed),
        note='This manifest excludes itself and later reviewer prose/test receipts. Private row-level output deliberately excluded. Hashes identify artifacts, not correctness or ownership.'))
    print(f'Finished in {time.perf_counter()-started:.1f}s; input bytes preserved.', flush=True)


if __name__ == '__main__':
    main()
