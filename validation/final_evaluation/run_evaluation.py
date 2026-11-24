"""New isolated evaluation of frozen predictions; no training/inference/notebooks."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from shared import HERE, REPO, PAIRED, IDENTITY_LIMITATION, align, add_scoring, metrics, digest
from verify_artifacts import FILENAMES, verify_inputs, verify_hash, check_output
from calibration import calibration_summary
from logloss_diagnostics import diagnose_logloss
from uncertainty import grouped_bootstrap

BASELINE_COMMIT = '981fa3d7b23187f8b7073b1d6eadf9490023f8c8'
EXPORT_SHA = 'e619b001e7dbd7c810a32a1fcd9453f8b844dc24f1c1b45424b5a85d60cd7105'


def write_json(path, obj):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(obj, f, indent=2, allow_nan=False,
                  default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
        f.write('\n')


def load_inputs(artifacts, sidecar=None, paired_export=None):
    hashes, reused_code = verify_inputs(artifacts, sidecar)
    paths = [Path(artifacts)/name for name in FILENAMES]
    v1 = joblib.load(paths[0])  # ONLY the hash-verified trusted DataFrame
    metadata = pd.read_csv(paths[2], float_precision='round_trip')
    if paired_export is None:
        v2 = pd.read_parquet(paths[1])
        route = dict(mode='original_parquet', current_run_decodes_V2_parquet=True)
    else:
        verify_hash(paired_export, EXPORT_SHA)
        export = pd.read_csv(paired_export, float_precision='round_trip')
        v2 = export[['dataset_row', 'match_id', 'playerA_id', 'playerB_id', 'playerA_wins', 'v2_probability']].rename(
            columns={'v2_probability': 'match_win_probA'})
        hashes['paired_export.csv'] = dict(sha256=EXPORT_SHA, bytes=Path(paired_export).stat().st_size)
        route = dict(mode='preserved_Point3_export', current_run_decodes_V2_parquet=False,
            note='Reuses Point 3 private aligned export; its hash first recorded in Point 4. '
              'V2 export lineage is inherited from the Point 3 run, not independently reauthenticated by this CSV. '
              'Original V2 Parquet bytes are verified; original V1 and metadata are loaded and alignment rechecked. '
              'Direct original-Parquet route is available with PyArrow 23+, but was not executed in this environment.')
    frame, ordered_v2, strict, alignment = align(v1, v2, metadata)
    frame['event_reference_date'] = frame.groupby('tourney_id').tourney_date.transform('min')
    if paired_export is not None:
        # Check the reused export against source V1 and every common metadata field,
        # without assigning new identities to the V1 array.
        if not np.array_equal(frame.dataset_row, export.dataset_row):
            raise ValueError('Export metadata order changed')
        if not np.array_equal(frame.v1_probability, export.v1_probability):
            raise ValueError('Export V1 probabilities changed')
        for col in metadata.columns:
            if col in frame and col in export:
                x, z = frame[col], export[col]
                if not (x.eq(z) | (x.isna() & z.isna())).all():
                    raise ValueError('Export/source metadata mismatch: '+col)
    if sidecar is not None:
        frame = add_scoring(frame, pd.read_csv(sidecar, float_precision='round_trip'))
    expected = json.loads((REPO/'reproduction/expected_metrics.json').read_text(encoding='utf8'))
    if len(frame) != expected['test_matches']:
        raise ValueError('Original test population changed')
    errors = {}
    for model, col in [('v1_1', 'v1_probability'), ('v2_1', 'v2_probability')]:
        errors[model] = {k: v-expected[model][k] for k, v in metrics(frame.playerA_wins, frame[col]).items()}
    if any(abs(x) > 1e-12 for m in errors.values() for x in m.values()):
        raise ValueError('Original metric reproduction failed')
    return frame, v1, strict, alignment, hashes, reused_code, route, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=REPO/'reproduction/artifacts')
    parser.add_argument('--scoring-sidecar', type=Path,
        help='Optional saved Point 2 sidecar; enables policy/completion subsets and scoring-only sensitivity')
    parser.add_argument('--paired-export', type=Path,
        help='Optional EXACT preserved Point 3 private CSV, hash checked; avoids current Parquet decoding')
    parser.add_argument('--output', type=Path, required=True, help='New aggregate output directory only')
    parser.add_argument('--replicates', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261124)
    args = parser.parse_args()
    if args.replicates < 20:
        parser.error('--replicates must be at least 20')
    output = check_output(args.output, args.artifacts,
        [p for p in [args.scoring_sidecar, args.paired_export] if p is not None])
    started = time.perf_counter()
    frame, v1, strict, alignment, hashes, reused_code, route, errors = load_inputs(
        args.artifacts, args.scoring_sidecar, args.paired_export)
    y = frame.playerA_wins.to_numpy()
    probabilities = {'V2.1': frame.v2_probability.to_numpy(), 'V1.1': frame.v1_probability.to_numpy()}
    print('Verified original input bytes, population, orientation and saved metrics.', flush=True)
    calibration = {m: calibration_summary(y, p) for m, p in probabilities.items()}
    uncertainty = {}
    for j, scheme in enumerate(['event', 'calendar_28d']):
        print('Calibration uncertainty: '+scheme, flush=True)
        uncertainty[scheme] = grouped_bootstrap(frame, probabilities, args.replicates, args.seed+j, scheme)
    subsets = {}
    masks = {'unique_shared_context': strict}
    if args.scoring_sidecar:
        masks.update(completed=frame.completed.to_numpy(), supported_policy=frame.supported_policy.to_numpy(),
                     completed_supported_policy=(frame.completed & frame.supported_policy).to_numpy())
    expected_sizes = dict(unique_shared_context=35, completed=50213, supported_policy=34367, completed_supported_policy=33294)
    for j, (name, mask) in enumerate(masks.items()):
        if int(mask.sum()) != expected_sizes[name]:
            raise ValueError('Sensitivity population changed: '+name)
        subsets[name] = dict(n=int(mask.sum()), calibration={m: calibration_summary(y[mask], p[mask])
                                                        for m, p in probabilities.items()},
                            logloss=diagnose_logloss(y[mask], probabilities['V2.1'][mask], probabilities['V1.1'][mask]))
        if name != 'unique_shared_context':
            # Reuse original full-population event reference anchors in subsets.
            subsets[name]['event_uncertainty'] = grouped_bootstrap(frame.loc[mask],
                {m: p[mask] for m, p in probabilities.items()}, args.replicates, args.seed+10+j, 'event')
    correction = None
    if args.scoring_sidecar:
        correction = dict(status='Separate scoring-only counterfactual; frozen serve p/q; not new ML predictions',
            changed_rows=int(frame.event_rule_correction.sum()),
            best_of_conflicts_excluded=4,
            corrected_V1=calibration_summary(y, frame.event_corrected_probability.to_numpy()),
            corrected_metrics=metrics(y, frame.event_corrected_probability.to_numpy()))
        if correction['changed_rows'] != 48:
            raise ValueError('Unexpected event-rule correction count')
    logloss = diagnose_logloss(y, probabilities['V2.1'], probabilities['V1.1'])
    previous = json.loads((PAIRED/'results/summary.json').read_text(encoding='utf8'))
    old = previous['diagnostics']
    top = next(t for t in logloss['tail_attribution'] if t['ranking'] == 'absolute_paired_difference' and t['nominal_fraction'] == .01)
    if abs(top['signed_contribution_to_full_mean']-old['concentration']['top_1pct_signed_contribution_to_mean']) > 1e-12:
        raise ValueError('Point 3 concentration finding no longer reproduces')
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(status='Frozen-prediction calibration and reproducibility verification',
        baseline_commit=BASELINE_COMMIT, n=len(frame), alignment=alignment,
        identity_limitation=IDENTITY_LIMITATION, data_route=route, reproduction_errors=errors,
        calibration=calibration, logloss=logloss, uncertainty=uncertainty, subsets=subsets,
        scoring_only_sensitivity=correction,
        saved_serve_probability_distribution={col: dict(mean=float(v1[col].mean()), sd=float(v1[col].std(ddof=0)),
            min=float(v1[col].min()), max=float(v1[col].max())) for col in ['p', 'q']},
        operations=dict(ML_training=0, ML_inference=0, preprocessing_refits=0, notebook_execution=0,
            existing_script_entrypoints_run=0, calibrators_fitted=0,
            diagnostic_logistic_estimates='Test-set diagnostic likelihood equations only; never applied to predictions'),
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__),
        limitations=[IDENTITY_LIMITATION,
            'Historical match/source availability unverified; ten-day cutoff is a conservative assumption, not proof.',
            '17,299 historical rule mappings inferred; four best-of conflicts outside scoring-only correction.',
            'Incomplete outcomes are part of primary frozen population; completion subset is exploratory.',
            'Pointwise fixed-model uncertainty excludes training/model-selection uncertainty and does not remove repeated-player dependence.',
            'ECE/MCE depend on bins, composition and finite-sample noise; grouping hides within-bin miscalibration.',
            'Logistic diagnostics use final-test outcomes; no unbiased recalibration performance is claimed.',
            'No suitable authenticated out-of-sample validation match-probability pair supplied; no calibrator fitted.',
            'Nonlinear scoring may accompany extreme probabilities; comparison cannot establish a Markov causal effect.',
            'Private row-level artifacts require rights clearance; hashes alone do not establish provenance or permissions.'],
        elapsed_seconds_before_figures=time.perf_counter()-started)
    from plots import make_figures
    import matplotlib
    summary['environment']['matplotlib'] = matplotlib.__version__
    try:
        import pyarrow
        summary['environment']['pyarrow'] = pyarrow.__version__
    except ImportError:
        summary['environment']['pyarrow'] = 'not used: preserved export route'
    make_figures(output/'figures', y, probabilities, summary)
    summary['elapsed_seconds'] = time.perf_counter()-started
    write_json(output/'summary.json', summary)
    bins = [dict(model=m, binning=t['scheme'], requested_bins=t['requested_bins'], **r)
            for m, c in calibration.items() for t in c['binning'] for r in t['rows']]
    pd.DataFrame(bins).to_csv(output/'reliability_bins.csv', index=False, mode='x', lineterminator='\n')
    pd.DataFrame(logloss['tail_attribution']).to_csv(output/'tail_attribution.csv', index=False, mode='x', lineterminator='\n')
    # Verify every actual source byte again after completing analysis.
    final_hashes, _ = verify_inputs(args.artifacts, args.scoring_sidecar)
    if any(final_hashes[name] != hashes[name] for name in final_hashes):
        raise RuntimeError('Frozen inputs changed')
    if args.paired_export:
        verify_hash(args.paired_export, EXPORT_SHA)
    write_json(output/'manifest.json', dict(algorithm='SHA-256', baseline_commit=BASELINE_COMMIT,
        inputs=hashes, reused_code=reused_code, preserved_inputs=True,
        code={p.name: digest(p) for p in sorted(HERE.glob('*.py'))},
        output={p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
        invocation=dict(replicates=args.replicates, seed=args.seed, route=route['mode'],
            scoring_sidecar=args.scoring_sidecar is not None),
        note='Manifest excludes itself and later prose/test receipts. Publication manifest covers those separately. '
          'Baseline is the immutable preceding GitHub source commit; publication commit contains these exact code bytes. '
          'A self-referential current commit SHA cannot be included in its own hashed contents.'))
    print(f'Completed in {time.perf_counter()-started:.1f}s. Aggregate output only: {output}', flush=True)


if __name__ == '__main__':
    main()
