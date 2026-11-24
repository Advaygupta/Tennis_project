"""Analytic synthetic checks, failure-path checks and optional frozen-input checks."""
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
from calibration import bin_assignment, reliability, logistic_diagnostic, confidence_groups, calibration_summary
from logloss_diagnostics import diagnose_logloss, tail_attribution
from verify_artifacts import verify_hash, verify_inputs, check_output
from uncertainty import grouped_bootstrap
from shared import IDENTITY_LIMITATION, REPO, metrics, losses


class CalibrationTests(unittest.TestCase):
    def test_known_bins(self):
        r = reliability([0, 1, 0, 1], [.1, .2, .6, .9], 2)
        self.assertEqual([x['n'] for x in r['rows']], [2, 2])
        self.assertAlmostEqual(r['rows'][0]['mean_probability'], .15)
        self.assertAlmostEqual(r['rows'][1]['observed_frequency'], .5)
        self.assertAlmostEqual(r['ECE'], .30)
        self.assertAlmostEqual(r['MCE'], .35)

    def test_empty_bins_retained(self):
        r = reliability([0, 1], [.01, .99], 10)
        self.assertEqual(len(r['rows']), 10)
        self.assertEqual(sum(x['n'] == 0 for x in r['rows']), 8)
        self.assertIsNone(r['rows'][3]['observed_frequency'])

    def test_boundaries(self):
        code, edge = bin_assignment([0, .5, 1], 2)
        np.testing.assert_array_equal(code, [0, 1, 1])
        self.assertEqual(reliability([0, 1], [0, 1], 10)['ECE'], 0.)

    def test_equal_frequency(self):
        p = np.arange(100)/100
        r = reliability(np.tile([0, 1], 50), p, 10, 'frequency')
        self.assertEqual([x['n'] for x in r['rows']], [10]*10)

    def test_frequency_ties(self):
        p = np.r_[np.full(90, .5), np.full(10, .9)]
        codes, edges = bin_assignment(p, 10, 'frequency')
        self.assertEqual(len(np.unique(codes[:90])), 1)
        self.assertLess(len(edges)-1, 10)

    def test_permutation_invariance(self):
        y = [0, 1, 0, 1]
        p = [.1, .2, .6, .9]
        for method in ['width', 'frequency']:
            self.assertAlmostEqual(reliability(y, p, 3, method)['ECE'], reliability(y[::-1], p[::-1], 3, method)['ECE'])

    def test_known_logistic_diagnostic(self):
        y = np.r_[np.ones(30), np.zeros(70), np.ones(70), np.zeros(30)]
        p = np.r_[np.full(100, .2), np.full(100, .8)]
        r = logistic_diagnostic(y, p)
        self.assertEqual(r['status'], 'converged')
        self.assertAlmostEqual(r['intercept'], 0., places=8)
        self.assertAlmostEqual(r['slope'], np.log(.7/.3)/np.log(4), places=8)

    def test_offset_intercept_analytic(self):
        r = logistic_diagnostic([1]*3+[0]*7, [.2]*10, offset_only=True)
        self.assertAlmostEqual(r['intercept'], np.log(.3/.7)-np.log(.2/.8), places=8)
        self.assertEqual(r['slope'], 1.)

    def test_perfect_group_calibration(self):
        y = np.r_[np.ones(20), np.zeros(80), np.ones(80), np.zeros(20)]
        p = np.r_[np.full(100, .2), np.full(100, .8)]
        r = logistic_diagnostic(y, p)
        self.assertAlmostEqual(r['intercept'], 0., places=8)
        self.assertAlmostEqual(r['slope'], 1., places=8)

    def test_constant_logit_undefined(self):
        self.assertIsNone(logistic_diagnostic([0, 1], [.5, .5])['slope'])

    def test_one_class_undefined(self):
        self.assertIsNone(logistic_diagnostic([1, 1], [.1, .9])['slope'])

    def test_separation_not_reported_finite(self):
        self.assertIsNone(logistic_diagnostic([0, 0, 1, 1], [.1, .2, .8, .9])['slope'])

    def test_boundary_diagnostic_finite(self):
        r = logistic_diagnostic([0, 1, 0, 1], [0, 0, 1, 1])
        self.assertEqual(r['status'], 'converged')
        self.assertEqual(r['clipped_logits'], 4)

    def test_no_probability_mutation(self):
        p = np.array([0., .2, .8, 1.])
        saved = p.copy()
        calibration_summary([0, 1, 0, 1], p)
        np.testing.assert_array_equal(p, saved)

    def test_winner_orientation(self):
        r = confidence_groups([1, 0, 0, 1], [.5, .1, .9, 1.])
        self.assertEqual(sum(x['n'] for x in r), 4)
        self.assertEqual(r[0]['accuracy'], 1.)  # .5 predicts A
        self.assertEqual(r[2]['accuracy'], .5)

    def test_invalid_probabilities(self):
        for p in [[np.nan], [np.inf], [-.1], [1.1]]:
            with self.subTest(p=p), self.assertRaises(ValueError):
                reliability([0], p)

    def test_invalid_outcomes(self):
        for y in [[2], [None], [np.nan]]:
            with self.subTest(y=y), self.assertRaises(ValueError):
                reliability(y, [.5])

    def test_bad_weights(self):
        with self.assertRaises(ValueError):
            logistic_diagnostic([0, 1], [.2, .8], weights=[-1, 1])

    def test_invalid_bins(self):
        with self.assertRaises(ValueError):
            bin_assignment([.5], 1)
        with self.assertRaises(ValueError):
            bin_assignment([.5], 2, 'bad')


class LossTests(unittest.TestCase):
    def test_logloss_analytic(self):
        r = diagnose_logloss([1, 0], [.5, .5], [.25, .75])
        self.assertAlmostEqual(r['A']['mean'], np.log(2))
        self.assertAlmostEqual(r['B']['mean'], np.log(4))
        self.assertAlmostEqual(r['A_minus_B']['mean'], -np.log(2))

    def test_tail_signed_absolute_differ(self):
        r = tail_attribution([8, 0, 0], [0, 10, 0], [.3])
        absolute = next(x for x in r if x['ranking'] == 'absolute_paired_difference')
        signed = next(x for x in r if x['ranking'] == 'signed_A_minus_B')
        self.assertEqual(absolute['selected_n'], 1)
        self.assertAlmostEqual(absolute['signed_contribution_to_full_mean'], -10/3)
        self.assertAlmostEqual(signed['signed_contribution_to_full_mean'], 8/3)

    def test_contribution_adds_to_mean(self):
        for r in tail_attribution([1, 2, 9, 4], [3, 4, 2, 1]):
            self.assertAlmostEqual(r['signed_contribution_to_full_mean']+r['remaining_contribution_to_full_mean'], 1.5)

    def test_positive_median_negative_mean(self):
        r = tail_attribution([1, 1, 1, 1, 0], [0, 0, 0, 0, 10], [.2])
        self.assertGreater(np.median(np.array([1, 1, 1, 1, -10])), 0)
        self.assertLess(sum([1, 1, 1, 1, -10])/5, 0)
        self.assertGreater(r[0]['share_of_net_mean_difference'], 1)

    def test_probability_boundaries_loss(self):
        r = diagnose_logloss([0, 1], [0, 1], [1, 0])
        self.assertTrue(np.isfinite(r['B']['mean']))
        self.assertGreater(r['B']['mean'], 30)

    def test_clipping_sensitivity(self):
        r = diagnose_logloss([0, 1], [.1, .9], [.2, .8])
        self.assertLess(np.ptp([x['delta'] for x in r['clipping_sensitivity']]), 1e-14)

    def test_observed_outcome_orientation(self):
        r = diagnose_logloss([0, 1, 1], [.99, .01, .9], [.1, .9, .5])
        a = next(x for x in r['adverse_predictions'] if x['model'] == 'V2.1' and x['observed_outcome_probability_below'] == .05)
        self.assertEqual(a['n'], 2)

    def test_outcome_regions_partition(self):
        r = diagnose_logloss([0, 1, 1], [.99, .01, .9], [.1, .9, .5])
        self.assertEqual(sum(x['n'] for x in r['B_true_outcome_regions']), 3)
        self.assertAlmostEqual(sum(x['signed_contribution_to_full_mean'] for x in r['B_true_outcome_regions']), r['A_minus_B']['mean'])


class ReproducibilityTests(unittest.TestCase):
    @staticmethod
    def sample():
        return pd.DataFrame(dict(playerA_wins=[0, 1, 0, 1, 0, 1], tourney_id=['a', 'a', 'b', 'b', 'c', 'c'],
            tourney_date=['2025-01-01']*2+['2025-02-01']*2+['2025-03-01']*2,
            event_reference_date=['2025-01-01']*2+['2025-02-01']*2+['2025-03-01']*2))

    def test_fixed_seed(self):
        f = self.sample()
        p = {'V2.1': np.array([.2, .6, .1, .8, .4, .7]), 'V1.1': np.array([.1, .9, .1, .7, .4, .8])}
        r = grouped_bootstrap(f, p, 30, 42)
        self.assertEqual(r, grouped_bootstrap(f, p, 30, 42))
        self.assertEqual(r['resampling_units'], 3)

    def test_identical_models_zero_difference(self):
        f = self.sample()
        p = np.array([.2, .6, .1, .8, .4, .7])
        r = grouped_bootstrap(f, {'V2.1': p, 'V1.1': p}, 30, 42)
        for e in r['binning_differences_A_minus_B']:
            self.assertEqual(e['ECE']['ci95'], [0., 0.])
        self.assertEqual(r['differences']['log_loss']['ci95'], [0., 0.])

    def test_cluster_bootstrap_against_explicit_rows(self):
        f = self.sample()
        p = np.array([.2, .6, .1, .8, .4, .7])
        r = grouped_bootstrap(f, {'V2.1': p, 'V1.1': p}, 30, 42)
        rng = np.random.default_rng(42)
        counts = rng.multinomial(3, [1/3]*3, size=30)
        y = f.playerA_wins.to_numpy()
        ece, bias = [], []
        for w in counts:
            ix = np.repeat(np.arange(6), np.repeat(w, 2))
            ece.append(reliability(y[ix], p[ix], 10, 'width')['ECE'])
            bias.append((p[ix]-y[ix]).mean())
        entry = next(x for x in r['binning'] if x['model'] == 'V2.1' and x['binning'] == 'width' and x['requested_bins'] == 10)
        np.testing.assert_allclose(entry['ECE']['ci95'], np.quantile(ece, [.025, .975]), atol=1e-15)
        np.testing.assert_allclose(r['models']['V2.1']['mean_bias']['ci95'], np.quantile(bias, [.025, .975]), atol=1e-15)

    def test_calendar_whole_event_anchor(self):
        from shared import cluster_codes
        f = self.sample()
        f.loc[1, 'tourney_date'] = '2025-04-01'
        code, k = cluster_codes(f, 'calendar_28d')
        self.assertEqual(code[0], code[1])

    def test_invalid_cluster_population(self):
        f = self.sample()
        f['tourney_id'] = 'single'
        with self.assertRaises(ValueError):
            grouped_bootstrap(f, {'V2.1': np.full(6, .5), 'V1.1': np.full(6, .5)}, 30, 42)

    def test_hash_acceptance_and_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'f'
            path.write_bytes(b'original')
            expected = hashlib.sha256(b'original').hexdigest()
            self.assertEqual(verify_hash(path, expected), expected)
            path.write_bytes(b'modified')
            with self.assertRaises(ValueError):
                verify_hash(path, expected)

    def test_missing_hash_file(self):
        with self.assertRaises(FileNotFoundError):
            verify_hash(Path('nonexistent.artifact'), 'abc')

    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                check_output(d, REPO/'reproduction/artifacts')

    def test_protected_output_rejected(self):
        with self.assertRaises(ValueError):
            check_output(REPO/'models/never_write', REPO/'reproduction/artifacts')

    def test_output_ancestor_rejected(self):
        with self.assertRaises(ValueError):
            check_output(REPO, REPO/'reproduction/artifacts')

    def test_identity_qualification_preserved(self):
        self.assertIn('conditional on preserved V1.1 row order', IDENTITY_LIMITATION)
        self.assertIn('do not independently authenticate', IDENTITY_LIMITATION)

    def test_no_calibrated_predictions_returned(self):
        r = calibration_summary([0, 1, 0, 1], [.1, .2, .8, .9])
        self.assertNotIn('adjusted_predictions', r)
        self.assertNotIn('calibrated_performance', r)

    def test_tampered_frozen_inputs_rejected_before_loading(self):
        from run_evaluation import load_inputs
        with tempfile.TemporaryDirectory() as d:
            for name in ['v1_1_results.joblib', 'v2_1_test_match_probabilities.parquet', 'matches_v1_1_metadata.csv']:
                (Path(d)/name).write_bytes(b'tampered')
            with patch('run_evaluation.joblib.load') as loader:
                with self.assertRaises(ValueError):
                    load_inputs(Path(d))
                loader.assert_not_called()

    def test_missing_population_rejected_by_shared_alignment(self):
        from shared import align
        t = pd.DataFrame(dict(dataset_row=[10, 11], match_id=[100, 101],
            tourney_id=['a', 'b'], tourney_name=['A', 'B'], tourney_date=['2025-01-01', '2025-02-01'],
            best_of=[3, 3], playerA_wins=[0, 1], playerA_id=[1, 3], playerB_id=[2, 4],
            round=['R32', 'R32'], _source_file=['x', 'x'], _source_row=[0, 1], split=['test', 'test']))
        a = t[['tourney_name', 'tourney_date', 'best_of', 'playerA_wins']].copy()
        a['match_win_probA'] = [.2, .8]
        a['deciding_set_rule'] = 'standard'
        b = t[['dataset_row', 'match_id', 'playerA_id', 'playerB_id', 'playerA_wins']].copy()
        b['match_win_probA'] = [.3, .7]
        with self.assertRaises(ValueError):
            align(a, b.iloc[:1], t)
        b.loc[0, 'playerA_wins'] = 1
        with self.assertRaises(ValueError):
            align(a, b, t)

    def test_package_allowlist_excludes_private_data(self):
        from build_package import package_files
        for path in package_files():
            self.assertNotIn('audit_private', path.parts)
            self.assertNotIn(path.suffix, ['.pkl', '.joblib', '.parquet', '.ipynb'])

    def test_result_comparison_rejects_changed_population(self):
        from compare_results import compare
        reference = dict(n=2, calibration={}, logloss={}, uncertainty={}, subsets={}, scoring_only_sensitivity=None)
        altered = dict(reference, n=1)
        with self.assertRaises(ValueError):
            compare(reference, altered)


if __name__ == '__main__':
    unittest.main()
