import unittest
import numpy as np
import pandas as pd
from alignment import align, add_scoring
from metrics import WeightedAUC, metrics, losses, discordance, validate
from paired_bootstrap import paired_bootstrap, replicate_weights, cluster_codes


def frames():
    t = pd.DataFrame(dict(dataset_row=[10, 11, 12, 13], match_id=[101, 102, 103, 104],
        tourney_id=['2025-A', '2025-A', '2025-B', '2025-B'],
        tourney_name=['A', 'A', 'B', 'B'], tourney_date=['2025-01-01']*2+['2025-02-01']*2,
        best_of=[3]*4, playerA_wins=[1, 1, 0, 1], playerA_id=[1, 3, 5, 7],
        playerB_id=[2, 4, 6, 8], round=['R32']*4, _source_file=['a']*4,
        _source_row=[1, 2, 3, 4], split=['test']*4, score=['6-3 6-4']*4))
    a = t[['tourney_name', 'tourney_date', 'best_of', 'playerA_wins']].copy()
    a['match_win_probA'] = [.7, .4, .2, .6]
    a['deciding_set_rule'] = 'standard'
    b = t[['dataset_row', 'match_id', 'playerA_id', 'playerB_id', 'playerA_wins']].copy()
    b['match_win_probA'] = [.6, .7, .3, .4]
    return a, b, t


class MetricTests(unittest.TestCase):
    def test_known_values(self):
        m = metrics([0, 0, 1, 1], [.1, .4, .35, .8])
        self.assertEqual(m['accuracy'], .75)
        self.assertEqual(m['roc_auc'], .75)
        self.assertAlmostEqual(m['brier_score'], (.01+.16+.4225+.04)/4)
        self.assertAlmostEqual(m['log_loss'], -np.log([.9, .6, .35, .8]).mean())

    def test_ties(self):
        m = metrics([0, 1], [.5, .5])
        self.assertEqual(m['accuracy'], .5)
        self.assertEqual(m['roc_auc'], .5)
        self.assertEqual(losses([1], [.5])[0, 0], 1)

    def test_single_class_auc(self):
        self.assertIsNone(metrics([1, 1], [.2, .3])['roc_auc'])

    def test_invalid_probabilities(self):
        for p in ([np.nan], [np.inf], [-.1], [1.1]):
            with self.assertRaises(ValueError): validate([1], p)

    def test_invalid_outcomes_and_lengths(self):
        for y, p in [([.9], [.2]), ([2], [.2]), ([], []), ([0, 1], [.4])]:
            with self.assertRaises(ValueError): validate(y, p)

    def test_boundary_clipping(self):
        self.assertTrue(np.isfinite(losses([0, 1, 0, 1], [0, 1, 1, 0])).all())

    def test_weighted_auc_matches_expanded_pairs(self):
        y = np.array([0, 1, 0, 1, 0])
        p = np.array([.1, .3, .3, .8, .9])
        for w in ([1, 2, 0, 3, 1], [0, 1, 3, 0, 2], [1, 1, 1, 1, 1]):
            ids = np.repeat(np.arange(len(y)), w)
            yp, pp = y[ids], p[ids]
            pos, neg = pp[yp==1], pp[yp==0]
            brute = np.mean((pos[:, None]>neg)+.5*(pos[:, None]==neg))
            self.assertAlmostEqual(WeightedAUC(y, p)(w)[0], brute)

    def test_mcnemar_counts_and_exact_probability(self):
        d = discordance([1]*4, [.8, .8, .8, .2], [.2, .2, .2, .2])
        self.assertEqual(d['A_only_correct'], 3)
        self.assertEqual(d['B_only_correct'], 0)
        self.assertEqual(d['both_wrong'], 1)
        self.assertAlmostEqual(d['mcnemar_exact_two_sided_p'], .25)
        self.assertEqual(discordance([0, 1], [.2, .8], [.2, .8])['mcnemar_exact_two_sided_p'], 1)

    def test_secondary_library_metric_check(self):
        # Optional secondary check, never used by the evaluator itself.
        try:
            from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score
        except ImportError:
            self.skipTest('Optional sklearn cross-check unavailable')
        rng = np.random.default_rng(17)
        y = rng.integers(0, 2, 200)
        p = np.round(rng.uniform(.001, .999, 200), 2)
        m = metrics(y, p)
        for key, value in dict(accuracy=accuracy_score(y, p>=.5), roc_auc=roc_auc_score(y, p),
                               log_loss=log_loss(y, p), brier_score=brier_score_loss(y, p)).items():
            self.assertAlmostEqual(m[key], value, places=13)


class AlignmentTests(unittest.TestCase):
    def test_ambiguity_is_retained_and_quantified(self):
        a, b, t = frames()
        aligned, _, strict, info = align(a, b, t)
        self.assertEqual(len(aligned), 4)
        self.assertEqual(strict.tolist(), [False, False, True, True])
        self.assertEqual(info['V1_ambiguous_positional'], 2)

    def test_reordered_v2_is_reconciled(self):
        a, b, t = frames()
        aligned, _, _, _ = align(a, b.iloc[::-1], t)
        np.testing.assert_allclose(aligned.v2_probability, b.match_win_probA)

    def test_v1_incorrect_order_is_rejected(self):
        a, b, t = frames()
        with self.assertRaises(ValueError): align(a.iloc[::-1], b, t)

    def test_undetectable_ambiguous_swap_never_becomes_unique(self):
        a, b, t = frames()
        a.loc[[0, 1], 'match_win_probA'] = [.4, .7]
        _, _, strict, info = align(a, b, t)
        self.assertFalse(strict[:2].any())
        self.assertEqual(info['V1_exact_match_id'], 0)

    def test_orientation_mismatch(self):
        a, b, t = frames()
        b.loc[0, ['playerA_id', 'playerB_id']] = [2, 1]
        with self.assertRaises(ValueError): align(a, b, t)

    def test_outcome_mismatch(self):
        a, b, t = frames()
        b.loc[0, 'playerA_wins'] = 0
        with self.assertRaises(ValueError): align(a, b, t)

    def test_duplicate_and_missing_ids(self):
        for value in (10, np.nan):
            a, b, t = frames()
            b.loc[1, 'dataset_row'] = value
            with self.assertRaises(ValueError): align(a, b, t)

    def test_missing_prediction_does_not_silently_intersect(self):
        a, b, t = frames()
        with self.assertRaises(ValueError): align(a, b.iloc[:3], t)

    def test_scoring_sidecar_validation(self):
        a, b, t = frames()
        frame, _, _, _ = align(a, b, t)
        side = pd.DataFrame(dict(dataset_row=t.dataset_row, saved_probability=a.match_win_probA,
            original_rule=['standard']*4, registry_ordinary=['standard']*4,
            registry_deciding=['standard']*4, reference_registry_probability=a.match_win_probA,
            flags=['']*4, registry_status=['verified']*4))
        result = add_scoring(frame, side.iloc[::-1])
        self.assertTrue(result.completed.all())
        side.loc[0, 'saved_probability'] = .01
        with self.assertRaises(ValueError): add_scoring(frame, side)

    def test_missing_inferred_counterfactual_is_not_used(self):
        a, b, t = frames()
        frame, _, _, _ = align(a, b, t)
        side = pd.DataFrame(dict(dataset_row=t.dataset_row, saved_probability=a.match_win_probA,
            original_rule=['standard']*4, registry_ordinary=['standard']*4,
            registry_deciding=['standard']*4, reference_registry_probability=[.7, np.nan, .2, .6],
            flags=['']*4, registry_status=['verified', 'inferred', 'verified', 'verified']))
        result = add_scoring(frame, side)
        self.assertEqual(result.event_corrected_probability.iloc[1], a.match_win_probA.iloc[1])
        self.assertFalse(result.supported_policy.iloc[1])

    def test_retirement_subset_and_bestof_preservation(self):
        a, b, t = frames()
        t.loc[0, 'score'] = '6-2 1-0 RET'
        frame, _, _, _ = align(a, b, t)
        side = pd.DataFrame(dict(dataset_row=t.dataset_row, saved_probability=a.match_win_probA,
            original_rule=['standard']*4, registry_ordinary=['standard']*4,
            registry_deciding=['standard']*4, reference_registry_probability=[.5, .4, .2, .6],
            flags=['best_of_conflict', '', '', ''], registry_status=['verified']*4))
        result = add_scoring(frame, side)
        self.assertEqual(int(result.completed.sum()), 3)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.event_corrected_probability.iloc[0], .7)


class BootstrapTests(unittest.TestCase):
    def test_shared_weights_preserve_pairing_and_clusters(self):
        w = replicate_weights(np.random.default_rng(5), np.array([0, 0, 1, 2, 2]), 3, 40)
        np.testing.assert_array_equal(w[:, 0], w[:, 1])
        np.testing.assert_array_equal(w[:, 3], w[:, 4])
        self.assertTrue(np.all(w[:, [0, 2, 3]].sum(axis=1)==3))

    def test_identical_models_have_zero_difference(self):
        result = paired_bootstrap([0, 1, 0, 1], [.2, .8, .3, .6], [.2, .8, .3, .6],
                                  np.arange(4), replicates=200)
        for entry in result['metrics'].values():
            self.assertEqual(entry['ci95'], [0., 0.])

    def test_seed_reproducibility(self):
        args = ([0, 1, 0, 1], [.2, .8, .4, .6], [.1, .3, .8, .9], np.arange(4))
        self.assertEqual(paired_bootstrap(*args, replicates=100, seed=5),
                         paired_bootstrap(*args, replicates=100, seed=5))

    def test_event_and_block_units_and_subset_consistency(self):
        _, _, t = frames()
        # A single event crossing reference dates retains the full-event anchor.
        t.loc[1, 'tourney_date'] = '2025-02-20'
        t['event_reference_date'] = t.groupby('tourney_id').tourney_date.transform('min')
        event, n = cluster_codes(t, 'event')
        self.assertEqual(n, 2)
        self.assertEqual(event[0], event[1])
        blocks, _ = cluster_codes(t, 'calendar_28d')
        self.assertEqual(blocks[0], blocks[1])
        subset = t.iloc[[1, 2]]
        sub, _ = cluster_codes(subset, 'calendar_28d')
        self.assertEqual(blocks[1]==blocks[2], sub[0]==sub[1])

    def test_one_class_replicates_counted(self):
        result = paired_bootstrap([0, 1], [.2, .8], [.4, .6], np.arange(2), replicates=100)
        self.assertGreater(result['metrics']['roc_auc']['undefined_replicates'], 0)
        self.assertEqual(result['metrics']['accuracy']['valid_replicates'], 100)

    def test_invalid_cluster_codes(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([0, 1], [.2, .8], [.4, .6], np.array([0, 2]), replicates=100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
