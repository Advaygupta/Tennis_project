"""Deterministic checks. No model fitting, tuning, prediction, or network calls."""
import ast
import contextlib
import io
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from availability import Evidence, decision, legacy_cutoff, timestamp
from notebook_adapter import ROOT, cv_definition, definitions, feature_outputs, source_cells
from reference import reconstruct


def synthetic():
    rng = np.random.default_rng(918)
    rows = []
    # Repeated dates, multi-round and simultaneous events; enough dates to exercise
    # 5/10-date Elo summaries and ambiguous rolling windows. All data are invented.
    for i in range(64):
        date = pd.Timestamp('2020-01-01')+pd.Timedelta(days=(i//3)*14)
        row = dict(match_id=i, tourney_date=date, tourney_id=f'E{i//6}',
                   tourney_name=f'Open{i%3}', tourney_level=('A', 'C', 'G', 'F')[i%4],
                   best_of=5 if i%4 == 2 else 3, surface=('Hard', 'Clay', 'Grass')[i%3],
                   playerA_id=1, playerB_id=2+i%4, playerA_wins=int(rng.integers(2)),
                   playerA_rank=8, playerB_rank=12 if i%2 else 80,
                   playerA_hand='R', playerB_hand='L' if i%2 else 'R')
        for side in ('A', 'B'):
            row.update({f'player{side}_{key}': value for key, value in
                        dict(ace=4, df=2, svpt=60, **{'1stIn': 36, '1stWon': 25,
                             '2ndWon': 12, 'bpFaced': 7, 'bpSaved': 4}).items()})
        rows.append(row)
    # Include another year and both strict lower/inclusive upper window boundaries.
    rows[-1]['tourney_date'] = pd.Timestamp('2022-02-01')
    rows[5]['surface'] = None
    rows[7]['playerA_svpt'] = np.nan
    rows[9]['playerB_bpSaved'] = 9  # invalid BP tuple must not enter conversion totals
    return pd.DataFrame(rows).sort_values('tourney_date', kind='stable').reset_index(drop=True)


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = definitions()
        cls.frame = synthetic()

    def test_independent_reference_all_history_columns(self):
        actual = feature_outputs(self.frame, self.ns)
        expected = reconstruct(self.frame)[actual.columns]
        assert_frame_equal(actual, expected, check_dtype=False, atol=1e-10, rtol=1e-12)

    def test_own_result_and_statistics_do_not_change_own_features(self):
        baseline = feature_outputs(self.frame, self.ns)
        changed = self.frame.copy()
        mid = 21
        changed.loc[changed.match_id.eq(mid), 'playerA_wins'] = 1-changed.loc[changed.match_id.eq(mid), 'playerA_wins']
        changed.loc[changed.match_id.eq(mid), 'playerA_1stWon'] = 20
        other = feature_outputs(changed, self.ns)
        assert_frame_equal(baseline.loc[[mid]], other.loc[[mid]])

    def test_future_mutation_does_not_change_earlier_features(self):
        before = self.frame.tourney_date < self.frame.tourney_date.iloc[40]
        changed = self.frame.copy()
        changed.loc[~before, 'playerA_wins'] = 1-changed.loc[~before, 'playerA_wins']
        changed.loc[~before, 'playerB_1stWon'] = 11
        ids = self.frame.loc[before, 'match_id']
        assert_frame_equal(feature_outputs(self.frame, self.ns).loc[ids], feature_outputs(changed, self.ns).loc[ids])

    def test_ambiguous_date_permutation(self):
        other = pd.concat([g.sample(frac=1, random_state=77) for _, g in self.frame.groupby('tourney_date', sort=True)])
        assert_frame_equal(feature_outputs(self.frame, self.ns), feature_outputs(other, self.ns), atol=1e-10, rtol=1e-12)

    def test_cutoff_inclusive_and_strict_window_boundaries(self):
        frame = self.frame.iloc[:3].copy()
        frame['tourney_date'] = pd.to_datetime(['2020-01-01', '2020-01-10', '2020-01-11'])
        result = feature_outputs(frame, self.ns)
        self.assertEqual(result.loc[1, 'playerA_career_matches'], 0)
        self.assertEqual(result.loc[2, 'playerA_career_matches'], 1)
        frame.loc[2, 'tourney_date'] = pd.Timestamp('2020-02-10')
        result = feature_outputs(frame, self.ns)
        self.assertEqual(result.loc[2, 'playerA_matches_30d'], 1)  # Jan 1 excluded, Jan 10 included

    def test_reproduces_same_event_with_inconsistent_dates_defect(self):
        frame = self.frame.iloc[:2].copy()
        frame['tourney_date'] = pd.to_datetime(['2020-01-01', '2020-01-15'])
        frame['tourney_id'] = 'same-edition'
        frame['tourney_name'] = 'same-name'
        result = feature_outputs(frame, self.ns)
        # This deliberately documents a production vulnerability, not a safe result.
        self.assertEqual(result.loc[1, 'playerA_career_matches'], 1)
        self.assertNotEqual(result.loc[1, 'playerA_elo'], 1500.)

    def test_legacy_elo_and_rolling_accept_a_synthetically_future_result(self):
        frame = self.frame.iloc[:2].copy()
        frame['tourney_date'] = pd.to_datetime(['2020-01-01', '2020-01-21'])
        frame['tourney_id'] = ['E1', 'E2']
        result = feature_outputs(frame, self.ns)
        e = Evidence('0', 'result', 'E1', '2020-01-01',
                     completion_lower=timestamp('2020-01-25T12:00:00Z'))
        status, allowed = decision(e, query_id='1', query_event='E2',
                                    prediction_at=timestamp('2020-01-21T00:00:00Z'),
                                    cutoff=timestamp('2020-01-11T00:00:00Z'))
        self.assertEqual(status, 'confirmed_future_result')
        self.assertFalse(allowed)
        self.assertEqual(result.loc[1, 'playerA_last10_matches'], 1)
        self.assertNotEqual(result.loc[1, 'playerA_elo'], 1500.)


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.prediction = timestamp('2020-01-21T00:00:00Z')
        self.cutoff = legacy_cutoff(self.prediction)
        self.evidence = Evidence('history-1', 'result', 'E1', '2020-01-01',
                                 completion_lower=timestamp('2020-01-02T00:00:00Z'),
                                 completion_upper=timestamp('2020-01-05T23:59:59Z'),
                                 public_available_upper=timestamp('2020-01-06T00:00:00Z'),
                                 kind='verified_match', source='synthetic-fixture', rationale='invented known times')

    def decide(self, evidence):
        return decision(evidence, query_id='query', query_event='E2', prediction_at=self.prediction, cutoff=self.cutoff)

    def test_verified_availability_inclusive_cutoff(self):
        self.assertTrue(self.decide(self.evidence)[1])
        self.assertTrue(self.decide(replace(self.evidence, public_available_upper=self.cutoff))[1])
        self.assertFalse(self.decide(replace(self.evidence, public_available_upper=self.prediction))[1])

    def test_ambiguous_availability_and_provider_snapshot_are_unresolved(self):
        e = replace(self.evidence, public_available_upper=None, provider_recorded_at=self.cutoff)
        self.assertEqual(self.decide(e), ('unresolved_public_availability', False))
        self.assertFalse(self.decide(replace(self.evidence, source=''))[1])

    def test_overlap_and_multiweek_completion_are_not_released(self):
        e = replace(self.evidence, completion_upper=timestamp('2020-01-25T00:00:00Z'))
        self.assertEqual(self.decide(e), ('not_established_by_cutoff', False))
        e = replace(e, completion_lower=timestamp('2020-01-22T00:00:00Z'),
                    public_available_upper=timestamp('2020-01-26T00:00:00Z'))
        self.assertEqual(self.decide(e), ('confirmed_future_result', False))

    def test_same_match_event_and_unknown_event_excluded(self):
        for e in (replace(self.evidence, observation_id='query'), replace(self.evidence, event_key='E2'),
                  replace(self.evidence, event_key='')):
            self.assertFalse(self.decide(e)[1])

    def test_result_proof_does_not_prove_statistics_or_rankings(self):
        e = replace(self.evidence, family='statistics', public_available_upper=None)
        self.assertFalse(self.decide(e)[1])
        e = replace(self.evidence, family='ranking', public_available_upper=None)
        self.assertFalse(self.decide(e)[1])

    def test_rejects_invalid_dates_and_bounds(self):
        with self.assertRaises(ValueError):
            timestamp('2020-01-01')
        with self.assertRaises(ValueError):
            replace(self.evidence, completion_lower=self.prediction)
        with self.assertRaises(ValueError):
            decision(self.evidence, query_id='q', query_event='E2', prediction_at=self.prediction, cutoff=self.prediction)


class EvaluationTests(unittest.TestCase):
    def test_original_cv_definitions_separate_true_dates(self):
        dates = pd.Series(pd.to_datetime(['2020-01-01']*2+['2020-01-15']*3+['2020-02-01']*2+
                                        ['2020-03-01']*4+['2020-04-01']*3+['2020-05-01']*2))
        for notebook, symbol in [('03_v1_1_model_training.ipynb', 'date_grouped_cv'),
                                 ('05_v2_1_model_training.ipynb', 'DateGroupedCV')]:
            cv = cv_definition(notebook, symbol, dates)
            folds = cv(n_splits=3) if symbol == 'date_grouped_cv' else cv(n_splits=3).split(pd.DataFrame(index=range(len(dates))))
            seen = set()
            for tr, va in folds:
                self.assertLess(dates.iloc[tr].max(), dates.iloc[va].min())
                self.assertTrue(seen.isdisjoint(set(va)))
                seen.update(va)

    def test_ridge_meta_dates_pass_explicit_original_oof_identity(self):
        # Inspect date plumbing without fitting even a toy estimator.
        sources = source_cells(ROOT/'notebooks/03_v1_1_model_training.ipynb')
        node = next(n for n in ast.parse(sources[163]).body if isinstance(n, ast.FunctionDef))
        self.assertIn('dates', [arg.arg for arg in node.args.args])
        self.assertIn('stack_dates = train_dates.iloc[oof_seen].reset_index(drop=True)', sources[164])
        self.assertIn('ridge_meta_model(X_stack_trainA, y_stack_trainA, dates=stack_dates)', sources[164])


if __name__ == '__main__':
    unittest.main(verbosity=2)
