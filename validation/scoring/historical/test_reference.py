"""Deterministic mathematics, state-transition, and historical-selector tests."""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import math
import unittest
import engine as ref
import registry
from production_adapter import extract, match_probability as original_match


def solve(matrix, rhs):
    """Small exact Gaussian elimination, used only as a second test oracle."""
    a=[[F(x) for x in row]+[F(v)] for row,v in zip(matrix,rhs)]
    for k in range(len(a)):
        pivot=next(i for i in range(k,len(a)) if a[i][k])
        a[k],a[pivot]=a[pivot],a[k]
        divisor=a[k][k]
        a[k]=[v/divisor for v in a[k]]
        for i in range(len(a)):
            if i!=k:
                x=a[i][k]
                a[i]=[u-x*v for u,v in zip(a[i],a[k])]
    return [row[-1] for row in a]


def rational_tb(p,q,first,target):
    """Independent finite cyclic chain: point lead and absolute service phase.

    No geometric-tail expression from engine.py or production is called here.
    Twelve tail states are solved using rational arithmetic, with zero truncation.
    """
    states=[(d,k) for d in (-1,0,1) for k in range(4)]
    index={s:i for i,s in enumerate(states)}
    matrix=[[F(i==j) for j in range(12)] for i in range(12)]
    rhs=[F(0) for _ in states]
    for i,(d,k) in enumerate(states):
        server=first if k in (0,3) else 1-first
        x=p if server==0 else 1-q
        for step,chance in ((1,x),(-1,1-x)):
            lead=d+step
            if lead==2:rhs[i]+=chance
            elif lead!=-2:matrix[i][index[lead,(k+1)%4]]-=chance
    solution=solve(matrix,rhs)
    @lru_cache(None)
    def point(a,b):
        if max(a,b)>=target and abs(a-b)>=2:return F(a>b)
        if a==b==target-1:return solution[index[0,(a+b)%4]]
        # Define the blocks independently from ref.tb_server.
        n=a+b
        server=first if n==0 or ((n-1)//2)%2==1 else 1-first
        x=p if server==0 else 1-q
        return x*point(a+1,b)+(1-x)*point(a,b+1)
    return point(0,0)


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original,cls.extraction=extract(Path(__file__).resolve().parents[3])

    def test_game_deuce_exact_oracle(self):
        for x in (F(0),F(1,10),F(1,2),F(9,10),F(1)):
            # State equations for deuce, server advantage, receiver advantage.
            d,_,_=solve([[1,-x,-(1-x)],[-(1-x),1,0],[-x,0,1]],[0,x,0])
            @lru_cache(None)
            def point(a,b):
                if max(a,b)>=4 and abs(a-b)>=2:return F(a>b)
                if a==b==3:return d
                return x*point(a+1,b)+(1-x)*point(a,b+1)
            self.assertAlmostEqual(ref.game_outcomes(float(x))[0],float(point(0,0)),places=14)
            self.assertAlmostEqual(sum(ref.game_outcomes(float(x))),1.,places=14)

    def test_no_ad_game(self):
        for x in (.1,.5,.7,.9):
            @lru_cache(None)
            def point(a,b):
                if a==4 or b==4:return float(a==4)
                return x*point(a+1,b)+(1-x)*point(a,b+1)
            self.assertAlmostEqual(ref.game_outcomes(x,True)[0],point(0,0),places=14)

    def test_tiebreak_exact_cyclic_chain(self):
        for p,q in ((F(1,2),F(1,2)),(F(3,5),F(7,10)),(F(9,10),F(2,5)),
                    (F(1,10**10),F(2,10**10)),(1-F(1,10**10),1-F(2,10**10))):
            # Compare rational oracles with the exact values represented by floats.
            p,q=F(float(p)),F(float(q))
            for target in (7,10):
                for first in (0,1):
                    self.assertAlmostEqual(ref.tiebreak(float(p),float(q),first,target),
                                           float(rational_tb(p,q,first,target)),places=12)

    def test_service_sequence_and_extended_phase(self):
        self.assertEqual([ref.tb_server(0,i) for i in range(13)],
                         [0,1,1,0,0,1,1,0,0,1,1,0,0])
        for i in range(100):
            self.assertEqual(ref.tb_server(1,i),1-ref.tb_server(0,i))
            self.assertEqual(ref.tb_server(0,i+4),ref.tb_server(0,i))

    def test_next_set_server(self):
        for first in (0,1):
            for n in (6,8,10,12,14,70):
                self.assertEqual(ref.next_set_server(first,n),first)
            for n in (7,9,11):
                self.assertEqual(ref.next_set_server(first,n),1-first)
            self.assertEqual(ref.next_set_server(first,13,True),1-first)

    def test_joint_transition_support(self):
        for rule in ref.RULES.values():
            for first in (0,1):
                t=ref.set_transitions(.63,.71,first,rule)
                self.assertAlmostEqual(sum(t),1,places=13)
                self.assertTrue(all(0<=x<=1 for x in t))
        # An advantage-set tail preserves game parity; no random server reset.
        t=ref.set_transitions(.63,.71,0,ref.RULES['advantage'])
        self.assertGreater(t[0],0)
        self.assertGreater(t[1],0)  # ordinary odd-length set endings also occur

    def test_all_rule_match_symmetries(self):
        for rule in ref.RULES.values():
            ordinary=rule if rule==ref.RULES['nextgen'] else ref.STANDARD
            for best in (3,5):
                for first in (0,1,None):
                    a=ref.match_probability(.63,.71,best,first,ordinary,rule)
                    b=ref.match_probability(.71,.63,best,None if first is None else 1-first,ordinary,rule)
                    self.assertAlmostEqual(a+b,1,places=12)
                    self.assertAlmostEqual(ref.match_probability(.5,.5,best,first,ordinary,rule),.5,places=12)

    def test_unknown_first_is_single_mixture(self):
        for rule in ref.RULES.values():
            self.assertAlmostEqual(ref.match_probability(.61,.73,5,None,deciding=rule),
                sum(ref.match_probability(.61,.73,5,s,deciding=rule) for s in (0,1))/2,places=14)

    def test_original_historical_grid(self):
        for p in (.05,.3,.5,.65,.9,.95):
            for q in (.05,.3,.5,.65,.9,.95):
                for name in ('standard','advantage','tb10','wimbledon_12'):
                    for best in (3,5):
                        expected=original_match(self.original,p,q,best,name)
                        for first in (0,1):
                            value=ref.match_probability(p,q,best,first,deciding=ref.RULES[name])
                            self.assertAlmostEqual(value,expected,delta=2e-12)

    def test_invalid_and_nonterminating_inputs(self):
        for p,q in ((-1,.5),(.5,1.1),(math.nan,.5),(math.inf,.5)):
            with self.assertRaises(ValueError):ref.match_probability(p,q)
        for p,q in ((0,0),(1,1)):
            with self.assertRaises(ref.NonTerminating):ref.tiebreak(p,q)
            with self.assertRaises(ref.NonTerminating):ref.set_transitions(p,q,rule=ref.RULES['advantage'])
        self.assertEqual(ref.match_probability(1,0),1)
        self.assertEqual(ref.match_probability(0,1),0)

    def test_match_tiebreak_differs_from_final_set_tiebreak(self):
        a=ref.match_probability(.63,.71,deciding=ref.RULES['match_tb10'])
        b=ref.match_probability(.63,.71,deciding=ref.RULES['tb10'])
        self.assertGreater(abs(a-b),1e-4)

    def test_nextgen_differs_from_standard(self):
        rule=ref.RULES['nextgen']
        a=ref.match_probability(.63,.71,5,ordinary=rule,deciding=rule)
        b=ref.match_probability(.63,.71,5)
        self.assertGreater(abs(a-b),1e-4)

    def test_best_of_five_three_four_five_set_paths(self):
        # With identical standard-set rules, explicit match-state propagation
        # agrees with disjoint endings in three, four and five sets.
        s=sum(ref.set_transitions(.62,.67,0)[:2])
        expected=s**3+3*s**3*(1-s)+6*s**3*(1-s)**2
        self.assertAlmostEqual(ref.match_probability(.62,.67,5),expected,places=13)

    def test_advantage_tail_exact_cyclic_game_chain(self):
        # Solve lead/server states directly instead of a geometric pair formula.
        ha,la=map(F,ref.game_outcomes(.63))
        hb,lb=map(F,ref.game_outcomes(.71))
        states=[(d,s) for d in (-1,0,1) for s in (0,1)]
        idx={s:i for i,s in enumerate(states)}
        m=[[F(i==j) for j in range(6)] for i in range(6)];rhs=[F(0)]*6
        for i,(d,s) in enumerate(states):
            for step,x in ((1,ha if s==0 else lb),(-1,la if s==0 else hb)):
                if d+step==2:rhs[i]+=x
                elif d+step!=-2:m[i][idx[d+step,1-s]]-=x
        solution=solve(m,rhs)
        pair=ha*lb/(ha*lb+la*hb)
        for s in (0,1):self.assertAlmostEqual(float(solution[idx[0,s]]),float(pair),places=14)

    def test_monotonicity_and_extreme_normalization(self):
        grid=(.001,.01,.1,.5,.9,.99,.999)
        for rule in ref.RULES.values():
            ordinary=rule if rule==ref.RULES['nextgen'] else ref.STANDARD
            values=[ref.match_probability(p,.65,3,ordinary=ordinary,deciding=rule) for p in grid]
            self.assertTrue(all(a<=b+2e-12 for a,b in zip(values,values[1:])))
            values=[ref.match_probability(.65,q,3,ordinary=ordinary,deciding=rule) for q in grid]
            self.assertTrue(all(a>=b-2e-12 for a,b in zip(values,values[1:])))
            for p,q in ((1e-8,2e-8),(1-1e-8,1-2e-8)):
                for first in (0,1):
                    t=ref.set_transitions(p,q,first,rule)
                    self.assertAlmostEqual(sum(t),1,places=12)
                    self.assertTrue(all(-2e-12<=x<=1+2e-12 for x in t))


class RegistryTests(unittest.TestCase):
    def row(self,**changes):
        r=dict(tourney_name='Example',tourney_date='2025-01-01',tourney_level='A',
               round='R32',best_of='3',_source_family='atp_matches',score='6-4 6-4')
        r.update(changes);return r

    def test_no_unknown_competition_default(self):
        a=registry.resolve(self.row(tourney_level='S'))
        self.assertEqual(a['status'],'unresolved')
        self.assertIsNone(a['deciding'])

    def test_grand_slam_boundaries(self):
        for name,boundary,old,new in [('Australian Open',2019,'advantage','tb10'),
            ('Roland Garros',2022,'advantage','tb10'),('Wimbledon',2019,'advantage','wimbledon_12'),
            ('Wimbledon',2022,'wimbledon_12','tb10'),('US Open',2022,'standard','tb10')]:
            for year,rule in ((boundary-1,old),(boundary,new)):
                a=registry.resolve(self.row(tourney_name=name,tourney_level='G',
                    best_of='5',tourney_date=f'{year}-06-01'))
                self.assertEqual(a['deciding'],rule)

    def test_qualifying_is_not_quarterfinal(self):
        for rnd,rule in [('Q1','match_tb10'),('Q2','match_tb10'),('QF','standard')]:
            a=registry.resolve(self.row(tourney_level='15',tourney_date='2024-08-01',round=rnd,
                _source_family='atp_matches_futures'))
            self.assertEqual(a['deciding'],rule)

    def test_unsupported_historical_qualifying_is_unresolved(self):
        a=registry.resolve(self.row(tourney_name='Roland Garros',tourney_level='G',
            tourney_date='2010-05-01',round='Q1'))
        self.assertEqual(a['status'],'unresolved')

    def test_dated_exceptions_override_generic_tour(self):
        a=registry.resolve(self.row(tourney_name='Next Gen Finals',best_of='5',tourney_level='F'))
        self.assertEqual((a['ordinary'],a['deciding']),('nextgen','nextgen'))
        b=registry.resolve(self.row(tourney_name='Laver Cup'))
        self.assertEqual(b['deciding'],'match_tb10')

    def test_best_of_conflicts_preserved_as_audit_override(self):
        r=self.row(tourney_name='Roland Garros',tourney_level='G')
        a=registry.resolve(r)
        self.assertEqual(r['best_of'],'3')
        self.assertTrue(a['format_override'])
        self.assertEqual(a['best_of'],5)

    def test_score_does_not_select_rule(self):
        r=self.row(tourney_name='Example',score='6-4 4-6 [10-8]')
        a=registry.resolve(r)
        r['score']='6-4 6-4'
        self.assertEqual(a,registry.resolve(r))

    def test_doha_alias(self):
        a=registry.resolve(self.row(tourney_name='Doha Aus Open Qualies',tourney_level='G',
            tourney_date='2021-01-10',round='Q1'))
        self.assertEqual(a['deciding'],'tb10')

    def test_original_selector_event_errors_and_alias(self):
        functions,_=extract(Path(__file__).resolve().parents[3])
        select=functions['deciding_set_rule']
        for name in ('Laver Cup','Next Gen Finals'):
            self.assertEqual(select(name,'2025-01-01',False),'standard')
        # 'aus open' contains the substring 'us open': this is misclassified,
        # rather than rejected as an unknown Grand Slam.
        self.assertEqual(select('Doha Aus Open Qualies','2021-01-10',True),'standard')


if __name__=='__main__':unittest.main(verbosity=2)
