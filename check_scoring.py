"""Standalone standard-library validation. Run: python check_scoring.py

Independent points; constant p=P(A wins point on A serve), q=P(B wins
point on B serve); advantage games; 7-point win-by-two tiebreak at 6-6 in
EVERY set; best of 3 or 5. Server 0=A, 1=B. No training/data dependencies.
"""
from functools import lru_cache
from pathlib import Path
import csv
import json
import math
import random
import hashlib
import original_scoring_formula as original


def hold(p):
    return sum(math.comb(3+k,k)*p**4*(1-p)**k for k in range(3)) + 20*p**3*(1-p)**3*p*p/(p*p+(1-p)**2)


def tb_server(first, points_played):
    if points_played == 0:
        return first
    return first ^ (((points_played-1)//2 + 1) % 2)


def tiebreak(p,q,first):
    # From 6-6, each successive two-point pair contains one serve each.
    win_pair, lose_pair = p*(1-q), (1-p)*q
    denominator=win_pair+lose_pair
    if denominator == 0:
        raise ValueError('Nonterminating tiebreak at p=q=0 or p=q=1; win probability undefined.')
    @lru_cache(None)
    def point(a,b):
        if max(a,b)>=7 and abs(a-b)>=2: return float(a>b)
        if a==b==6: return win_pair/denominator
        chance=p if tb_server(first,a+b)==0 else 1-q
        return chance*point(a+1,b)+(1-chance)*point(a,b+1)
    return point(0,0)


def set_transitions(p,q,first):
    """Joint distribution (A,next A),(A,next B),(B,next A),(B,next B)."""
    ha,hb=hold(p),hold(q)
    @lru_cache(None)
    def game(a,b,server):
        if max(a,b)>=6 and abs(a-b)>=2:
            vector=[0.]*4
            vector[(0 if a>b else 2)+server]=1.
            return tuple(vector)
        if a==b==6:
            win=tiebreak(p,q,server)
            vector=[0.]*4
            # Receiver of the first tiebreak point serves first next set.
            vector[1-server]=win
            vector[2+1-server]=1-win
            return tuple(vector)
        chance=ha if server==0 else 1-hb
        won,lost=game(a+1,b,1-server),game(a,b+1,1-server)
        return tuple(chance*x+(1-chance)*y for x,y in zip(won,lost))
    return game(0,0,first)


def match_probability(p,q,best_of,first):
    transitions=[set_transitions(p,q,s) for s in [0,1]]
    target=best_of//2+1
    @lru_cache(None)
    def match(a,b,server):
        if a==target:return 1.
        if b==target:return 0.
        t=transitions[server]
        return sum(t[s]*match(a+1,b,s)+t[2+s]*match(a,b+1,s) for s in [0,1])
    return match(0,0,first)


def polynomial(s,best_of):
    return 3*s*s-2*s**3 if best_of==3 else s**3+3*s**3*(1-s)+6*s**3*(1-s)**2


def simulate(p,q,best_of,first,count,rng):
    def point(server):return rng.random() < (p if server==0 else 1-q)
    def game(server):
        a=b=0
        while max(a,b)<4 or abs(a-b)<2:
            if point(server):a+=1
            else:b+=1
        return a>b
    wins=0
    for _ in range(count):
        sets_a=sets_b=0
        server=first
        while max(sets_a,sets_b)<best_of//2+1:
            a=b=0
            while True:
                if a==b==6:
                    x=y=0
                    first_tb=server
                    while max(x,y)<7 or abs(x-y)<2:
                        if point(tb_server(first_tb,x+y)):x+=1
                        else:y+=1
                    won=x>y
                    server=1-first_tb
                    break
                if game(server):a+=1
                else:b+=1
                server=1-server
                if max(a,b)>=6 and abs(a-b)>=2:
                    won=a>b
                    break
            if won:sets_a+=1
            else:sets_b+=1
        wins+=sets_a>sets_b
    return wins/count


def main():
    out=Path(__file__).resolve().parent
    grid=[i/20 for i in range(21)]
    rows=[]
    checks=0
    max_error={k:0. for k in ['game','first_server_set','first_server_match','notebook_set','notebook_match','polynomial','symmetry']}
    def close(x,y,key,tolerance=2e-12):
        nonlocal checks
        checks+=1
        error=abs(x-y)
        max_error[key]=max(max_error[key],error)
        if error>tolerance: failures.append({'check':key,'p':p,'q':q,'left':x,'right':y,'error':error})
    failures=[]
    for p in grid:
        for q in grid:
            if (p,q) in [(0,0),(1,1)]:continue
            close(hold(p),original.game_win_probab(p),'game')
            transitions=[set_transitions(p,q,s) for s in [0,1]]
            sets=[sum(t[:2]) for t in transitions]
            for t in transitions:
                assert abs(sum(t)-1)<2e-12 and all(-2e-12<=x<=1+2e-12 for x in t)
            close(sets[0],sets[1],'first_server_set')
            old=original.set_win_probab(p,q)
            close(old,sets[0],'notebook_set')
            close(sets[0],1-sum(set_transitions(q,p,1)[:2]),'symmetry')
            for best in [3,5]:
                probabilities=[match_probability(p,q,best,s) for s in [0,1]]
                close(probabilities[0],probabilities[1],'first_server_match')
                close(probabilities[0],polynomial(sets[0],best),'polynomial')
                close(probabilities[0],polynomial(old,best),'notebook_match')
                rows.append(dict(p=p,q=q,best_of=best,set_A_first=sets[0],set_B_first=sets[1],
                                 notebook_set=old,match_A_first=probabilities[0],match_B_first=probabilities[1],
                                 notebook_match=polynomial(old,best)))
    # Monotonicity and equal-player fairness are independent of notebook agreement.
    for q in grid[1:-1]:
        for best in [3,5]:
            values=[match_probability(p,q,best,0) for p in grid]
            assert all(a<=b+2e-12 for a,b in zip(values,values[1:]))
            values=[match_probability(q,p,best,0) for p in grid]
            assert all(a>=b-2e-12 for a,b in zip(values,values[1:]))
            assert abs(match_probability(q,q,best,0)-.5)<2e-12
    for best in [3,5]:
        assert match_probability(1,0,best,0)==1
        assert match_probability(0,1,best,0)==0
    assert [tb_server(0,k) for k in range(7)]==[0,1,1,0,0,1,1]
    for p,q in [(0,0),(1,1)]:
        try:
            tiebreak(p,q,0)
        except ValueError:
            pass
        else:
            raise AssertionError('Nonterminating corner should be rejected')
    rng=random.Random(20261019)
    simulations=[]
    for p in [.55,.65,.75]:
        for q in [.55,.65,.75]:
            for best in [3,5]:
                for first in [0,1]:
                    exact=match_probability(p,q,best,first)
                    observed=simulate(p,q,best,first,1000,rng)
                    se=math.sqrt(exact*(1-exact)/1000)
                    passed=abs(exact-observed)<=6*se+1/1000
                    simulations.append(dict(p=p,q=q,best_of=best,first_server='AB'[first],
                                            n=1000,exact=exact,simulated=observed,standard_error=se,
                                            within_six_standard_errors=passed))
                    assert passed
    with (out/'grid_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'seeded_simulations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(simulations[0]));w.writeheader();w.writerows(simulations)
    summary=dict(grid_points=439,grid_step=.05,formats=[3,5],absolute_tolerance=2e-12,
                 scalar_comparisons=checks,maximum_absolute_errors=max_error,
                 failures=failures,simulation_seed=20261019,simulation_matches=36000,
                 simulation_cases=36,all_simulation_checks_passed=True,
                 symmetry_boundary_monotonicity_checks_passed=True,
                 nonterminating_corners_excluded=[[0,0],[1,1]],
                 extracted_formula_sha256=hashlib.sha256((out/'original_scoring_formula.py').read_bytes()).hexdigest())
    (out/'results.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    assert not failures, 'Formula agreement checks failed; see results.json'


if __name__=='__main__':main()
