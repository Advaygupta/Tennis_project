"""Independent singles scoring reference; no project probability functions imported.

Server 0 is A and server 1 is B. All calculations are conditional on fixed,
independent point probabilities p and q. Infinite tails are summed exactly in
real arithmetic, rather than truncated. Inputs are never silently clipped.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import comb, fsum, isfinite


class NonTerminating(ValueError):
    """The requested win-by-two contest has a nonabsorbing closed class."""


def validate(p, q):
    if not all(isfinite(float(x)) and 0 <= x <= 1 for x in (p, q)):
        raise ValueError("Serve probabilities must be finite and in [0,1]")


def game_outcomes(x, no_ad=False):
    """Compute BOTH winning tails directly, avoiding subtraction near 0 or 1."""
    validate(x, x)
    y = 1-x
    deuce = comb(6, 3)*x**3*y**3
    denominator = x*x+y*y
    continuation = x if no_ad else x*x/denominator
    opposite = y if no_ad else y*y/denominator
    win = fsum(comb(3+k, k)*x**4*y**k for k in range(3)) + deuce*continuation
    lose = fsum(comb(3+k, k)*y**4*x**k for k in range(3)) + deuce*opposite
    return win/(win+lose), lose/(win+lose)


def tb_server(first, points_played):
    if first not in (0, 1) or not isinstance(points_played, int) or points_played < 0:
        raise ValueError("Invalid service state")
    return (first, 1-first, 1-first, first)[points_played % 4]


def next_set_server(first, games_played, tiebreak=False):
    return 1-first if tiebreak else first ^ (games_played % 2)


def tiebreak(p, q, first=0, target=7):
    validate(p, q)
    if first not in (0, 1) or not isinstance(target, int) or target < 2:
        raise ValueError("Invalid tiebreak format")
    win_pair, lose_pair = p*(1-q), (1-p)*q
    if win_pair+lose_pair == 0:
        raise NonTerminating("Nonterminating win-by-two tiebreak")
    # Once tied at target-1, each two-point block contains one serve each.
    # Split blocks return to a tie, so their geometric series sums to this ratio.
    tail = win_pair/(win_pair+lose_pair)

    @lru_cache(None)
    def point(a, b):
        if max(a,b) >= target and abs(a-b) >= 2:
            return float(a>b)
        if a == b == target-1:
            return tail
        x = p if tb_server(first, a+b) == 0 else 1-q
        return x*point(a+1,b) + (1-x)*point(a,b+1)
    return point(0,0)


@dataclass(frozen=True)
class SetRule:
    games: int = 6
    tiebreak_at: int | None = 6
    tiebreak_points: int = 7
    no_ad: bool = False
    match_tiebreak: bool = False


STANDARD = SetRule()
RULES = {
    "standard": STANDARD,
    "advantage": SetRule(tiebreak_at=None),
    "tb10": SetRule(tiebreak_points=10),
    "wimbledon_12": SetRule(tiebreak_at=12),
    "match_tb10": SetRule(match_tiebreak=True, tiebreak_points=10),
    "nextgen": SetRule(games=4, tiebreak_at=3, no_ad=True),
}


def set_transitions(p, q, first=0, rule=STANDARD):
    """Joint masses [A,next A; A,next B; B,next A; B,next B]."""
    validate(p, q)
    if first not in (0,1):
        raise ValueError("Unknown first server")
    if rule.match_tiebreak:
        x=tiebreak(p,q,first,rule.tiebreak_points)
        v=[0.]*4
        v[1-first],v[3-first]=x,1-x
        return tuple(v)
    if rule.games < 1 or (rule.tiebreak_at is not None and rule.tiebreak_at < rule.games-1):
        raise ValueError("Unsupported set rule")
    ha,la=game_outcomes(p,rule.no_ad)
    hb,lb=game_outcomes(q,rule.no_ad)

    @lru_cache(None)
    def game(a,b,server):
        if max(a,b)>=rule.games and abs(a-b)>=2:
            v=[0.]*4
            v[(0 if a>b else 2)+server]=1.
            return tuple(v)
        if rule.tiebreak_at is not None and a==b==rule.tiebreak_at:
            x=tiebreak(p,q,server,rule.tiebreak_points)
            v=[0.]*4
            v[1-server],v[3-server]=x,1-x
            return tuple(v)
        if rule.tiebreak_at is None and a==b==rule.games:
            # An extended advantage set ends after an even number of games.
            # Both possible absorbing outcomes therefore retain this next server.
            w,l=ha*lb,la*hb
            if w+l==0:
                raise NonTerminating("Nonterminating advantage set")
            v=[0.]*4
            v[server],v[2+server]=w/(w+l),l/(w+l)
            return tuple(v)
        x,y=(ha,la) if server==0 else (lb,hb)
        won,lost=game(a+1,b,1-server),game(a,b+1,1-server)
        return tuple(x*u+y*v for u,v in zip(won,lost))
    return game(0,0,first)


def match_probability(p,q,best_of=3,first=None,ordinary=STANDARD,deciding=STANDARD):
    """Propagate service state; unknown first server is an explicit 1/2 mixture."""
    if best_of not in (3,5) or first not in (None,0,1):
        raise ValueError("Unsupported match state")
    validate(p,q)
    target=best_of//2+1
    transitions={}
    for r in {ordinary,deciding}:
        for s in (0,1):
            transitions[r,s]=set_transitions(p,q,s,r)

    @lru_cache(None)
    def match(a,b,server):
        if a==target:return 1.
        if b==target:return 0.
        rule=deciding if a==b==target-1 else ordinary
        t=transitions[rule,server]
        return fsum(t[s]*match(a+1,b,s)+t[2+s]*match(a,b+1,s) for s in (0,1))
    return .5*(match(0,0,0)+match(0,0,1)) if first is None else match(0,0,first)
