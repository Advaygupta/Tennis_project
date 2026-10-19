# Standalone validation of the tennis scoring formula

## Finding

The notebook's scalar set formula and its best-of-three/best-of-five aggregation are consistent with an independent service-order-aware dynamic program under the scoring model specified below. Service-order carryover does not invalidate these expressions within that model. This is a mathematical/model-implementation finding, not validation of the trained probabilities or reported predictive performance.

No training notebook, data-preparation notebook, model artifact or tennis dataset was executed or loaded for this validation. The notebook JSON was read, and only the two function definitions `game_win_probab` and `set_win_probab` were extracted from original code cell 230 (zero-based) into `original_scoring_formula.py`. Those extracted scalar functions were executed in isolation. The notebook's match expressions in cell 234 were inspected for agreement with the polynomials used in the checker. The original files were not changed. The notebook SHA-256 is recorded in `extraction_manifest.json`.

## Assumptions and rules

- Points are independent, conditional on fixed player-specific serve probabilities.
- p is A's point-win probability on A's serve; q is B's point-win probability on B's serve. These remain constant throughout the match.
- Games use advantage scoring, with a two-point winning margin after deuce.
- Sets are first to six games with a two-game margin, with a first-to-seven-points, win-by-two tiebreak at 6-6.
- Every set uses the same rules, including the deciding set.
- Service alternates between ordinary games. Tiebreak serves follow one point, then alternating blocks of two. The receiver of the first tiebreak point serves first in the next set.
- Matches are best of three or best of five; retirement and walkover rules are outside this model.

This does not assert that historical tournament formats all followed these rules, or that real point probabilities are constant. Advantage deciding sets, ten-point deciding tiebreaks, match tiebreaks and other formats require separate treatment. The dataset includes multiple years and event categories; format coverage has not been audited here.

## Why service carryover can cancel

Newton and Keller prove that, in the independent-point model and scoring format they analyse, the probability of winning a set is independent of the first server. Their match derivation explicitly accounts for service rotation and then reduces to the familiar expressions. See *Probability of Winning at Tennis I. Theory and Data*, Studies in Applied Mathematics 114 (2005), 241–269, especially the first-server analysis and equations (51)–(52):

https://www.cis.upenn.edu/~bhusnur4/cit592_fall2013/NeKe2005.pdf

Let s denote A's set-win probability under the assumptions above. The next set's first server depends on the preceding games, but its conditional win probability is s for either server. Independent future points, constant p/q and identical set rules therefore give the same s after every completed-set history.

For best of three, A can win in two sets (probability s²), or split the first two sets and win the third (probability 2s²(1-s)). Their sum is:

P(A wins best of three) = 3s² − 2s³.

For best of five, A wins in three, four or five sets:

P(A wins best of five) = s³ + 3s³(1-s) + 6s³(1-s)².

These expressions can be exact within this model. It is not necessary to assume that the next set begins with a freshly randomised server. A finite numerical grid alone is not a proof for the continuum; the mathematical justification comes from the stated result and conditional-probability argument, supported by implementation checks.

## Independent implementation

`check_scoring.py` uses a separate dynamic program to enumerate game-score states. Each completed-set transition records the joint distribution of (set winner, next set's first server). A match recursion then propagates both set score and next server without using the repeated-set polynomial. Tiebreak probabilities use a separate point-state recursion. At 6-6 points, the infinite tail is summed analytically using the chance of winning or losing both points in a two-point pair.

A separate point-by-point Monte Carlo implementation keeps service rotation across ordinary games, tiebreak points and subsequent sets. It uses no game, set or match win-probability formula. This replaces the original notebook's simulation as the validation reference; it does not repair that notebook's simulation.

## Results

- Deterministic grid: p and q from 0 to 1 in increments of 0.05; 439 defined pairs after excluding two nonterminating corners. Both initial servers and both match formats are evaluated.
- 4,390 scalar agreement checks; tolerance 2 × 10⁻¹²; no failures.
- Maximum absolute error between extracted notebook set formula and dynamic program: 6.67 × 10⁻¹⁶ (rounded upward).
- Maximum absolute error between notebook-based match polynomial and service-aware match recursion: 1.23 × 10⁻¹⁵ (rounded upward).
- First-server invariance, equal-player fairness, player-exchange set symmetry, probability bounds/transition normalisation, dominance boundaries and monotonicity checks passed. Monotonicity is checked on the grid, not proved symbolically by the script.
- Seeded simulation: seed 20261019; p,q in {0.55, 0.65, 0.75}; both initial servers and both match formats; 1,000 matches per case, 36 cases, 36,000 matches total. All estimates lay within six theoretical Monte Carlo standard errors plus one observation of their dynamic-programming probabilities. This is a stochastic sanity check, not a tight error guarantee or an empirical prediction assessment.
- At p=q=0 and p=q=1, a win-by-two tiebreak never terminates. These are undefined match-win cases under these rules, not automatic 50/50 outcomes. The checker rejects them; the original formula has a zero denominator there. This boundary handling still needs to be added to production code if such values can occur.

Detailed results are in `results.json`, `grid_comparison.csv` and `seeded_simulations.csv`. These files contain synthetic probability inputs only.

## Reproduction

Keep `check_scoring.py` and `original_scoring_formula.py` together and run:

    python check_scoring.py

Only the Python standard library is required. No private Drive, packages, trained models, downloads or original notebook is needed. The checker overwrites only its own three result files in its directory. Results are generated in seconds rather than requiring model retraining.
