# Independent Markov scoring and historical rules audit

23 November 2026. Baseline: `1d02053b56f09016c4b012ff825f4114b9cb2b1a`.

**All 51,666 saved match probabilities agree with independently calculated
probabilities under their original assigned rules. Historical rule selection is
not fully verified: 48 test matches have confirmed event-format mismatches, four
test matches have best-of metadata conflicts, and 17,299 test mappings remain
inferred.** This is a scoring audit, not completion of every verification gate.

No ML model was fitted, tuned, loaded or used for inference. No preprocessing was
refitted. No existing notebook or script was rerun. Six isolated scoring function
definitions were extracted from notebook JSON and executed after advance notice.
Original notebook, metadata and prediction hashes were checked before and after;
their bytes did not change. Saved model checkpoints were not opened or modified.

## Existing implementation

Notebook `03_v1_1_model_training.ipynb`, zero-based cell 213, contains
`game_win_probab`, `set_win_probab`, `probability_reach_6_6`,
`tiebreak_win_probab`, `deciding_set_win_probab`, and `deciding_set_rule`.
Cell 210 clips ensemble serve probabilities; cells 215/217 calculate set/match
probabilities; cell 232 saves `anal_data` in `v1_1_results.joblib`.

The point model uses constant independent p=P(A wins on A serve) and
q=P(B wins on B serve); hence A wins on B serve with probability 1-q. A game's
infinite deuce continuation is summed algebraically. The standard-set expression
enumerates ordinary game-score endings and a seven-point tiebreak at 6-6. Special
deciding sets replace only its 6-6 continuation. If s is the ordinary-set
probability and d the deciding-set probability, the match expressions are
`s^2+2s(1-s)d` and `s^3+3s^3(1-s)+6s^2(1-s)^2 d`.

Production does not explicitly carry server state between sets or marginalize
over it; its tiebreak function begins with A serving. This is not automatically
an error. The independent engine conditions on both first servers and explicitly
propagates their joint transition distributions; the tested scalar reductions
agree to floating-point precision for the supported standard/historical formats.
No conclusion that arbitrary service sequences are interchangeable is implied.

The earlier checker in `validation/scoring/` validates only ordinary seven-point
tiebreak sets. Its extracted game/set function syntax still matches this notebook.
Its recorded 4,390 checks and simulation results were read, not rerun or counted
as new tests here. It did not validate the later historical-rule extensions.

The older `markov/analytical_prototype.py` and
`markov/monte_carlo_prototype.py` are exploratory scripts, not the saved analytical
evaluation path. Notebook cell 203's older simulation contains visibly defective
tiebreak service updates and an unconditional server flip between sets; its
evaluation call in cell 205 is commented out. These files were inspected, not
executed, and are not used as this audit's reference or evidence of correctness.
No production analytical service-transition error was demonstrated on the tested
interior grid or saved inputs.

## Independent mathematical method and service states

`engine.py` imports no production scoring function. It computes game-win and
game-loss probabilities separately, including deuce, to avoid cancellation in
`1-hold` near extreme probabilities. No-ad games use a single deciding point.
Tiebreaks use a memoized point-score recursion with the legal A,B,B,A,A,... service
sequence, and an analytical repeating-tail absorption probability. A separate
test oracle solves twelve cyclic (point lead, service phase) states using exact
rational Gaussian elimination. It does not use the engine's geometric-tail formula.

Set recursion retains `(games A, games B, next server)`. Its terminal output is
the **joint distribution of set winner and next server**, not only a scalar set
probability. Ordinary games alternate servers, including across set boundaries.
After a tiebreak the receiver of its opening point serves first in the next set,
regardless of tiebreak length [ITF_RULES]. Advantage-set tails are summed in
two-game blocks with the correct next-server parity, independently checked against
a six-state cyclic game-chain solver. Finite 12-12 extensions are enumerated.

Match recursion retains `(sets A, sets B, next server)`, covers endings in three,
four and five sets, and uses a special rule only at the deciding-set state. A
replacement match tiebreak starts immediately at one set all, not after six games
all. Unknown initial service is an explicit 50/50 mixture **once**, without
resetting or re-randomizing service after sets. Short/no-ad sets are separate rules.

There is no point/game-count truncation: the mathematical tail-truncation error is
zero. This is not a universal IEEE-754 roundoff guarantee. The absolute comparison
tolerance is `2e-12`; rational-oracle extreme tests and separately disclosed
production stress failures characterize numerical accuracy on tested inputs.
At p=q=0 or p=q=1 the relevant win-by-two contests do not terminate; the reference
raises `NonTerminating`, rather than inventing a half probability or silently clipping.

## Automated and synthetic verification

- **24 deterministic tests passed**, with zero failures or errors.
- 49 synthetic p/q pairs x four historical deciding rules x two match formats =
  **392 cases / 784 comparisons** with both starting servers. No discrepancies
  exceeded `2e-12`; maximum absolute difference `4.50e-15`.
- Maximum first-server difference on that grid: `4.44e-16`.
- Tests include game deuce, no-ad, exact cyclic tiebreak oracles for 7/10 points,
  extended service phases, joint-transition normalization, next-set service,
  unknown-first-server marginalization, player-exchange complementarity, symmetry,
  dominance, nontermination, three/four/five-set paths, and distinct replacement
  match tiebreak/Next Gen formats. Registry tests cover boundary years, aliases,
  qualifying versus quarterfinals, unsupported formats and metadata conflicts.

The 32-case numerical-stress file identifies 13 defined production/reference
disagreements beyond the comparison tolerance and eight nonfinite production
match outputs at nonterminating corners. For example, p=1e-8, q=2e-8 with an
advantage deciding set gives approximately 0.55556 in production versus 0.13725
in the stable reference. This arises from cancellation in game-loss probabilities
and the production fallback at a zero continuation denominator. The independent
engine supplies the correction and rejects undefined corners.

These stress inputs are outside the saved artifact's ranges: p approximately
0.38254-0.78840 and q approximately 0.38681-0.78796. No corresponding failure
was found in the saved evaluation. Clipping future probabilities is not a proof
of numerical correctness; use the stable engine for scoring-layer recomputation.

## Historical registry and evidence

`sources.json` records primary URLs, dates/sections and retrieval limitations.
Some official sites exposed only indexed excerpts (403/404/502 on direct fetch);
these limits are disclosed, and source documents are not redistributed. This
audit does not claim source-page bytes were archived or cryptographically frozen.

| Competition/period | Ordinary sets | Deciding format | Evidence/status |
|---|---|---|---|
| Australian Open 2019 onward, main/qualifying | Standard | TB10 at 6-6 | AO_2019, GS_2022/2025; published policy verified |
| All four Slams from RG 2022 | Standard | TB10 at 6-6 | GS_2022/2025; published policy verified |
| Wimbledon main draw 2019-2021 | Standard | TB7 at 12-12 | WIM_2019; published policy verified |
| US Open main draw before 2022 | Standard | TB7 at 6-6 | US_HISTORY; historical policy verified |
| Earlier AO/RG/Wimbledon main draw | Standard | Advantage | Retrospective inference; annual source coverage incomplete |
| Earlier Slam qualifying except evidenced AO periods | Not assigned | Not assigned | Unresolved; no silent default |
| ATP/Challenger 2024-2026, documented regular formats | Standard | Standard | ATP year-specific policy; event exceptions take priority |
| Earlier ATP/Challenger/Futures | Standard candidate | Standard candidate | Explicitly inferred, not historically verified |
| ITF main/qualifying 2024 and 2026 | Standard | Standard / replacement TB10 | ITF_2024/2026; approved emergency exceptions not enumerated |
| ITF 2025 | Standard candidate | Draw-specific candidate | Full mid-year snapshot retrieved; amendment continuity across all earlier events remains inferred |
| Next Gen 2024-2025 | Four games, no-ad, TB7 at 3-3 | Same short-set format | NEXTGEN_2024/2025 |
| Laver Cup 2024-2025 | Standard | Replacement TB10 | LAVER_2024/2025 |
| Davis Cup, supported 2016-2018 BO5 and post-2019 BO3 | Standard | Standard | DAVIS_HISTORY; other zone/dead-rubber cases unresolved |
| Paris Olympics 2024 singles | Standard | Standard | OLYMPIC_2024; earlier editions unresolved |

The registry also checks best-of metadata. Men's Grand Slam main-draw matches
are best of five under GS_2025. Wimbledon Q3 is a documented best-of-five exception
to ordinary best-of-three qualifying; earlier-year source gaps remain disclosed.
Actual score strings do not select a prediction rule; they only flag contradictions
or retirements for review. These are retrospective audits, not model selection.

An additional selector defect is independently reproduced: the 2021 alias
`Doha Aus Open Qualies` contains the substring `us open`, so production assigns
the US Open's standard rule instead of AO's TB10 rule. The supplied training
metadata contains 110 such rows. They are not current test predictions, and no
historical scoring artifacts were regenerated. The new registry uses explicit
canonical aliases. An earlier inspection statement that this alias would raise
an exception was incorrect; the isolated-function test establishes the actual
substring misclassification.

The actual event/year/draw/round inventory is supplied as a compressed CSV. Scope
is the **259,407 target-eligible modeling rows**, not all 855,130 historical rows
that contributed feature histories; the latter complete source file is not local.

| Partition | Supported governing policy | Inferred | Unresolved |
|---|---:|---:|---:|
| Train | 6,250 | 171,966 | 3,416 |
| Validation | 9,079 | 17,015 | 15 |
| Test | 34,367 | 17,299 | 0 |

Thus **17,299 test mappings remain unverified**, even though they have a labelled
candidate rule. Across all modeling metadata there are 36 best-of conflicts with
registry expectations; four occur in the test set. Historical/inferred conflicts
are candidates, not automatically confirmed metadata errors. Governing-policy
verification does not establish that every event followed it without a permitted
variation; additional event fact sheets would be needed for that stronger claim.

## Full saved-probability comparison

The frozen V1 artifact contains p, q, best_of, outcomes, set probabilities, rule
labels, tournament name/date and match probabilities. All 51,666 rows align in
ordered tournament name/date/best_of/outcome with the test metadata. The saved V1
file lacks unique match/row IDs: positional alignment is corroborated, but unique
identity cannot be established independently from those shared fields alone.

| Check | Matching | Discrepant | Maximum absolute difference |
|---|---:|---:|---:|
| Isolated original functions versus saved outputs | 51,666 | 0 | 1.22e-15 |
| Independent engine under original assigned rules | 51,666 | 0 | 3.77e-15 |

Original saved assignments were 50,237 standard and 1,429 TB10. Therefore this
saved test set does not directly exercise older advantage/12-12 branches; those
branches are covered by the synthetic verification, not historical saved predictions.

### Confirmed corrections and conditional sensitivities

| Category | Test matches | Changed above tolerance | Largest probability change |
|---|---:|---:|---:|
| Next Gen short/no-ad format | 30 | 30 | 0.07014768 |
| Laver Cup replacement match tiebreak | 18 | 18 | 0.03641264 |
| RG 2025 recorded BO3 versus prescribed BO5 | 4 | 4 | 0.05304850 |

All three categories were routed through supported-but-inappropriate production
assumptions. The four RG records are retirements, so scoring the intended BO5
match still does not model the actual retirement mechanism. Their correction is
reported as a separate **frozen-input best-of sensitivity**, not a fully corrected
predictor: `best_of` is also an ML feature. Repairing the earlier model inputs is
outside this audit, would invalidate frozen-input equivalence, and was not attempted.

The new engine and registry are the separate corrected scoring implementation.
Legacy notebooks and their outputs were preserved. Full row-level original versus
reference and registry-counterfactual probabilities are written separately to the
ignored private output folder. Public files contain aggregate results and hashes.

| Full 51,666-row scenario | Accuracy | ROC-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| Original saved output | 0.68904115 | 0.75988840 | 0.20670643 | 0.63115991 |
| Event-rule correction, original best_of retained | 0.68904115 | 0.75990844 | 0.20668944 | 0.63110567 |
| Event-rule plus four frozen-p/q best_of corrections | 0.68904115 | 0.75990478 | 0.20669226 | 0.63111640 |

Inferred policies retain the original probability in these whole-test scenarios.
No selection or calibration was performed using these outcomes. The figures are
scoring sensitivities, not independently verified new predictive-performance claims.
There are 1,453 retirement/incomplete test records outside the completed-match point
model; they remain in the whole-test scenarios for comparability. Additional
completed-match/verified-policy subset results are provided in `summary.json`.

## Reproduction, preservation and notebook reruns

See [README.md](README.md) for the public synthetic command and private-artifact
comparison command. `summary.json` records input/code hashes, extracted function
hashes, dependency versions and measured counts; `manifest.json` covers final
published code, documentation and result files. Full source availability and format
coverage are explicitly limited as above; this audit does not reopen temporal,
calibration or statistical-uncertainty gates.

**No existing notebook cells need to be rerun to reproduce this audit or obtain
the separate scoring corrections.** No existing artifact is overwritten. Updating
historical notebook displays is optional and was not performed. If requested later,
it should load existing p/q, score using the new module and write a new named result;
it must not execute cell 209 (which performs ML inference) or the existing cell 232
save operation against the original artifact. Base-model/preprocessing corrections,
if ever pursued, require a separate task; no training is authorized here.
