# Reviewer summary: scoring verification

23 November 2026. Audited baseline: `1d02053`.

The audit independently checks the conversion from fixed serve-point win
probabilities to match probabilities. It performs **zero ML fits or inference
calls**, executes no notebook, and preserves every original model and prediction.

## Mathematical and implementation results

A separate point/game/set/match engine explicitly propagates the server, including
the joint distribution of set winner and next-set server. It supports standard,
advantage, 10-point final-set, Wimbledon 12-12, replacement match-tiebreak and
Next Gen short/no-ad formats. Infinite tails are summed analytically, without
truncation. Independent exact-rational cyclic chains check the tiebreak and
advantage-set tails.

- 24 deterministic tests passed.
- 784 synthetic comparisons across four historical rules, both first servers and
  BO3/BO5: zero differences above the `2e-12` tolerance.
- All **51,666 saved V1.1 probabilities** agree with the independent engine under
  their **original assigned rules**; maximum difference `3.77e-15`.
- Extreme synthetic inputs expose cancellation/nontermination weaknesses in the
  legacy code. They are outside the saved serve-probability range. The separate
  new engine provides stable scoring and explicit nontermination handling.

Agreement under an assigned rule does not verify that the assigned rule was the
one historically used.

## Historical-rule findings

Primary-source URLs and applicability notes accompany an event/year/draw/round
registry covering all 259,407 supplied modeling records. In the test set, 34,367
records have supported published governing policies and 17,299 remain inferred;
approved individual-event variations are not exhaustively documented.

| Confirmed test-format issue | Matches | Maximum frozen-p/q probability change |
|---|---:|---:|
| Next Gen incorrectly treated as standard long/advantage sets | 30 | 0.07015 |
| Laver Cup incorrectly given a full third set | 18 | 0.03641 |
| RG 2025 main-draw retirement records labelled BO3 | 4 | 0.05305 |

The new scoring module and registry calculate separate counterfactual outputs.
Correcting the 48 event-rule assignments leaves full-test accuracy unchanged
(`0.68904115`); log loss changes from `0.63115991` to `0.63110567`.
The four best-of corrections are reported in a separate scenario because best_of
also enters ML inputs; frozen p/q cannot repair those inputs. No retraining was done.

The old selector also misidentifies the 2021 Doha Australian Open qualifying alias
through a `us open` substring (110 training metadata records). The explicit registry
corrects the mapping; no saved historical scoring outputs were available for comparison.

## Limits and reproduction

Historical qualifying/other edition gaps remain counted rather than defaulted.
There are 1,453 retirement/incomplete test records outside the completed-match
point model. The V1 prediction artifact lacks unique match IDs, so ordered
name/date/best_of/outcome alignment is corroborated but not independently keyed.
Rule-only sensitivity figures are not new claims of fully verified model performance.

[Full methods, findings and limitations](report.md), [source registry](sources.json),
[measured results and hashes](results/summary.json), and
[reproduction commands](README.md) are provided. Public synthetic checks need no
private data. Repeating the full comparison needs the two existing frozen private
artifacts and their recorded hashes. **No existing notebook cells need rerunning.**
