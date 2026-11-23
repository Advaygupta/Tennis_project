# Tennis V1.1/V2.1: independent paired evaluation and uncertainty

**Point 3 audit, 23 November 2026. Baseline: `65e1b58811ebeb44e1be513590a7009a1dd37b91`.**

All four originally reported metrics reproduce within 3.33e-15. The 0.14323
percentage-point accuracy difference and 0.00059229 AUC difference do not establish
a material predictive advantage: accuracy intervals include zero under every
design, and AUC intervals include zero under event and calendar-block resampling.
The iid-match AUC interval is narrowly positive, making that inference sensitive
to the dependence assumption. Brier score and log loss favour V2.1 under all three
resampling schemes, conditional on the frozen population and row correspondence.

**Identity qualification:** V2.1 has exact row/match/player correspondence with
the supplied metadata. V1.1 lacks match and player identifiers. Only 35 records
have unique shared context within the intended test population; 51,631 remain
ambiguous by those fields. Thus the full paired inference is conditional on
preserved V1.1 order, not an independently authenticated pairing of every match.
No uncertainty method repairs this evidence gap.

## 1. Original comparison and available evidence

Model A is the **V2.1 equal-weight CatBoost, LightGBM, XGBoost and SGD classifier
ensemble**. Model B is **V1.1's inverse-validation-MAE Elastic Net, LightGBM,
XGBoost and CatBoost serve-regression ensemble plus analytical Markov scoring**.
This is the pre-existing README comparison, not a scoring-correction comparison.

Both probability fields mean `P(Player A wins)`, and the binary outcome is
`playerA_wins`. Accuracy predicts A at probabilities **greater than or equal to
0.5**, including ties. Recorded retirements are retained in the primary outcome.
The chronological splits are 181,632 training, 26,109 validation and 51,666 test
records; test tournament reference dates run from 2024-07-08 to 2026-06-01.
Notebook source/outputs describe final train-plus-validation fits with test
reserved for evaluation. This audit verifies saved outputs and source structure;
it does not independently authenticate prediction-generation history or prove
leakage was absent.

The inventory also locates V2.1 stacking and the four individual classifiers in
the saved Parquet. Their descriptive metrics are in `results/summary.json`.
They share the same outcomes, orientation, threshold and test split. They were
not used to select another winner. No frozen V1.1 individual-regressor match
probability artifacts or standalone Elo/rank baseline predictions were supplied;
such comparisons cannot be reconstructed under the no-inference constraint.
`results/inventory.json` records precise artifacts, versions, reported metrics,
and notebook cell/source hashes. Notebook cell indices there are zero-based.

## 2. Alignment and population accounting

| Check | Records |
|---|---:|
| Original V1.1 saved predictions | 51,666 |
| Original V2.1 saved predictions | 51,666 |
| Original metadata, all splits | 259,407 |
| Intended test metadata | 51,666 |
| V2.1 exact dataset row, match ID, oriented player IDs and outcome | 51,666 |
| V1.1 independently saved match/player IDs | 0 |
| V1.1 unique shared-context corroboration | 35 |
| V1.1 ambiguous positional records | 51,631 |
| Unmatched rows or silently discarded full-population rows | 0 |

Metadata `dataset_row`, test `match_id`, source-file/source-row provenance and
V2.1 IDs are unique and nonmissing. Same-event/same-round oriented player-pair
collisions are rejected. V2.1 is merged one-to-one by `dataset_row`, then checked
against match ID, both player IDs and outcome. Its row ordering may differ;
the join explicitly restores metadata order.

V1.1's ordered tournament name, tournament date, best-of and outcome agree with
metadata. Those four shared fields are also reconciled independently as composite
keys: a record is classified as unique only when its key occurs once in **both**
V1.1 and test metadata. The outcome is part of this fallback context. No players
are saved in V1.1; even singleton corroboration is limited to uniqueness within
this frozen population. Within ambiguous groups, a probability permutation can
pass all shared-context checks. The 35-record subset excludes exactly 51,631
records, is context/label-selected and unrepresentative, and is not promoted to
the main out-of-sample result. Its intervals are tabulated below.

## 3. Independent metric reproduction and primary uncertainty

The following A/B values reproduce the original saved four-metric table; maximum
absolute discrepancy is 3.33e-15. The CI column uses **event-edition resampling**.
All differences are A minus B: positive favours A for accuracy/AUC; negative
favours A for Brier/log loss.

| Metric | Model A | Model B | Paired difference | 95% CI |
|---|---:|---:|---:|---|
| Accuracy | 0.69047343 | 0.68904115 | +0.00143228 | [-0.00021374, +0.00310760] |
| ROC-AUC | 0.76048069 | 0.75988840 | +0.00059229 | [-0.00004480, +0.00123070] |
| Brier score | 0.19887455 | 0.20670643 | -0.00783189 | [-0.00852311, -0.00713335] |
| Log loss | 0.58104872 | 0.63115991 | -0.05011119 | [-0.05421161, -0.04600166] |

The evaluator uses direct correctness and squared-error means, natural-log loss,
and a separate exact tie-aware ranking AUC implementation. Probabilities must
be finite and in [0,1], and outcomes exactly binary. Only log arguments are clipped
at float64 epsilon and one minus epsilon. AUC uses continuous scores and gives
half credit to ties; it is not computed from thresholded predictions. These
computations do not call the original evaluator or sklearn. A secondary
synthetic test checks against sklearn.

### Resampling comparison

All intervals below use **5,000 paired replicates**, a fixed recorded seed per
analysis, and the 2.5th/97.5th percentiles of metric differences. Both models and
outcomes always receive the same multiplicities. AUC is recomputed exactly with
resampled positive/negative weights in every replicate, including tied scores.

| Metric | Match reference | Event editions | 28-day blocks |
|---|---|---|---|
| Accuracy | [-0.00015484, +0.00315488] | [-0.00021374, +0.00310760] | [-0.00014572, +0.00297490] |
| ROC-AUC | [+0.00002141, +0.00117930] | [-0.00004480, +0.00123070] | [-0.00003089, +0.00117250] |
| Brier score | [-0.00846733, -0.00721342] | [-0.00852311, -0.00713335] | [-0.00859635, -0.00699921] |
| Log loss | [-0.05351760, -0.04681244] | [-0.05421161, -0.04600166] | [-0.05595456, -0.04406506] |

- **Match reference:** 51,666 resampling units; assumes independent match pairs.
- **Event editions:** 1,407 `tourney_id` units. Each ID includes its edition year;
  all dates/draws with that ID remain together, including any year crossing.
- **Calendar blocks:** 25 nonempty 28-day units on a fixed grid originating
  Monday 1970-01-05. Entire events are assigned to the block containing their
  earliest full-test reference date; subset analyses retain those anchors.
  0 event IDs span more than one recorded reference date.

Event/block draws sample K whole units with replacement and calculate match-weighted
metrics over the resulting variable number of matches. They do not average event
metrics equally. Resampling units are **assumed**, not proven, independent.
Repeated players across events/blocks and dependence exceeding 28 days remain.
Only 25 time units are available, and the grid/length is one prespecified
sensitivity, not a comprehensive serial-dependence analysis. Tournament dates
are reference dates, not verified individual match dates. No full-population
bootstrap replicate has undefined AUC. Half-run percentile estimates in the
JSON provide a Monte Carlo stability diagnostic, not additional inference.

Accuracy intervals include zero under all three designs. For AUC the iid-match
interval narrowly excludes zero, whereas event and calendar-block intervals
include zero. Thus an iid significance conclusion for AUC is not robust to the
specified dependence-aware resampling, and the observed AUC change is only
0.00059229. These results do not establish a material accuracy/AUC advantage and
are **not proof of equivalence**. The paired Brier/log-loss intervals stay below zero, supporting
better probability scoring by frozen V2.1 under the stated conditions. No
practical materiality threshold or downstream decision utility was specified;
statistical exclusion of zero is not a claim of practical superiority.

## 4. Paired correctness and probability diagnostics

| Classification outcome | Matches |
|---|---:|
| Both models correct | 34,711 |
| Both models wrong | 15,103 |
| V2.1 alone correct | 963 |
| V1.1 alone correct | 889 |

Exact two-sided McNemar p = **0.0898022** for
equal marginal accuracy, using a Binomial(discordant count, 0.5) reference.
Its independent-match-pair assumption is not secured for tennis; this is a
descriptive reference alongside dependence-aware intervals. No iid DeLong
p-value is used for AUC; paired bootstrap intervals supply AUC uncertainty.
No bootstrap sign frequency is mislabelled as a null-test p-value.

Absolute probability differences have median 0.072719, 95th
percentile 0.142458 and maximum 0.293021. Per-match
log-loss differences (A minus B) have median +0.068250, 1st
percentile -1.817925 and 99th percentile +0.227214.
The largest 1% of absolute loss differences account for
14.66% of total
absolute loss difference and contribute
-0.027487 to the overall mean
loss difference. The ten largest absolute net event differences account for
4.30%
of absolute net event differences. These are concentration diagnostics, not
grounds for selective exclusion. Wrongly confident predictions assigning the
observed outcome below 0.05 occur 46
times for A and 1,092 times for B.
The positive median loss difference alongside the negative mean shows that
V2.1's aggregate log-loss gain comes from reducing large adverse losses, rather
than improving the typical match's log loss. The largest 1% of differences
contribute about 55% of the net mean gain; V1.1's loss exceeds V2.1's by more
than one natural-log unit on 1,417 records, with zero records in the reverse
direction. This tail sensitivity matters when interpreting the loss advantage.
Year, surface and tournament-level summaries are exploratory and stored in JSON;
no subgroup was searched to declare significance or change the selected model.

## 5. Prespecified sensitivity populations

The primary population retains all 1,453 retirement/incomplete records and all
17,299 inferred scoring-policy mappings. Completed-only removes the former;
supported-policy restricts to 34,367 records whose governing policy has collected
primary-source support. This **reuses Point 2's evidence classification**, not a
new verification of every scorecard or individual-event exception. The combined
subset has 33,294 records. All sensitivity pairs remain conditional on the same
V1.1 correspondence, and none is relabelled the primary result.

| Population | N | Metric | A minus B | Event 95% CI | Block 95% CI |
|---|---:|---|---:|---|---|
| Completed | 50,213 | Accuracy | +0.00123474 | [-0.00042446, +0.00295890] | [-0.00016422, +0.00271342] |
| Completed | 50,213 | ROC-AUC | +0.00055179 | [-0.00006863, +0.00117378] | [-0.00005978, +0.00113593] |
| Completed | 50,213 | Brier score | -0.00713441 | [-0.00783304, -0.00644398] | [-0.00789527, -0.00630059] |
| Completed | 50,213 | Log loss | -0.04630002 | [-0.05040808, -0.04217014] | [-0.05230138, -0.04000317] |
| Supported policy | 34,367 | Accuracy | +0.00212413 | [+0.00000000, +0.00424170] | [+0.00016287, +0.00393430] |
| Supported policy | 34,367 | ROC-AUC | +0.00077492 | [-0.00002971, +0.00161062] | [+0.00001393, +0.00145336] |
| Supported policy | 34,367 | Brier score | -0.00629931 | [-0.00712329, -0.00550173] | [-0.00713377, -0.00556552] |
| Supported policy | 34,367 | Log loss | -0.03625456 | [-0.04070597, -0.03211705] | [-0.04359168, -0.02979592] |
| Completed + supported | 33,294 | Accuracy | +0.00189223 | [-0.00017777, +0.00408847] | [+0.00009501, +0.00364809] |
| Completed + supported | 33,294 | ROC-AUC | +0.00071439 | [-0.00010853, +0.00152038] | [-0.00006000, +0.00137129] |
| Completed + supported | 33,294 | Brier score | -0.00554520 | [-0.00636605, -0.00475568] | [-0.00635265, -0.00480563] |
| Completed + supported | 33,294 | Log loss | -0.03253707 | [-0.03701771, -0.02836483] | [-0.03977919, -0.02583309] |

The completed subset retains 1,407 event units; supported and combined subsets
each retain 814. Each retains all 25 time units. All use 5,000 paired replicates.
The subset JSON includes each model's four metric values as well as differences.

Completed-only accuracy/AUC intervals include zero under both designs. Some
supported-policy block intervals narrowly exclude zero, while corresponding
event intervals include or touch zero; the combined subset also has a positive
block accuracy interval but an event interval spanning zero. These are
population/design-sensitive exploratory findings, not robust evidence of a
material predictive advantage. Brier/log-loss intervals remain negative in
each of these three subsets under both dependence-aware designs.

### Unique shared-context subset (35 records)

This subset has 25 event units but only **five** nonempty time blocks. Its sparse,
context/label-selected composition cannot validate the full-population comparison.

| Metric | Model A | Model B | Paired difference | 95% CI |
|---|---:|---:|---:|---|
| Accuracy | 0.54285714 | 0.51428571 | +0.02857143 | [+0.00000000, +0.08823529] |
| ROC-AUC | 0.74000000 | 0.72666667 | +0.01333333 | [-0.03461806, +0.07894737] |
| Brier score | 0.22182187 | 0.24336229 | -0.02154042 | [-0.04318506, -0.00200921] |
| Log loss | 0.62075189 | 0.66469854 | -0.04394665 | [-0.11466060, +0.01410101] |

| Metric | Match reference | Event editions | 28-day blocks |
|---|---|---|---|
| Accuracy | [+0.00000000, +0.08571429] | [+0.00000000, +0.08823529] | [+0.00000000, +0.03658537] |
| ROC-AUC | [-0.03623188, +0.06884058] | [-0.03461806, +0.07894737] | [-0.14814815, +0.03819444] |
| Brier score | [-0.04363541, -0.00152281] | [-0.04318506, -0.00200921] | [-0.06777929, -0.00892737] |
| Log loss | [-0.11419153, +0.01323852] | [-0.11466060, +0.01410101] | [-0.15230911, -0.01292142] |

Undefined single-class AUC draws are counted and excluded **only from AUC
interval calculation**, not from other metrics. Three of 5,000 block draws have
undefined AUC, leaving 4,997 valid AUC replicates. All other metrics/designs
in this subset retain all 5,000 draws. Such conditional AUC intervals
are particularly fragile with few blocks; nonsignificance is not equivalence.

## 6. Separate scoring-only correction sensitivity

Here A means **V1.1 with 48 confirmed Next Gen/Laver event-rule corrections**
and B means **the original saved V1.1**, both using the same already frozen p/q.
These are saved counterfactuals from Point 2, joined by dataset row and checked
against the original probability and rule. No scoring or ML pipeline was rerun.
Four best-of metadata conflicts are left unchanged: their ML input-feature effect
cannot be repaired with frozen p/q. Inferred mappings retain original values.

| Metric | Model A | Model B | Paired difference | 95% CI |
|---|---:|---:|---:|---|
| Accuracy | 0.68904115 | 0.68904115 | +0.00000000 | [+0.00000000, +0.00000000] |
| ROC-AUC | 0.75990844 | 0.75988840 | +0.00002004 | [-0.00000875, +0.00006732] |
| Brier score | 0.20668944 | 0.20670643 | -0.00001699 | [-0.00004703, +0.00000013] |
| Log loss | 0.63110567 | 0.63115991 | -0.00005424 | [-0.00014401, -0.00000309] |

| Metric | Match reference | Event editions | 28-day blocks |
|---|---|---|---|
| Accuracy | [+0.00000000, +0.00000000] | [+0.00000000, +0.00000000] | [+0.00000000, +0.00000000] |
| ROC-AUC | [+0.00000025, +0.00004198] | [-0.00000875, +0.00006732] | [-0.00000837, +0.00007578] |
| Brier score | [-0.00003038, -0.00000537] | [-0.00004703, +0.00000013] | [-0.00005331, +0.00000012] |
| Log loss | [-0.00010199, -0.00001457] | [-0.00014401, -0.00000309] | [-0.00016370, -0.00000306] |

Exactly 48 confirmed event-rule rows are changed;
accuracy stays unchanged. These corrections are not a new predictor or independent
evidence of model improvement. Their small and event-concentrated uncertainty
must not be conflated with the V1.1/V2.1 comparison.
Event/block AUC and Brier intervals include zero, while the log-loss interval
remains negative. Only four event editions supply the changes, so the latter
interval should be interpreted particularly cautiously. The exactly zero
accuracy interval describes unchanged classifications in these frozen rows;
it is not a general equivalence result.

## 7. Reproducibility, rights and remaining limits

**No rerun is required on the author's end.** All needed frozen inputs were
available. Model fits, ML inference calls, notebook executions and existing-script
reruns are all **zero**. Original inputs were SHA-256 checked before and after
evaluation and their bytes were preserved. The temporal and scoring audits were
not modified. The new suite passes **26 tests**, including ambiguity-preserving
alignment, orientation/order errors, missing/duplicate IDs, weighted tied AUC,
McNemar counts, seeds, paired cluster multiplicities and subset construction.
The recorded evaluator took **274.8 seconds** on CPU.

See [README.md](README.md) for exact installation/test/run commands and new-output
directory requirements. `results/manifest.json` records the input and executed
code hashes, seeds, environment and aggregate hashes. `package_manifest.json`
also covers documentation and test receipts. Required frozen inputs:

| Input | SHA-256 |
|---|---|
| `v1_1_results.joblib` | `ed2982fda03f9c3735b6694040451f29f833f5a305a05ab854a8fe2fb5b75912` |
| `v2_1_test_match_probabilities.parquet` | `1342c40da095d4551cd6f0a89f0981fb294b33c0a069f8df734faef1ab01a021` |
| `matches_v1_1_metadata.csv` | `943813de874ded92c15d6542b7e4684b7a2b992ea1871bb99d152819533c989f` |
| `saved_probability_comparison.csv` | `f8f6d5008042ece8184effe841f55e138db04c71e730f3544f642145e3283eef` |

Public output contains only code, aggregate results, synthetic tests and hashes.
Aligned predictions/outcomes remain in ignored `audit_private/` and are not
redistributed. Authorized reviewers need the frozen private files; hashes alone
do not reproduce results or grant data rights.

Intervals condition on fixed fitted models and exclude training, tuning and
selection uncertainty. Earlier notebooks contain exploratory test displays;
absence of all historical test-guided choices is not established. Intervals are
pointwise, not family-wise: no multiplicity-adjusted superiority claim is made.
Ancillary models, subsets and corrections are explicitly exploratory. The earlier
ten-day history convention and uncertain source availability are unchanged;
matching old metrics does not prove a leakage-free generation process.

Definitive V1.1 identity requires a preserved ID-bearing output or independently
authenticated export lineage. Attaching metadata IDs to the old array by position,
rerunning the old metric cell, or producing new model predictions would not
authenticate those historical pairs. The present numerical/uncertainty audit is
complete **with that explicit unresolved qualification**.

## Method references

- [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html): paired-index resampling and percentile interval conventions; the implementation here is independent.
- [statsmodels McNemar documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html): exact binomial reference and equal-marginal null; not called by this evaluator.
- [Cameron and Miller (2015), A Practitioner's Guide to Cluster-Robust Inference](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf): motivation for treating within-unit dependence separately from assumed between-unit independence. This reference is not a validation of tennis-specific clustering choices.
