# Independent paired evaluation (professor's Point 3)

This package evaluates **frozen saved predictions only**. It does not import the
old evaluator, load model checkpoints, execute notebooks, fit models, or perform
ML inference. The original comparison is V2.1's equal-weight direct classifier
ensemble versus V1.1's inverse-validation-MAE serve-regression ensemble followed
by Markov scoring. See [report.md](report.md) for findings and limitations.

## What the audit establishes

- Independently reproduced accuracy, ROC-AUC, Brier score and log loss.
- Exact V2.1-to-metadata row/match/player/outcome checks.
- Explicit distinction between V1.1's unique shared context and ambiguous
  positional correspondence. No row IDs are invented for V1.1 predictions.
- Paired confidence intervals for metric differences, using ordinary matches,
  entire event editions, and fixed 28-day calendar blocks.
- Completed-match and supported-policy sensitivity populations, and a separate
  scoring-only correction comparison. Original predictions stay unchanged.

**Full-population pairing remains conditional on the saved V1.1 order.**
An audit passing numerical checks does not authenticate that order within
ambiguous context groups or establish absence of leakage.

## Inputs (private; do not commit)

The three existing files are listed in
[`reproduction/artifacts/README.md`](../../reproduction/artifacts/README.md):

1. `reproduction/artifacts/v1_1_results.joblib`
2. `reproduction/artifacts/v2_1_test_match_probabilities.parquet`
3. `reproduction/artifacts/matches_v1_1_metadata.csv`

The optional **already saved** scoring sidecar is
`validation/scoring/historical/audit_private/saved_probability_comparison.csv`.
It supplies historical policy flags and scoring counterfactuals from Point 2.
It is matched by `dataset_row`, with original probabilities and scoring rules
cross-checked. Inferred-policy counterfactuals are intentionally absent; only
the 48 confirmed Next Gen/Laver corrections are used. Four best-of conflicts
are excluded from the scoring correction. If this sidecar is missing, the
runner reports only analyses available without it. Never regenerate it by
running a model. Its exact hash is in `results/manifest.json`.

The saved joblib is a trusted user-supplied DataFrame, not a checkpoint. Use the
exact frozen input hashes in the manifest to reproduce this run. Do not substitute
new prediction versions and expect the old comparison to remain identical.

## Reproduce without training or inference

From the repository root, in an environment with Python 3.10 or later:

```bash
python -m pip install -r validation/paired_evaluation/requirements.txt
python -m unittest discover -s validation/paired_evaluation -p test_evaluation.py -v
python validation/paired_evaluation/run_audit.py \
  --output validation/paired_evaluation/reviewer_run \
  --private-output validation/paired_evaluation/audit_private/reviewer_run \
  --replicates 5000 --seed 20261123
```

Choose **new** output directory names each time. The runner refuses existing
directories and locations overlapping original notebooks, artifacts, models,
or previous audits. PowerShell users can place the command on one line.
The public `reviewer_run` contains aggregates, input/code hashes and the
comparison inventory. `audit_private` contains the aligned row-level table,
including ambiguous positional assignments explicitly labelled as such.
Keep private output ignored and do not publish it without rights clearance.

The tested environment used Python 3.13.9, NumPy 2.1.3, pandas 2.2.3,
joblib 1.5.3 and PyArrow 23.0.1. PyArrow 19 failed to read the supplied
Parquet with a repetition-level histogram error; using 23.0.1 read the
unchanged file successfully. The runner does not refit preprocessing or
regenerate a prediction file to work around a reader-version issue.

CPU computation uses batches of 16 bootstrap replicates to avoid a
5,000-by-51,666 dense allocation. GPU is unnecessary. Runtime for the recorded
run is in `results/summary.json`; expect minutes on a laptop and comfortably
under 1 GB of working memory in this implementation, depending on package overhead.

## Files

- `inventory.py`: read-only model, artifact and notebook-source inventory.
- `alignment.py`: identifier, orientation, context, population and sidecar checks.
- `metrics.py`: independent metrics, weighted tie-aware AUC, McNemar reference.
- `paired_bootstrap.py`: exact multinomial paired resampling and clustering.
- `diagnostics.py`: descriptive discordance, confidence and concentration checks.
- `run_audit.py`: isolated runner; new output directories only.
- `test_evaluation.py`: deterministic synthetic tests and optional sklearn metric cross-check.
- `results/`: aggregate results and frozen-input/code manifest.

## Statistical conventions

All primary differences are **V2.1 minus V1.1**. Positive accuracy/AUC favours
V2.1; negative Brier/log loss favours V2.1. Both probability columns mean
`P(Player A wins)`. A score of exactly 0.5 predicts Player A. Log loss clips
only its logarithm arguments to float64 epsilon and one minus epsilon;
accuracy, AUC and Brier use the original probabilities.

Each bootstrap draw uses the same multiplicity for both models and outcomes.
Match resampling draws N matches; event/block resampling draws K entire
units with replacement, computes match-weighted metrics using the resulting
variable number of matches, and recomputes AUC exactly in each draw. The
reported intervals are the 2.5th and 97.5th percentiles of paired differences.
No IID DeLong p-value is used. The exact binomial McNemar p-value is provided
only as an independent-match reference, with its null and limitation recorded.

Event clusters use complete `tourney_id` editions, keeping main/qualifying
draws together when they share that ID. Dates and year crossings do not split
an event ID. Calendar clusters use non-overlapping 28-day bins from Monday
1970-01-05. Every event is anchored to its earliest reference date in the full
test population, including in subset analyses, so an edition is never split
between blocks. These are reference dates, not claimed exact match dates.

All designs assume independence **between their resampling units**. Repeated
players across units and longer serial dependence are not eliminated. Intervals
condition on fixed fitted models and exclude training/selection uncertainty.
They are pointwise intervals, not simultaneous or multiplicity-adjusted claims.
Additional models, scoring scenarios and subsets are descriptive/exploratory.
No equivalence margin or practical materiality threshold was pre-specified.

## Reruns on the author's end

**None is required for this audit.** All three frozen inputs and the saved
Point 2 sidecar were available. No existing script or notebook cell was run.
Missing V1.1 match/player IDs cannot be recovered by rerunning the metric cell
or attaching metadata to the old array position. Definitive resolution would
need an already-preserved ID-bearing V1.1 output or independently authenticated
export lineage. Generating new predictions would violate this task's scope and
would not authenticate the historical saved output.
