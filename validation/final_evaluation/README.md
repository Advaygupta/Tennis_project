# Final frozen-prediction audit: calibration and log loss

This is the professor's Point 4: a thin layer over the existing Point 3 evaluator.
It reads frozen probabilities/outcomes, produces aggregate diagnostics and four
figures, and never trains, performs ML inference, refits preprocessing, executes
notebooks, modifies scoring code, or changes original artifacts.

Read [report.md](report.md) for findings,
[inventory.md](inventory.md) for reused/missing evidence, and
[rights_and_provenance.md](rights_and_provenance.md) for access restrictions.
**Full-population paired conclusions remain conditional on preserved V1.1 order.**

## Fixed source and package

Baseline: `981fa3d7b23187f8b7073b1d6eadf9490023f8c8`. The publication commit
containing this directory is the reproducibility pin: copy its full SHA from
GitHub's commit view and check it out, instead of relying on a moving `main`.
The result manifest hashes the actual code and reused helpers; the publication
manifest hashes reports, figures, receipts and results. A commit cannot contain
its own SHA inside its hashed contents, so the run manifest identifies the
immutable preceding baseline and the publication commit supplies the final pin.

The optional compact ZIP is produced by `build_package.py` from an explicit
allowlist. It contains this directory, only the four reused Point 3 helpers,
Point 3 summary/manifest, and expected metrics. It excludes all raw/private
rows, notebooks, models, libraries and the large upstream datasets. Extract it
and run from the extracted root using the same commands below. Building the ZIP
is a new utility; no prior pipeline needs rerunning.

## Inputs

Place the exact private files in `reproduction/artifacts/`:

1. `v1_1_results.joblib` — trusted frozen pandas DataFrame, **not a model**.
2. `v2_1_test_match_probabilities.parquet` — original classifier predictions.
3. `matches_v1_1_metadata.csv` — all split metadata; runner selects the original test rows.

For the full recorded sensitivity results, also supply the existing private
`validation/scoring/historical/audit_private/saved_probability_comparison.csv`.
It is optional for primary calibration; omission excludes policy/completion
subsets and scoring-only correction, so such a run will not reproduce the entire
committed summary. Input access must follow the rights inventory. No Google
Drive, Colab, raw-history download or GPU is required.

## Install, verify, test and reproduce

Use Python 3.12 or newer in an isolated environment. From the repository or
extracted package root:

```bash
python -m pip install -r validation/final_evaluation/requirements.txt
python validation/final_evaluation/verify_artifacts.py --scoring-sidecar validation/scoring/historical/audit_private/saved_probability_comparison.csv
python -m unittest discover -s validation/final_evaluation -p test_final_evaluation.py -v
python validation/final_evaluation/run_evaluation.py --output validation/final_evaluation/reviewer_run --scoring-sidecar validation/scoring/historical/audit_private/saved_probability_comparison.csv --replicates 5000 --seed 20261124
python validation/final_evaluation/compare_results.py --actual validation/final_evaluation/reviewer_run/summary.json
```

The new runner defaults to reading the **original Parquet** with PyArrow.
Use a new output directory each time; existing locations and preserved audit/input
locations are rejected. Missing, altered or mismatched inputs fail before joblib
deserialization. Only this known trusted DataFrame is loaded; never replace it
with an untrusted pickle. Invalid probabilities/outcomes and changed populations
also fail rather than dropping rows.

The numeric comparison uses absolute and relative tolerance `1e-10` on scientific
outputs, including all populations and bootstrap arrays summarized into intervals.
Original reported metrics must agree within `1e-12`. Environment, elapsed time
and execution-route descriptions are excluded from comparison. Exact byte hashes
of generated SVGs or JSON are not promised across dependency/platform versions;
the recorded environment is in `results/summary.json`.

## Recorded execution route

The available runtime was Python 3.12, while the locally installed PyArrow
binary was for Python 3.13. Consequently **this Point 4 execution reused the
preserved Point 3 private aligned CSV instead of decoding the original V2
Parquet again**. The original Parquet's bytes were still hashed before/after.
V1 was loaded from its original joblib, metadata from its original CSV, and
the reused export was checked against every common metadata field, V1
probabilities, orientation, counts, scoring sidecar and all expected metrics.
V2's export lineage is inherited from Point 3; the export's own hash is first
recorded here and is not contemporaneous historical authentication.

To reproduce that exact recorded route, privately obtain the preserved Point 3
CSV and add:

```bash
--paired-export validation/paired_evaluation/audit_private/run_20261123/aligned_comparison.csv
```

Its expected SHA-256 is
`e619b001e7dbd7c810a32a1fcd9453f8b844dc24f1c1b45424b5a85d60cd7105`.
No row-level CSV is supplied publicly. The original-Parquet route is provided
for independent reviewers but was not exercised in this runtime; it should be
checked against the committed numeric summary using the comparison command.
No existing notebook or model needs rerunning to do that optional verification.

## Methods and outputs

- Equal-width and quantile reliability tables use 5/10/20 bins. Quantile ties
  remain together; duplicate cuts collapse explicitly. Empty bins are retained,
  means are null, and bins with fewer than 100 observations are flagged.
- ECE is count-weighted absolute bin deviation; MCE is its occupied-bin maximum.
  Brier uses original probabilities; log loss and diagnostic logits use float64
  epsilon safeguards. These summaries depend on grouping and population.
- Joint diagnostic `logit E[Y] = intercept + slope * logit(p)` is estimated from
  test outcomes. Calibration-in-the-large separately estimates an offset
  intercept with slope fixed to 1. No fitted diagnostic is deployed or used to
  report improved test scores. No calibrator is fitted.
- Winner confidence is `max(p,1-p)`, with outcome `1[(p>=0.5)==y]`.
  Groups are `[.5,.6)`, `[.6,.8)`, `[.8,.95)`, `[.95,.99)`, `[.99,1]`.
  Each match appears exactly once in this transformation.
- The primary calibration bootstrap uses 5,000 paired whole-event draws and
  5,000 calendar-block draws, with Point 3's full-event reference-date anchors.
  Bins/cutpoints are fixed to the original sample, including quantile cuts.
  Event-only bootstrap is added for the three exploratory sensitivity sets.
  No iid-only inference or re-estimation of test-fitted calibration models is used.
- Loss tails distinguish descending absolute difference, both signed difference
  directions and each individual model's loss. Tail contributions divide by full
  N and retain all matches; they are not selected-test-set improvements.

`results/summary.json`, `reliability_bins.csv` and `tail_attribution.csv` contain
aggregates. `figures/` contains SVGs. `results/manifest.json` records inputs,
code, output hashes, environment/seed references and preservation checks.
Its output hashes refer to the original run-root layout: published tables and
summary live under `results/`, while figure members live under `figures/`.
The publication manifest uses the final repository-relative layout directly.

## Tests and reruns

The receipt records **43 passing tests** on analytic synthetic examples and
failure paths, including explicit repeated-row checks of the clustered bootstrap,
boundary probabilities, bin ties/emptiness, known diagnostic coefficients,
tail additivity, hashes, invalid outcomes, population/orientation rejection,
output protection, public-package allowlisting and the V1 qualification.
These tests do not prove historical assumptions or authenticate identities.

No rerun is required on the author's side to obtain the published diagnostics.
Optional independent original-Parquet verification uses this **new** runner only.
Training, inference, feature rebuilding and notebook execution remain outside scope.
