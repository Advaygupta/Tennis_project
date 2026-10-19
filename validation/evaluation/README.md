# Original v1 evaluation reproduction

This package reproduces four metrics from 51,885 saved v1 test predictions without private Drive access, trained models or retraining.

Run from the repository root after placing these files in `validation/evaluation/`:

    python validation/evaluation/evaluate_v1.py

Only the Python standard library is required. The script reads files relative to its own location and checks results against `expected_metrics.json` within absolute tolerance 1e-12. Tested runtime versions and source hashes are in `manifest.json`.

## Contents

- `v1_results.csv`: saved p/q, match format, binary outcome and set/match probabilities, plus reconstructed match IDs and tournament dates. No names are present. Row order is preserved from the supplied saved artifact.
- `evaluate_v1.py`: accuracy with threshold >= 0.5, natural-log loss, Brier score and probability-ranking ROC-AUC with average ranks for ties.
- `expected_metrics.json`: independently calculated expected values.
- `manifest.json`: artifact provenance, hashes and scope limitations.
- `split_manifest.json`: original v1 split counts, tournament-date ranges, reconstruction procedure and validation checks.
- `split_membership.csv`: each eligible match ID, split, row within split and tournament date. The script checks counts/date ranges and test-prediction membership against these records.
- `source_data_manifest.json`: content hashes for all 108 local source CSVs, a snapshot identifier, and limits of source provenance. Raw source data is not redistributed.
- `requirements.txt`: no third-party packages are required for evaluation; the tested Python version is recorded.

## Results and scope

Accuracy: 0.6958080370049147. Log loss: 0.615575567168235. Brier score: 0.2032140249967398. ROC-AUC: 0.7665526737159448.

These are original, provisional predictions. Reproducing their metrics does not validate feature availability, preprocessing, data independence or predictive superiority. Upstream corrections have not been applied. Split membership was reconstructed from local cleaned data and checked against saved local split labels, match formats and serve-point targets (within float32 rounding). Original predictions had no match IDs; their ID/date mapping is reconstructed through the matching local test sequence rather than independently recorded at prediction time. Dates are tournament-week labels, not match dates. Adjacent splits share boundary dates; this records the original splits rather than fixing them. No baseline comparison is supplied. V2's available table contains validation predictions and is excluded from a supposed common-test comparison.

The source manifest identifies current local file contents but cannot recover the exact upstream Git commit or prove these files were unchanged since the original download. It does not itself provide access to the frozen raw snapshot. Source provenance and access therefore remain partial. Original Colab training dependency versions are not established by the runtime versions used for these reproduction scripts.

The source artifact was a joblib-serialized DataFrame named `v1_results.parquet`. It was converted to CSV without modifying probabilities, outcomes or row order. Conversion was checked by reading back the CSV and restoring the original numeric dtypes. Evaluation thresholds and metric definitions are explicit in the script. Log loss clips exact boundaries at float64 machine epsilon; none of this artifact's probabilities requires clipping.

Confirm applicable source-data redistribution terms before publishing the prediction table; omission of names alone does not establish redistribution rights.
