# Reproduction package

## Verify the committed notebooks

```bash
python reproduction/check_package.py
```

This standard-library check validates notebook JSON, hashes, executed outputs, corrected split sizes and saved headline metrics.

## Recompute paired test metrics

Copy the three files described in `artifacts/README.md` into `reproduction/artifacts/`, then run:

```bash
pip install -r reproduction/requirements.txt
python reproduction/evaluate_predictions.py
```

The evaluator reconstructs V1.1's `dataset_row` from ordered test metadata, validates labels and lengths, merges V2.1 one-to-one, recomputes both metric rows, and writes `reproduction/paired_metrics.csv`.

This verifies saved predictions; it does not retrain the models. The supplied artifacts were used successfully, and the resulting aggregate table is committed as `paired_metrics.csv`. Their hashes are recorded in `artifact_checksums.json`; the row-level files themselves remain ignored.

