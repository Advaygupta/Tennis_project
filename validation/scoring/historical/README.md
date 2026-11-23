# Independent historical scoring audit

Start with [report.md](report.md). This audit extends the older standard-set
checker one directory above. It does not run that checker, any notebook, ML
inference, preprocessing, tuning, or training. Original artifacts are immutable.

## Reproduce

Use Python 3.12 or newer and a separate environment:

```bash
python -m pip install -r validation/scoring/historical/requirements.txt
python validation/scoring/historical/audit.py --synthetic-only --output /tmp/scoring-audit --private-output /tmp/scoring-audit-private
```

On Windows, replace `/tmp/...` with separate local output directories. Run the
commands from the repository root. The output directories are created if absent;
files with the audit's output names in them are replaced. Production files are
never changed. To run just the 24 automated tests:

```bash
python -m unittest discover -s validation/scoring/historical -p test_reference.py -v
```

The full comparison additionally needs the **existing**, trusted private files
`reproduction/artifacts/matches_v1_1_metadata.csv` and `v1_1_results.joblib`.
Expected SHA-256 values are in `results/summary.json`. Do not load untrusted pickle
or joblib files. The full frozen input files are not redistributed in this audit.

```bash
python validation/scoring/historical/audit.py --output /tmp/scoring-audit --private-output /tmp/scoring-audit-private
```

No notebook cells need to be rerun. `production_adapter.py` reads notebook JSON
and executes only six allowlisted **function definitions**, currently in
zero-based cell 213. It never executes the cell's top-level statements. The
scoring-only polynomial from cell 217 is reproduced in the comparison harness.
The captured notebook/function hashes bind this comparison to its source.

## Files

- `engine.py`: independent game, point-state tiebreak, joint set-transition and
  match-state calculations. Supports advantage sets, 7/10-point tiebreaks at 6-6,
  a 7-point tiebreak at 12-12, replacement match tiebreaks, and Next Gen short/no-ad sets.
- `test_reference.py`: exact rational cyclic-chain oracles and deterministic tests.
- `registry.py`: explicit, evidence-scoped selection; unknown formats have no default.
- `sources.json`: primary-source URLs, applicability, locators and retrieval limits.
- `audit.py`: synthetic checks, metadata inventory and saved-output comparison.
- `results/summary.json`: coverage, differences, metrics, environment and hashes.
- `results/synthetic_grid.csv`: 392 synthetic cases, both initial servers.
- `results/numerical_stress.json`: defined extreme cases and nontermination cases.
- `results/rule_coverage.csv`: category counts by train/validation/test split.
- `results/event_rule_inventory.csv.gz`: complete aggregate event/year/draw/round
  registry for the supplied 259,407-row modeling metadata. It contains no player
  identities, individual outcomes or row-level predictions. Read it with
  `pandas.read_csv(path, compression='gzip')`, or decompress using any gzip reader.
- `results/tests.txt`: actual automated test output.
- `manifest.json`: final code/document/output hashes (excludes itself).

The ignored `audit_private/` directory contains row-level alignment/comparison,
metadata flags and the largest changes. It is never part of the public commit.
`reference_registry_probability` in its comparison CSV is a separate scoring
counterfactual; blank means the registry did not establish a verified policy.

## Interpretation

Passing numerical checks verifies arithmetic **conditional on a specified rule**.
`verified` registry status means the published governing policy is supported by
primary evidence; it does not authenticate every scorecard or rule exception.
`inferred` and `unresolved` remain unverified. All probability comparisons use
fixed saved p/q, including format-conflict scenarios; they do not repair ML input
features, model retirement, or establish new generalization performance.
