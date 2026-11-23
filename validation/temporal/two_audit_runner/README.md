# Run these two audits

Keep all files together after extracting the ZIP. Only NumPy and pandas are required.
No model is loaded or trained; no GPU is used; the original CSVs are read-only.

## Easiest: VS Code/Jupyter

Open Run_Two_Audits.ipynb, select your normal Python kernel and run its cells.
DATA_ROOT is prefilled for the local tennis data directory already located on this
computer. BUNDLE_DIR must point to this extracted folder if your notebook's working
directory is elsewhere. Install requirements only if they are missing.

## Command line

From this extracted folder:

    python -m pip install -r requirements.txt
    python run_two_audits.py --data-root "D:\Advay\Additional Learning\Machine Learning\projects\tennis\data"

You can specify --history PATH and --saved-features PATH instead. Required originals:

- created_data/matches_1991_to_2026_cleaned.csv (complete cleaned history, before target filtering)
- created_data/matches_1991_2026_features_added.csv (pre-imputation feature export)

The defaults compare 154 historical columns for up to 240 representative saved rows,
with date-frozen Elo replayed over the complete preceding history. Sampling covers
years, surfaces and levels where feasible; actual coverage is reported. This is not
an exhaustive value comparison across every row. The event/identity scan covers all
rows of the supplied cleaned history. The script reports issues instead of silently
repairing or dropping source rows. It never downloads a fresh upstream dataset.

The saved 1.4 GB CSV is read in chunks to avoid loading the entire feature matrix.
The complete cleaned history is loaded, so this uses CPU and RAM, although it does
not fit models. Progress is printed throughout the Elo replay; leave the cell running.
Runtime depends on the machine and history size and is not promised in advance.

## Return these outputs

1. audit_outputs/audit_results.zip: aggregate report, machine-readable summary and
   per-feature comparison counts. Return this even if the audit reports an error.
2. audit_outputs/private_diagnostics.zip: row-level mismatch/candidate diagnostics
   and logs. Return this too if investigation is needed; keep it off public GitHub.

If inputs are invalid, inspect/share the generated outputs rather than modifying the
source CSVs to force a pass. For reruns use a different --output directory; existing
results are preserved. The final report is deliberately restricted to these two
checks and does not claim a complete review of all modeling procedures.

Agreement confirms the implemented ten-day reference-date policy on the sampled
features, not verified public availability. Exact match dates/availability evidence
are still a separate limitation. No date is fabricated by this runner.
