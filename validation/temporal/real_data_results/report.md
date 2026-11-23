# Real-data feature reconstruction and full-history consistency

**Both scoped checks completed.** All 36,960 sampled feature values matched, and
the full-history checks found no flagged identity/event inconsistencies. These check
calculations and recorded identities, not historical public availability. No models
were fitted and no original data or predictions were changed.

The project author ran the supplied package on 23 November 2026 and returned aggregate
results and private diagnostics. Output review checked the per-feature CSV, diagnostic
counts and run code hashes against the summary. The full audit was not rerun during
this output review.

## Inputs and method

| Input | Rows | Role |
|---|---:|---|
| `matches_1991_to_2026_cleaned.csv` | 855,130 | Complete supplied cleaned A/B history, including matches without serve targets |
| `matches_1991_2026_features_added.csv` | 855,077 | Saved pre-imputation feature export |

The 53-row difference is the existing upstream missing-surface exclusion; this audit
introduced no additional row removal. The complete supplied history remains the
historical input, including its unknown-surface records.

The separately written reference implementation does not call the original notebook
feature-generation functions. It reconstructs counts, ratios and reference-date
windows by directly selecting eligible player histories, and independently replays
date-frozen Elo over preceding history. It uses the existing ten-day reference-date
policy, including its documented date-tie/window conventions.

The comparison uses a deterministic sample prioritizing reference-year, surface/level
coverage and stratum boundaries. It does not compare every saved feature row. The
event/identity scan covers every row in the supplied cleaned history.

## Audit 1: independently reconstructed historical values

| Measure | Result |
|---|---:|
| Sampled saved matches | 240 |
| Historical feature columns per match | 154 |
| Values compared / matching values | 36,960 / 36,960 |
| Mismatching values / sampled rows with mismatches | 0 / 0 |
| Maximum finite absolute difference | 4.55e-13 |
| Absolute / relative comparison tolerance | 1e-6 / 1e-7 |

The columns cover matchup/H2H and opponent-quality records; overall, surface and
blended Elo plus summaries; career/recent/surface form; career/matchup/surface serve
statistics; return and break-point conversion; workload; and tournament/level experience.
Matching missing values are explicitly treated as agreement.

The sample covers all 37 reference-date years from 1990 through 2026, four surfaces
(Carpet, Clay, Grass, Hard), and ten recorded level codes (15, 25, A, C, D, F, G, M, O, S).
These are reference-date years and recorded codes, not independently verified actual
match dates or tournament classifications.

**Interpretation:** the sampled saved values agree with a separately implemented
calculation of the existing historical policy. This supports implementation consistency
for those values. It does not establish that every saved row is correct or that every
historical input was available at prediction time.

## Audit 2: full-history event and identity consistency

| Check | Result |
|---|---:|
| Historical rows inspected | 855,130 |
| Distinct recorded event IDs inspected | 27,179 |
| Missing or duplicate match IDs | 0 |
| Invalid reference dates or nonchronological ordering | 0 |
| Missing player IDs or the same player on both sides | 0 |
| Invalid binary outcomes | 0 |
| Missing or duplicate source-file/row identities | 0 |
| Missing event IDs | 0 |
| Event IDs with multiple reference dates | 0 |
| Event IDs spanning at least ten reference days | 0 |
| Potential same-event query rows identified by the scan | 0 |
| Same-event identity collision rows | 0 |
| Cross-event alias candidate rows | 0 |

The same-event collision check examines recorded event ID, round and unordered player
pair. Cross-event candidates use shared reference date, normalized name, round and
unordered player pair with differing event IDs. The same-event contribution scan
looks for repeated player appearances within an event whose reference dates satisfy
the ten-day filter.

**Interpretation:** no candidates were detected by these defined checks. This does not
prove all possible aliases/duplicates are absent. Aliases with different dates/names
may escape the candidate rule. Consistent reference dates also do not establish actual
tournament completion times.

Private discrepancy, invalid-identity, missing-event, same-event and cross-event
diagnostic CSVs all contain zero data rows. The private event inventory contains
27,179 rows and the sample manifest 240 rows, agreeing with the public summary.

## Timestamp limitation and performance implications

The supplied files do not provide verified individual completion/publication timestamps.
Where independent timestamps or documented availability bounds cannot be established,
conservative reference-date rules are a practical fallback. Existing tournament/date
batching and the ten-calendar-day lag reduce risk but cannot guarantee that every
historical input was genuinely public before prediction.

Larger gaps can be checked for changes in eligible history and feature values without
training. Their effect on accuracy remains unverified; a performance sensitivity
experiment would generally require regenerating affected inputs and fitting models.
That experiment has not been run, given the time and resource cost. These audit results
do not constitute new model-performance estimates.

## Reproduction and evidence

The executed Python/JSON hashes match the sources in
[`../two_audit_runner/`](../two_audit_runner/). Run its notebook or command-line entry
point with the original frozen inputs. Original input SHA-256 digests/sizes, actual
environment versions, coverage and measured counts are in [`summary.json`](summary.json).
The returned run used Python 3.13.14, pandas 3.0.3 and NumPy 2.4.3.

[`feature_comparison_summary.csv`](feature_comparison_summary.csv) records all 154
per-feature counts. [`review_receipt.json`](review_receipt.json) records archive hashes,
code-hash agreement and diagnostic cross-checks. The submitted generated report is
retained as [`generated_report.md`](generated_report.md).

Only aggregate results, implementation code and hashes are published. Row-level
sample/event inventories and original tennis records are not redistributed here.
