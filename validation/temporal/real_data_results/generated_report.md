# Two focused feature-construction audits

Scope: independent reconstruction of saved historical feature values, and consistency
checks across the complete supplied history. This report does not claim to be a
complete review of every modeling procedure. No model was trained or modified.

## Independent feature reconstruction

Status: completed.
Historical input rows: 855130.
Sampled saved rows compared: 240.
Historical feature columns: 154.
Compared values: 36960.
Mismatching values: 0.
Rows with a mismatch: 0.

The reference implementation uses separately written calculations instead of calling
the original feature-generation functions. Elo is replayed over the full preceding
history; other values use direct filtering of each sampled player's history.
This is a stratified sample audit, not an exhaustive comparison of every feature row.
Mismatches require investigation; they are not automatically evidence of leakage.

## Full-history event and identity consistency

Recorded event IDs: 27179.
Event IDs with multiple reference dates: 0.
Event IDs spanning at least ten reference days: 0.
Potential same-event query rows: 0.
Same-event identity collision rows: 0.
Cross-event alias candidate rows: 0.
Missing event-ID rows: 0.
Original rows dropped: 0.

Candidates are reported for investigation. Event aliases cannot be certified solely
from similar names, and consistent recorded IDs cannot rule out all unrecognized aliases.
Private diagnostic CSVs retain row identities; the public output contains aggregates only.

## Timestamp limitation

These checks use recorded tournament reference dates, not verified individual match
completion/publication timestamps. Without those timestamps or supported availability
bounds, conservative rules are a practical fallback. The existing ten-calendar-day
eligibility lag and date batching reduce risk but do not guarantee that leakage is absent.
Larger gaps can be audited for their effect on eligible history without training;
their effect on model accuracy remains unmeasured and would generally require new
features and model fits. No such fitting is performed by this runner.

Input and code hashes, environment versions, coverage and any run error are recorded
in summary.json. No conclusions should be drawn from a failed or incomplete stage.
