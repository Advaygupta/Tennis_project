# Point 3: paired evaluation — reviewer summary

**23 November 2026.** The existing comparison is V2.1's equal-weight direct
classifier ensemble (A) versus V1.1's validation-weighted serve-regression
ensemble plus Markov scoring (B). All four metrics reproduce on the frozen
51,666-record test population within **3.33 × 10⁻¹⁵**. No training, inference,
notebook execution, or rerun of an existing script was performed.

| Metric | V2.1 (A) | V1.1 (B) | A minus B | Event-bootstrap 95% CI |
|---|---:|---:|---:|---|
| Accuracy | 0.69047343 | 0.68904115 | +0.00143228 | [−0.00021374, +0.00310760] |
| ROC-AUC | 0.76048069 | 0.75988840 | +0.00059229 | [−0.00004480, +0.00123070] |
| Brier score | 0.19887455 | 0.20670643 | −0.00783189 | [−0.00852311, −0.00713335] |
| Log loss | 0.58104872 | 0.63115991 | −0.05011119 | [−0.05421161, −0.04600166] |

Each design uses **5,000 paired bootstrap replicates**. Event resampling uses
1,407 editions; the ordinary match reference uses 51,666 units, and a separate
calendar sensitivity uses 25 nonempty 28-day blocks. Resampling units are assumed
independent within each design; repeated players and longer dependence remain.

Accuracy intervals include zero under every full-population design. AUC is
borderline positive under iid-match resampling but includes zero under event and
calendar resampling. Thus the reported small accuracy/AUC differences do not
establish a material predictive advantage. Exact McNemar p = 0.0898022 is supplied
only as an independent-match reference. Nonsignificance is not equivalence.

Brier/log-loss intervals favour V2.1 under all three designs and on the completed,
supported-policy, and combined sensitivity subsets. These are conditional frozen
probability-score differences, not a demonstrated practical utility gain. The
log-loss benefit reflects reduced extreme adverse losses: the largest 1% of
differences contribute about 55% of the net mean improvement.

**Unresolved identity qualification:** V2.1's row IDs, match IDs, oriented players
and outcomes match metadata. V1.1 lacks match/player IDs. All saved context agrees
in order, but only 35 records have unique tournament/date/best-of/outcome context;
51,631 remain ambiguous. Full-population paired inference is conditional on the
preserved V1.1 order. The tiny, unrepresentative singleton subset cannot
authenticate the full comparison. A metric rerun or newly attached positional
IDs cannot resolve this; preserved ID-bearing output or authenticated lineage
would be needed.

The original population is retained. Separate analyses exclude 1,453 incomplete
matches, restrict to 34,367 supported-policy records, or combine both restrictions
(33,294). The 48 Next Gen/Laver scoring corrections are a separate frozen-p/q
sensitivity, with no accuracy change. Four best-of feature conflicts are left
unchanged. Policy support reuses the earlier audit and is not authentication of
every individual-event exception. Subsets and ancillary comparisons are exploratory;
intervals are pointwise, with no multiplicity-adjusted superiority claim.

**26 tests pass.** Original inputs are byte-preserved. Public outputs contain
aggregates, code, tests and hashes; row-level artifacts remain private. See the
[full report](report.md), [run instructions](README.md), and
[input/code manifest](results/manifest.json). **No author-side rerun is required.**
