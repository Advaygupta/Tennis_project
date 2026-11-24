# Rights, provenance and redistribution inventory

Checked 24 November 2026. This inventory records evidence and limits; it does
not infer permissions from a file extension, a SHA-256 match, or local creation.
No raw records, row-level derived data, predictions, private sidecars, model
checkpoints or external source documents are added to the public package.

## Data and evidence

| Material | Source/provenance evidence | Licence or usage evidence | Publication decision and reviewer access |
|---|---|---|---|
| Raw ATP, qualifying/Challenger and Futures CSV families | First project notebook and `data/README.md` identify Jeff Sackmann / Tennis Abstract's `tennis_atp` | Historically attributed to CC BY-NC-SA 4.0. Original repository URL returned 404 during this inspection; an authenticated original licence snapshot and precise upstream commit for all supplied files are not preserved in this compact package | Exclude. Retain original notices and confirm file-specific rights and upstream snapshot before any sharing; noncommercial academic purpose alone is not blanket clearance |
| Other historical row sources | No additional row-data source established by the inspected pipeline description | No independent licence clearance demonstrated for undisclosed additions or third-party underlying rights | Exclude; author must identify any extra source and its terms before distribution |
| Cleaned history and engineered matrices | Project transforms raw records, date batches, historical features and preprocessing | Derived status does not remove upstream conditions; extraction/database/adaptation rights may remain relevant | Exclude; not required for frozen-output evaluation. Full temporal reconstruction requires additional authorized history |
| Original V1 joblib and V2 Parquet predictions | Frozen author-supplied evaluation exports; exact hashes in Point 3 and Point 4 manifests | Generated locally from upstream records; no independently established blanket redistribution permission | Private, author-controlled transfer only after confirming applicable terms; do not publish automatically |
| Match metadata | Frozen `matches_v1_1_metadata.csv`, containing outcomes and identifiers | Row-level upstream-derived content; rights not cleared by this audit | Private; authorized reviewer obtains exact frozen bytes from author, verifies hashes |
| Historical scoring sidecar | Prior Point 2 output, linked by dataset row and checked against original probabilities/rules | Contains row-level outcomes/probabilities and upstream metadata | Private; optional for primary diagnostics, required for recorded policy/completion subsets and scoring sensitivity |
| Preserved Point 3 aligned CSV | Existing private run output; hash first recorded in Point 4 | Same row-level restrictions; no independent proof of its V2 export-generation history | Private, optional route; never in ZIP or Git publication |
| Aggregate tables and figures added here | New descriptive calculations from frozen predictions; no match/player identifiers or individual probability vectors | Analytical summaries, with underlying source attribution retained. No claim that aggregation grants unrestricted rights to protected upstream material | Publish these aggregate findings for academic review; applicable upstream conditions continue to govern protected content |
| Synthetic tests | Small invented outcomes/probabilities in new test file | Authored mathematical examples, independent of real tennis records | Publish with code; no source dataset required |
| Project/audit code | Existing project code and newly authored evaluation scripts | No repository-wide software licence was found at the inspected commit. Public GitHub access is not a blanket software reuse licence | Published at author's request for reviewer inspection/reproduction; no invented MIT/BSD grant or relicensing of old code. Contact author for broader reuse terms |
| Scoring policy documents | ITF, ATP and event publications linked in Point 2 `sources.json` | Documents may carry their own copyrights and website conditions; no redistribution licence established here | Link and paraphrase prior findings only; do not bundle source PDFs or copy long passages |
| Previously saved checkpoints | Original project artifacts; not loaded by this audit | Training-derived rights are not established merely by publication elsewhere in the repository | Exclude from compact package; this audit does not retroactively clear existing repository content |

The historical Sackmann licence attribution requires credit, noncommercial use
and applicable ShareAlike conditions. Creative Commons' primary legal text also
limits the grant to rights the licensor can grant; other permissions may still
be needed. The original source URL was unavailable, so this audit does **not**
claim freshly authenticated file-specific upstream licence coverage or replace
missing provenance with a mirror's statement. No new upstream data were acquired.

Primary references (linked, not redistributed):

- [Jeff Sackmann / Tennis Abstract original source](https://github.com/JeffSackmann/tennis_atp) — unavailable during this check.
- [CC BY-NC-SA 4.0 legal text](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en) — Sections 2, 3 and 4 describe scope, conditions and database rights.
- [Point 2 official policy-source inventory](../scoring/historical/sources.json).

No contact to the upstream author or an external reviewer was made by this audit.

## Dependencies

Packages are installed independently by reviewers and are not bundled. Their own
licences/notices apply; installation does not license the tennis data. These
primary project licence references should be checked for the installed release:

| Package | Project licence family | Primary reference |
|---|---|---|
| NumPy | BSD 3-Clause | [NumPy licence](https://numpy.org/doc/stable/license.html) |
| pandas | BSD 3-Clause | [pandas licence](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| joblib | BSD 3-Clause | [joblib licence](https://github.com/joblib/joblib/blob/main/LICENSE.txt) |
| Apache Arrow / PyArrow | Apache 2.0, with dependency notices | [Arrow licence](https://github.com/apache/arrow/blob/main/LICENSE.txt) |
| Matplotlib | Project PSF-based licence and historical notices | [Matplotlib licence](https://matplotlib.org/stable/project/license.html) |
| Python | PSF licence and included notices | [Python licence](https://docs.python.org/3/license.html) |

The compact ZIP uses an explicit code/documentation/aggregate allowlist. It
does not zip the repository, vendor libraries, grant upstream permissions,
or claim that all existing repository artifacts have been legally cleared.

## Data access and hash interpretation

The author must provide an authorized reviewer the exact frozen bytes through
an approved private transfer or another route permitted by the applicable terms.
The reviewer needs neither access to the author's Google Drive nor Colab.
If sharing is not cleared, the public code and synthetic tests remain inspectable,
but full real-data reproduction is gated by authorized input access.

Hash verification demonstrates byte identity only. It cannot establish the
authenticity of historical generation, V1 row identity, correctness, public
availability at prediction time, ownership or redistribution rights. In particular,
the four best-of feature conflicts and the V1 missing-ID issue are not licensing
or hashing problems and remain unresolved by this package.
