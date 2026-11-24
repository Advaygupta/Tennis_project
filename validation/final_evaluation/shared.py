"""Reuse Point 3's verified arithmetic and alignment without running its entry point."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAIRED = HERE.parent/'paired_evaluation'
sys.path.insert(0, str(PAIRED))
from metrics import validate, metrics, losses, EPS  # noqa: E402
from alignment import align, add_scoring  # noqa: E402
from paired_bootstrap import cluster_codes, replicate_weights  # noqa: E402
from inventory import digest  # noqa: E402

IDENTITY_LIMITATION = ('Full-population pairing remains conditional on preserved V1.1 row order. '
    'V1.1 has no saved match/player IDs; positional metadata and exact metric reproduction '
    'do not independently authenticate that order.')
