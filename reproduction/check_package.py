"""Verify committed notebooks and their recorded correction outputs."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "01_deduplication_feature_engineering.ipynb",
    "02_v1_1_preprocessing.ipynb",
    "03_v1_1_model_training.ipynb",
    "04_v1_1_model_analysis.ipynb",
    "05_v2_1_model_training.ipynb",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def notebook_text(notebook: dict) -> str:
    parts: list[str] = []
    for cell in notebook.get("cells", []):
        parts.extend(cell.get("source", []))
        for output in cell.get("outputs", []):
            parts.extend(output.get("text", []))
            plain = output.get("data", {}).get("text/plain", [])
            parts.extend(plain if isinstance(plain, list) else [plain])
    return "\n".join(parts)


def require(pattern: str, text: str, label: str) -> None:
    if not re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
        raise AssertionError(f"Missing expected evidence: {label}")


def main() -> int:
    manifest = json.loads((Path(__file__).with_name("notebook_manifest.json")).read_text())
    combined: dict[str, str] = {}
    for name in NOTEBOOKS:
        path = ROOT / "notebooks" / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4:
            raise AssertionError(f"Unexpected notebook format: {name}")
        actual = sha256(path)
        expected = manifest["notebooks"][name]["sha256"]
        if actual != expected:
            raise AssertionError(f"Hash mismatch for {name}: {actual}")
        if not any(cell.get("outputs") for cell in notebook.get("cells", [])):
            raise AssertionError(f"No executed outputs found in {name}")
        combined[name] = notebook_text(notebook)
        print(f"OK {name} {actual[:12]}")

    require(r"Original records:\s*855194", combined[NOTEBOOKS[0]], "source row count")
    require(r"Records remaining:\s*855130", combined[NOTEBOOKS[0]], "deduplicated row count")
    require(r"Valid serve probability targets:\s*259,407", combined[NOTEBOOKS[0]], "eligible sample")
    require(r"Train:\s*181,632", combined[NOTEBOOKS[1]], "train split")
    require(r"Validation:\s*26,109", combined[NOTEBOOKS[1]], "validation split")
    require(r"Test:\s*51,666", combined[NOTEBOOKS[1]], "test split")
    require(r"Accuracy:\s*0\.6890411489180506", combined[NOTEBOOKS[2]], "V1.1 accuracy")
    require(r"ROC AUC Score:\s*0\.7598884003716795", combined[NOTEBOOKS[2]], "V1.1 ROC-AUC")
    require(r"Test matches:\s*51,666", combined[NOTEBOOKS[4]], "V2.1 test count")
    require(r"0\.69047\s+0\.76048\s+0\.19887\s+0\.58105", combined[NOTEBOOKS[4]], "V2.1 metrics")
    print("All package checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)

