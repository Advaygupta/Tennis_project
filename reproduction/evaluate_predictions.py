"""Recompute paired V1.1 and V2.1 test metrics from saved artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
EXPECTED = json.loads((HERE / "expected_metrics.json").read_text())


def metrics(name: str, y_true: pd.Series, y_prob: pd.Series) -> dict[str, float | str]:
    probability = pd.to_numeric(y_prob, errors="raise").to_numpy(dtype=float)
    outcome = pd.to_numeric(y_true, errors="raise").to_numpy(dtype=int)
    if not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all():
        raise ValueError(f"{name} contains invalid probabilities")
    return {
        "approach": name,
        "accuracy": accuracy_score(outcome, probability >= 0.5),
        "roc_auc": roc_auc_score(outcome, probability),
        "brier_score": brier_score_loss(outcome, probability),
        "log_loss": log_loss(outcome, probability),
    }


def main() -> None:
    v1 = joblib.load(ARTIFACTS / "v1_1_results.joblib")
    v2 = pd.read_parquet(ARTIFACTS / "v2_1_test_match_probabilities.parquet")
    metadata = pd.read_csv(ARTIFACTS / "matches_v1_1_metadata.csv")
    test_meta = metadata.loc[metadata["split"].eq("test"), ["dataset_row", "playerA_wins"]].reset_index(drop=True)

    for label, frame, required in [
        ("V1.1", v1, {"playerA_wins", "match_win_probA"}),
        ("V2.1", v2, {"dataset_row", "playerA_wins", "match_win_probA"}),
        ("metadata", test_meta, {"dataset_row", "playerA_wins"}),
    ]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")

    expected_n = EXPECTED["test_matches"]
    if not len(v1) == len(v2) == len(test_meta) == expected_n:
        raise ValueError(f"Expected {expected_n} rows, got V1={len(v1)}, V2={len(v2)}, metadata={len(test_meta)}")
    if not test_meta["dataset_row"].is_unique or not v2["dataset_row"].is_unique:
        raise ValueError("dataset_row must be unique")
    if not np.array_equal(v1["playerA_wins"].astype(int), test_meta["playerA_wins"].astype(int)):
        raise ValueError("V1.1 outcomes do not match ordered test metadata")

    v1_aligned = pd.DataFrame({
        "dataset_row": test_meta["dataset_row"],
        "playerA_wins": v1["playerA_wins"].astype(int),
        "v1_probability": v1["match_win_probA"],
    })
    paired = v1_aligned.merge(
        v2[["dataset_row", "playerA_wins", "match_win_probA"]].rename(
            columns={"playerA_wins": "v2_outcome", "match_win_probA": "v2_probability"}
        ),
        on="dataset_row",
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != expected_n or not np.array_equal(paired["playerA_wins"], paired["v2_outcome"]):
        raise ValueError("Paired prediction alignment failed")

    result = pd.DataFrame([
        metrics("V1.1 regression plus Markov", paired["playerA_wins"], paired["v1_probability"]),
        metrics("V2.1 equal classifier ensemble", paired["playerA_wins"], paired["v2_probability"]),
    ])
    result.to_csv(HERE / "paired_metrics.csv", index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print(f"Verified {expected_n:,} paired test matches.")


if __name__ == "__main__":
    main()

