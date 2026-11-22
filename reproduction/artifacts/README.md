# Required saved artifacts

Copy these files from Google Drive into this directory:

1. `v1_1_results.joblib`
   - Drive path: `/content/drive/MyDrive/Projects/Tennis/v1_1/v1_1_results.joblib`
   - Required columns: `playerA_wins`, `match_win_probA`
2. `v2_1_test_match_probabilities.parquet`
   - Drive path: `/content/drive/MyDrive/Projects/Tennis/v2_1/predictions/v2_1_test_match_probabilities.parquet`
   - Required columns: `dataset_row`, `playerA_wins`, `match_win_probA`
3. `matches_v1_1_metadata.csv`
   - Drive path: `/content/drive/MyDrive/Projects/Tennis/v1_1/datasets/matches_v1_1_metadata.csv`
   - Required columns: `dataset_row`, `split`, `playerA_wins`

These files are ignored by Git. They contain row-level outcomes and predictions and should be published only after confirming the applicable redistribution terms.

