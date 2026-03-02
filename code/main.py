# main.py

import os
import sys
import pandas as pd

# Ensure imports resolve from code/ root regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_WEIGHTS, OUTPUT_DIR
from data.loader import load_production_data, load_process_data
from data.cleaner import fill_missing_with_median, remove_outliers_iqr
from data.feature_engineering import generate_process_features
from scoring.composite_score import compute_scores
from scoring.golden_signature import generate_golden_signature


def run_pipeline() -> None:
    """Execute the full OptiBatch scoring pipeline.

    Steps
    -----
    1. Load production and process data.
    2. Clean production data (median imputation + IQR outlier removal).
    3. Engineer per-batch process features.
    4. Merge on Batch_ID and compute composite scores.
    5. Generate golden-batch signature.
    6. Persist outputs to the outputs/ directory.
    """
    try:
        # ── 1. Load ───────────────────────────────────────────────────────────
        prod    = load_production_data()
        process = load_process_data()

        # ── 2. Clean production data ──────────────────────────────────────────
        prod = fill_missing_with_median(prod)
        prod = remove_outliers_iqr(prod, exclude_cols=["Batch_ID"])

        # ── 3. Engineer features ──────────────────────────────────────────────
        process_features = generate_process_features(process)

        # ── 4. Merge + score ──────────────────────────────────────────────────
        df: pd.DataFrame = pd.merge(prod, process_features, on="Batch_ID", how="inner")
        df = compute_scores(df, DEFAULT_WEIGHTS)

        # ── 5. Golden signature ───────────────────────────────────────────────
        golden_mean, golden_std = generate_golden_signature(df)

        # ── 6. Save outputs ───────────────────────────────────────────────────
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        scored_path = os.path.join(OUTPUT_DIR, "scored_batches.csv")
        mean_path   = os.path.join(OUTPUT_DIR, "golden_signature_mean.csv")
        std_path    = os.path.join(OUTPUT_DIR, "golden_signature_std.csv")

        df.to_csv(scored_path, index=False)
        golden_mean.to_csv(mean_path)
        golden_std.to_csv(std_path)

        print(
            f"\n✅ Pipeline complete.\n"
            f"   Batches scored  : {len(df)}\n"
            f"   Outputs saved to: {OUTPUT_DIR}\n"
            f"     • scored_batches.csv\n"
            f"     • golden_signature_mean.csv\n"
            f"     • golden_signature_std.csv"
        )

    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"\n❌ Pipeline failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()