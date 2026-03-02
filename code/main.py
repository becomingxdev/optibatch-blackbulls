from config import DEFAULT_WEIGHTS
from data.loader import load_production_data, load_process_data
from data.cleaner import fill_missing_with_median, remove_outliers_iqr
from data.feature_engineering import generate_process_features
from scoring.composite_score import compute_scores
from scoring.golden_signature import generate_golden_signature

import pandas as pd
import os

def run_pipeline():

    prod = load_production_data()
    process = load_process_data()

    prod = fill_missing_with_median(prod)
    prod = remove_outliers_iqr(prod, exclude_cols=["batch_id"])

    process_features = generate_process_features(process)

    df = pd.merge(prod, process_features, on="batch_id", how="inner")

    df = compute_scores(df, DEFAULT_WEIGHTS)

    golden_mean, golden_std = generate_golden_signature(df)

    os.makedirs("../outputs", exist_ok=True)

    df.to_csv("../outputs/scored_batches.csv", index=False)
    golden_mean.to_csv("../outputs/golden_signature_mean.csv")
    golden_std.to_csv("../outputs/golden_signature_std.csv")

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()