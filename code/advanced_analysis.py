# code/advanced_analysis.py

import os
import pandas as pd
import numpy as np
from config import RAW_DATA_DIR, ADVANCED_ANALYSIS_DIR

def run_advanced_analysis():
    # ── 1. Load Data ──────────────────────────────────────────────────────────
    scored_path = os.path.join(RAW_DATA_DIR, "scored_batches.csv")
    mean_path = os.path.join(RAW_DATA_DIR, "golden_signature_mean.csv")
    std_path = os.path.join(RAW_DATA_DIR, "golden_signature_std.csv")

    if not all(os.path.exists(p) for p in [scored_path, mean_path, std_path]):
        print("Error: Required output files for analysis not found.")
        return

    df = pd.read_csv(scored_path)
    # Golden mean/std come in vertical format with no header or index column 0
    golden_mean = pd.read_csv(mean_path, index_col=0).iloc[:, 0].to_dict()
    golden_std = pd.read_csv(std_path, index_col=0).iloc[:, 0].to_dict()

    # ── 2. Feature Engineering: Z-Scores ──────────────────────────────────────
    z_score_df = df.copy()
    z_score_cols = []
    
    for col, mean_val in golden_mean.items():
        if col in df.columns:
            std_val = golden_std.get(col, 0)
            z_col_name = f"z_{col}"
            if std_val > 0:
                z_score_df[z_col_name] = (df[col] - mean_val) / std_val
            else:
                z_score_df[z_col_name] = 0.0
            z_score_cols.append(z_col_name)

    # ── 3. Anomaly Detection ──────────────────────────────────────────────────
    # Flag Batch as anomalous if any feature deviates by more than 2 SD
    z_abs = z_score_df[z_score_cols].abs()
    z_score_df["max_z_score"] = z_abs.max(axis=1)
    z_score_df["worst_feature"] = z_abs.idxmax(axis=1).str.replace("z_", "")
    z_score_df["is_anomalous"] = z_score_df["max_z_score"] > 2.0

    # ── 4. Scoring Refinement ──────────────────────────────────────────────────
    # Create a 'Deviation Penalty' - higher total deviation lowers the score
    # We'll normalize the mean absolute z-score across features
    mean_abs_z = z_abs.mean(axis=1)
    deviation_penalty = (mean_abs_z / mean_abs_z.max()) * 20  # Max 20 point penalty
    z_score_df["refined_score"] = (z_score_df["composite_score"] - deviation_penalty).clip(0, 100)

    # ── 5. Rank and Sort ──────────────────────────────────────────────────────
    z_score_df = z_score_df.sort_values(by="refined_score", ascending=False)
    
    # ── 6. Save Detailed Output ───────────────────────────────────────────────
    os.makedirs(ADVANCED_ANALYSIS_DIR, exist_ok=True)
    analysis_output_path = os.path.join(ADVANCED_ANALYSIS_DIR, "batch_advanced_analysis.csv")
    z_score_df.to_csv(analysis_output_path, index=False)
    
    # ── 7. Generate Summary Statistics ────────────────────────────────────────
    summary_stats = z_score_df[["refined_score", "max_z_score", "composite_score"]].describe()
    summary_path = os.path.join(ADVANCED_ANALYSIS_DIR, "analysis_summary_stats.csv")
    summary_stats.to_csv(summary_path)

    # ── 8. Heatmap Data (Top Deviating Features for Low Performers) ───────────
    # We'll take the bottom 20% batches and see which features have highest avg z-score
    bottom_20 = z_score_df.tail(max(1, int(len(z_score_df)*0.2)))
    feature_deviations = bottom_20[z_score_cols].abs().mean().sort_values(ascending=False).head(10)
    
    print("\n✅ Advanced Analysis Complete.")
    print(f"   Output saved to: {analysis_output_path}")
    print(f"   Anomalies detected: {z_score_df['is_anomalous'].sum()} out of {len(df)} batches")
    
    print("\nTop Features causing deviations in low-performing batches:")
    for feat, val in feature_deviations.items():
        print(f"   • {feat.replace('z_', ''):<25}: Avg Deviation {val:.2f} SD")

    # Save Heatmap info as text table
    heatmap_path = os.path.join(ADVANCED_ANALYSIS_DIR, "feature_deviation_table.txt")
    with open(heatmap_path, "w") as f:
        f.write("TOP DEVIATING FEATURES IN LOW-PERFORMING BATCHES\n")
        f.write("==============================================\n")
        for feat, val in feature_deviations.items():
            f.write(f"{feat.replace('z_', ''):<30} | {val:.2f} SD\n")

if __name__ == "__main__":
    run_advanced_analysis()
