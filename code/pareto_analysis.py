# code/pareto_analysis.py
"""
Golden Signature Pareto Analysis
- Ranks features by cumulative deviation from the golden signature.
- Identifies the top-20% features causing ~80% of total deviation (Pareto principle).
- Saves a refined golden signature for critical features only.
- Generates a Pareto chart (bar + cumulative % line).
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR

# ── Constants ──────────────────────────────────────────────────────────────────
ANALYSIS_CSV   = os.path.join(OUTPUT_DIR, "batch_advanced_analysis.csv")
MEAN_CSV       = os.path.join(OUTPUT_DIR, "golden_signature_mean.csv")
STD_CSV        = os.path.join(OUTPUT_DIR, "golden_signature_std.csv")
RANKED_OUT     = os.path.join(OUTPUT_DIR, "pareto_feature_ranking.csv")
PARETO_SIG_OUT = os.path.join(OUTPUT_DIR, "golden_signature_pareto.csv")
CHART_OUT      = os.path.join(OUTPUT_DIR, "pareto_chart.png")

# Score/meta columns that are NOT raw features
_EXCLUDE_COLS: set[str] = {
    "Batch_ID", "is_anomalous", "worst_feature",
    "max_z_score", "refined_score", "composite_score",
    "quality_score", "yield_score", "performance_score", "energy_score",
    "Friability_inv",        # derived column, skip to avoid duplication
}

PARETO_THRESHOLD = 0.80      # capture 80% of total cumulative deviation
ALLOWABLE_SIGMA  = 1.5       # ±N × std defines the recommended operating range


# ─────────────────────────────────────────────────────────────────────────────
def _load_data() -> tuple[pd.DataFrame, dict, dict]:
    """Load analysis CSV plus golden mean / std dictionaries."""
    for path in [ANALYSIS_CSV, MEAN_CSV, STD_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    df         = pd.read_csv(ANALYSIS_CSV)
    golden_mean = pd.read_csv(MEAN_CSV,  index_col=0).iloc[:, 0].to_dict()
    golden_std  = pd.read_csv(STD_CSV,   index_col=0).iloc[:, 0].to_dict()
    return df, golden_mean, golden_std


# ─────────────────────────────────────────────────────────────────────────────
def _extract_z_cols(df: pd.DataFrame) -> list[str]:
    """Return z-score column names already present in the DataFrame."""
    return [c for c in df.columns if c.startswith("z_")]


# ─────────────────────────────────────────────────────────────────────────────
def _pareto_ranking(df: pd.DataFrame, z_cols: list[str]) -> pd.DataFrame:
    """Compute per-feature total absolute z-score and cumulative Pareto share.

    Parameters
    ----------
    df      : DataFrame containing all z-score columns.
    z_cols  : List of column names like 'z_Temperature_C_mean'.

    Returns
    -------
    ranked : DataFrame with columns:
        feature, total_abs_z, pct_contribution, cumulative_pct, is_pareto_critical
    """
    # Sum of |z| across all batches per feature
    total_abs_z: pd.Series = df[z_cols].abs().sum().rename("total_abs_z")
    total_abs_z.index = total_abs_z.index.str.replace("^z_", "", regex=True)

    # Sort descending
    ranked = (
        total_abs_z
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    grand_total = ranked["total_abs_z"].sum()
    ranked["pct_contribution"]  = ranked["total_abs_z"] / grand_total * 100
    ranked["cumulative_pct"]    = ranked["pct_contribution"].cumsum()
    ranked["is_pareto_critical"] = ranked["cumulative_pct"] <= (PARETO_THRESHOLD * 100 + ranked["pct_contribution"])

    # Precise Pareto cut: first row that tips cumulative_pct over threshold
    cut_idx = (ranked["cumulative_pct"] > PARETO_THRESHOLD * 100).idxmax()
    ranked["is_pareto_critical"] = ranked.index <= cut_idx

    return ranked


# ─────────────────────────────────────────────────────────────────────────────
def _build_refined_signature(
    df:          pd.DataFrame,
    ranked:      pd.DataFrame,
    golden_mean: dict,
    golden_std:  dict,
) -> pd.DataFrame:
    """Build a refined golden signature table for Pareto-critical features only.

    For each critical feature the table includes:
    - golden_mean, golden_std
    - actual_mean / actual_std across all 56 batches
    - recommended_min  = golden_mean - ALLOWABLE_SIGMA × golden_std
    - recommended_max  = golden_mean + ALLOWABLE_SIGMA × golden_std
    - total_abs_z  (deviation impact score from the Pareto ranking)
    - pct_contribution
    """
    critical_features = ranked.loc[ranked["is_pareto_critical"], "feature"].tolist()

    rows = []
    for feat in critical_features:
        if feat not in golden_mean:
            continue

        g_mean = golden_mean[feat]
        g_std  = golden_std.get(feat, 0)
        a_mean = df[feat].mean() if feat in df.columns else np.nan
        a_std  = df[feat].std()  if feat in df.columns else np.nan
        total_z = ranked.loc[ranked["feature"] == feat, "total_abs_z"].values[0]
        pct     = ranked.loc[ranked["feature"] == feat, "pct_contribution"].values[0]

        rows.append({
            "feature":           feat,
            "golden_mean":       round(g_mean, 4),
            "golden_std":        round(g_std, 4),
            "actual_mean":       round(a_mean, 4),
            "actual_std":        round(a_std, 4),
            "recommended_min":   round(g_mean - ALLOWABLE_SIGMA * g_std, 4),
            "recommended_max":   round(g_mean + ALLOWABLE_SIGMA * g_std, 4),
            "total_abs_z":       round(total_z, 2),
            "pct_contribution":  round(pct, 2),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def _draw_pareto_chart(ranked: pd.DataFrame) -> None:
    """Generate and save a Pareto bar + cumulative-line chart."""
    top_n = min(20, len(ranked))          # show at most 20 bars for readability
    data  = ranked.head(top_n).copy()

    labels   = data["feature"].str.replace("_", "\n", n=1)   # wrap long names
    bar_vals = data["pct_contribution"].values
    cum_vals = data["cumulative_pct"].values

    fig, ax1 = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#0f0f1a")
    ax1.set_facecolor("#0f0f1a")

    # ── Bars ──────────────────────────────────────────────────────────────────
    bar_colors = [
        "#f97316" if ranked["is_pareto_critical"].values[i] else "#334155"
        for i in range(top_n)
    ]
    bars = ax1.bar(
        range(top_n), bar_vals,
        color=bar_colors, edgecolor="#1e293b", linewidth=0.5, zorder=3
    )

    # ── Cumulative line ────────────────────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.set_facecolor("#0f0f1a")
    ax2.plot(range(top_n), cum_vals, color="#38bdf8", linewidth=2.2,
             marker="o", markersize=5, zorder=4, label="Cumulative %")
    ax2.axhline(80, color="#f43f5e", linewidth=1.2, linestyle="--",
                label="80% threshold", zorder=5)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.set_ylim(0, 110)
    ax2.tick_params(axis="y", colors="#94a3b8", labelsize=9)

    # ── Styling ────────────────────────────────────────────────────────────────
    ax1.set_xticks(range(top_n))
    ax1.set_xticklabels(labels, rotation=40, ha="right",
                        fontsize=7.5, color="#cbd5e1")
    ax1.set_ylabel("% Contribution to Total Deviation", color="#94a3b8", fontsize=10)
    ax1.set_xlabel("Feature", color="#94a3b8", fontsize=10)
    ax1.tick_params(axis="y", colors="#94a3b8", labelsize=9)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax1.grid(axis="y", color="#1e293b", linewidth=0.6, zorder=0)
    ax1.set_ylim(0, max(bar_vals) * 1.25)

    # Legend patches
    import matplotlib.patches as mpatches
    critical_patch   = mpatches.Patch(color="#f97316", label="Pareto Critical (top 80%)")
    secondary_patch  = mpatches.Patch(color="#334155", label="Secondary features")
    lines2, labels2  = ax2.get_legend_handles_labels()
    ax1.legend(
        handles=[critical_patch, secondary_patch] + lines2,
        labels=["Pareto Critical (top 80%)", "Secondary features"] + labels2,
        loc="upper right", facecolor="#1e293b", edgecolor="#334155",
        labelcolor="#e2e8f0", fontsize=9
    )

    plt.title(
        "Pareto Analysis — Feature Contribution to Batch Deviations",
        color="#f1f5f9", fontsize=14, fontweight="bold", pad=14
    )
    plt.tight_layout()
    fig.savefig(CHART_OUT, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def run_pareto_analysis() -> None:
    """Entry point: run full Pareto golden-signature analysis and save outputs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load
    df, golden_mean, golden_std = _load_data()

    # 2. Identify z-score columns
    z_cols = _extract_z_cols(df)
    if not z_cols:
        raise ValueError(
            "No z-score columns found in batch_advanced_analysis.csv. "
            "Re-run advanced_analysis.py first."
        )

    # 3. Pareto ranking
    ranked = _pareto_ranking(df, z_cols)

    n_critical     = ranked["is_pareto_critical"].sum()
    n_total        = len(ranked)
    pct_features   = n_critical / n_total * 100
    captured_pct   = ranked.loc[ranked["is_pareto_critical"], "cumulative_pct"].max()

    # 4. Refined golden signature
    refined_sig = _build_refined_signature(df, ranked, golden_mean, golden_std)

    # 5. Save CSVs
    ranked.to_csv(RANKED_OUT, index=False)
    refined_sig.to_csv(PARETO_SIG_OUT, index=False)

    # 6. Draw chart
    _draw_pareto_chart(ranked)

    # ── Terminal Summary ───────────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print("  PARETO ANALYSIS — GOLDEN SIGNATURE DEVIATION REPORT")
    print("═" * 58)
    print(f"  Total features analysed : {n_total}")
    print(f"  Pareto-critical features: {n_critical} ({pct_features:.0f}% of features)")
    print(f"  Cumulative deviation captured: {captured_pct:.1f}%")
    print("─" * 58)
    print(f"  {'Rank':<5} {'Feature':<32} {'Impact':>8}  {'Cum%':>7}  {'Critical'}")
    print("─" * 58)
    for _, row in ranked.iterrows():
        flag = "  ◄ CRITICAL" if row["is_pareto_critical"] else ""
        print(
            f"  {int(_)+1:<5} {row['feature']:<32} "
            f"{row['total_abs_z']:>8.1f}  {row['cumulative_pct']:>6.1f}%{flag}"
        )
    print("═" * 58)
    print(f"\n  Outputs saved to: {OUTPUT_DIR}")
    print(f"    • pareto_feature_ranking.csv")
    print(f"    • golden_signature_pareto.csv")
    print(f"    • pareto_chart.png")
    print()


if __name__ == "__main__":
    run_pareto_analysis()
