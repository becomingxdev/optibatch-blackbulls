# code/batch_monitor.py
"""
Real-Time Batch Monitoring & Alert Generation
─────────────────────────────────────────────
• Compares every batch in scored_batches.csv against the Pareto golden ranges.
• Flags anomalous batches and lists which critical features caused the deviation.
• Computes a Pareto-weighted Deviation Score per batch.
• Saves a detailed alert CSV and a heatmap PNG.
"""

import os
import sys
import warnings
import textwrap
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches     import Patch
from matplotlib.patheffects import withStroke

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DATA_DIR, PARETO_DIR, MONITORING_DIR

# ── File paths ────────────────────────────────────────────────────────────────
SCORED_CSV      = os.path.join(RAW_DATA_DIR, "scored_batches.csv")
PARETO_SIG_CSV  = os.path.join(PARETO_DIR, "golden_signature_pareto.csv")
ALERT_OUT       = os.path.join(MONITORING_DIR, "batch_monitoring_alerts.csv")
HEATMAP_OUT     = os.path.join(MONITORING_DIR, "batch_deviation_heatmap.png")
REPORT_OUT      = os.path.join(MONITORING_DIR, "monitoring_report.txt")

# ── Constants ──────────────────────────────────────────────────────────────────
ANOMALY_THRESHOLD = 1          # any feature out-of-range → anomaly
SEVERITY_BINS     = [0, 1, 3, 6, float("inf")]
SEVERITY_LABELS   = ["OK", "LOW", "MEDIUM", "HIGH"]


# ─────────────────────────────────────────────────────────────────────────────
def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    for p in [SCORED_CSV, PARETO_SIG_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file missing: {p}")
    df     = pd.read_csv(SCORED_CSV)
    pareto = pd.read_csv(PARETO_SIG_CSV)
    return df, pareto


# ─────────────────────────────────────────────────────────────────────────────
def _evaluate_batches(
    df: pd.DataFrame,
    pareto: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    For every Pareto-critical feature, determine if each batch's value falls
    inside the [recommended_min, recommended_max] range.

    Returns
    -------
    results      : Main alert DataFrame (one row per batch).
    deviation_mat: Matrix of per-feature z-scores for heatmap (batchs × features).
    crit_features: List of critical feature names that exist in the scored CSV.
    """
    # Features that actually exist in scored data
    crit_features: list[str] = [
        f for f in pareto["feature"].tolist() if f in df.columns
    ]

    # Build lookup dicts
    f_min   = pareto.set_index("feature")["recommended_min"].to_dict()
    f_max   = pareto.set_index("feature")["recommended_max"].to_dict()
    f_mean  = pareto.set_index("feature")["golden_mean"].to_dict()
    f_std   = pareto.set_index("feature")["golden_std"].to_dict()
    f_pct   = pareto.set_index("feature")["pct_contribution"].to_dict()

    # Per-feature out-of-range boolean matrix  (n_batches × n_features)
    oor_mat = pd.DataFrame(index=df.index, columns=crit_features, dtype=bool)
    # Signed z-score matrix for heatmap
    z_mat   = pd.DataFrame(index=df.index, columns=crit_features, dtype=float)

    for feat in crit_features:
        vals = df[feat]
        oor_mat[feat] = (vals < f_min[feat]) | (vals > f_max[feat])
        std_val = f_std[feat] if f_std[feat] > 0 else 1e-9
        z_mat[feat]   = (vals - f_mean[feat]) / std_val

    # ── Aggregate per batch ───────────────────────────────────────────────────
    results = pd.DataFrame()
    results["Batch_ID"]      = df["Batch_ID"]
    results["composite_score"] = df["composite_score"].round(2)

    # n_features out of range
    results["features_oor"]  = oor_mat.sum(axis=1).astype(int)

    # Weighted deviation score  = Σ |z| × pct_contribution  for OOR features
    def _weighted_dev(row_idx: int) -> float:
        total = 0.0
        for feat in crit_features:
            if oor_mat.loc[row_idx, feat]:
                total += abs(z_mat.loc[row_idx, feat]) * f_pct.get(feat, 1)
        return round(total, 3)

    results["deviation_score"] = [_weighted_dev(i) for i in df.index]

    # Anomaly flag
    results["is_anomalous"]   = results["features_oor"] >= ANOMALY_THRESHOLD

    # Severity level based on number of features OOR
    results["severity"] = pd.cut(
        results["features_oor"],
        bins=SEVERITY_BINS,
        labels=SEVERITY_LABELS,
        right=False,
    ).astype(str)

    # Which features are OOR — comma-separated list
    def _oor_list(row_idx: int) -> str:
        oor = [f for f in crit_features if oor_mat.loc[row_idx, f]]
        return "; ".join(oor) if oor else "None"

    results["critical_features_oor"] = [_oor_list(i) for i in df.index]

    # Direction of each OOR feature: HIGH / LOW
    def _direction(row_idx: int) -> str:
        parts = []
        for feat in crit_features:
            if oor_mat.loc[row_idx, feat]:
                val = df.loc[row_idx, feat]
                direction = "HIGH" if val > f_max[feat] else "LOW"
                parts.append(f"{feat}:{direction}")
        return "; ".join(parts) if parts else "None"

    results["feature_directions"] = [_direction(i) for i in df.index]

    # Sort by deviation_score descending
    results = results.sort_values("deviation_score", ascending=False).reset_index(drop=True)
    results.insert(0, "rank", results.index + 1)

    return results, z_mat, crit_features


# ─────────────────────────────────────────────────────────────────────────────
def _draw_heatmap(
    df:           pd.DataFrame,
    z_mat:        pd.DataFrame,
    results:      pd.DataFrame,
    crit_features: list[str],
) -> None:
    """Render a styled deviation heatmap (batches × Pareto features)."""

    # Align z_mat rows with sorted results
    batch_order = results["Batch_ID"].tolist()
    df_idx      = df.set_index("Batch_ID")
    z_plot      = (
        z_mat
        .assign(Batch_ID=df["Batch_ID"].values)
        .set_index("Batch_ID")
        .loc[batch_order]
        [crit_features]
    )

    # Shorten feature names for axis labels
    def _shorten(name: str, max_len: int = 22) -> str:
        return name if len(name) <= max_len else name[:max_len - 1] + "…"

    feat_labels = [_shorten(f) for f in crit_features]

    n_batches  = len(z_plot)
    n_features = len(crit_features)

    fig_w = max(16, n_features * 0.72)
    fig_h = max(10, n_batches  * 0.38)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0a0a14")
    ax.set_facecolor("#0a0a14")

    # Custom diverging colormap (green → black → red)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "traffic",
        ["#22c55e", "#16a34a", "#0f172a", "#0f172a", "#dc2626", "#ef4444"],
        N=512,
    )

    z_data = z_plot.values.astype(float)
    vmax   = max(abs(np.nanmax(z_data)), abs(np.nanmin(z_data)), 1.5)

    im = ax.imshow(z_data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Z-Score vs Golden Signature", color="#94a3b8", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#64748b")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#94a3b8", fontsize=8)

    # ── Axes labels and ticks ─────────────────────────────────────────────────
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feat_labels, rotation=45, ha="right",
                       fontsize=7.5, color="#cbd5e1")
    ax.set_yticks(range(n_batches))

    # Batch labels coloured by severity
    severity_col = results.set_index("Batch_ID")["severity"]
    sev_palette  = {"OK": "#4ade80", "LOW": "#facc15",
                    "MEDIUM": "#fb923c", "HIGH": "#f43f5e"}

    y_labels = []
    y_colors = []
    for bid in batch_order:
        sev = severity_col.get(bid, "OK")
        y_labels.append(bid)
        y_colors.append(sev_palette.get(sev, "#94a3b8"))

    ax.set_yticklabels(y_labels, fontsize=7.5)
    for label, color in zip(ax.get_yticklabels(), y_colors):
        label.set_color(color)

    # Grid lines between cells
    ax.set_xticks(np.arange(-0.5, n_features, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_batches,  1), minor=True)
    ax.grid(which="minor", color="#1e293b", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor="#4ade80", label="OK  (0 features OOR)"),
        Patch(facecolor="#facc15", label="LOW  (1–2 features OOR)"),
        Patch(facecolor="#fb923c", label="MEDIUM  (3–5 features OOR)"),
        Patch(facecolor="#f43f5e", label="HIGH  (6+ features OOR)"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper right",
        facecolor="#1e293b", edgecolor="#334155",
        labelcolor="#e2e8f0", fontsize=8,
        bbox_to_anchor=(1.0, -0.09), ncol=4,
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax.set_title(
        f"Batch Deviation Heatmap — Pareto-Critical Features  [{ts}]",
        color="#f1f5f9", fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Pareto-Critical Features", color="#94a3b8", fontsize=10, labelpad=8)
    ax.set_ylabel("Batch ID  (sorted by deviation severity)", color="#94a3b8",
                  fontsize=10, labelpad=8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(HEATMAP_OUT, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def _write_report(results: pd.DataFrame, crit_features: list[str]) -> None:
    """Write a human-readable monitoring report."""
    total      = len(results)
    anomalous  = results["is_anomalous"].sum()
    ok         = total - anomalous
    sev_counts = results["severity"].value_counts()

    high_risk = results[results["severity"] == "HIGH"]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "═" * 64

    lines = [
        sep,
        "  REAL-TIME BATCH MONITORING REPORT",
        f"  Generated: {ts}",
        sep,
        f"  Total batches evaluated : {total}",
        f"  Anomalous batches        : {anomalous}  ({anomalous/total*100:.0f}%)",
        f"  Fully compliant batches  : {ok}",
        "",
        "  Severity Breakdown:",
        f"    OK      : {sev_counts.get('OK',0)}",
        f"    LOW     : {sev_counts.get('LOW',0)}",
        f"    MEDIUM  : {sev_counts.get('MEDIUM',0)}",
        f"    HIGH    : {sev_counts.get('HIGH',0)}",
        sep,
        "  TOP 10 HIGHEST-DEVIATION BATCHES",
        "─" * 64,
        f"  {'Rank':<5} {'Batch':<8} {'Dev.Score':>10} {'Features OOR':>13} {'Severity':<10}",
        "─" * 64,
    ]

    for _, row in results.head(10).iterrows():
        lines.append(
            f"  {int(row['rank']):<5} {row['Batch_ID']:<8} "
            f"{row['deviation_score']:>10.2f} {row['features_oor']:>13}   {row['severity']}"
        )

    if not high_risk.empty:
        lines += [
            sep,
            "  HIGH-RISK BATCH DETAILS",
            "─" * 64,
        ]
        for _, row in high_risk.iterrows():
            lines.append(f"\n  ► Batch {row['Batch_ID']}  |  Score {row['composite_score']}  |  {row['features_oor']} features OOR")
            # Word-wrap long OOR list
            oor_wrapped = textwrap.fill(
                f"    OOR: {row['critical_features_oor']}",
                width=62, subsequent_indent="         "
            )
            lines.append(oor_wrapped)
            lines.append(f"    Direction: {row['feature_directions'][:80]}…")

    lines += [
        sep,
        f"  Outputs saved to: {MONITORING_DIR}",
        f"    • batch_monitoring_alerts.csv",
        f"    • batch_deviation_heatmap.png",
        f"    • monitoring_report.txt",
        sep,
    ]

    report_text = "\n".join(lines)
    with open(REPORT_OUT, "w") as f:
        f.write(report_text)

    print(report_text)


# ─────────────────────────────────────────────────────────────────────────────
def run_batch_monitor() -> None:
    """Main entry point."""
    os.makedirs(MONITORING_DIR, exist_ok=True)

    # 1. Load
    df, pareto = _load_inputs()

    # 2. Evaluate
    results, z_mat, crit_features = _evaluate_batches(df, pareto)

    # 3. Save alert CSV
    results.to_csv(ALERT_OUT, index=False)

    # 4. Draw heatmap
    _draw_heatmap(df, z_mat, results, crit_features)

    # 5. Write report (also prints to terminal)
    _write_report(results, crit_features)


if __name__ == "__main__":
    run_batch_monitor()
