# code/ops_dashboard.py
"""
Operational Predictive Dashboard
──────────────────────────────────────────────────────────────────────────────
Panel 1 — Live Deviation Score History + ML next-batch overlay
Panel 2 — Severity timeline with ML predicted severity flag
Panel 3 — Top Δ/σ feature contributions (from feature importance)
Panel 4 — Batch-level predicted vs actual KPI
Panel 5 — Next-batch risk card (score, severity, top 5 explanations)
Panel 6 — Corrective actions table for HIGH-risk upcoming batches

Outputs:
  outputs/monitoring/ops_dashboard.png
  outputs/monitoring/ops_alerts_summary.csv
"""

import os, sys, warnings, textwrap
from datetime import datetime
from collections import Counter

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import matplotlib.patches   as mpatches
import matplotlib.ticker    as mticker

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MONITORING_DIR, ML_MODELS_DIR, RAW_DATA_DIR, PARETO_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
ALERTS_CSV  = os.path.join(MONITORING_DIR, "batch_monitoring_alerts.csv")
PREDS_CSV   = os.path.join(ML_MODELS_DIR,  "ml_predictions.csv")
IMP_CSV     = os.path.join(ML_MODELS_DIR,  "ml_feature_importance.csv")
SCORED_CSV  = os.path.join(RAW_DATA_DIR,   "scored_batches.csv")
PARETO_CSV  = os.path.join(PARETO_DIR,     "golden_signature_pareto.csv")
DASH_OUT    = os.path.join(MONITORING_DIR, "ops_dashboard.png")
ALERT_SUM   = os.path.join(MONITORING_DIR, "ops_alerts_summary.csv")

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#07090f"; PANEL = "#0d1117"; GRID = "#1a2035"
TEXT     = "#e2e8f0"; MUTED = "#64748b"
ACCENT   = "#38bdf8"; ORANGE = "#f97316"; GREEN = "#22c55e"
RED      = "#ef4444"; PURPLE = "#a78bfa"; YELLOW = "#facc15"
SEV_CLR  = {"OK": GREEN, "LOW": YELLOW, "MEDIUM": ORANGE, "HIGH": RED}
ROLLING  = 5

# ── Corrective Action Map ─────────────────────────────────────────────────────
ACTIONS = {
    "Compression_Force_kN_mean_delta": {"HIGH": "Reduce compression force setpoint by ~5%",
                                         "LOW":  "Increase compression force setpoint by ~5%"},
    "Disintegration_Time_delta":        {"HIGH": "Increase superdisintegrant level; review granule porosity",
                                         "LOW":  "Reduce disintegrant level; check coating integrity"},
    "Humidity_Percent_min_delta":       {"HIGH": "Lower HVAC humidity setpoint; increase desiccant capacity",
                                         "LOW":  "Increase HVAC humidity setpoint"},
    "Humidity_Percent_max_roll_std":    {"HIGH": "Stabilise HVAC cycling; check thermostat hunting",
                                         "LOW":  "Humidity variation acceptable"},
    "Tablet_Weight_delta":              {"HIGH": "Adjust fill depth cam on tablet press",
                                         "LOW":  "Increase fill cam depth; check hopper feed rate"},
    "Friability_delta":                 {"HIGH": "Increase compression force; review binder concentration",
                                         "LOW":  "Reduce compression force to prevent over-hardening"},
    "Pressure_Bar_mean_delta":          {"HIGH": "Reduce line pressure — check upstream regulator valve",
                                         "LOW":  "Increase line pressure — inspect pump"},
    "Flow_Rate_LPM_std_delta":          {"HIGH": "Inspect spray nozzle for blockage or wear",
                                         "LOW":  "Flow consistency improved — monitor"},
    "n_oor_roll_mean":                  {"HIGH": "Multiple parameters trending OOR — escalate to engineer",
                                         "LOW":  "Rolling OOR count is reducing — continue monitoring"},
    "Moisture_Content_roll_mean":       {"HIGH": "Extend drying time; check oven humidity exhaust",
                                         "LOW":  "Reduce drying time to preserve granule integrity"},
}
DEFAULT_ACTION = "Review parameter trend with process engineer and compare against SOP"


def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.6, zorder=0)
    if title:  ax.set_title(title, color=TEXT, fontsize=9.5, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=8)


def _batch_num(bid: str) -> int:
    s = "".join(c for c in str(bid) if c.isdigit())
    return int(s) if s else 0


def _action_for(feat: str, direction: str) -> str:
    return ACTIONS.get(feat, {}).get(direction.upper(), DEFAULT_ACTION)


# ── 1. Load Data ──────────────────────────────────────────────────────────────
def load_data():
    for p in [ALERTS_CSV, PREDS_CSV, IMP_CSV, SCORED_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    alerts  = pd.read_csv(ALERTS_CSV)
    preds   = pd.read_csv(PREDS_CSV)
    imp     = pd.read_csv(IMP_CSV)
    scored  = pd.read_csv(SCORED_CSV)
    pareto  = pd.read_csv(PARETO_CSV) if os.path.exists(PARETO_CSV) else pd.DataFrame()

    # Chronological sort
    for df in [alerts, preds, scored]:
        df["_n"] = df["Batch_ID"].apply(_batch_num)
    alerts = alerts.sort_values("_n").reset_index(drop=True)
    preds  = preds[preds["Batch_ID"] != "NEXT (forecast)"].copy()
    preds  = preds.sort_values("_n").reset_index(drop=True)
    scored = scored.sort_values("_n").reset_index(drop=True)

    # Separate next-batch forecast row
    next_df = pd.read_csv(PREDS_CSV)
    next_df = next_df[next_df["Batch_ID"] == "NEXT (forecast)"].reset_index(drop=True)

    return alerts, preds, imp, scored, pareto, next_df


# ── 2. Build Alert Summary ─────────────────────────────────────────────────────
def build_alert_summary(alerts, preds, imp):
    # Top 5 Δ features by avg importance
    delta_feats = imp[imp["feature"].str.endswith("_delta")].head(5)

    alerts_merged = alerts.merge(
        preds[["Batch_ID", "pred_dev_score", "pred_severity"]],
        on="Batch_ID", how="left"
    )

    rows = []
    for _, row in alerts_merged.iterrows():
        top_actions = []
        oor_feats = [f.strip() for f in str(row.get("critical_features_oor", "")).split(";")
                     if f.strip() and f.strip().lower() != "none"]
        dir_map = {}
        for part in str(row.get("feature_directions","")).split(";"):
            part = part.strip()
            if ":" in part:
                f, d = part.split(":", 1)
                dir_map[f.strip()] = d.strip()

        for feat, _ in delta_feats[["feature", "avg_importance"]].values:
            base = feat.replace("_delta","").replace("_roll_mean","").replace("_roll_std","")
            direction = dir_map.get(base, "HIGH")
            top_actions.append(f"[{feat}] {_action_for(feat, direction)}")

        rows.append({
            "Batch_ID":            row["Batch_ID"],
            "actual_severity":     row["severity"],
            "predicted_severity":  row.get("pred_severity", "N/A"),
            "deviation_score":     row["deviation_score"],
            "pred_dev_score":      row.get("pred_dev_score", np.nan),
            "features_oor":        row["features_oor"],
            "preemptive_actions":  " | ".join(top_actions[:3]),
        })

    return pd.DataFrame(rows)


# ── 3. Draw Dashboard ─────────────────────────────────────────────────────────
def draw_ops_dashboard(alerts, preds, imp, scored, next_df, alert_sum):
    fig = plt.figure(figsize=(24, 28), facecolor=BG)
    gs  = gridspec.GridSpec(
        5, 2, figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.07, right=0.97,
        top=0.95, bottom=0.03,
    )

    batches = alerts["Batch_ID"].tolist()
    x       = np.arange(len(batches))

    # ── Panel 1 (full width): Deviation Score + ML Prediction Overlay ─────────
    ax1 = fig.add_subplot(gs[0, :])
    dev_scores = alerts["deviation_score"].values
    roll_dev   = pd.Series(dev_scores).rolling(ROLLING, min_periods=2).mean()
    roll_std   = pd.Series(dev_scores).rolling(ROLLING, min_periods=2).std().fillna(0)

    ax1.fill_between(x, (roll_dev - roll_std).clip(0), roll_dev + roll_std,
                     alpha=0.12, color=ACCENT)
    ax1.bar(x, dev_scores, color=[SEV_CLR.get(s, MUTED) for s in alerts["severity"]],
            alpha=0.45, width=0.75, label="Actual deviation", zorder=2)
    ax1.plot(x, roll_dev, color=ACCENT, linewidth=2.2, label=f"Rolling {ROLLING}-batch trend", zorder=3)

    # Overlay ML predicted scores
    common = preds.merge(alerts[["Batch_ID"]], on="Batch_ID", how="inner")
    x_ml   = [batches.index(bid) for bid in common["Batch_ID"] if bid in batches]
    ax1.plot(x_ml, common["pred_dev_score"].values,
             color=PURPLE, linewidth=1.6, linestyle="--",
             marker="D", markersize=4, label="ML predicted score", zorder=4)

    # Next-batch prediction marker
    if not next_df.empty:
        nxt_score = float(next_df.iloc[0].get("pred_dev_score", 0))
        nxt_sev   = str(next_df.iloc[0].get("pred_severity", "HIGH"))
        ax1.annotate(
            f"▶ NEXT BATCH  Pred: {nxt_score:.0f}  [{nxt_sev}]",
            xy=(len(x), nxt_score),
            xytext=(len(x) - 5, nxt_score + 50),
            arrowprops=dict(arrowstyle="->", color=SEV_CLR.get(nxt_sev, RED), lw=1.8),
            color=SEV_CLR.get(nxt_sev, RED), fontsize=8.5, fontweight="bold",
        )
        ax1.scatter([len(x)], [nxt_score],
                    color=SEV_CLR.get(nxt_sev, RED), s=100, zorder=5,
                    marker="*", edgecolors="white", linewidth=0.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(batches, rotation=45, ha="right", fontsize=6.5)
    ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8, loc="upper right", ncol=4)
    _ax_style(ax1,
              title="① Live Deviation Score History — ML Predicted Overlay (coloured by actual severity)",
              ylabel="Deviation Score")

    # ── Panel 2 (left): Severity Timeline — Actual vs ML Predicted ───────────
    ax2 = fig.add_subplot(gs[1, 0])
    sev_order = ["LOW", "MEDIUM", "HIGH"]
    def sev_idx(s): return sev_order.index(s) if s in sev_order else 0

    common2 = preds.merge(alerts[["Batch_ID", "severity"]], on="Batch_ID", how="inner")
    x2 = np.arange(len(common2))
    ax2.scatter(x2 - 0.15, [sev_idx(s) for s in common2["actual_severity"]],
                color=ACCENT, s=45, label="Actual", marker="o", zorder=3)
    ax2.scatter(x2 + 0.15, [sev_idx(s) for s in common2["pred_severity"]],
                color=ORANGE, s=45, label="ML Predicted", marker="D", zorder=3)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(sev_order, color=TEXT, fontsize=8)
    ax2.set_xticks(x2[::5])
    ax2.set_xticklabels(common2["Batch_ID"].iloc[::5].values, rotation=45, ha="right", fontsize=6.5)
    ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    _ax_style(ax2, title="② Severity Timeline — Actual vs ML Predicted",
              xlabel="Batch", ylabel="Severity Class")

    # ── Panel 3 (right): Top Δ / rolling σ Features ──────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    top_imp = imp.head(15)
    feat_labels = (
        top_imp["feature"]
        .str.replace("_roll_mean", " [μ]")
        .str.replace("_roll_std",  " [σ]")
        .str.replace("_delta",     " [Δ]")
        .str.replace("_", " ")
    )
    bar_colors = [ORANGE if "Δ" in fl else PURPLE if "σ" in fl else ACCENT
                  for fl in feat_labels]
    bars = ax3.barh(range(len(top_imp)), top_imp["avg_importance"].values,
                    color=bar_colors, edgecolor=PANEL, linewidth=0.3)
    ax3.set_yticks(range(len(top_imp)))
    ax3.set_yticklabels(feat_labels.str[:28], fontsize=7, color=TEXT)
    ax3.invert_yaxis()
    for b, v in zip(bars, top_imp["avg_importance"].values):
        ax3.text(v + 0.002, b.get_y() + b.get_height()/2,
                 f"{v:.3f}", va="center", color=TEXT, fontsize=6.5)
    # Legend patches
    p1 = mpatches.Patch(color=ORANGE, label="Δ (rate of change)")
    p2 = mpatches.Patch(color=PURPLE, label="σ (volatility)")
    p3 = mpatches.Patch(color=ACCENT, label="μ (rolling mean)")
    ax3.legend(handles=[p1, p2, p3], facecolor=PANEL, edgecolor=GRID,
               labelcolor=TEXT, fontsize=7.5, loc="lower right")
    _ax_style(ax3, title="③ Top Predictive Features — Δ/σ/μ Breakdown",
              xlabel="Avg Feature Importance")

    # ── Panel 4 (left): Predicted vs Actual score scatter ────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    common4 = preds.copy()
    ax4.scatter(common4["actual_dev_score"], common4["pred_dev_score"],
                c=[SEV_CLR.get(s, MUTED) for s in common4["actual_severity"]],
                s=55, alpha=0.85, edgecolors="none", zorder=3)
    lims = [min(common4[["actual_dev_score","pred_dev_score"]].min()),
            max(common4[["actual_dev_score","pred_dev_score"]].max())]
    ax4.plot(lims, lims, color=MUTED, linewidth=1.0, linestyle="--")
    r2 = np.corrcoef(common4["actual_dev_score"], common4["pred_dev_score"])[0,1]**2
    ax4.annotate(f"R²={r2:.3f}", xy=(0.05, 0.91), xycoords="axes fraction",
                 color=PURPLE, fontsize=9, fontweight="bold")
    for sev, col in SEV_CLR.items():
        ax4.scatter([], [], c=col, s=35, label=sev)
    ax4.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7.5, ncol=2)
    ax4.set_xlim(lims); ax4.set_ylim(lims)
    _ax_style(ax4, title="④ ML Predicted vs Actual Deviation Score",
              xlabel="Actual Score", ylabel="Predicted Score")

    # ── Panel 5 (right): Composite Score Trend with golden band ──────────────
    ax5 = fig.add_subplot(gs[2, 1])
    comp = alerts["composite_score"].values
    roll_comp = pd.Series(comp).rolling(ROLLING, min_periods=2).mean()
    ax5.plot(x, comp, color=MUTED, linewidth=0.9, alpha=0.6)
    ax5.plot(x, roll_comp, color=GREEN, linewidth=2.0, label=f"Rolling {ROLLING}-batch avg")
    ax5.scatter(x, comp, c=[SEV_CLR.get(s, MUTED) for s in alerts["severity"]],
                s=40, zorder=3, edgecolors="none")
    ax5.axhline(60, color=YELLOW, linewidth=1.0, linestyle="--", alpha=0.8, label="Target ≥60")
    ax5.axhline(80, color=GREEN,  linewidth=1.0, linestyle=":",  alpha=0.8, label="Golden ≥80")
    ax5.set_xticks(x[::5])
    ax5.set_xticklabels(batches[::5], rotation=45, ha="right", fontsize=6.5)
    ax5.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    _ax_style(ax5, title="⑤ Composite Score Trend — Severity Coloured",
              xlabel="Batch", ylabel="Composite Score (0–100)")

    # ── Panel 6 (full width): Next-Batch Risk Card + Corrective Actions ───────
    ax6 = fig.add_subplot(gs[3:, :])
    ax6.set_facecolor(PANEL)
    ax6.axis("off")
    ax6.set_title(
        "⑥  NEXT-BATCH RISK CARD & PREEMPTIVE CORRECTIVE ACTIONS  (top predicted HIGH-risk batches)",
        color=TEXT, fontsize=10, fontweight="bold", pad=10, loc="left"
    )

    # Next-batch summary box (left)
    if not next_df.empty:
        row0   = next_df.iloc[0]
        ns     = str(row0.get("pred_severity", "HIGH"))
        nc     = SEV_CLR.get(ns, RED)
        nd     = float(row0.get("pred_dev_score", 0))

        ax6.text(0.01, 0.92, "NEXT BATCH FORECAST", color=ACCENT,
                 fontsize=9, fontweight="bold", transform=ax6.transAxes)
        ax6.text(0.01, 0.84, f"Deviation Score: {nd:.0f}", color=ORANGE,
                 fontsize=11, fontweight="bold", transform=ax6.transAxes)
        ax6.text(0.01, 0.76, f"Severity Class: {ns}", color=nc,
                 fontsize=12, fontweight="bold", transform=ax6.transAxes)

        # Risk bar
        risk_pct = min(nd / 750, 1.0)
        bar_col  = RED if risk_pct > 0.6 else ORANGE if risk_pct > 0.3 else GREEN
        ax6.add_patch(mpatches.FancyBboxPatch(
            (0.01, 0.62), 0.22, 0.07,
            boxstyle="round,pad=0.002", fc=GRID, transform=ax6.transAxes, zorder=1))
        ax6.add_patch(mpatches.FancyBboxPatch(
            (0.01, 0.62), 0.22 * risk_pct, 0.07,
            boxstyle="round,pad=0.002", fc=bar_col, transform=ax6.transAxes, zorder=2))
        ax6.text(0.12, 0.60, f"{risk_pct*100:.0f}% of max risk",
                 ha="center", color=bar_col, fontsize=7.5, transform=ax6.transAxes)

    # Corrective actions table (right)
    high_risk = alert_sum[alert_sum["actual_severity"] == "HIGH"].head(6)
    col_labels = ["Batch", "Actual\nSev", "Pred\nSev", "Dev.", "Actions (top Δ/σ features)"]
    col_x      = [0.26, 0.35, 0.44, 0.53, 0.62]
    col_w      = [0.09, 0.09, 0.09, 0.09, 0.36]

    y0 = 0.92; rh = 0.12
    for ci, (label, cx) in enumerate(zip(col_labels, col_x)):
        ax6.text(cx, y0, label, ha="left", va="center", fontsize=7.5,
                 fontweight="bold", color=ACCENT, transform=ax6.transAxes)

    ax6.plot([0.25, 0.99], [y0 - 0.018, y0 - 0.018],
             color=GRID, linewidth=1.0, transform=ax6.transAxes)

    for ri, (_, rr) in enumerate(high_risk.iterrows()):
        y = y0 - (ri + 1) * rh
        if y < 0.02: break
        bg = "#111827" if ri % 2 == 0 else PANEL
        ax6.add_patch(mpatches.FancyBboxPatch(
            (0.25, y - rh * 0.45), 0.745, rh * 0.88,
            boxstyle="round,pad=0.002", fc=bg, transform=ax6.transAxes, zorder=0
        ))
        sc = SEV_CLR.get(str(rr["actual_severity"]), MUTED)
        pc = SEV_CLR.get(str(rr["predicted_severity"]), MUTED)
        vals   = [rr["Batch_ID"],
                  str(rr["actual_severity"]),
                  str(rr["predicted_severity"]),
                  f"{rr['deviation_score']:.0f}",
                  textwrap.shorten(str(rr["preemptive_actions"]), width=130, placeholder="…")]
        colors = [TEXT, sc, pc, ORANGE, TEXT]
        for ci, (val, cx, clr) in enumerate(zip(vals, col_x, colors)):
            ax6.text(cx, y, val, ha="left", va="center",
                     fontsize=7 if ci < 4 else 6.5, color=clr,
                     transform=ax6.transAxes)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.99, 0.01, f"Generated: {ts}  |  OptiBatch Ops Dashboard",
             ha="right", va="bottom", fontsize=7.5, color=MUTED)
    fig.suptitle("OptiBatch — Integrated Operational + Predictive Dashboard",
                 fontsize=16, fontweight="bold", color=TEXT, y=0.975)

    fig.savefig(DASH_OUT, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [ops_dashboard] Saved: {DASH_OUT}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_ops_dashboard():
    os.makedirs(MONITORING_DIR, exist_ok=True)
    alerts, preds, imp, scored, pareto, next_df = load_data()
    alert_sum = build_alert_summary(alerts, preds, imp)
    alert_sum.to_csv(ALERT_SUM, index=False)
    print(f"  [ops_dashboard] Saved: {ALERT_SUM}")
    draw_ops_dashboard(alerts, preds, imp, scored, next_df, alert_sum)
    print("\n✅ Operational dashboard complete.")
    print(f"   • {DASH_OUT}")
    print(f"   • {ALERT_SUM}")


if __name__ == "__main__":
    run_ops_dashboard()
