# code/predictive_dashboard.py
"""
Predictive Trend & Root-Cause Dashboard
────────────────────────────────────────
Panel 1 — Composite Score Trend with rolling anomaly risk band
Panel 2 — Severity Distribution over time (stacked area)
Panel 3 — Top recurring deviating features (horizontal bar)
Panel 4 — Deviation score trend + predictive forecast (rolling window)
Panel 5 — Corrective Actions summary for upcoming / high-risk batches
Outputs:
  predictive_dashboard.png
  predictive_alerts_with_actions.csv
  root_cause_report.txt
"""

import os
import sys
import warnings
import textwrap
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors   as mcolors
import matplotlib.patches  as mpatches
import matplotlib.ticker   as mticker

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
ALERTS_CSV    = os.path.join(OUTPUT_DIR, "batch_monitoring_alerts.csv")
SCORED_CSV    = os.path.join(OUTPUT_DIR, "scored_batches.csv")
PARETO_CSV    = os.path.join(OUTPUT_DIR, "golden_signature_pareto.csv")
ADV_CSV       = os.path.join(OUTPUT_DIR, "batch_advanced_analysis.csv")
DASH_OUT      = os.path.join(OUTPUT_DIR, "predictive_dashboard.png")
ACTION_CSV    = os.path.join(OUTPUT_DIR, "predictive_alerts_with_actions.csv")
REPORT_OUT    = os.path.join(OUTPUT_DIR, "root_cause_report.txt")

# ── Style constants ────────────────────────────────────────────────────────────
BG        = "#080b14"
PANEL_BG  = "#0d1117"
GRID_CLR  = "#1a2035"
TEXT_CLR  = "#e2e8f0"
MUTED     = "#64748b"
ACCENT    = "#38bdf8"
ORANGE    = "#f97316"
GREEN     = "#22c55e"
RED       = "#ef4444"
YELLOW    = "#facc15"
PURPLE    = "#a78bfa"

SEV_PALETTE = {"OK": GREEN, "LOW": YELLOW, "MEDIUM": ORANGE, "HIGH": RED}

ROLLING_N = 5      # rolling window for trend analysis and prediction

# ── Corrective actions lookup ──────────────────────────────────────────────────
ACTIONS: dict[str, dict[str, str]] = {
    "Pressure_Bar_mean":       {"HIGH": "Reduce line pressure — check upstream regulator valve",
                                 "LOW":  "Increase line pressure — inspect pump output and seals"},
    "Humidity_Percent_min":    {"HIGH": "Lower HVAC humidity setpoint or increase desiccant capacity",
                                 "LOW":  "Increase HVAC humidity setpoint or add humidifier feed"},
    "Lubricant_Conc":          {"HIGH": "Reduce lubricant dosing rate — recalibrate dispensing pump",
                                 "LOW":  "Increase lubricant dosing rate — check pump blockage"},
    "Friability":              {"HIGH": "Increase compression force; review binder concentration",
                                 "LOW":  "Reduce compression force to prevent over-hardening"},
    "Friability_inv":          {"HIGH": "Tablet strength optimal — no action needed",
                                 "LOW":  "Tablet too brittle — increase compression force"},
    "Drying_Time":             {"HIGH": "Shorten drying cycle; raise drying temperature by 2–3 °C",
                                 "LOW":  "Extend drying cycle; check airflow and heater output"},
    "Machine_Speed":           {"HIGH": "Reduce tablet press RPM to target range",
                                 "LOW":  "Increase tablet press RPM to target range"},
    "Compression_Force":       {"HIGH": "Reduce compression force to prevent tablet cracking",
                                 "LOW":  "Increase compression force to improve tablet hardness"},
    "Binder_Amount":           {"HIGH": "Reduce binder quantity per batch; check metering valve",
                                 "LOW":  "Increase binder quantity; inspect granulator feed line"},
    "Hardness":                {"HIGH": "Reduce compression force; adjust tooling clearance",
                                 "LOW":  "Increase compression force or binder concentration"},
    "Granulation_Time":        {"HIGH": "Shorten granulation cycle; verify mixer timer calibration",
                                 "LOW":  "Extend granulation time for uniform granule size"},
    "Dissolution_Rate":        {"HIGH": "Check disintegrant concentration; review coating parameters",
                                 "LOW":  "Reduce coating thickness or increase disintegrant level"},
    "Compression_Force_kN_std":{"HIGH": "Investigate press force variability — check cam track wear",
                                 "LOW":  "Force consistency good — monitor for further drops"},
    "Content_Uniformity":      {"HIGH": "Review blending time and mixer speed",
                                 "LOW":  "Increase blending time or API particle size control"},
    "Compression_Force_kN_max":{"HIGH": "Check upper punch tooling for wear or misalignment",
                                 "LOW":  "Verify press load cell calibration"},
    "Drying_Temp":             {"HIGH": "Reduce oven setpoint; avoid API degradation",
                                 "LOW":  "Increase oven setpoint; verify thermocouple calibration"},
    "Moisture_Content":        {"HIGH": "Extend drying time; check oven humidity exhaust",
                                 "LOW":  "Reduce drying time to preserve granule integrity"},
    "Tablet_Weight":           {"HIGH": "Adjust fill depth cam on tablet press",
                                 "LOW":  "Increase fill cam depth; check hopper feed rate"},
    "Disintegration_Time":     {"HIGH": "Increase superdisintegrant level; review granule porosity",
                                 "LOW":  "Reduce disintegrant level; check coating integrity"},
    "Flow_Rate_LPM_mean":      {"HIGH": "Reduce binder spray rate; check atomising air pressure",
                                 "LOW":  "Increase spray nozzle flow; check pump speed"},
    "Humidity_Percent_max":    {"HIGH": "Seal processing area; increase dehumidification capacity",
                                 "LOW":  "HVAC is over-drying — reduce exhaust airflow"},
    "Humidity_Percent_std":    {"HIGH": "Stabilise HVAC cycling; check thermostat hunting",
                                 "LOW":  "Humidity variation acceptable"},
    "Flow_Rate_LPM_std":       {"HIGH": "Inspect spray nozzle for blockage or wear",
                                 "LOW":  "Flow consistency good"},
    "Compression_Force_kN_mean":{"HIGH":"Reduce compression force setpoint",
                                  "LOW": "Increase compression force setpoint"},
    "Pressure_Bar_std":        {"HIGH": "Investigate pressure oscillation — check pulsation dampener",
                                 "LOW":  "Pressure stability acceptable"},
    "total_energy_kwh":        {"HIGH": "Audit machine idle time; optimise run schedule",
                                 "LOW":  "Energy usage acceptable"},
}

DEFAULT_ACTION = "Review parameter with process engineer and compare against SOP"


# ─────────────────────────────────────────────────────────────────────────────
def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    for p in [ALERTS_CSV, SCORED_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    alerts = pd.read_csv(ALERTS_CSV)
    scored = pd.read_csv(SCORED_CSV)
    # Sort both by Batch_ID (numeric part) to get chronological order
    def _batch_num(bid: str) -> int:
        return int(bid[1:]) if bid[1:].isdigit() else 0
    alerts["_n"] = alerts["Batch_ID"].apply(_batch_num)
    scored["_n"] = scored["Batch_ID"].apply(_batch_num)
    alerts = alerts.sort_values("_n").reset_index(drop=True)
    scored = scored.sort_values("_n").reset_index(drop=True)
    return alerts, scored


# ─────────────────────────────────────────────────────────────────────────────
def _feature_frequency(alerts: pd.DataFrame) -> pd.Series:
    """Count how often each feature appears as OOR across all batches."""
    counter: Counter = Counter()
    for cell in alerts["critical_features_oor"].dropna():
        if cell.strip().lower() == "none":
            continue
        for feat in cell.split(";"):
            feat = feat.strip()
            if feat:
                counter[feat] += 1
    return pd.Series(counter).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
def _rolling_forecast(series: pd.Series, n: int = ROLLING_N) -> pd.Series:
    """Extend series by one step using rolling mean as naive forecast."""
    roll_mean = series.rolling(n, min_periods=2).mean()
    last_pred = roll_mean.dropna().iloc[-1] if not roll_mean.dropna().empty else series.mean()
    extension = pd.Series(
        [last_pred],
        index=[len(series)],
        name=series.name,
    )
    return pd.concat([roll_mean, extension])


# ─────────────────────────────────────────────────────────────────────────────
def _corrective_action(feat: str, direction: str) -> str:
    lookup = ACTIONS.get(feat, {})
    return lookup.get(direction.upper(), DEFAULT_ACTION)


# ─────────────────────────────────────────────────────────────────────────────
def _build_action_csv(alerts: pd.DataFrame) -> pd.DataFrame:
    """Create per-batch action CSV with predicted risk and suggestions."""
    rows: list[dict] = []
    dev_series = alerts["deviation_score"]
    roll_dev   = dev_series.rolling(ROLLING_N, min_periods=2).mean()

    for i, row in alerts.iterrows():
        # Predicted risk = rolling avg deviation over last N batches
        pred_risk  = roll_dev.iloc[i] if not pd.isna(roll_dev.iloc[i]) else dev_series.iloc[i]
        predicted_anomaly = pred_risk > dev_series.median()

        # Parse OOR features + directions
        oor_feats = [f.strip() for f in str(row["critical_features_oor"]).split(";")
                     if f.strip() and f.strip().lower() != "none"]
        dir_map: dict[str, str] = {}
        for part in str(row["feature_directions"]).split(";"):
            part = part.strip()
            if ":" in part:
                f, d = part.split(":", 1)
                dir_map[f.strip()] = d.strip()

        # Top 3 corrective actions
        actions = []
        for feat in oor_feats[:3]:
            direction = dir_map.get(feat, "")
            action    = _corrective_action(feat, direction)
            actions.append(f"[{feat}:{direction}] {action}")

        rows.append({
            "Batch_ID":           row["Batch_ID"],
            "composite_score":    row["composite_score"],
            "deviation_score":    row["deviation_score"],
            "features_oor":       row["features_oor"],
            "severity":           row["severity"],
            "is_anomalous":       row["is_anomalous"],
            "predicted_risk":     round(float(pred_risk), 2),
            "predicted_anomaly":  predicted_anomaly,
            "top_features_oor":   "; ".join(oor_feats[:5]),
            "corrective_actions": " | ".join(actions) if actions else "No immediate action required",
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def _ax_style(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines[:].set_color(GRID_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.5, alpha=0.6)
    if title:
        ax.set_title(title, color=TEXT_CLR, fontsize=10, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
def _draw_dashboard(alerts: pd.DataFrame, freq: pd.Series, action_df: pd.DataFrame) -> None:
    batches   = alerts["Batch_ID"].tolist()
    x         = np.arange(len(batches))
    sev_order = ["OK", "LOW", "MEDIUM", "HIGH"]

    fig = plt.figure(figsize=(22, 26), facecolor=BG)
    gs  = gridspec.GridSpec(
        5, 2,
        figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.07, right=0.97,
        top=0.95, bottom=0.04,
    )

    # ── Panel 1 (full width): Composite Score Trend ───────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    scores     = alerts["composite_score"].values
    roll_score = pd.Series(scores).rolling(ROLLING_N, min_periods=2).mean().values
    # Risk band: rolling mean ± rolling std
    roll_std   = pd.Series(scores).rolling(ROLLING_N, min_periods=2).std().fillna(0).values

    ax1.fill_between(x, roll_score - roll_std, roll_score + roll_std,
                     alpha=0.18, color=ACCENT, label="±1σ risk band")
    ax1.plot(x, scores,     color=MUTED,   linewidth=1.0, alpha=0.7, zorder=2)
    ax1.scatter(x, scores,  c=[SEV_PALETTE.get(s, MUTED) for s in alerts["severity"]],
                s=55, zorder=3, edgecolors="none")
    ax1.plot(x, roll_score, color=ACCENT,  linewidth=2.2, label=f"Rolling {ROLLING_N}-batch trend", zorder=4)

    # Predicted next-batch score
    next_pred = roll_score[-1]
    ax1.annotate(
        f"Next pred: {next_pred:.1f}",
        xy=(len(x), next_pred), xytext=(len(x) - 2.5, next_pred + 8),
        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5),
        color=PURPLE, fontsize=8, fontweight="bold",
    )
    ax1.axhline(60, color=YELLOW, linewidth=1.0, linestyle="--", alpha=0.7, label="Target score 60")

    ax1.set_xticks(x)
    ax1.set_xticklabels(batches, rotation=45, ha="right", fontsize=7)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
               fontsize=8, loc="upper right")
    _ax_style(ax1,
              title="① Composite Score Trend — All Batches (chronological, coloured by severity)",
              ylabel="Composite Score (0–100)")

    # Severity colour legend
    for sev, col in SEV_PALETTE.items():
        ax1.scatter([], [], c=col, s=40, label=sev)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
               fontsize=8, loc="lower right", ncol=6)

    # ── Panel 2 (left): Severity stacked area ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    sev_dummies = pd.get_dummies(alerts["severity"])
    for s in sev_order:
        if s not in sev_dummies.columns:
            sev_dummies[s] = 0

    cum_bottom = np.zeros(len(alerts))
    for sev in sev_order:
        vals = sev_dummies[sev].values.astype(float)
        ax2.fill_between(x, cum_bottom, cum_bottom + vals,
                         color=SEV_PALETTE[sev], alpha=0.82, label=sev)
        cum_bottom += vals

    ax2.set_xticks(x[::5])
    ax2.set_xticklabels(batches[::5], rotation=45, ha="right", fontsize=6.5)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
               fontsize=7.5, loc="upper left")
    _ax_style(ax2, title="② Severity Distribution over Batch Timeline",
              xlabel="Batch", ylabel="Anomaly Count (per batch)")

    # ── Panel 3 (right): Feature frequency bar ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    top_feats = freq.head(15)
    colors_bar = [ORANGE if i < 5 else ACCENT if i < 10 else MUTED
                  for i in range(len(top_feats))]
    bars = ax3.barh(
        range(len(top_feats)), top_feats.values,
        color=colors_bar, edgecolor=PANEL_BG, linewidth=0.4
    )
    ax3.set_yticks(range(len(top_feats)))
    ax3.set_yticklabels(
        [f.replace("_", " ")[:28] for f in top_feats.index],
        fontsize=7.5, color=TEXT_CLR
    )
    ax3.invert_yaxis()
    # Value labels
    for rect, val in zip(bars, top_feats.values):
        ax3.text(val + 0.3, rect.get_y() + rect.get_height() / 2,
                 str(int(val)), va="center", color=TEXT_CLR, fontsize=7)
    _ax_style(ax3, title="③ Top Recurring Deviation Features",
              xlabel="# Batches OOR", ylabel="")

    # ── Panel 4 (left): Deviation Score + forecast ────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    dev_scores  = alerts["deviation_score"].values
    roll_dev    = pd.Series(dev_scores).rolling(ROLLING_N, min_periods=2).mean()
    roll_dev_std= pd.Series(dev_scores).rolling(ROLLING_N, min_periods=2).std().fillna(0)
    pred_val    = roll_dev.dropna().iloc[-1] if not roll_dev.dropna().empty else dev_scores.mean()

    # Extend x for forecast
    x_ext = np.append(x, len(x))
    pred_series = np.append(roll_dev.values, pred_val)

    ax4.fill_between(x, (roll_dev - roll_dev_std).clip(0), roll_dev + roll_dev_std,
                     alpha=0.18, color=RED)
    ax4.bar(x, dev_scores, color=[SEV_PALETTE.get(s, MUTED) for s in alerts["severity"]],
            alpha=0.55, width=0.7)
    ax4.plot(x, roll_dev.values, color=ORANGE, linewidth=2.0, label="Rolling trend")
    ax4.plot([len(x) - 1, len(x)], [roll_dev.values[-1], pred_val],
             color=PURPLE, linewidth=2.0, linestyle="--", marker="D",
             markersize=6, label=f"Forecast next: {pred_val:.1f}")
    ax4.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=8)
    ax4.set_xticks(x[::5])
    ax4.set_xticklabels(batches[::5], rotation=45, ha="right", fontsize=6.5)
    _ax_style(ax4, title="④ Deviation Score Trend + Predictive Forecast",
              xlabel="Batch", ylabel="Weighted Deviation Score")

    # ── Panel 5 (right): Features OOR count per batch scatter ───────────────
    ax5 = fig.add_subplot(gs[2, 1])
    oor_counts = alerts["features_oor"].values
    roll_oor   = pd.Series(oor_counts).rolling(ROLLING_N, min_periods=2).mean()

    ax5.bar(x, oor_counts,
            color=[SEV_PALETTE.get(s, MUTED) for s in alerts["severity"]],
            alpha=0.65, width=0.7)
    ax5.plot(x, roll_oor.values, color=ACCENT, linewidth=2.0, label="Rolling avg")
    ax5.axhline(6, color=RED, linewidth=1.0, linestyle="--", alpha=0.8, label="HIGH threshold")
    ax5.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=8)
    ax5.set_xticks(x[::5])
    ax5.set_xticklabels(batches[::5], rotation=45, ha="right", fontsize=6.5)
    _ax_style(ax5, title="⑤ Features Out-of-Range per Batch",
              xlabel="Batch", ylabel="# Critical Features OOR")

    # ── Panel 6 (full width): Top-10 Corrective Actions Table ─────────────────
    ax6 = fig.add_subplot(gs[3:, :])
    ax6.set_facecolor(PANEL_BG)
    ax6.axis("off")
    ax6.set_title("⑥ Predictive Corrective Actions — Top Batches by Predicted Risk",
                  color=TEXT_CLR, fontsize=10, fontweight="bold", pad=8, loc="left")

    top10 = action_df.sort_values("predicted_risk", ascending=False).head(10)
    col_labels = ["Batch", "Score", "Dev.", "Features\nOOR", "Severity",
                  "Pred.\nRisk", "Pred.\nAnomalous", "Top Corrective Actions"]
    col_widths = [0.07, 0.06, 0.06, 0.05, 0.07, 0.06, 0.07, 0.54]

    y_start = 0.92
    row_h   = 0.082

    # Header
    x_pos = 0.0
    for label, w in zip(col_labels, col_widths):
        ax6.text(x_pos + w / 2, y_start, label,
                 ha="center", va="center", fontsize=7.5, fontweight="bold",
                 color=ACCENT, transform=ax6.transAxes)
        x_pos += w

    ax6.plot([0, 1], [y_start - 0.008, y_start - 0.008],
             color=GRID_CLR, linewidth=1.0, transform=ax6.transAxes)

    # Rows
    for r_idx, (_, row) in enumerate(top10.iterrows()):
        y = y_start - (r_idx + 1) * row_h
        bg_col = "#111827" if r_idx % 2 == 0 else PANEL_BG
        rect = mpatches.FancyBboxPatch(
            (0, y - row_h * 0.5), 1.0, row_h * 0.92,
            boxstyle="round,pad=0.002", linewidth=0,
            facecolor=bg_col, transform=ax6.transAxes, zorder=0
        )
        ax6.add_patch(rect)

        sev_col = SEV_PALETTE.get(str(row["severity"]), MUTED)
        values = [
            row["Batch_ID"],
            f"{row['composite_score']:.1f}",
            f"{row['deviation_score']:.0f}",
            str(int(row["features_oor"])),
            row["severity"],
            f"{row['predicted_risk']:.0f}",
            "⚠ YES" if row["predicted_anomaly"] else "✓ OK",
            textwrap.shorten(str(row["corrective_actions"]), width=120, placeholder="…"),
        ]
        colors = [TEXT_CLR, TEXT_CLR, TEXT_CLR, TEXT_CLR,
                  sev_col, ORANGE,
                  RED if row["predicted_anomaly"] else GREEN,
                  TEXT_CLR]

        x_pos = 0.0
        for val, w, clr in zip(values, col_widths, colors):
            is_long = len(val) >= 20
            tx = x_pos + 0.01 if is_long else x_pos + w / 2
            ax6.text(tx, y, val,
                     ha="left" if is_long else "center",
                     va="center", fontsize=6.8, color=clr,
                     transform=ax6.transAxes)
            x_pos += w

    # Timestamp
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.99, 0.01, f"Generated: {ts}  |  OptiBatch Monitoring System",
             ha="right", va="bottom", fontsize=7.5, color=MUTED)
    fig.suptitle(
        "OptiBatch — Predictive Trend & Root-Cause Dashboard",
        fontsize=16, fontweight="bold", color=TEXT_CLR, y=0.975
    )

    fig.savefig(DASH_OUT, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def _write_root_cause_report(alerts: pd.DataFrame, freq: pd.Series,
                              action_df: pd.DataFrame) -> None:
    ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep    = "═" * 68
    n_all  = len(alerts)
    n_anom = alerts["is_anomalous"].sum()

    # % batches each feature is OOR
    pct_feat = (freq / n_all * 100).round(1)

    # Rolling risk prediction for next batch
    pred_dev  = pd.Series(action_df["deviation_score"]) \
                  .rolling(ROLLING_N, min_periods=2).mean().iloc[-1]
    pred_sev  = "HIGH" if pred_dev > 300 else "MEDIUM" if pred_dev > 100 else "LOW"

    lines = [
        sep,
        "  OPTIBATCH ROOT-CAUSE & PREDICTIVE TREND REPORT",
        f"  Generated : {ts}",
        sep,
        f"  Batches analysed  : {n_all}",
        f"  Anomalous batches : {n_anom} ({n_anom/n_all*100:.0f}%)",
        f"  Predicted next batch risk : {pred_dev:.1f}  →  severity FORECAST = {pred_sev}",
        sep,
        "  TOP RECURRING DEVIATION DRIVERS",
        "─" * 68,
        f"  {'Rank':<5} {'Feature':<32} {'Batches OOR':>12} {'% of all':>10}",
        "─" * 68,
    ]
    for i, (feat, cnt) in enumerate(freq.head(15).items(), 1):
        lines.append(f"  {i:<5} {feat:<32} {cnt:>12d} {pct_feat[feat]:>9.1f}%")

    lines += [
        sep,
        "  CONSISTENT DEVIATION PATTERNS DETECTED",
        "─" * 68,
    ]
    # Features OOR in >80% of anomalous batches
    threshold_80 = n_anom * 0.80
    systemic = freq[freq >= threshold_80]
    if not systemic.empty:
        lines.append("  ► Systemic issues (OOR in >80% of anomalous batches):")
        for feat, cnt in systemic.items():
            direction_counts = Counter()
            for cell in alerts["feature_directions"].dropna():
                for part in cell.split(";"):
                    part = part.strip()
                    if feat in part and ":" in part:
                        direction_counts[part.split(":")[-1].strip()] += 1
            dominant_dir = direction_counts.most_common(1)[0][0] if direction_counts else "UNKNOWN"
            action = _corrective_action(feat, dominant_dir)
            lines.append(f"\n    • {feat}  [{dominant_dir} in {cnt} batches]")
            lines.append(f"      Recommendation: {action}")
    else:
        lines.append("  No systemic (>80%) single-feature issues detected.")

    lines += [
        sep,
        "  PREDICTIVE ALERTS — NEXT BATCH",
        "─" * 68,
        f"  Forecast deviation score  : {pred_dev:.1f}",
        f"  Forecast severity         : {pred_sev}",
        "  Pre-emptive actions to take BEFORE next batch run:",
    ]

    # Top 3 systemic features → pre-emptive actions
    for feat, cnt in freq.head(3).items():
        lines.append(f"\n  [{feat}]")
        for direction in ["HIGH", "LOW"]:
            action = ACTIONS.get(feat, {}).get(direction, DEFAULT_ACTION)
            lines.append(f"    If {direction}: {action}")

    lines += [
        sep,
        f"  Full action CSV saved to : {ACTION_CSV}",
        f"  Dashboard saved to       : {DASH_OUT}",
        sep,
    ]

    report_text = "\n".join(lines)
    with open(REPORT_OUT, "w") as f:
        f.write(report_text)
    print(report_text)


# ─────────────────────────────────────────────────────────────────────────────
def run_predictive_dashboard() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    alerts, scored = _load()
    freq           = _feature_frequency(alerts)
    action_df      = _build_action_csv(alerts)

    action_df.to_csv(ACTION_CSV, index=False)
    _draw_dashboard(alerts, freq, action_df)
    _write_root_cause_report(alerts, freq, action_df)

    print(f"\n✅ Predictive Dashboard complete.")
    print(f"   • {DASH_OUT}")
    print(f"   • {ACTION_CSV}")
    print(f"   • {REPORT_OUT}")


if __name__ == "__main__":
    run_predictive_dashboard()
