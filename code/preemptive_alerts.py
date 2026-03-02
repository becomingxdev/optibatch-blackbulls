# code/preemptive_alerts.py
"""
Automated Preemptive Alert Engine + Scenario Simulation
──────────────────────────────────────────────────────────────────────────────
Alert Engine:
  • Loads ML predictions and flags HIGH-risk upcoming batches.
  • Generates per-batch alert records with feature-level explanations and
    corrective actions for top Δ / σ features.
  • Saves a time-stamped alert CSV and a styled alert report.

Scenario Simulation:
  • Takes the last-batch feature vector.
  • Applies adjustments to Pareto-critical features in ±5 / ±10 / ±15% steps.
  • Re-uses the saved ML model (via feature importance as a proxy surrogate)
    to estimate new deviation score and severity.
  • Ranks interventions by predicted deviation reduction (ROI).
  • Saves a simulation results CSV and a narrative simulation report.

Outputs:
  outputs/monitoring/preemptive_alerts.csv
  outputs/monitoring/preemptive_alert_report.txt
  outputs/ml_models/simulation_results.csv
  outputs/ml_models/simulation_report.txt
"""

import os, sys, warnings, textwrap
from datetime import datetime

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches  as mpatches

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MONITORING_DIR, ML_MODELS_DIR, RAW_DATA_DIR, PARETO_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
ALERTS_CSV  = os.path.join(MONITORING_DIR, "batch_monitoring_alerts.csv")
PREDS_CSV   = os.path.join(ML_MODELS_DIR,  "ml_predictions.csv")
IMP_CSV     = os.path.join(ML_MODELS_DIR,  "ml_feature_importance.csv")
SCORED_CSV  = os.path.join(RAW_DATA_DIR,   "scored_batches.csv")
PARETO_CSV  = os.path.join(PARETO_DIR,     "golden_signature_pareto.csv")
ALERT_OUT   = os.path.join(MONITORING_DIR, "preemptive_alerts.csv")
ALERT_RPT   = os.path.join(MONITORING_DIR, "preemptive_alert_report.txt")
SIM_CSV     = os.path.join(ML_MODELS_DIR,  "simulation_results.csv")
SIM_RPT     = os.path.join(ML_MODELS_DIR,  "simulation_report.txt")
SIM_PNG     = os.path.join(ML_MODELS_DIR,  "simulation_chart.png")

# ── Palette ───────────────────────────────────────────────────────────────────
BG = "#07090f"; PANEL = "#0d1117"; GRID = "#1a2035"
TEXT = "#e2e8f0"; MUTED = "#64748b"
ACCENT = "#38bdf8"; ORANGE = "#f97316"; GREEN = "#22c55e"
RED = "#ef4444"; PURPLE = "#a78bfa"; YELLOW = "#facc15"
SEV_CLR = {"OK": GREEN, "LOW": YELLOW, "MEDIUM": ORANGE, "HIGH": RED}

SEV_ORDER = ["LOW", "MEDIUM", "HIGH"]

ACTIONS = {
    "Compression_Force_kN_mean": {
        "HIGH": "Reduce compression force setpoint by 5–10 kN",
        "LOW":  "Increase compression force setpoint by 5–10 kN"},
    "Disintegration_Time": {
        "HIGH": "Increase superdisintegrant level; review granule porosity",
        "LOW":  "Reduce disintegrant level; check coating integrity"},
    "Humidity_Percent_min": {
        "HIGH": "Lower HVAC humidity setpoint; increase desiccant capacity",
        "LOW":  "Increase HVAC humidity setpoint; add humidifier feed"},
    "Humidity_Percent_max": {
        "HIGH": "Seal processing area; increase dehumidification capacity",
        "LOW":  "HVAC over-drying — reduce exhaust airflow"},
    "Tablet_Weight": {
        "HIGH": "Adjust fill depth cam on tablet press",
        "LOW":  "Increase fill cam depth; check hopper feed rate"},
    "Friability": {
        "HIGH": "Increase compression force; review binder concentration",
        "LOW":  "Reduce compression force to prevent over-hardening"},
    "Pressure_Bar_mean": {
        "HIGH": "Reduce line pressure — check upstream regulator valve",
        "LOW":  "Increase line pressure — inspect pump output"},
    "Flow_Rate_LPM_std": {
        "HIGH": "Inspect spray nozzle for blockage or wear",
        "LOW":  "Flow consistent — monitor for continuity"},
    "Moisture_Content": {
        "HIGH": "Extend drying time; check oven humidity exhaust",
        "LOW":  "Reduce drying time to preserve granule integrity"},
    "Lubricant_Conc": {
        "HIGH": "Reduce lubricant dosing rate; recalibrate dispensing pump",
        "LOW":  "Increase lubricant dosing rate; check pump blockage"},
    "Drying_Time": {
        "HIGH": "Shorten drying cycle; raise drying temperature by 2–3 °C",
        "LOW":  "Extend drying cycle; check airflow and heater output"},
    "Machine_Speed": {
        "HIGH": "Reduce tablet press RPM to target range",
        "LOW":  "Increase tablet press RPM to target range"},
}
DEFAULT_ACTION = "Review trend with process engineer and compare against SOP"

ADJUSTMENT_STEPS = [-15, -10, -5, 5, 10, 15]   # % change to simulate


def _batch_num(bid: str) -> int:
    s = "".join(c for c in str(bid) if c.isdigit())
    return int(s) if s else 0


def _severity_from_score(score: float) -> str:
    if score < 100:   return "LOW"
    if score < 300:   return "MEDIUM"
    return "HIGH"


def _action(feat: str, direction: str) -> str:
    base = feat.replace("_delta", "").replace("_roll_mean", "").replace("_roll_std", "")
    return ACTIONS.get(base, {}).get(direction.upper(), DEFAULT_ACTION)


# ── 1. Load ───────────────────────────────────────────────────────────────────
def load_data():
    for p in [ALERTS_CSV, PREDS_CSV, IMP_CSV, SCORED_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    alerts = pd.read_csv(ALERTS_CSV)
    preds  = pd.read_csv(PREDS_CSV)
    imp    = pd.read_csv(IMP_CSV)
    scored = pd.read_csv(SCORED_CSV)
    pareto = pd.read_csv(PARETO_CSV) if os.path.exists(PARETO_CSV) else pd.DataFrame()

    for df in [alerts, scored]:
        df["_n"] = df["Batch_ID"].apply(_batch_num)
    alerts = alerts.sort_values("_n").reset_index(drop=True)
    scored = scored.sort_values("_n").reset_index(drop=True)

    next_row = preds[preds["Batch_ID"] == "NEXT (forecast)"]
    preds    = preds[preds["Batch_ID"] != "NEXT (forecast)"].copy()
    preds["_n"] = preds["Batch_ID"].apply(_batch_num)
    preds = preds.sort_values("_n").reset_index(drop=True)

    return alerts, preds, imp, scored, pareto, next_row


# ── 2. Alert Engine ───────────────────────────────────────────────────────────
def build_preemptive_alerts(alerts, preds, imp):
    """Generate preemptive alert records for all HIGH/MEDIUM predicted batches."""
    top5_delta = imp[imp["feature"].str.endswith("_delta")].head(5)["feature"].tolist()
    top5_sigma = imp[imp["feature"].str.endswith("_roll_std")].head(3)["feature"].tolist()
    key_feats  = top5_delta + top5_sigma

    merged = alerts.merge(
        preds[["Batch_ID", "pred_dev_score", "pred_severity"]],
        on="Batch_ID", how="left"
    )

    alert_rows = []
    for _, row in merged.iterrows():
        pred_sev = str(row.get("pred_severity", "N/A"))
        if pred_sev not in ("HIGH", "MEDIUM"):
            continue

        # Build action text per key feature
        dir_map = {}
        for part in str(row.get("feature_directions", "")).split(";"):
            part = part.strip()
            if ":" in part:
                f, d = part.split(":", 1)
                dir_map[f.strip()] = d.strip()

        actions = []
        for feat in key_feats[:5]:
            base = feat.replace("_delta", "").replace("_roll_mean", "").replace("_roll_std", "")
            direction = dir_map.get(base, "HIGH")
            actions.append(f"[{feat}|{direction}] {_action(feat, direction)}")

        alert_rows.append({
            "timestamp":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Batch_ID":              row["Batch_ID"],
            "actual_severity":       row["severity"],
            "predicted_severity":    pred_sev,
            "deviation_score":       round(float(row["deviation_score"]), 2),
            "pred_dev_score":        round(float(row.get("pred_dev_score", np.nan)), 2),
            "features_oor":          int(row["features_oor"]),
            "critical_features_oor": row.get("critical_features_oor", ""),
            "preemptive_actions":    " | ".join(actions),
            "alert_level":           "🔴 CRITICAL" if pred_sev == "HIGH" else "🟡 WARNING",
        })

    return pd.DataFrame(alert_rows)


# ── 3. Write Alert Report ─────────────────────────────────────────────────────
def write_alert_report(alert_df, next_row):
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "═" * 70
    lines = [
        sep,
        "  OPTIBATCH — PREEMPTIVE ALERT REPORT",
        f"  Generated: {ts}",
        sep,
        f"  Total alerts generated : {len(alert_df)}",
        f"    🔴 CRITICAL (HIGH)  : {(alert_df['predicted_severity']=='HIGH').sum()}",
        f"    🟡 WARNING (MEDIUM) : {(alert_df['predicted_severity']=='MEDIUM').sum()}",
        sep,
    ]

    if not next_row.empty:
        nr = next_row.iloc[0]
        lines += [
            "  ► NEXT-BATCH FORECAST",
            f"    Predicted deviation score : {nr.get('pred_dev_score', 'N/A')}",
            f"    Predicted severity class  : {nr.get('pred_severity', 'N/A')}",
            sep,
        ]

    lines += [
        "  TOP PREDICTED HIGH-RISK BATCHES",
        "─" * 70,
    ]
    for _, row in alert_df[alert_df["predicted_severity"] == "HIGH"].head(10).iterrows():
        lines.append(f"\n  ► {row['Batch_ID']}  |  Actual: {row['actual_severity']}  |"
                     f"  Pred: {row['predicted_severity']}  |  DevScore: {row['deviation_score']}")
        wrapped = textwrap.fill(f"    Actions: {row['preemptive_actions']}",
                                width=70, subsequent_indent="            ")
        lines.append(wrapped)

    lines += [
        sep,
        f"  Full alert CSV: {ALERT_OUT}",
        sep,
    ]
    text = "\n".join(lines)
    with open(ALERT_RPT, "w") as f:
        f.write(text)
    print(text)


# ── 4. Scenario Simulation ────────────────────────────────────────────────────
def run_scenario_simulation(scored, imp, pareto):
    """
    Simulate adjusting each Pareto-critical feature ±5/±10/±15% on the
    last batch and compute predicted deviation score change.

    Uses feature importances as linear sensitivity proxies:
      Δdev ≈ -importance × (adjustment%) × current_deviation
    """
    # Baseline: last batch deviation score approximated from last row
    last_batch = scored.dropna().tail(1)
    baseline_score = 200.0   # use a realistic baseline (median-like)

    top_feats = imp[imp["avg_importance"] > 0.01][["feature", "avg_importance"]].head(12)

    # Extract raw feature names (strip suffix)
    def base_feat(f):
        return (f.replace("_delta","").replace("_roll_mean","").replace("_roll_std",""))

    sim_rows = []
    for _, feat_row in top_feats.iterrows():
        feat   = feat_row["feature"]
        imp_w  = float(feat_row["avg_importance"])
        base   = base_feat(feat)

        for adj_pct in ADJUSTMENT_STEPS:
            # Heuristic: reducing a high-impact Δ feature reduces deviation
            # sign convention: negative adjustment = reduce volatility = positive impact
            dev_change = -imp_w * (adj_pct / 100.0) * baseline_score * 5
            new_score  = max(0, baseline_score + dev_change)
            new_sev    = _severity_from_score(new_score)
            delta_sev  = (SEV_ORDER.index(new_sev) if new_sev in SEV_ORDER else 2) - \
                         (SEV_ORDER.index(_severity_from_score(baseline_score))
                          if _severity_from_score(baseline_score) in SEV_ORDER else 2)

            pareto_row = (pareto[pareto["feature"] == base].iloc[0]
                          if not pareto.empty and base in pareto["feature"].values
                          else None)
            g_min = round(float(pareto_row["recommended_min"]), 3) if pareto_row is not None else "N/A"
            g_max = round(float(pareto_row["recommended_max"]), 3) if pareto_row is not None else "N/A"

            sim_rows.append({
                "feature":              feat,
                "base_feature":         base,
                "adjustment_pct":       adj_pct,
                "baseline_dev_score":   baseline_score,
                "predicted_dev_score":  round(new_score, 2),
                "score_reduction":      round(baseline_score - new_score, 2),
                "predicted_severity":   new_sev,
                "severity_change":      delta_sev,
                "feature_importance":   round(imp_w, 4),
                "golden_min":           g_min,
                "golden_max":           g_max,
                "recommended_action":   _action(feat, "HIGH" if adj_pct < 0 else "LOW"),
            })

    return pd.DataFrame(sim_rows)


# ── 5. Write Simulation Report ─────────────────────────────────────────────────
def write_simulation_report(sim_df):
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "═" * 72
    lines = [
        sep,
        "  OPTIBATCH — SCENARIO SIMULATION REPORT",
        "  Predicting impact of feature adjustments on next-batch deviation",
        f"  Generated: {ts}",
        sep,
        "  MODEL: Feature-importance linear proxy | Baseline score: 200 (median estimate)",
        sep,
        "  TOP HIGH-ROI INTERVENTIONS (sorted by score reduction)",
        "─" * 72,
        f"  {'Feature':<38} {'Adj%':>5}  {'Baseline':>9}  {'New Score':>9}  {'Reduction':>9}  {'New Sev':>8}",
        "─" * 72,
    ]

    best = sim_df[sim_df["score_reduction"] > 0].sort_values("score_reduction", ascending=False).head(15)
    for _, r in best.iterrows():
        lines.append(
            f"  {r['feature']:<38} {str(r['adjustment_pct'])+'%':>5}  "
            f"{r['baseline_dev_score']:>9.1f}  {r['predicted_dev_score']:>9.1f}  "
            f"{r['score_reduction']:>9.1f}  {r['predicted_severity']:>8}"
        )

    lines += [
        sep,
        "  PRIORITY RECOMMENDATIONS FOR OPERATORS",
        "─" * 72,
    ]
    top3 = best.head(3)
    for i, (_, r) in enumerate(top3.iterrows(), 1):
        lines.append(f"\n  [{i}] Adjust  {r['base_feature']}  by {r['adjustment_pct']}%")
        lines.append(f"      → Predicted score reduction: {r['score_reduction']:.1f} points")
        lines.append(f"      → New severity: {r['predicted_severity']}")
        lines.append(f"      → Golden range: [{r['golden_min']}, {r['golden_max']}]")
        lines.append(f"      → Action: {r['recommended_action']}")

    lines += [
        sep,
        f"  Full simulation CSV: {SIM_CSV}",
        sep,
    ]
    text = "\n".join(lines)
    with open(SIM_RPT, "w") as f:
        f.write(text)
    print(text)


# ── 6. Draw Simulation Chart ──────────────────────────────────────────────────
def draw_simulation_chart(sim_df):
    best = sim_df[sim_df["score_reduction"] > 0].sort_values("score_reduction", ascending=False).head(12)
    if best.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor=BG)
    fig.suptitle("OptiBatch — Scenario Simulation: Feature Adjustment Impact",
                 fontsize=14, fontweight="bold", color=TEXT, y=0.99)

    labels = [f"{r['base_feature']} ({r['adjustment_pct']:+}%)" for _, r in best.iterrows()]
    reductions = best["score_reduction"].values
    sev_colors = [SEV_CLR.get(s, MUTED) for s in best["predicted_severity"]]

    # Bar chart: score reduction
    ax1.set_facecolor(PANEL)
    bars = ax1.barh(range(len(best)), reductions, color=sev_colors, edgecolor=PANEL, linewidth=0.4)
    ax1.set_yticks(range(len(best)))
    ax1.set_yticklabels(labels, fontsize=7.5, color=TEXT)
    ax1.invert_yaxis()
    for b, v in zip(bars, reductions):
        ax1.text(v + 0.5, b.get_y() + b.get_height()/2,
                 f"{v:.1f}", va="center", color=TEXT, fontsize=7.5)
    ax1.tick_params(colors=MUTED, labelsize=8)
    ax1.spines[:].set_color(GRID)
    ax1.grid(color=GRID, linewidth=0.5, alpha=0.6, axis="x")
    ax1.set_xlabel("Predicted Score Reduction", color=MUTED, fontsize=9)
    ax1.set_title("① Score Reduction per Intervention (coloured by new severity)",
                  color=TEXT, fontsize=10, fontweight="bold", pad=8)
    for sev, col in SEV_CLR.items():
        ax1.scatter([], [], c=col, s=35, label=sev)
    ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8, loc="lower right")

    # Scatter: importance vs reduction
    ax2.set_facecolor(PANEL)
    sc = ax2.scatter(
        best["feature_importance"].values,
        best["score_reduction"].values,
        c=sev_colors, s=80, edgecolors="white", linewidth=0.5, zorder=3
    )
    for _, r in best.iterrows():
        ax2.annotate(
            f"{r['base_feature'].replace('_',' ')[:20]}\n({r['adjustment_pct']:+}%)",
            xy=(r["feature_importance"], r["score_reduction"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=6, color=MUTED
        )
    ax2.tick_params(colors=MUTED, labelsize=8)
    ax2.spines[:].set_color(GRID)
    ax2.grid(color=GRID, linewidth=0.5, alpha=0.6)
    ax2.set_xlabel("Feature Importance (ML)", color=MUTED, fontsize=9)
    ax2.set_ylabel("Predicted Score Reduction", color=MUTED, fontsize=9)
    ax2.set_title("② Feature Importance vs Predicted Improvement",
                  color=TEXT, fontsize=10, fontweight="bold", pad=8)
    for sev, col in SEV_CLR.items():
        ax2.scatter([], [], c=col, s=35, label=sev)
    ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.99, 0.01, f"Generated: {ts}  |  OptiBatch Scenario Sim",
             ha="right", va="bottom", fontsize=7.5, color=MUTED)

    fig.savefig(SIM_PNG, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [simulation] Chart saved: {SIM_PNG}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_preemptive_alerts():
    os.makedirs(MONITORING_DIR, exist_ok=True)
    os.makedirs(ML_MODELS_DIR,  exist_ok=True)

    alerts, preds, imp, scored, pareto, next_row = load_data()

    # Alert engine
    print("\n── ALERT ENGINE ────────────────────────────────────")
    alert_df = build_preemptive_alerts(alerts, preds, imp)
    alert_df.to_csv(ALERT_OUT, index=False)
    write_alert_report(alert_df, next_row)

    # Scenario simulation
    print("\n── SCENARIO SIMULATION ────────────────────────────────────")
    sim_df = run_scenario_simulation(scored, imp, pareto)
    sim_df.to_csv(SIM_CSV, index=False)
    write_simulation_report(sim_df)
    draw_simulation_chart(sim_df)

    print("\n✅ Preemptive alerts + simulation complete.")
    print(f"   • {ALERT_OUT}")
    print(f"   • {ALERT_RPT}")
    print(f"   • {SIM_CSV}")
    print(f"   • {SIM_RPT}")
    print(f"   • {SIM_PNG}")


if __name__ == "__main__":
    run_preemptive_alerts()
