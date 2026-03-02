# code/feedback_loop.py
"""
Continuous Feedback Loop — Retraining, Golden Signature Update, Performance Tracking
──────────────────────────────────────────────────────────────────────────────────────
1. RETRAINING TRIGGER
   • Loads latest actual batch outcomes.
   • Compares last N actual vs predicted scores / severities.
   • If accuracy drops below threshold → retrains both models.
   • Logs retraining events (timestamp, trigger reason, before/after metrics).

2. GOLDEN SIGNATURE UPDATE
   • After retraining, if new top-quartile batches show improved scores,
     updates the golden signature mean/std with the new top performers.
   • Logs golden signature adjustments (which features changed, by how much).

3. MODEL PERFORMANCE TRACKING
   • Appends cumulative performance log (per run: RMSE, MAE, R², Accuracy, F1).
   • Flags performance degradation if rolling accuracy < threshold.
   • Generates a performance trend chart.

Outputs:
  outputs/ml_models/retraining_log.csv
  outputs/ml_models/model_performance_log.csv
  outputs/ml_models/performance_trend.png
  outputs/raw_batches/golden_signature_mean_updated.csv   (if updated)
  outputs/raw_batches/golden_signature_std_updated.csv    (if updated)
  outputs/ml_models/golden_signature_change_log.csv
"""

import os, sys, warnings, json
from datetime import datetime

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble      import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics       import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MONITORING_DIR, ML_MODELS_DIR, RAW_DATA_DIR, PARETO_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
ALERTS_CSV    = os.path.join(MONITORING_DIR, "batch_monitoring_alerts.csv")
SCORED_CSV    = os.path.join(RAW_DATA_DIR,   "scored_batches.csv")
PARETO_CSV    = os.path.join(PARETO_DIR,     "golden_signature_pareto.csv")
IMP_CSV       = os.path.join(ML_MODELS_DIR,  "ml_feature_importance.csv")
PREDS_CSV     = os.path.join(ML_MODELS_DIR,  "ml_predictions.csv")
MEAN_CSV      = os.path.join(RAW_DATA_DIR,   "golden_signature_mean.csv")
STD_CSV       = os.path.join(RAW_DATA_DIR,   "golden_signature_std.csv")

RETRAIN_LOG   = os.path.join(ML_MODELS_DIR,  "retraining_log.csv")
PERF_LOG      = os.path.join(ML_MODELS_DIR,  "model_performance_log.csv")
PERF_CHART    = os.path.join(ML_MODELS_DIR,  "performance_trend.png")
GS_MEAN_UPD   = os.path.join(RAW_DATA_DIR,   "golden_signature_mean_updated.csv")
GS_STD_UPD    = os.path.join(RAW_DATA_DIR,   "golden_signature_std_updated.csv")
GS_CHANGE_LOG = os.path.join(ML_MODELS_DIR,  "golden_signature_change_log.csv")

# ── Thresholds ────────────────────────────────────────────────────────────────
ROLLING_N        = 5
MIN_TRAIN_SIZE   = 10
ACC_THRESHOLD    = 0.70    # Retrain if rolling accuracy drops below this
R2_THRESHOLD     = 0.50    # Retrain if R² drops below this
GOLDEN_PCTILE    = 0.85    # Top 85th percentile = golden batches
SEV_ORDER        = ["LOW", "MEDIUM", "HIGH"]

# ── Palette ───────────────────────────────────────────────────────────────────
BG = "#07090f"; PANEL = "#0d1117"; GRID = "#1a2035"
TEXT = "#e2e8f0"; MUTED = "#64748b"
ACCENT = "#38bdf8"; ORANGE = "#f97316"; GREEN = "#22c55e"
RED = "#ef4444"; PURPLE = "#a78bfa"; YELLOW = "#facc15"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _batch_num(bid: str) -> int:
    s = "".join(c for c in str(bid) if c.isdigit())
    return int(s) if s else 0


def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.6)
    if title:  ax.set_title(title, color=TEXT, fontsize=9.5, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=8)


# ── 1. Load + Engineer Features ───────────────────────────────────────────────
def load_and_engineer():
    for p in [ALERTS_CSV, SCORED_CSV, PARETO_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    alerts = pd.read_csv(ALERTS_CSV)
    scored = pd.read_csv(SCORED_CSV)
    pareto = pd.read_csv(PARETO_CSV)

    alerts["_n"] = alerts["Batch_ID"].apply(_batch_num)
    scored["_n"] = scored["Batch_ID"].apply(_batch_num)
    alerts = alerts.sort_values("_n").reset_index(drop=True)
    scored = scored.sort_values("_n").reset_index(drop=True)

    crit_feats = [f for f in pareto["feature"].tolist()
                  if f in scored.columns and f != "Friability_inv"]

    scored_slim = scored[["Batch_ID", "_n"] + crit_feats].copy()
    alerts_slim = alerts[["Batch_ID", "deviation_score", "severity",
                           "features_oor", "composite_score"]].copy()
    df = scored_slim.merge(alerts_slim, on="Batch_ID", how="inner")
    df = df.sort_values("_n").reset_index(drop=True)

    # Engineer features
    for feat in crit_feats:
        s = df[feat]
        df[f"{feat}_roll_mean"] = s.rolling(ROLLING_N, min_periods=2).mean()
        df[f"{feat}_roll_std"]  = s.rolling(ROLLING_N, min_periods=2).std().fillna(0)
        df[f"{feat}_delta"]     = s.diff().fillna(0)

    df["dev_roll_mean"]   = df["deviation_score"].rolling(ROLLING_N, min_periods=2).mean()
    df["dev_roll_std"]    = df["deviation_score"].rolling(ROLLING_N, min_periods=2).std().fillna(0)
    df["n_oor_roll_mean"] = df["features_oor"].rolling(ROLLING_N, min_periods=2).mean()

    df = df.dropna(subset=[f"{crit_feats[0]}_roll_mean"]).reset_index(drop=True)

    # Encode severity
    df["severity_mapped"] = df["severity"].replace("OK", "LOW")
    le = LabelEncoder()
    le.fit(SEV_ORDER)
    df["severity_enc"] = df["severity_mapped"].apply(
        lambda s: le.transform([s])[0] if s in le.classes_ else 0
    )

    eng_feats = (
        [f"{f}_roll_mean" for f in crit_feats] +
        [f"{f}_roll_std"  for f in crit_feats] +
        [f"{f}_delta"     for f in crit_feats] +
        ["dev_roll_mean", "dev_roll_std", "n_oor_roll_mean"]
    )
    feature_cols = [c for c in eng_feats if c in df.columns]

    return df, feature_cols, le, crit_feats, scored, pareto


# ── 2. Evaluate Current Models ────────────────────────────────────────────────
def evaluate_current_performance(df, feature_cols, le):
    """Walk-forward CV on all available data to get current performance."""
    n = len(df)
    y_reg_true, y_reg_pred = [], []
    y_clf_true, y_clf_pred = [], []

    reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                     learning_rate=0.1, subsample=0.8, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                      learning_rate=0.1, subsample=0.8, random_state=42)

    for i in range(MIN_TRAIN_SIZE, n):
        train = df.iloc[:i]
        test  = df.iloc[[i]]
        X_tr  = train[feature_cols].fillna(train[feature_cols].mean())
        X_te  = test[feature_cols].fillna(train[feature_cols].mean())
        reg.fit(X_tr, train["deviation_score"])
        clf.fit(X_tr, train["severity_enc"])
        y_reg_true.append(test["deviation_score"].values[0])
        y_reg_pred.append(float(reg.predict(X_te)[0]))
        y_clf_true.append(test["severity_enc"].values[0])
        y_clf_pred.append(int(clf.predict(X_te)[0]))

    rmse = float(np.sqrt(mean_squared_error(y_reg_true, y_reg_pred)))
    mae  = float(mean_absolute_error(y_reg_true, y_reg_pred))
    r2   = float(r2_score(y_reg_true, y_reg_pred))
    acc  = float(accuracy_score(y_clf_true, y_clf_pred))
    f1   = float(f1_score(y_clf_true, y_clf_pred, average="weighted", zero_division=0))
    prec = float(precision_score(y_clf_true, y_clf_pred, average="weighted", zero_division=0))
    rec  = float(recall_score(y_clf_true, y_clf_pred, average="weighted", zero_division=0))

    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2,
               "Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec}

    return metrics, reg, clf


# ── 3. Log Performance ────────────────────────────────────────────────────────
def log_performance(metrics: dict, trigger: str):
    row = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trigger":      trigger,
        "RMSE":         round(metrics["RMSE"], 3),
        "MAE":          round(metrics["MAE"], 3),
        "R2":           round(metrics["R2"], 4),
        "Accuracy":     round(metrics["Accuracy"], 4),
        "F1":           round(metrics["F1"], 4),
        "Precision":    round(metrics["Precision"], 4),
        "Recall":       round(metrics["Recall"], 4),
        "acc_flag":     "⚠ DEGRADED" if metrics["Accuracy"] < ACC_THRESHOLD else "✅ OK",
        "r2_flag":      "⚠ DEGRADED" if metrics["R2"] < R2_THRESHOLD else "✅ OK",
    }
    if os.path.exists(PERF_LOG):
        perf_df = pd.read_csv(PERF_LOG)
        perf_df = pd.concat([perf_df, pd.DataFrame([row])], ignore_index=True)
    else:
        perf_df = pd.DataFrame([row])
    perf_df.to_csv(PERF_LOG, index=False)
    return perf_df


# ── 4. Retrain Decision ───────────────────────────────────────────────────────
def should_retrain(perf_df: pd.DataFrame) -> tuple[bool, str]:
    if len(perf_df) < 2:
        return False, "not enough history"
    last = perf_df.iloc[-1]
    if float(last["Accuracy"]) < ACC_THRESHOLD:
        return True, f"Accuracy={last['Accuracy']:.3f} < threshold={ACC_THRESHOLD}"
    if float(last["R2"]) < R2_THRESHOLD:
        return True, f"R²={last['R2']:.3f} < threshold={R2_THRESHOLD}"
    prev = perf_df.iloc[-2]
    if float(last["Accuracy"]) < float(prev["Accuracy"]) - 0.05:
        return True, f"Accuracy dropped {float(prev['Accuracy']):.3f} → {float(last['Accuracy']):.3f}"
    return False, "performance stable"


# ── 5. Log Retraining Event ───────────────────────────────────────────────────
def log_retrain(trigger_reason: str, before: dict, after: dict):
    row = {
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trigger_reason":    trigger_reason,
        "before_RMSE":       round(before["RMSE"], 3),
        "after_RMSE":        round(after["RMSE"], 3),
        "before_R2":         round(before["R2"], 4),
        "after_R2":          round(after["R2"], 4),
        "before_Accuracy":   round(before["Accuracy"], 4),
        "after_Accuracy":    round(after["Accuracy"], 4),
        "before_F1":         round(before["F1"], 4),
        "after_F1":          round(after["F1"], 4),
        "improvement_RMSE":  round(before["RMSE"] - after["RMSE"], 3),
        "improvement_Acc":   round(after["Accuracy"] - before["Accuracy"], 4),
    }
    if os.path.exists(RETRAIN_LOG):
        df = pd.read_csv(RETRAIN_LOG)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(RETRAIN_LOG, index=False)
    print(f"  [feedback] Retraining logged. ΔAcc={row['improvement_Acc']:+.4f}  ΔRMSE={row['improvement_RMSE']:+.3f}")


# ── 6. Update Golden Signature ────────────────────────────────────────────────
def update_golden_signature(scored, crit_feats, pareto):
    """Recompute golden mean/std from updated top-scoring batches."""
    if "composite_score" not in scored.columns:
        print("  [feedback] composite_score not in scored CSV — skipping GS update.")
        return None, None

    threshold = scored["composite_score"].quantile(GOLDEN_PCTILE)
    top_batches = scored[scored["composite_score"] >= threshold]

    if len(top_batches) < 3:
        print("  [feedback] Too few golden batches for update.")
        return None, None

    old_mean = pd.read_csv(MEAN_CSV, index_col=0).iloc[:, 0] if os.path.exists(MEAN_CSV) else pd.Series()
    old_std  = pd.read_csv(STD_CSV,  index_col=0).iloc[:, 0] if os.path.exists(STD_CSV)  else pd.Series()

    valid_feats = [f for f in crit_feats if f in top_batches.columns]
    new_mean = top_batches[valid_feats].mean()
    new_std  = top_batches[valid_feats].std().fillna(0)

    # Save updated GS files
    new_mean.to_csv(GS_MEAN_UPD, header=["mean"])
    new_std.to_csv(GS_STD_UPD,   header=["std"])

    # Log changes
    change_rows = []
    for feat in valid_feats:
        old_m = float(old_mean.get(feat, np.nan))
        new_m = float(new_mean.get(feat, np.nan))
        old_s = float(old_std.get(feat, np.nan))
        new_s = float(new_std.get(feat, np.nan))
        change_rows.append({
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feature":     feat,
            "old_mean":    round(old_m, 4),
            "new_mean":    round(new_m, 4),
            "mean_change": round(new_m - old_m, 4) if not np.isnan(old_m) else "N/A",
            "old_std":     round(old_s, 4),
            "new_std":     round(new_s, 4),
            "std_change":  round(new_s - old_s, 4) if not np.isnan(old_s) else "N/A",
        })

    change_df = pd.DataFrame(change_rows)
    if os.path.exists(GS_CHANGE_LOG):
        existing = pd.read_csv(GS_CHANGE_LOG)
        change_df = pd.concat([existing, change_df], ignore_index=True)
    change_df.to_csv(GS_CHANGE_LOG, index=False)

    print(f"  [feedback] Golden Signature updated from {len(top_batches)} top-scoring batches.")
    print(f"  [feedback] Saved: {GS_MEAN_UPD}")
    print(f"  [feedback] Saved: {GS_STD_UPD}")
    print(f"  [feedback] Change log: {GS_CHANGE_LOG}")

    return new_mean, new_std


# ── 7. Performance Trend Chart ─────────────────────────────────────────────────
def draw_performance_chart(perf_df: pd.DataFrame):
    if len(perf_df) < 2:
        print("  [feedback] Not enough log entries for performance chart yet.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
    fig.suptitle("OptiBatch — Model Performance Tracking Over Time",
                 fontsize=14, fontweight="bold", color=TEXT, y=0.99)
    axes = axes.flatten()

    x     = np.arange(len(perf_df))
    runs  = perf_df["timestamp"].str[5:16].tolist()   # MM-DD HH:MM

    metrics = [
        ("R2",       "R² Score",        ACCENT,  R2_THRESHOLD,  "R² Threshold"),
        ("RMSE",     "RMSE",            ORANGE,  None,          None),
        ("Accuracy", "Accuracy",        GREEN,   ACC_THRESHOLD, "Acc Threshold"),
        ("F1",       "F1 Score (wtd)",  PURPLE,  None,          None),
    ]
    for ax, (col, label, color, thresh, thresh_label) in zip(axes, metrics):
        ax.set_facecolor(PANEL)
        ax.plot(x, perf_df[col].values, color=color, linewidth=2.2,
                marker="o", markersize=5, zorder=3, label=label)
        if thresh is not None:
            ax.axhline(thresh, color=RED, linewidth=1.0, linestyle="--",
                       alpha=0.8, label=thresh_label)
        ax.fill_between(x, perf_df[col].values,
                        alpha=0.12, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=7)
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines[:].set_color(GRID)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.6)
        ax.set_title(f"{label} over Pipeline Runs", color=TEXT, fontsize=9.5,
                     fontweight="bold", pad=7)
        ax.set_ylabel(label, color=MUTED, fontsize=8)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.99, 0.01, f"Generated: {ts}  |  OptiBatch Feedback Loop",
             ha="right", va="bottom", fontsize=7.5, color=MUTED)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(PERF_CHART, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [feedback] Performance chart: {PERF_CHART}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_feedback_loop():
    os.makedirs(ML_MODELS_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR,  exist_ok=True)

    print("\n── LOADING + ENGINEERING FEATURES ─────────────────────")
    df, feature_cols, le, crit_feats, scored, pareto = load_and_engineer()

    print(f"  [feedback] Loaded {len(df)} batches | {len(feature_cols)} features")

    print("\n── EVALUATING MODEL PERFORMANCE ────────────────────────")
    metrics, reg_model, clf_model = evaluate_current_performance(df, feature_cols, le)
    print(f"  [feedback] RMSE={metrics['RMSE']:.2f}  R²={metrics['R2']:.3f}  "
          f"Acc={metrics['Accuracy']:.2%}  F1={metrics['F1']:.2%}")

    # Log performance
    perf_df = log_performance(metrics, trigger="scheduled_run")

    # Check if retraining needed
    do_retrain, reason = should_retrain(perf_df)
    print(f"\n── RETRAIN CHECK ───────────────────────────────────────")
    print(f"  [feedback] Retrain needed: {do_retrain}  |  Reason: {reason}")

    if do_retrain:
        before_metrics = metrics.copy()
        print("  [feedback] Retraining models on latest data …")
        reg_model.fit(df[feature_cols].fillna(df[feature_cols].mean()),
                      df["deviation_score"])
        clf_model.fit(df[feature_cols].fillna(df[feature_cols].mean()),
                      df["severity_enc"])
        after_metrics, _, _ = evaluate_current_performance(df, feature_cols, le)
        log_retrain(reason, before_metrics, after_metrics)
        log_performance(after_metrics, trigger="retrain")
        perf_df = pd.read_csv(PERF_LOG)
    else:
        print("  [feedback] Models are performing within thresholds — no retraining required.")

    # Golden Signature update
    print("\n── GOLDEN SIGNATURE UPDATE ─────────────────────────────")
    update_golden_signature(scored, crit_feats, pareto)

    # Performance Chart
    print("\n── PERFORMANCE TREND CHART ─────────────────────────────")
    draw_performance_chart(perf_df)

    # Summary
    sep = "═" * 60
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{sep}")
    print(f"  OPTIBATCH FEEDBACK LOOP SUMMARY  [{ts}]")
    print(sep)
    print(f"  RMSE      : {metrics['RMSE']:.2f}")
    print(f"  R²        : {metrics['R2']:.4f}  {'⚠ DEGRADED' if metrics['R2'] < R2_THRESHOLD else '✅ OK'}")
    print(f"  Accuracy  : {metrics['Accuracy']:.2%}  {'⚠ DEGRADED' if metrics['Accuracy'] < ACC_THRESHOLD else '✅ OK'}")
    print(f"  F1        : {metrics['F1']:.2%}")
    print(f"  Retrained : {'Yes — ' + reason if do_retrain else 'No (stable)'}")
    print(sep)
    print(f"  Outputs saved to:")
    print(f"    • {RETRAIN_LOG}")
    print(f"    • {PERF_LOG}")
    print(f"    • {PERF_CHART}")
    print(f"    • {GS_MEAN_UPD}")
    print(f"    • {GS_STD_UPD}")
    print(f"    • {GS_CHANGE_LOG}")
    print(sep)


if __name__ == "__main__":
    run_feedback_loop()
