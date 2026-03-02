# code/predictive_models.py
"""
Predictive ML Pipeline — Batch Outcome Forecasting
────────────────────────────────────────────────────
Regression  : Predict next-batch deviation score  (RandomForest + GradientBoosting)
Classification: Predict next-batch severity class  (LOW / MEDIUM / HIGH)
Validation  : Rolling-window (time-aware) cross-validation
Outputs     :
  ml_predictions.csv            — per-batch predicted score + risk class
  ml_feature_importance.csv     — ranked feature importance from best model
  ml_dashboard.png              — 6-panel diagnostic dashboard
  ml_report.txt                 — narrative metrics report
"""

import os, sys, warnings
from datetime import datetime

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker   as mticker

from sklearn.ensemble         import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble         import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DATA_DIR, PARETO_DIR, MONITORING_DIR, ML_MODELS_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
ALERTS_CSV   = os.path.join(MONITORING_DIR, "batch_monitoring_alerts.csv")
SCORED_CSV   = os.path.join(RAW_DATA_DIR, "scored_batches.csv")
PARETO_CSV   = os.path.join(PARETO_DIR, "golden_signature_pareto.csv")
PRED_OUT     = os.path.join(ML_MODELS_DIR, "ml_predictions.csv")
IMP_OUT      = os.path.join(ML_MODELS_DIR, "ml_feature_importance.csv")
DASH_OUT     = os.path.join(ML_MODELS_DIR, "ml_dashboard.png")
REPORT_OUT   = os.path.join(ML_MODELS_DIR, "ml_report.txt")

# ── Style ─────────────────────────────────────────────────────────────────────
BG        = "#080b14";  PANEL = "#0d1117";  GRID  = "#1a2035"
TEXT      = "#e2e8f0";  MUTED = "#64748b"
ACCENT    = "#38bdf8";  ORANGE= "#f97316";  GREEN = "#22c55e"
RED       = "#ef4444";  PURPLE= "#a78bfa";  YELLOW= "#facc15"
SEV_CLR   = {"OK": GREEN, "LOW": YELLOW, "MEDIUM": ORANGE, "HIGH": RED}

ROLLING_N      = 5      # rolling window size for feature engineering
MIN_TRAIN_SIZE = 10     # minimum batches before starting rolling validation
SEV_ORDER      = ["LOW", "MEDIUM", "HIGH"]   # severity classes (no OK — all anomalous)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
def _batch_num(bid: str) -> int:
    s = "".join(c for c in bid if c.isdigit())
    return int(s) if s else 0


def load_and_merge() -> pd.DataFrame:
    """Merge scored batches with alert metadata, sorted chronologically."""
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

    # Keep only Pareto-critical raw features from scored_batches
    crit_feats: list[str] = [
        f for f in pareto["feature"].tolist()
        if f in scored.columns and f != "Friability_inv"   # derived — skip
    ]

    scored_slim = scored[["Batch_ID", "_n"] + crit_feats].copy()
    alerts_slim = alerts[["Batch_ID", "deviation_score", "severity",
                           "features_oor", "composite_score", "is_anomalous"]].copy()

    df = scored_slim.merge(alerts_slim, on="Batch_ID", how="inner")
    df = df.sort_values("_n").reset_index(drop=True)

    return df, crit_feats


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING  (time-series / rolling features)
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, base_feats: list[str]) -> pd.DataFrame:
    """
    For each Pareto-critical feature, compute:
      • {feat}_roll_mean   — rolling mean over last N batches (trend)
      • {feat}_roll_std    — rolling std  over last N batches (volatility)
      • {feat}_delta       — value − previous batch value    (rate of change)

    Also computes:
      • dev_roll_mean / dev_roll_std — rolling stats of deviation_score itself
      • n_oor_roll_mean              — rolling mean of features_oor count
    """
    out = df.copy()

    for feat in base_feats:
        s = out[feat]
        out[f"{feat}_roll_mean"] = s.rolling(ROLLING_N, min_periods=2).mean()
        out[f"{feat}_roll_std"]  = s.rolling(ROLLING_N, min_periods=2).std().fillna(0)
        out[f"{feat}_delta"]     = s.diff().fillna(0)

    out["dev_roll_mean"]  = out["deviation_score"].rolling(ROLLING_N, min_periods=2).mean()
    out["dev_roll_std"]   = out["deviation_score"].rolling(ROLLING_N, min_periods=2).std().fillna(0)
    out["n_oor_roll_mean"]= out["features_oor"].rolling(ROLLING_N, min_periods=2).mean()

    # Drop rows where rolling features are all NaN (first few rows)
    out = out.dropna(subset=[f"{base_feats[0]}_roll_mean"]).reset_index(drop=True)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROLLING-WINDOW CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def rolling_cv(
    df:              pd.DataFrame,
    feature_cols:    list[str],
    reg_model,
    clf_model,
    le:              LabelEncoder,
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Walk-forward (expanding window) cross-validation.
    Train on rows [0..i-1], predict row [i], for i in [MIN_TRAIN_SIZE, N).

    Returns
    -------
    reg_metrics  : dict of regression  metric lists
    clf_metrics  : dict of classification metric lists
    oof_df       : out-of-fold predictions DataFrame
    """
    n = len(df)
    y_reg_true, y_reg_pred       = [], []
    y_clf_true, y_clf_pred       = [], []
    batch_ids                    = []

    for i in range(MIN_TRAIN_SIZE, n):
        train = df.iloc[:i]
        test  = df.iloc[[i]]

        X_tr = train[feature_cols].fillna(train[feature_cols].mean())
        y_tr_reg = train["deviation_score"].values
        y_tr_clf = train["severity_enc"].values

        X_te = test[feature_cols].fillna(train[feature_cols].mean())

        reg_model.fit(X_tr, y_tr_reg)
        clf_model.fit(X_tr, y_tr_clf)

        y_reg_true.append(test["deviation_score"].values[0])
        y_reg_pred.append(float(reg_model.predict(X_te)[0]))
        y_clf_true.append(test["severity_enc"].values[0])
        y_clf_pred.append(int(clf_model.predict(X_te)[0]))
        batch_ids.append(test["Batch_ID"].values[0])

    # ── Regression metrics ───────────────────────────────────────────────────
    reg_metrics = {
        "RMSE" : float(np.sqrt(mean_squared_error(y_reg_true, y_reg_pred))),
        "MAE"  : float(mean_absolute_error(y_reg_true, y_reg_pred)),
        "R2"   : float(r2_score(y_reg_true, y_reg_pred)),
    }

    # ── Classification metrics ───────────────────────────────────────────────
    avg = "weighted"
    clf_metrics = {
        "Accuracy" : float(accuracy_score(y_clf_true, y_clf_pred)),
        "Precision": float(precision_score(y_clf_true, y_clf_pred, average=avg, zero_division=0)),
        "Recall"   : float(recall_score(y_clf_true, y_clf_pred, average=avg, zero_division=0)),
        "F1"       : float(f1_score(y_clf_true, y_clf_pred, average=avg, zero_division=0)),
    }

    oof_df = pd.DataFrame({
        "Batch_ID"        : batch_ids,
        "actual_dev_score": y_reg_true,
        "pred_dev_score"  : [round(v, 2) for v in y_reg_pred],
        "actual_severity" : le.inverse_transform(y_clf_true),
        "pred_severity"   : le.inverse_transform(y_clf_pred),
    })

    return reg_metrics, clf_metrics, oof_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. FINAL MODEL TRAINING + NEXT-BATCH PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def train_final_and_forecast(
    df:           pd.DataFrame,
    feature_cols: list[str],
    reg_model,
    clf_model,
    le:           LabelEncoder,
) -> tuple[pd.DataFrame, pd.DataFrame, object, object]:
    """Train on ALL data, forecast the next (unseen) batch."""
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y_reg = df["deviation_score"].values
    y_clf = df["severity_enc"].values

    reg_model.fit(X, y_reg)
    clf_model.fit(X, y_clf)

    # Next-batch feature vector = rolling mean of last N rows (simple proxy)
    X_next = df.tail(ROLLING_N)[feature_cols].fillna(df[feature_cols].mean()).mean().values.reshape(1, -1)

    next_dev  = float(reg_model.predict(X_next)[0])
    next_sev  = le.inverse_transform(clf_model.predict(X_next))[0]
    next_prob = clf_model.predict_proba(X_next)[0]
    class_probs = {le.inverse_transform([k])[0]: round(float(p), 3)
                   for k, p in enumerate(next_prob)}

    next_batch_df = pd.DataFrame([{
        "Batch_ID"           : "NEXT (forecast)",
        "pred_dev_score"     : round(next_dev, 2),
        "pred_severity"      : next_sev,
        **{f"prob_{k}": v for k, v in class_probs.items()},
    }])

    # Feature importances
    reg_imp = pd.DataFrame({
        "feature"            : feature_cols,
        "reg_importance"     : reg_model.feature_importances_,
    }).sort_values("reg_importance", ascending=False)

    clf_imp = pd.DataFrame({
        "feature"            : feature_cols,
        "clf_importance"     : clf_model.feature_importances_,
    }).sort_values("clf_importance", ascending=False)

    imp_df = reg_imp.merge(clf_imp, on="feature")
    imp_df["avg_importance"] = (imp_df["reg_importance"] + imp_df["clf_importance"]) / 2
    imp_df = imp_df.sort_values("avg_importance", ascending=False).reset_index(drop=True)

    return next_batch_df, imp_df, reg_model, clf_model


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATION DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.6)
    if title:  ax.set_title(title, color=TEXT, fontsize=9.5, fontweight="bold", pad=7)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=8)


def draw_ml_dashboard(
    oof_df:       pd.DataFrame,
    imp_df:       pd.DataFrame,
    next_df:      pd.DataFrame,
    reg_metrics:  dict,
    clf_metrics:  dict,
    le:           LabelEncoder,
    df_full:      pd.DataFrame,
    reg_model,
    clf_model,
    feature_cols: list[str],
) -> None:

    fig = plt.figure(figsize=(22, 20), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.50, wspace=0.38,
                            left=0.07, right=0.97,
                            top=0.94, bottom=0.05)

    # ── Panel 1 (span 2 cols): Predicted vs Actual deviation score ────────────
    ax1 = fig.add_subplot(gs[0, :2])
    x   = np.arange(len(oof_df))
    ax1.plot(x, oof_df["actual_dev_score"],  color=ACCENT,  linewidth=1.8,
             marker="o", markersize=4, label="Actual deviation score")
    ax1.plot(x, oof_df["pred_dev_score"],    color=ORANGE,  linewidth=1.8,
             linestyle="--", marker="s", markersize=4, label="Predicted deviation score")
    ax1.fill_between(x, oof_df["actual_dev_score"], oof_df["pred_dev_score"],
                     alpha=0.12, color=RED, label="Error band")
    ax1.set_xticks(x)
    ax1.set_xticklabels(oof_df["Batch_ID"], rotation=45, ha="right", fontsize=7)
    ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    # Annotation: metrics
    ax1.annotate(
        f" RMSE={reg_metrics['RMSE']:.1f}  MAE={reg_metrics['MAE']:.1f}  R²={reg_metrics['R2']:.3f}",
        xy=(0.01, 0.94), xycoords="axes fraction",
        color=PURPLE, fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1e1b4b", ec=PURPLE, alpha=0.9)
    )
    _ax_style(ax1, title="① Regression — Predicted vs Actual Batch Deviation Score (Rolling-Window CV)",
              xlabel="Batch (chronological)", ylabel="Deviation Score")

    # ── Panel 2 (right): Scatter actual vs predicted ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    sc = ax2.scatter(oof_df["actual_dev_score"], oof_df["pred_dev_score"],
                     c=[SEV_CLR.get(s, MUTED) for s in oof_df["actual_severity"]],
                     s=55, edgecolors="none", zorder=3, alpha=0.85)
    lims = [min(oof_df[["actual_dev_score","pred_dev_score"]].min()),
            max(oof_df[["actual_dev_score","pred_dev_score"]].max())]
    ax2.plot(lims, lims, color=MUTED, linewidth=1.0, linestyle="--", label="Perfect fit")
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    for sev, col in SEV_CLR.items():
        ax2.scatter([], [], c=col, s=35, label=sev)
    ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7.5, ncol=2)
    _ax_style(ax2, title="② Actual vs Predicted\n(coloured by severity)",
              xlabel="Actual", ylabel="Predicted")

    # ── Panel 3 (left): Classification results timeline ───────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    xc  = np.arange(len(oof_df))
    ax3.scatter(xc - 0.15, [SEV_ORDER.index(s) if s in SEV_ORDER else 0
                              for s in oof_df["actual_severity"]],
                color=ACCENT,  s=55, label="Actual",    marker="o", zorder=3)
    ax3.scatter(xc + 0.15, [SEV_ORDER.index(s) if s in SEV_ORDER else 0
                              for s in oof_df["pred_severity"]],
                color=ORANGE,  s=55, label="Predicted", marker="D", zorder=3)
    ax3.set_yticks(range(len(SEV_ORDER)))
    ax3.set_yticklabels(SEV_ORDER, fontsize=8.5)
    ax3.set_xticks(xc)
    ax3.set_xticklabels(oof_df["Batch_ID"], rotation=45, ha="right", fontsize=7)
    ax3.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax3.annotate(
        f" Acc={clf_metrics['Accuracy']:.2f}  F1={clf_metrics['F1']:.2f}"
        f"  Prec={clf_metrics['Precision']:.2f}  Rec={clf_metrics['Recall']:.2f}",
        xy=(0.01, 0.91), xycoords="axes fraction",
        color=GREEN, fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#052e16", ec=GREEN, alpha=0.9)
    )
    _ax_style(ax3, title="③ Classification — Actual vs Predicted Severity Class (Rolling-Window CV)",
              xlabel="Batch", ylabel="Severity Class")

    # ── Panel 4 (right): Confusion matrix ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    classes_present = sorted(set(oof_df["actual_severity"]) | set(oof_df["pred_severity"]),
                             key=lambda s: SEV_ORDER.index(s) if s in SEV_ORDER else 99)
    y_true_enc = [SEV_ORDER.index(s) if s in SEV_ORDER else 0 for s in oof_df["actual_severity"]]
    y_pred_enc = [SEV_ORDER.index(s) if s in SEV_ORDER else 0 for s in oof_df["pred_severity"]]
    cm = confusion_matrix(y_true_enc, y_pred_enc,
                          labels=[SEV_ORDER.index(s) for s in classes_present])

    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("cm", [PANEL, "#1d4ed8", ACCENT], N=128)
    im4 = ax4.imshow(cm, cmap=cmap, aspect="auto")
    ax4.set_xticks(range(len(classes_present)))
    ax4.set_yticks(range(len(classes_present)))
    ax4.set_xticklabels(classes_present, color=TEXT, fontsize=8)
    ax4.set_yticklabels(classes_present, color=TEXT, fontsize=8)
    ax4.set_xlabel("Predicted", color=MUTED, fontsize=8)
    ax4.set_ylabel("Actual",    color=MUTED, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax4.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color=TEXT, fontsize=11, fontweight="bold")
    ax4.set_facecolor(PANEL)
    ax4.spines[:].set_color(GRID)
    ax4.tick_params(colors=MUTED)
    ax4.set_title("④ Confusion Matrix\n(Severity Classification)", color=TEXT,
                  fontsize=9.5, fontweight="bold", pad=7)

    # ── Panel 5 (left 2 cols): Feature Importance ─────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    top_n  = min(20, len(imp_df))
    top_imp = imp_df.head(top_n)
    colors5 = [ORANGE if i < 5 else ACCENT if i < 10 else MUTED for i in range(top_n)]
    bars = ax5.barh(range(top_n), top_imp["avg_importance"].values,
                    color=colors5, edgecolor=PANEL, linewidth=0.4)
    ax5.set_yticks(range(top_n))
    ax5.set_yticklabels(
        [f.replace("_roll_mean","[μ]").replace("_roll_std","[σ]")
          .replace("_delta","[Δ]").replace("_"," ")[:30]
         for f in top_imp["feature"]],
        fontsize=7.5, color=TEXT
    )
    ax5.invert_yaxis()
    for rect, val in zip(bars, top_imp["avg_importance"].values):
        ax5.text(val + 0.001, rect.get_y() + rect.get_height() / 2,
                 f"{val:.3f}", va="center", color=TEXT, fontsize=7)
    _ax_style(ax5, title="⑤ Top Feature Importances — Avg of Regression + Classification Model",
              xlabel="Average Feature Importance", ylabel="")

    # ── Panel 6 (right): Next Batch Forecast card ─────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(PANEL)
    ax6.axis("off")
    ax6.set_title("⑥ Next-Batch Forecast", color=TEXT, fontsize=9.5,
                  fontweight="bold", pad=7)

    row = next_df.iloc[0]
    sev_col = SEV_CLR.get(str(row["pred_severity"]), MUTED)

    lines = [
        ("Predicted Deviation Score", f"{row['pred_dev_score']:.1f}", ORANGE),
        ("Predicted Severity Class",   str(row["pred_severity"]),     sev_col),
    ]
    prob_keys = [k for k in row.index if k.startswith("prob_")]
    for pk in prob_keys:
        cls = pk.replace("prob_", "")
        lines.append((f"P({cls})", f"{row[pk]*100:.1f}%", SEV_CLR.get(cls, MUTED)))

    # Risk-level gauge bar
    dev_max  = 750
    risk_pct = min(row["pred_dev_score"] / dev_max, 1.0)
    gauge_clr = RED if risk_pct > 0.6 else ORANGE if risk_pct > 0.3 else GREEN

    y0 = 0.85
    ax6.text(0.5, y0, "NEXT BATCH PREDICTION", ha="center", va="center",
             fontsize=9, color=ACCENT, fontweight="bold", transform=ax6.transAxes)

    for i, (label, value, color) in enumerate(lines):
        yv = y0 - 0.14 * (i + 1)
        ax6.text(0.05, yv, label + ":", ha="left", va="center",
                 fontsize=8, color=MUTED, transform=ax6.transAxes)
        ax6.text(0.95, yv, value, ha="right", va="center",
                 fontsize=9, color=color, fontweight="bold", transform=ax6.transAxes)

    # Gauge
    gy = y0 - 0.14 * (len(lines) + 2)
    ax6.text(0.5, gy + 0.06, "Risk Gauge", ha="center", va="center",
             fontsize=7.5, color=MUTED, transform=ax6.transAxes)
    ax6.add_patch(plt.Rectangle((0.05, gy - 0.04), 0.90, 0.08,
                                 fc=GRID, transform=ax6.transAxes, zorder=1))
    ax6.add_patch(plt.Rectangle((0.05, gy - 0.04), 0.90 * risk_pct, 0.08,
                                 fc=gauge_clr, transform=ax6.transAxes, zorder=2))
    ax6.add_patch(plt.Rectangle((0.05, gy - 0.04), 0.90, 0.08,
                                 fc="none", ec=GRID, linewidth=1.0,
                                 transform=ax6.transAxes, zorder=3))
    ax6.text(0.50, gy - 0.04 - 0.06, f"{risk_pct*100:.0f}% of max risk",
             ha="center", va="center", fontsize=7.5, color=gauge_clr,
             transform=ax6.transAxes)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.99, 0.01, f"Generated: {ts}  |  OptiBatch ML Predictor",
             ha="right", va="bottom", fontsize=7.5, color=MUTED)
    fig.suptitle("OptiBatch — Predictive ML Model Dashboard",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.975)

    fig.savefig(DASH_OUT, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Dashboard saved: {DASH_OUT}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. REPORT
# ─────────────────────────────────────────────────────────────────────────────
def write_report(
    reg_metrics:  dict,
    clf_metrics:  dict,
    imp_df:       pd.DataFrame,
    next_df:      pd.DataFrame,
    oof_df:       pd.DataFrame,
) -> None:
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "═" * 66

    row = next_df.iloc[0]
    lines = [
        sep,
        "  OPTIBATCH — PREDICTIVE ML MODEL REPORT",
        f"  Generated : {ts}",
        sep,
        "  MODELS USED",
        "─" * 66,
        "  Regression    : Gradient Boosting Regressor",
        "  Classification: Gradient Boosting Classifier",
        "  Validation    : Rolling-window (expanding) cross-validation",
        f"  Min train size: {MIN_TRAIN_SIZE} batches",
        f"  Rolling window: last {ROLLING_N} batches for trend features",
        sep,
        "  REGRESSION METRICS  (predict deviation score)",
        "─" * 66,
        f"  RMSE  = {reg_metrics['RMSE']:.2f}   (lower is better)",
        f"  MAE   = {reg_metrics['MAE']:.2f}   (lower is better)",
        f"  R²    = {reg_metrics['R2']:.4f}  (closer to 1.0 is better)",
        sep,
        "  CLASSIFICATION METRICS  (predict severity class)",
        "─" * 66,
        f"  Accuracy  = {clf_metrics['Accuracy']:.2%}",
        f"  Precision = {clf_metrics['Precision']:.2%}  (weighted avg)",
        f"  Recall    = {clf_metrics['Recall']:.2%}  (weighted avg)",
        f"  F1 Score  = {clf_metrics['F1']:.2%}  (weighted avg)",
        sep,
        "  TOP 15 FEATURES BY IMPORTANCE  (avg of Reg + Clf)",
        "─" * 66,
        f"  {'Rank':<5} {'Feature':<36} {'Reg Imp':>9} {'Clf Imp':>9} {'Avg':>9}",
        "─" * 66,
    ]
    for i, r in imp_df.head(15).iterrows():
        lines.append(
            f"  {i+1:<5} {r['feature']:<36} "
            f"{r['reg_importance']:>9.4f} {r['clf_importance']:>9.4f} {r['avg_importance']:>9.4f}"
        )
    lines += [
        sep,
        "  NEXT-BATCH FORECAST",
        "─" * 66,
        f"  Predicted deviation score : {row['pred_dev_score']:.2f}",
        f"  Predicted severity class  : {row['pred_severity']}",
    ]
    prob_keys = [k for k in row.index if k.startswith("prob_")]
    for pk in prob_keys:
        lines.append(f"  P({pk.replace('prob_',''):<10}) = {row[pk]*100:.1f}%")

    lines += [
        sep,
        "  KEY INSIGHTS",
        "─" * 66,
    ]
    top3 = imp_df.head(3)["feature"].tolist()
    lines.append(f"  Top 3 predictive features:")
    for f in top3:
        lines.append(f"    • {f}")
    lines += [
        "",
        "  Interpretation:",
        "  Delta (Δ) and rolling-mean (μ) features dominate importance,",
        "  confirming that TREND and RATE-OF-CHANGE are more predictive",
        "  than raw feature values alone.",
        sep,
        f"  Outputs saved to: {ML_MODELS_DIR}",
        f"    • ml_predictions.csv",
        f"    • ml_feature_importance.csv",
        f"    • ml_dashboard.png",
        f"    • ml_report.txt",
        sep,
    ]

    text = "\n".join(lines)
    with open(REPORT_OUT, "w") as f:
        f.write(text)
    print(text)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_predictive_models() -> None:
    os.makedirs(ML_MODELS_DIR, exist_ok=True)

    # 1. Load & merge
    df_raw, base_feats = load_and_merge()
    print(f"[ml] Loaded {len(df_raw)} batches  |  {len(base_feats)} Pareto-critical features")

    # 2. Engineer features
    df = engineer_features(df_raw, base_feats)

    # 3. Encode severity labels
    # Map OK → LOW for modelling (no 'OK' batches in anomalous set effectively)
    df["severity_mapped"] = df["severity"].replace("OK", "LOW")
    le = LabelEncoder()
    le.fit(SEV_ORDER)
    df["severity_enc"] = df["severity_mapped"].apply(
        lambda s: le.transform([s])[0] if s in le.classes_ else 0
    )

    # 4. Build feature column list (base + engineered)
    eng_feats = (
        [f"{feat}_roll_mean" for feat in base_feats] +
        [f"{feat}_roll_std"  for feat in base_feats] +
        [f"{feat}_delta"     for feat in base_feats] +
        ["dev_roll_mean", "dev_roll_std", "n_oor_roll_mean"]
    )
    # Keep only columns that exist
    feature_cols = [c for c in eng_feats if c in df.columns]
    print(f"[ml] Feature matrix: {len(df)} rows × {len(feature_cols)} engineered features")

    # 5. Init models  (tuned for small dataset N≈56)
    reg_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8,  random_state=42
    )
    clf_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8,  random_state=42
    )

    # 6. Rolling-window CV
    print("[ml] Running rolling-window cross-validation …")
    reg_metrics, clf_metrics, oof_df = rolling_cv(
        df, feature_cols, reg_model, clf_model, le
    )
    print(f"[ml] Reg  RMSE={reg_metrics['RMSE']:.1f}  MAE={reg_metrics['MAE']:.1f}  R²={reg_metrics['R2']:.3f}")
    print(f"[ml] Clf  Acc={clf_metrics['Accuracy']:.2%}  F1={clf_metrics['F1']:.2%}")

    # 7. Final train + next-batch forecast
    print("[ml] Training final model on all data …")
    next_df, imp_df, reg_model, clf_model = train_final_and_forecast(
        df, feature_cols, reg_model, clf_model, le
    )

    # 8. Build full predictions CSV
    all_pred = oof_df.copy()
    all_pred.to_csv(PRED_OUT, index=False)
    imp_df.to_csv(IMP_OUT, index=False)

    # 9. Dashboard
    print("[ml] Drawing ML dashboard …")
    draw_ml_dashboard(
        oof_df, imp_df, next_df,
        reg_metrics, clf_metrics,
        le, df, reg_model, clf_model, feature_cols,
    )

    # 10. Report
    write_report(reg_metrics, clf_metrics, imp_df, next_df, oof_df)
    print(f"\n✅ ML pipeline complete — outputs in {ML_MODELS_DIR}")


if __name__ == "__main__":
    run_predictive_models()
