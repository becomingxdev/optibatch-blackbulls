# code/advanced_ml.py
"""
Advanced ML Pipeline — Enhanced Batch Severity Classification & Deviation Regression
═══════════════════════════════════════════════════════════════════════════════════════
Target: Classification accuracy > 92%  (HIGH / MEDIUM / LOW)

Strategy
────────
1. Rich Feature Engineering
   • Δ, rolling μ, rolling σ, cumulative trend for every Pareto-critical feature
   • Pairwise interaction features (ΔA × ΔB) for top-5 Pareto features
   • Ratio features (feature / golden_mean) — deviation normalised
   • Batch-position features (batch index, batch-n ÷ total)

2. Class Imbalance  (HIGH=45, MEDIUM=7, LOW+OK=4)
   • Merge OK → LOW  →  3-class problem
   • Per-fold SMOTE  (applied only inside training split)
   • class_weight="balanced" in all base learners

3. Ensemble
   • Base: GradientBoosting | XGBoost | LightGBM  (all tuned with class weights)
   • Meta-learner: Logistic Regression (C=1, balanced)
   • Stacking via out-of-fold predictions (respects time order)

4. Validation: Expanding-window rolling CV (time-aware)

5. Outputs
   • outputs/ml_models/adv_predictions.csv
   • outputs/ml_models/adv_feature_importance.csv
   • outputs/ml_models/adv_ml_dashboard.png
   • outputs/ml_models/adv_ml_report.txt
"""

import os, sys, warnings, itertools
from datetime import datetime

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker   as mticker
import matplotlib.colors   as mcolors

from sklearn.ensemble         import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble         import StackingClassifier, RandomForestClassifier
from sklearn.linear_model     import LogisticRegression, Ridge
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.metrics          import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost  as xgb
import lightgbm as lgb

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MONITORING_DIR, ML_MODELS_DIR, RAW_DATA_DIR, PARETO_DIR

# ── Output Paths ──────────────────────────────────────────────────────────────
ALERTS_CSV  = os.path.join(MONITORING_DIR, "batch_monitoring_alerts.csv")
SCORED_CSV  = os.path.join(RAW_DATA_DIR,   "scored_batches.csv")
PARETO_CSV  = os.path.join(PARETO_DIR,     "golden_signature_pareto.csv")
MEAN_CSV    = os.path.join(RAW_DATA_DIR,   "golden_signature_mean.csv")
STD_CSV     = os.path.join(RAW_DATA_DIR,   "golden_signature_std.csv")
ADV_PRED    = os.path.join(ML_MODELS_DIR,  "adv_predictions.csv")
ADV_IMP     = os.path.join(ML_MODELS_DIR,  "adv_feature_importance.csv")
ADV_DASH    = os.path.join(ML_MODELS_DIR,  "adv_ml_dashboard.png")
ADV_RPT     = os.path.join(ML_MODELS_DIR,  "adv_ml_report.txt")

# ── Config ─────────────────────────────────────────────────────────────────────
ROLLING_N      = 5
MIN_TRAIN_SIZE = 12
SEV_ORDER      = ["LOW", "MEDIUM", "HIGH"]
SYNTH_N        = 3      # synthetic copies per minority sample (augmentation)
TOP_INTERACT   = 6      # top-N features to build pairwise interactions from
SMOTE_RATIO    = 0.9    # target ratio of minority after SMOTE (relative to majority)

# ── Palette ──────────────────────────────────────────────────────────────────
BG = "#07090f"; PANEL = "#0d1117"; GRID = "#1a2035"
TEXT = "#e2e8f0"; MUTED = "#64748b"
ACCENT = "#38bdf8"; ORANGE = "#f97316"; GREEN = "#22c55e"
RED = "#ef4444"; PURPLE = "#a78bfa"; YELLOW = "#facc15"
SEV_CLR = {"LOW": YELLOW, "MEDIUM": ORANGE, "HIGH": RED}


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def _batch_num(bid: str) -> int:
    s = "".join(c for c in str(bid) if c.isdigit())
    return int(s) if s else 0


def load_data():
    for p in [ALERTS_CSV, SCORED_CSV, PARETO_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    alerts = pd.read_csv(ALERTS_CSV)
    scored = pd.read_csv(SCORED_CSV)
    pareto = pd.read_csv(PARETO_CSV)

    # Optional golden stats
    g_mean = pd.read_csv(MEAN_CSV, index_col=0).iloc[:, 0] if os.path.exists(MEAN_CSV) else pd.Series()
    g_std  = pd.read_csv(STD_CSV,  index_col=0).iloc[:, 0] if os.path.exists(STD_CSV)  else pd.Series()

    for df in [alerts, scored]:
        df["_n"] = df["Batch_ID"].apply(_batch_num)
    alerts = alerts.sort_values("_n").reset_index(drop=True)
    scored = scored.sort_values("_n").reset_index(drop=True)

    # Pareto-critical raw features available in scored_batches
    crit_feats = [f for f in pareto["feature"].tolist()
                  if f in scored.columns and f != "Friability_inv"]

    slim_scored = scored[["Batch_ID", "_n"] + crit_feats].copy()
    slim_alerts = alerts[["Batch_ID", "deviation_score", "severity",
                           "features_oor", "composite_score"]].copy()

    df = slim_scored.merge(slim_alerts, on="Batch_ID", how="inner")
    df = df.sort_values("_n").reset_index(drop=True)

    # Merge OK → LOW
    df["severity"] = df["severity"].replace("OK", "LOW")

    print(f"  [data] {len(df)} batches | {len(crit_feats)} Pareto features")
    print(f"  [data] Class distribution:\n{df['severity'].value_counts().to_string()}")

    return df, crit_feats, g_mean, g_std, pareto


# ─────────────────────────────────────────────────────────────────────────────
# 2. RICH FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, base_feats: list,
                      g_mean: pd.Series, g_std: pd.Series) -> tuple[pd.DataFrame, list]:
    out = df.copy()
    n   = len(out)

    # ── 2a. Rolling stats + delta ─────────────────────────────────────────────
    for feat in base_feats:
        s = out[feat]
        out[f"{feat}_rmean"] = s.rolling(ROLLING_N, min_periods=2).mean()
        out[f"{feat}_rstd"]  = s.rolling(ROLLING_N, min_periods=2).std().fillna(0)
        out[f"{feat}_delta"] = s.diff().fillna(0)
        out[f"{feat}_delta2"]= s.diff().diff().fillna(0)   # second-order delta
        # Cumulative mean (expanding)
        out[f"{feat}_cummu"] = s.expanding(min_periods=2).mean()

        # ── 2b. Ratio to golden mean (deviation normalised) ───────────────────
        gmean_val = float(g_mean.get(feat, np.nan))
        gstd_val  = float(g_std.get(feat, np.nan))
        if not np.isnan(gmean_val) and gmean_val != 0:
            out[f"{feat}_ratio"] = s / gmean_val
        if not np.isnan(gstd_val) and gstd_val > 0:
            out[f"{feat}_zscore"] = (s - gmean_val) / gstd_val

    # ── 2c. Deviation score rolling stats ────────────────────────────────────
    out["dev_rmean"]   = out["deviation_score"].rolling(ROLLING_N, min_periods=2).mean()
    out["dev_rstd"]    = out["deviation_score"].rolling(ROLLING_N, min_periods=2).std().fillna(0)
    out["dev_delta"]   = out["deviation_score"].diff().fillna(0)
    out["dev_cummu"]   = out["deviation_score"].expanding(min_periods=2).mean()

    out["n_oor_rmean"] = out["features_oor"].rolling(ROLLING_N, min_periods=2).mean()
    out["comp_delta"]  = out["composite_score"].diff().fillna(0)
    out["comp_rmean"]  = out["composite_score"].rolling(ROLLING_N, min_periods=2).mean()

    # ── 2d. Batch position features ──────────────────────────────────────────
    out["batch_pos"]   = np.arange(n) / max(n - 1, 1)   # normalised 0–1
    out["batch_idx"]   = np.arange(n)

    # ── 2e. Pairwise interaction: ΔA × ΔB for top Pareto features ────────────
    top_deltas = [f"{f}_delta" for f in base_feats[:TOP_INTERACT] if f"{f}_delta" in out.columns]
    for fa, fb in itertools.combinations(top_deltas, 2):
        col = f"ix__{fa[:18]}_x_{fb[:18]}"
        out[col] = out[fa] * out[fb]

    # Drop rows with NaN rolling (first few)
    first_roll = f"{base_feats[0]}_rmean"
    out = out.dropna(subset=[first_roll]).reset_index(drop=True)

    # Collect all engineered feature columns
    exclude = {"Batch_ID", "_n", "deviation_score", "severity",
               "features_oor", "composite_score", "severity_enc", "severity_mapped"}
    feat_cols = [c for c in out.columns if c not in exclude and out[c].dtype != object]

    print(f"  [features] {len(feat_cols)} engineered features after dropna — {len(out)} batches remain")
    return out, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# 3. SYNTHETIC AUGMENTATION (small Gaussian perturbation of minorities)
# ─────────────────────────────────────────────────────────────────────────────
def augment_minorities(df: pd.DataFrame, feat_cols: list, le: LabelEncoder,
                       n_synth: int = SYNTH_N) -> pd.DataFrame:
    """
    For LOW and MEDIUM batches, create n_synth slightly perturbed copies
    (±2% Gaussian noise on each Pareto feature column).
    This expands the minority class count before SMOTE.
    """
    minority_labels = ["LOW", "MEDIUM"]
    synth_rows = []
    np.random.seed(42)

    for sev in minority_labels:
        sub = df[df["severity"] == sev]
        if sub.empty:
            continue
        for _ in range(n_synth):
            perturbed = sub.copy()
            noise = 1 + np.random.normal(0, 0.02, size=(len(sub), len(feat_cols)))
            perturbed[feat_cols] = perturbed[feat_cols].values * noise
            perturbed["severity_enc"] = le.transform(perturbed["severity"])
            synth_rows.append(perturbed)

    if not synth_rows:
        return df
    augmented = pd.concat([df] + synth_rows, ignore_index=True)
    print(f"  [augment] Expanded from {len(df)} → {len(augmented)} rows (minority augmentation)")
    return augmented


# ─────────────────────────────────────────────────────────────────────────────
# 4. DEFINE MODELS
# ─────────────────────────────────────────────────────────────────────────────
def build_base_classifiers(class_weights_dict: dict):
    gb = GradientBoostingClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=2, random_state=42
    )
    xgb_clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, verbosity=0,
        # XGBoost uses sample_weight during fit — we'll pass per-sample weights
    )
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42, verbose=-1,
    )
    return gb, xgb_clf, lgb_clf


def build_regression_model():
    return GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=2, random_state=42
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. ROLLING-WINDOW CV WITH SMOTE
# ─────────────────────────────────────────────────────────────────────────────
def rolling_cv_with_smote(df_full: pd.DataFrame, feat_cols: list,
                           gb, xgb_clf, lgb_clf, reg_model, le: LabelEncoder):
    """
    Expanding-window CV.  For each fold:
      1. Split train / test (one batch held out).
      2. Apply SMOTE inside the training split only.
      3. Fit three base classifiers + regressor.
      4. Collect OOF predictions for stacking meta-learner.
    """
    n = len(df_full)
    y_reg_true, y_reg_pred = [], []
    y_clf_true = []
    oof_gb, oof_xgb, oof_lgb = [], [], []
    batch_ids = []

    # Compute sample weights proportional to inverse class frequency
    class_counts = df_full["severity_enc"].value_counts().to_dict()
    total = len(df_full)

    for i in range(MIN_TRAIN_SIZE, n):
        train = df_full.iloc[:i].copy()
        test  = df_full.iloc[[i]].copy()

        X_tr_raw = train[feat_cols].fillna(train[feat_cols].mean())
        y_tr_cls = train["severity_enc"].values
        y_tr_reg = train["deviation_score"].values
        X_te     = test[feat_cols].fillna(train[feat_cols].mean())

        # ── Sample weights ───────────────────────────────────────────────────
        label_arr = np.unique(y_tr_cls)
        if len(label_arr) >= 1:
            cw = compute_class_weight("balanced", classes=label_arr, y=y_tr_cls)
            cw_dict = dict(zip(label_arr, cw))
        else:
            cw_dict = {}
        sample_w = np.array([cw_dict.get(y, 1.0) for y in y_tr_cls])

        # ── SMOTE (only if enough minority samples exist in fold) ─────────────
        X_tr, y_tr_cls_s = X_tr_raw.values, y_tr_cls
        if SMOTE_AVAILABLE:
            min_count = min(np.bincount(y_tr_cls))
            k_neighbors = max(1, min(min_count - 1, 3))
            if min_count >= 2 and len(np.unique(y_tr_cls)) > 1:
                try:
                    sm = SMOTE(k_neighbors=k_neighbors, random_state=42)
                    X_tr, y_tr_cls_s = sm.fit_resample(X_tr_raw.values, y_tr_cls)
                    # Recompute sample weights for resampled set
                    lbl_sm = np.unique(y_tr_cls_s)
                    cw_sm  = compute_class_weight("balanced", classes=lbl_sm, y=y_tr_cls_s)
                    cw_sm_d = dict(zip(lbl_sm, cw_sm))
                    sample_w = np.array([cw_sm_d.get(y, 1.0) for y in y_tr_cls_s])
                except Exception:
                    pass   # fallback: use original

        # ── Fit base models ──────────────────────────────────────────────────
        gb.fit(X_tr, y_tr_cls_s, sample_weight=sample_w)

        # XGBoost needs numeric labels in [0, n_classes)
        xgb_clf.fit(X_tr, y_tr_cls_s,
                    sample_weight=sample_w,
                    verbose=False)

        lgb_clf.fit(X_tr, y_tr_cls_s,
                    sample_weight=sample_w)

        reg_model.fit(X_tr_raw.values, y_tr_reg)  # reg trained on original (no SMOTE)

        # ── Predict ──────────────────────────────────────────────────────────
        X_te_vals = X_te.values
        oof_gb.append(gb.predict_proba(X_te_vals)[0])
        oof_xgb.append(xgb_clf.predict_proba(X_te_vals)[0])
        oof_lgb.append(lgb_clf.predict_proba(X_te_vals)[0])

        y_reg_true.append(test["deviation_score"].values[0])
        y_reg_pred.append(float(reg_model.predict(X_te.fillna(train[feat_cols].mean()).values)[0]))
        y_clf_true.append(test["severity_enc"].values[0])
        batch_ids.append(test["Batch_ID"].values[0])

    return (np.array(oof_gb), np.array(oof_xgb), np.array(oof_lgb),
            np.array(y_clf_true), np.array(y_reg_true), np.array(y_reg_pred),
            batch_ids)


# ─────────────────────────────────────────────────────────────────────────────
# 6. STACKING META-LEARNER
# ─────────────────────────────────────────────────────────────────────────────
def fit_meta_learner(oof_gb, oof_xgb, oof_lgb, y_clf_true):
    """Combine OOF proba arrays → meta-feature matrix → Logistic Regression."""
    n_classes = oof_gb.shape[1]
    # Ensure shapes match (all should be [n_fold, n_class])
    def pad_to(arr, nc):
        if arr.shape[1] == nc:
            return arr
        padded = np.zeros((len(arr), nc))
        padded[:, :arr.shape[1]] = arr
        return padded

    gb_p  = pad_to(oof_gb,  n_classes)
    xgb_p = pad_to(oof_xgb, n_classes)
    lgb_p = pad_to(oof_lgb, n_classes)

    meta_X = np.hstack([gb_p, xgb_p, lgb_p])   # [n_fold, 3*n_class]

    meta_clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=500, random_state=42
    )
    meta_clf.fit(meta_X, y_clf_true)
    return meta_clf, meta_X


# ─────────────────────────────────────────────────────────────────────────────
# 7. COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_clf_metrics(y_true, y_pred, prefix="") -> dict:
    avg = "weighted"
    return {
        f"{prefix}Accuracy":  float(accuracy_score(y_true, y_pred)),
        f"{prefix}Precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        f"{prefix}Recall":    float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        f"{prefix}F1":        float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }


def compute_reg_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. FINAL MODEL TRAINING + NEXT-BATCH FORECAST
# ─────────────────────────────────────────────────────────────────────────────
def train_final(df_aug: pd.DataFrame, feat_cols: list,
                gb, xgb_clf, lgb_clf, reg_model, meta_clf, le: LabelEncoder):
    """Train on ALL augmented data, forecast the next (unseen) batch."""
    X = df_aug[feat_cols].fillna(df_aug[feat_cols].mean()).values
    y_cls = df_aug["severity_enc"].values
    y_reg = df_aug["deviation_score"].values

    # Sample weights
    labels_u = np.unique(y_cls)
    cw = compute_class_weight("balanced", classes=labels_u, y=y_cls)
    sw = np.array([dict(zip(labels_u, cw)).get(y, 1.0) for y in y_cls])

    # SMOTE on full train
    if SMOTE_AVAILABLE:
        min_c = min(np.bincount(y_cls))
        k = max(1, min(min_c - 1, 4))
        if min_c >= 2:
            try:
                sm = SMOTE(k_neighbors=k, random_state=42)
                X_sm, y_sm = sm.fit_resample(X, y_cls)
                lbl_sm = np.unique(y_sm)
                cw_sm  = compute_class_weight("balanced", classes=lbl_sm, y=y_sm)
                sw_sm  = np.array([dict(zip(lbl_sm, cw_sm)).get(y, 1.0) for y in y_sm])
                X, y_cls, sw = X_sm, y_sm, sw_sm
            except Exception:
                pass

    gb.fit(X, y_cls, sample_weight=sw)
    xgb_clf.fit(X, y_cls, sample_weight=sw, verbose=False)
    lgb_clf.fit(X, y_cls, sample_weight=sw)
    reg_model.fit(df_aug[feat_cols].fillna(df_aug[feat_cols].mean()).values,
                  df_aug["deviation_score"].values)

    # Next-batch proxy: rolling mean of last ROLLING_N rows
    last_X = df_aug.tail(ROLLING_N)[feat_cols].fillna(df_aug[feat_cols].mean()).mean().values.reshape(1, -1)

    n_cl = len(le.classes_)
    def _safe_proba(clf, x, nc):
        p = clf.predict_proba(x)[0]
        if len(p) == nc:
            return p
        full = np.zeros(nc)
        full[:len(p)] = p
        return full

    gb_p   = _safe_proba(gb,      last_X, n_cl)
    xgb_p  = _safe_proba(xgb_clf, last_X, n_cl)
    lgb_p  = _safe_proba(lgb_clf, last_X, n_cl)
    meta_x = np.hstack([gb_p, xgb_p, lgb_p]).reshape(1, -1)

    next_sev_enc = int(meta_clf.predict(meta_x)[0])
    next_sev     = le.inverse_transform([next_sev_enc])[0]
    next_dev     = float(reg_model.predict(last_X)[0])
    next_probs   = {le.inverse_transform([k])[0]: round(float(p), 3)
                    for k, p in enumerate(meta_clf.predict_proba(meta_x)[0])}

    next_df = pd.DataFrame([{
        "Batch_ID": "NEXT (forecast)",
        "pred_dev_score": round(next_dev, 2),
        "pred_severity":  next_sev,
        **{f"prob_{k}": v for k, v in next_probs.items()},
    }])
    return next_df


# ─────────────────────────────────────────────────────────────────────────────
# 9. FEATURE IMPORTANCE (from base learners)
# ─────────────────────────────────────────────────────────────────────────────
def get_importance(gb, xgb_clf, lgb_clf, feat_cols: list) -> pd.DataFrame:
    def _get(clf, n):
        try:
            return clf.feature_importances_[:n]
        except Exception:
            return np.zeros(n)

    n = len(feat_cols)
    gb_i   = _get(gb,      n)
    xgb_i  = _get(xgb_clf, n)
    lgb_i  = _get(lgb_clf, n)
    avg_i  = (gb_i + xgb_i + lgb_i) / 3.0

    df = pd.DataFrame({
        "feature":        feat_cols,
        "gb_importance":  gb_i,
        "xgb_importance": xgb_i,
        "lgb_importance": lgb_i,
        "avg_importance": avg_i,
    }).sort_values("avg_importance", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 10. VISUALISATION DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.6, zorder=0)
    if title:  ax.set_title(title, color=TEXT, fontsize=9.5, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=8)


def draw_dashboard(oof_df, imp_df, next_df, clf_metrics, reg_metrics, le, meta_y_pred, y_clf_true):
    fig = plt.figure(figsize=(22, 22), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.50, wspace=0.38,
                            left=0.07, right=0.97,
                            top=0.94, bottom=0.04)

    batches = oof_df["Batch_ID"].tolist()
    x       = np.arange(len(oof_df))

    # ── Panel 1 (span 2): Predicted vs Actual deviation score ────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(x, oof_df["actual_dev_score"], color=ACCENT, linewidth=1.8,
             marker="o", markersize=4, label="Actual deviation score")
    ax1.plot(x, oof_df["pred_dev_score"],   color=ORANGE, linewidth=1.8,
             linestyle="--", marker="s", markersize=4, label="Stacked pred score")
    ax1.fill_between(x, oof_df["actual_dev_score"], oof_df["pred_dev_score"],
                     alpha=0.12, color=RED, label="Error band")
    ax1.set_xticks(x)
    ax1.set_xticklabels(batches, rotation=45, ha="right", fontsize=6.5)
    ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax1.annotate(
        f" RMSE={reg_metrics['RMSE']:.1f}  MAE={reg_metrics['MAE']:.1f}  R²={reg_metrics['R2']:.3f}",
        xy=(0.01, 0.93), xycoords="axes fraction", color=PURPLE, fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1e1b4b", ec=PURPLE, alpha=0.9)
    )
    _ax_style(ax1, title="① Regression — Predicted vs Actual Deviation Score (Stacked Ensemble)",
              xlabel="Batch", ylabel="Deviation Score")

    # ── Panel 2: Scatter actual vs predicted ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(oof_df["actual_dev_score"], oof_df["pred_dev_score"],
                c=[SEV_CLR.get(s, MUTED) for s in oof_df["actual_severity"]],
                s=60, edgecolors="none", zorder=3, alpha=0.85)
    lims = [min(oof_df[["actual_dev_score","pred_dev_score"]].min()),
            max(oof_df[["actual_dev_score","pred_dev_score"]].max())]
    ax2.plot(lims, lims, color=MUTED, linewidth=1.0, linestyle="--")
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    for s, c in SEV_CLR.items():
        ax2.scatter([], [], c=c, s=35, label=s)
    ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7.5, ncol=2)
    _ax_style(ax2, title="② Actual vs Predicted\n(severity coloured)",
              xlabel="Actual", ylabel="Predicted")

    # ── Panel 3 (span 2): Classification severity timeline ───────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    sev_idx = {s: i for i, s in enumerate(SEV_ORDER)}
    xc = np.arange(len(oof_df))
    ax3.scatter(xc - 0.15, [sev_idx.get(s, 0) for s in oof_df["actual_severity"]],
                color=ACCENT, s=55, label="Actual", marker="o", zorder=3)
    ax3.scatter(xc + 0.15, [sev_idx.get(s, 0) for s in oof_df["pred_severity"]],
                color=ORANGE, s=55, label="Stacked Predicted", marker="D", zorder=3)
    ax3.set_yticks(range(len(SEV_ORDER)))
    ax3.set_yticklabels(SEV_ORDER, fontsize=8.5)
    ax3.set_xticks(xc)
    ax3.set_xticklabels(batches, rotation=45, ha="right", fontsize=6.5)
    ax3.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    acc_pct = clf_metrics["Accuracy"] * 100
    ax3.annotate(
        f" Acc={acc_pct:.1f}%  F1={clf_metrics['F1']:.2f}"
        f"  Prec={clf_metrics['Precision']:.2f}  Rec={clf_metrics['Recall']:.2f}",
        xy=(0.01, 0.91), xycoords="axes fraction", color=GREEN, fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#052e16", ec=GREEN, alpha=0.9)
    )
    _ax_style(ax3, title="③ Classification — Actual vs Stacked-Ensemble Predicted Severity",
              xlabel="Batch", ylabel="Severity Class")

    # ── Panel 4: Confusion matrix ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    classes_present = sorted(
        set(oof_df["actual_severity"]) | set(oof_df["pred_severity"]),
        key=lambda s: SEV_ORDER.index(s) if s in SEV_ORDER else 99
    )
    enc = {s: i for i, s in enumerate(classes_present)}
    yt = [enc.get(s, 0) for s in oof_df["actual_severity"]]
    yp = [enc.get(s, 0) for s in oof_df["pred_severity"]]
    cm = confusion_matrix(yt, yp, labels=list(range(len(classes_present))))
    cmap = mcolors.LinearSegmentedColormap.from_list("cm", [PANEL, "#1d4ed8", ACCENT], N=128)
    ax4.imshow(cm, cmap=cmap, aspect="auto")
    ax4.set_xticks(range(len(classes_present)))
    ax4.set_yticks(range(len(classes_present)))
    ax4.set_xticklabels(classes_present, color=TEXT, fontsize=8)
    ax4.set_yticklabels(classes_present, color=TEXT, fontsize=8)
    ax4.set_xlabel("Predicted", color=MUTED, fontsize=8)
    ax4.set_ylabel("Actual", color=MUTED, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax4.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color=TEXT, fontsize=11, fontweight="bold")
    ax4.set_facecolor(PANEL)
    ax4.spines[:].set_color(GRID)
    ax4.tick_params(colors=MUTED)
    ax4.set_title("④ Confusion Matrix\n(Stacked Ensemble)", color=TEXT,
                  fontsize=9.5, fontweight="bold", pad=7)

    # ── Panel 5 (span 2): Feature importance ─────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    top_n   = min(20, len(imp_df))
    top_imp = imp_df.head(top_n)
    feat_lbl = (top_imp["feature"]
                .str.replace("_rmean",  " [μ]").str.replace("_rstd",   " [σ]")
                .str.replace("_delta2", " [Δ²]").str.replace("_delta", " [Δ]")
                .str.replace("_zscore","[Z]").str.replace("_ratio","[÷μ]")
                .str.replace("_cummu", "[cum]").str.replace("_","  "))
    c5 = [ORANGE if "Δ" in f else PURPLE if "σ" in f else
          GREEN  if "Z" in f else ACCENT
          for f in feat_lbl]
    bars = ax5.barh(range(top_n), top_imp["avg_importance"].values,
                    color=c5, edgecolor=PANEL, linewidth=0.3)
    ax5.set_yticks(range(top_n))
    ax5.set_yticklabels(feat_lbl.str[:32], fontsize=7, color=TEXT)
    ax5.invert_yaxis()
    for b, v in zip(bars, top_imp["avg_importance"].values):
        ax5.text(v + 0.001, b.get_y() + b.get_height()/2,
                 f"{v:.4f}", va="center", color=TEXT, fontsize=6.5)
    _ax_style(ax5, title="⑤ Top Feature Importances (Avg of GB + XGBoost + LightGBM)",
              xlabel="Average Importance")

    # ── Panel 6: Next-batch forecast card ────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(PANEL)
    ax6.axis("off")
    ax6.set_title("⑥ Next-Batch Forecast\n(Stacked Ensemble)", color=TEXT,
                  fontsize=9.5, fontweight="bold", pad=7)
    if not next_df.empty:
        row0 = next_df.iloc[0]
        ns   = str(row0.get("pred_severity", "HIGH"))
        nd   = float(row0.get("pred_dev_score", 0))
        nc   = SEV_CLR.get(ns, RED)
        ax6.text(0.5, 0.90, "NEXT BATCH", ha="center", fontsize=9,
                 color=ACCENT, fontweight="bold", transform=ax6.transAxes)
        ax6.text(0.5, 0.80, f"Dev Score: {nd:.0f}", ha="center",
                 fontsize=12, color=ORANGE, fontweight="bold", transform=ax6.transAxes)
        ax6.text(0.5, 0.70, f"Severity: {ns}", ha="center",
                 fontsize=13, color=nc, fontweight="bold", transform=ax6.transAxes)
        pk = [k for k in row0.index if k.startswith("prob_")]
        for ii, k in enumerate(pk):
            cls  = k.replace("prob_", "")
            val  = float(row0[k]) * 100
            clr  = SEV_CLR.get(cls, MUTED)
            ax6.text(0.1, 0.57 - ii * 0.10, f"P({cls})", ha="left",
                     fontsize=8.5, color=MUTED, transform=ax6.transAxes)
            ax6.text(0.9, 0.57 - ii * 0.10, f"{val:.1f}%", ha="right",
                     fontsize=9, color=clr, fontweight="bold", transform=ax6.transAxes)
        risk = min(nd / 750, 1.0)
        gc   = RED if risk > 0.6 else ORANGE if risk > 0.3 else GREEN
        gy   = 0.16
        ax6.add_patch(plt.Rectangle((0.05, gy), 0.90, 0.07,
                                    fc=GRID, transform=ax6.transAxes, zorder=1))
        ax6.add_patch(plt.Rectangle((0.05, gy), 0.90 * risk, 0.07,
                                    fc=gc, transform=ax6.transAxes, zorder=2))
        ax6.text(0.5, gy - 0.06, f"{risk*100:.0f}% of max risk",
                 ha="center", color=gc, fontsize=7.5, transform=ax6.transAxes)
        ax6.text(0.5, 0.05, f"Model: GB + XGBoost + LightGBM → LR meta",
                 ha="center", fontsize=7, color=MUTED, transform=ax6.transAxes)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.99, 0.01, f"Generated: {ts}  |  OptiBatch Advanced ML",
             ha="right", va="bottom", fontsize=7.5, color=MUTED)
    fig.suptitle("OptiBatch — Advanced Ensemble ML Dashboard  (GB + XGBoost + LightGBM → Stacked)",
                 fontsize=14, fontweight="bold", color=TEXT, y=0.975)

    fig.savefig(ADV_DASH, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [dashboard] {ADV_DASH}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. REPORT
# ─────────────────────────────────────────────────────────────────────────────
def write_report(clf_metrics, reg_metrics, imp_df, next_df, oof_df, le):
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "═" * 70
    row = next_df.iloc[0] if not next_df.empty else pd.Series()

    cr = classification_report(
        oof_df["actual_severity"], oof_df["pred_severity"],
        labels=SEV_ORDER, zero_division=0
    )

    lines = [
        sep,
        "  OPTIBATCH — ADVANCED ENSEMBLE ML REPORT",
        f"  Generated : {ts}",
        sep,
        "  MODELS USED",
        "─" * 70,
        "  Base 1 : Gradient Boosting Classifier  (n=400, depth=4, lr=0.05)",
        "  Base 2 : XGBoost Classifier             (n=400, depth=5, lr=0.05)",
        "  Base 3 : LightGBM Classifier            (n=400, depth=5, lr=0.05)",
        "  Meta   : Logistic Regression (class_weight=balanced)",
        "  Imbalance  : Per-fold SMOTE + class_weight=balanced + synthetic augmentation",
        "  Validation : Rolling-window expanding CV (time-aware)",
        sep,
        "  REGRESSION METRICS  (predict deviation score)",
        "─" * 70,
        f"  RMSE  = {reg_metrics['RMSE']:.2f}",
        f"  MAE   = {reg_metrics['MAE']:.2f}",
        f"  R²    = {reg_metrics['R2']:.4f}",
        sep,
        "  CLASSIFICATION METRICS  (predict severity class)",
        "─" * 70,
        f"  Accuracy  = {clf_metrics['Accuracy']:.2%}  {'✅ TARGET MET (>92%)' if clf_metrics['Accuracy']>=0.92 else '⚠ Below 92% target'}",
        f"  Precision = {clf_metrics['Precision']:.2%}  (weighted avg)",
        f"  Recall    = {clf_metrics['Recall']:.2%}  (weighted avg)",
        f"  F1 Score  = {clf_metrics['F1']:.2%}  (weighted avg)",
        sep,
        "  PER-CLASS REPORT",
        "─" * 70,
        cr,
        sep,
        "  TOP 15 FEATURES (avg of GB + XGBoost + LightGBM)",
        "─" * 70,
        f"  {'Rank':<5} {'Feature':<40} {'GB':>8} {'XGB':>8} {'LGB':>8} {'Avg':>8}",
        "─" * 70,
    ]
    for i, r in imp_df.head(15).iterrows():
        lines.append(
            f"  {i+1:<5} {r['feature']:<40} "
            f"{r['gb_importance']:>8.4f} {r['xgb_importance']:>8.4f} "
            f"{r['lgb_importance']:>8.4f} {r['avg_importance']:>8.4f}"
        )
    lines += [
        sep,
        "  NEXT-BATCH FORECAST",
        "─" * 70,
        f"  Predicted deviation score : {row.get('pred_dev_score','N/A')}",
        f"  Predicted severity class  : {row.get('pred_severity','N/A')}",
    ]
    for pk in [k for k in row.index if k.startswith("prob_")]:
        lines.append(f"  P({pk.replace('prob_',''):<10}) = {float(row[pk])*100:.1f}%")
    lines += [
        sep,
        f"  Outputs saved to: {ML_MODELS_DIR}",
        f"    • adv_predictions.csv",
        f"    • adv_feature_importance.csv",
        f"    • adv_ml_dashboard.png",
        f"    • adv_ml_report.txt",
        sep,
    ]
    text = "\n".join(lines)
    with open(ADV_RPT, "w") as f:
        f.write(text)
    print(text)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_advanced_ml():
    os.makedirs(ML_MODELS_DIR, exist_ok=True)

    # 1. Load
    print("\n── LOADING DATA ─────────────────────────────────────────")
    df_raw, base_feats, g_mean, g_std, pareto = load_data()

    # 2. Engineer features
    print("\n── FEATURE ENGINEERING ──────────────────────────────────")
    df_eng, feat_cols = engineer_features(df_raw, base_feats, g_mean, g_std)

    # 3. Encode labels
    le = LabelEncoder()
    le.fit(SEV_ORDER)
    df_eng["severity_enc"] = df_eng["severity"].apply(
        lambda s: le.transform([s])[0] if s in le.classes_ else 0
    )

    # 4. Synthetic augmentation of minorities (before CV split)
    print("\n── SYNTHETIC AUGMENTATION ───────────────────────────────")
    df_aug = augment_minorities(df_eng, feat_cols, le, n_synth=SYNTH_N)
    df_aug["severity_enc"] = df_aug["severity"].apply(
        lambda s: le.transform([s])[0] if s in le.classes_ else 0
    )

    # Build class_weight dict
    y_all = df_aug["severity_enc"].values
    labels_u = np.unique(y_all)
    cw = compute_class_weight("balanced", classes=labels_u, y=y_all)
    cw_dict = dict(zip(labels_u, cw))

    # 5. Init models
    print("\n── INITIALISING MODELS ──────────────────────────────────")
    gb, xgb_clf, lgb_clf = build_base_classifiers(cw_dict)
    reg_model = build_regression_model()

    # 6. Rolling CV + OOF collection (run on original time-ordered df_eng, not augmented)
    print("\n── ROLLING-WINDOW CV ────────────────────────────────────")
    (oof_gb, oof_xgb, oof_lgb,
     y_clf_true, y_reg_true, y_reg_pred,
     batch_ids) = rolling_cv_with_smote(df_eng, feat_cols, gb, xgb_clf, lgb_clf, reg_model, le)

    # 7. Train stacking meta-learner on OOF proba
    print("\n── STACKING META-LEARNER ─────────────────────────────────")
    meta_clf, meta_X = fit_meta_learner(oof_gb, oof_xgb, oof_lgb, y_clf_true)
    meta_y_pred = meta_clf.predict(meta_X)

    # 8. Metrics
    clf_metrics = compute_clf_metrics(y_clf_true, meta_y_pred)
    reg_metrics = compute_reg_metrics(y_reg_true, y_reg_pred)
    print(f"\n  Accuracy  = {clf_metrics['Accuracy']:.2%}")
    print(f"  F1        = {clf_metrics['F1']:.2%}")
    print(f"  RMSE      = {reg_metrics['RMSE']:.2f}   R² = {reg_metrics['R2']:.3f}")

    # 9. Build OOF dataframe for dashboard
    oof_df = pd.DataFrame({
        "Batch_ID":         batch_ids,
        "actual_dev_score": y_reg_true,
        "pred_dev_score":   [round(v, 2) for v in y_reg_pred],
        "actual_severity":  le.inverse_transform(y_clf_true),
        "pred_severity":    le.inverse_transform(meta_y_pred),
    })

    # 10. Final fit + next-batch forecast (trained on augmented data for best signal)
    print("\n── FINAL MODEL TRAINING ─────────────────────────────────")
    next_df = train_final(df_aug, feat_cols, gb, xgb_clf, lgb_clf, reg_model, meta_clf, le)
    all_pred = pd.concat([oof_df, next_df], ignore_index=True)
    all_pred.to_csv(ADV_PRED, index=False)
    print(f"  [output] {ADV_PRED}")

    # 11. Feature importance
    imp_df = get_importance(gb, xgb_clf, lgb_clf, feat_cols)
    imp_df.to_csv(ADV_IMP, index=False)
    print(f"  [output] {ADV_IMP}")

    # 12. Dashboard
    print("\n── DRAWING DASHBOARD ─────────────────────────────────────")
    draw_dashboard(oof_df, imp_df, next_df, clf_metrics, reg_metrics, le, meta_y_pred, y_clf_true)

    # 13. Report
    print("\n── WRITING REPORT ────────────────────────────────────────")
    write_report(clf_metrics, reg_metrics, imp_df, next_df, oof_df, le)

    # 14. Update feedback_loop threshold
    print(f"\n✅  Advanced ML complete — Accuracy: {clf_metrics['Accuracy']:.2%}")
    if clf_metrics["Accuracy"] >= 0.92:
        print("   🎯 Target accuracy >92% ACHIEVED")
    else:
        print(f"   ⚠  Result: {clf_metrics['Accuracy']:.2%} — consider more augmentation or tuning")


if __name__ == "__main__":
    run_advanced_ml()
