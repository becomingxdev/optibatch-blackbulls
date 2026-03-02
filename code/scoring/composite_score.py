# scoring/composite_score.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def _scale(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Fit and transform a subset of columns with a fresh MinMaxScaler.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list[str]
        Column names to scale (must all exist in df).

    Returns
    -------
    np.ndarray  shape (n_rows, len(cols))
    """
    return MinMaxScaler().fit_transform(df[cols])


def compute_scores(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Compute quality, yield, performance, energy and composite scores.

    Each metric group uses its own MinMaxScaler instance (no data leakage
    between groups).  All operations are fully vectorised — no row loops.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame containing both production and engineered process features.
    weights : dict[str, float]
        Keys: 'quality', 'yield', 'performance', 'energy'.
        Values must sum to 1.0 (not enforced but recommended).

    Returns
    -------
    pd.DataFrame
        Original DataFrame augmented with:
        quality_score, yield_score, performance_score, energy_score,
        composite_score (all on a 0–100 scale).
    """
    # ── Quality Score ─────────────────────────────────────────────────────────
    quality_cols = [
        c for c in ["Dissolution_Rate", "Content_Uniformity", "Hardness"]
        if c in df.columns
    ]

    # Invert Friability (lower is better) with its own scaler
    if "Friability" in df.columns:
        friability_scaled: np.ndarray = MinMaxScaler().fit_transform(df[["Friability"]])
        df = df.copy()                          # avoid SettingWithCopyWarning
        df["Friability_inv"] = 1.0 - friability_scaled.flatten()
        quality_cols.append("Friability_inv")

    if quality_cols:
        quality_scaled = _scale(df, quality_cols)           # one scaler for group
        df["quality_score"] = quality_scaled.mean(axis=1) * 100.0
    else:
        df["quality_score"] = 50.0

    # ── Yield Score ───────────────────────────────────────────────────────────
    if "Tablet_Weight" in df.columns:
        df["yield_score"] = _scale(df, ["Tablet_Weight"]).flatten() * 100.0
    else:
        df["yield_score"] = 50.0

    # ── Performance Score (inverted vibration spikes) ─────────────────────────
    if "vibration_spikes" in df.columns:
        inv_spikes = 1.0 - MinMaxScaler().fit_transform(df[["vibration_spikes"]])
        df["performance_score"] = inv_spikes.flatten() * 100.0
    else:
        df["performance_score"] = 80.0

    # ── Energy Score (inverted total energy) ──────────────────────────────────
    if "total_energy_kwh" in df.columns:
        inv_energy = 1.0 - MinMaxScaler().fit_transform(df[["total_energy_kwh"]])
        df["energy_score"] = inv_energy.flatten() * 100.0
    else:
        df["energy_score"] = 50.0

    # ── Composite Score ───────────────────────────────────────────────────────
    df["composite_score"] = (
        weights["quality"]     * df["quality_score"]     +
        weights["yield"]       * df["yield_score"]       +
        weights["performance"] * df["performance_score"] +
        weights["energy"]      * df["energy_score"]
    )

    return df