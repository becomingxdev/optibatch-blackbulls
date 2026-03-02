# scoring/golden_signature.py

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GOLDEN_PERCENTILE

# Columns that must never appear in the golden signature stats
_EXCLUDE_COLS: set[str] = {
    "Batch_ID",
    "quality_score",
    "yield_score",
    "performance_score",
    "energy_score",
    "composite_score",
}


def generate_golden_signature(
    df: pd.DataFrame,
    percentile: float = GOLDEN_PERCENTILE,
) -> tuple[pd.Series, pd.Series]:
    """Identify top-performing batches and compute their statistical fingerprint.

    Parameters
    ----------
    df : pd.DataFrame
        Fully scored DataFrame (must contain 'composite_score').
    percentile : float
        Quantile threshold for "golden" batch selection (default 0.90 → top 10%).

    Returns
    -------
    golden_mean : pd.Series
        Column-wise mean of numeric features for top batches.
    golden_std : pd.Series
        Column-wise std  of numeric features for top batches.

    Raises
    ------
    KeyError
        If 'composite_score' is absent from df.
    ValueError
        If no batches meet the golden threshold.
    """
    if "composite_score" not in df.columns:
        raise KeyError("'composite_score' column not found in DataFrame.")

    threshold = df["composite_score"].quantile(percentile)
    top_batches = df.loc[df["composite_score"] >= threshold]

    if top_batches.empty:
        raise ValueError(
            f"No batches found at or above the {percentile:.0%} composite score threshold."
        )

    # Restrict to numeric columns, excluding score/ID columns
    feature_cols = [
        c for c in top_batches.select_dtypes(include=[np.number]).columns
        if c not in _EXCLUDE_COLS
    ]

    golden_mean: pd.Series = top_batches[feature_cols].mean()
    golden_std:  pd.Series = top_batches[feature_cols].std()

    return golden_mean, golden_std