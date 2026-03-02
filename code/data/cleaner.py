# data/cleaner.py

import numpy as np
import pandas as pd


def fill_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values in numeric columns with each column's median.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (not mutated in place).

    Returns
    -------
    pd.DataFrame
        DataFrame with numeric NaNs replaced by column medians.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    medians = df[numeric_cols].median()
    return df.fillna(medians)


def remove_outliers_iqr(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Remove rows where any numeric column falls outside [Q1-1.5·IQR, Q3+1.5·IQR].

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    exclude_cols : list[str] | None
        Columns to skip during outlier filtering (e.g. ID columns).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with outlier rows removed.
    """
    exclude_cols = exclude_cols or []
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    if not numeric_cols:
        return df

    q1 = df[numeric_cols].quantile(0.25)
    q3 = df[numeric_cols].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # Vectorised row mask — no per-column loop filtering
    mask = ((df[numeric_cols] >= lower) & (df[numeric_cols] <= upper)).all(axis=1)
    return df.loc[mask].reset_index(drop=True)