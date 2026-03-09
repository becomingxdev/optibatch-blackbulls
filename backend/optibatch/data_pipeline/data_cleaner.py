"""
This file is responsible for handling operations related to data cleaner.
It is part of the data_pipeline module and will later contain the implementation for features associated with data cleaner.
"""

import pandas as pd
import numpy as np

def clean_batch_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw batch datasets.
    Steps:
    - Remove duplicate rows
    - Handle missing values (numerical -> median, categorical -> mode)
    - Detect extreme outliers using IQR method and clip them
    - Convert incorrect datatypes
    - Ensure numeric process parameters are numeric
    """
    df_clean = df.copy()

    # Remove duplicate rows
    df_clean.drop_duplicates(inplace=True)

    # Convert incorrect datatypes / Ensure numeric
    # Let's try to infer numeric objects
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                # Try converting to numeric
                df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
            except Exception:
                pass

    # Handle missing values & outliers
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            # Fill missing with median
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
            # Detect and clip extreme outliers using IQR method
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Clip outliers
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
        else:
            # Fill missing with mode for categorical
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                
    return df_clean
