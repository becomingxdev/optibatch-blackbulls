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
    df_clean = df_clean.drop_duplicates()

    # Convert incorrect datatypes / Ensure numeric
    for col in df_clean.columns:
        # Check if column is not already numeric
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            try:
                # Try converting to numeric (coerce turns errors into NaN)
                temp_numeric = pd.to_numeric(df_clean[col], errors='coerce')
                # Only keep the conversion if it successfully converted at least one value 
                # (and the column wasn't already entirely empty)
                if temp_numeric.notnull().any():
                    df_clean[col] = temp_numeric
            except Exception:
                pass

    # Handle missing values & outliers
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            # Fill missing with median
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
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
                mode_res = df_clean[col].mode()
                if not mode_res.empty:
                    df_clean[col] = df_clean[col].fillna(mode_res[0])
                
    return df_clean
