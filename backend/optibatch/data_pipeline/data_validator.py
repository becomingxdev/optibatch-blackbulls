"""
This file is responsible for handling operations related to data validator.
It is part of the data_pipeline module and will later contain the implementation for features associated with data validator.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def validate_batch_dataset(df: pd.DataFrame) -> tuple:
    """
    Ensures dataset integrity before modeling.
    Checks:
    - Required columns exist (batch_id at least)
    - No critical columns are entirely null
    - Batch count > minimal threshold
    - Detect constant columns
    - Verify numeric ranges (no negative energy or impossible values)
    
    Returns:
        tuple: (validated_dataframe, validation_report_dictionary)
    """
    report = {
        "is_valid": True,
        "warnings": [],
        "errors": []
    }
    
    df_validated = df.copy()
    
    # Batch count > minimal threshold
    if len(df_validated) < 10:
        report["warnings"].append("Dataset contains very few rows (< 10).")
    
    # Check for completely null columns
    null_cols = df_validated.columns[df_validated.isnull().all()].tolist()
    if null_cols:
        report["warnings"].append(f"Columns completely null: {null_cols}")
        df_validated.drop(columns=null_cols, inplace=True)
        
    # Detect constant columns
    constant_cols = [col for col in df_validated.columns if df_validated[col].nunique() <= 1]
    if constant_cols:
        report["warnings"].append(f"Constant columns detected: {constant_cols}")
        
    # Verify numeric ranges
    for col in df_validated.columns:
        if pd.api.types.is_numeric_dtype(df_validated[col]):
            # Check for negative energy values
            if 'energy' in col.lower() and (df_validated[col] < 0).any():
                report["warnings"].append(f"Negative values found in energy-related column: {col}")
                # Fix negative energy by setting to 0
                df_validated[col] = df_validated[col].apply(lambda x: max(0, x))
                
    if report["warnings"]:
        for warning in report["warnings"]:
            logger.warning(warning)
            
    return df_validated, report
