"""
This file is responsible for handling operations related to feature engineering.
It is part of the data_pipeline module and will later contain the implementation for features associated with feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Generates advanced features useful for prediction and optimization.
    Returns:
        tuple: (engineered_dataframe, feature_metadata_dictionary)
    """
    if df.empty:
        return df, {"new_features": []}

    df_eng = df.copy()
    metadata = {"new_features": []}
    
    # Derived features
    # Process efficiency indicators: energy_per_batch, energy_per_output, process_efficiency_ratio
    energy_col = next((c for c in df_eng.columns if 'energy' in c.lower()), None)
    output_col = next((c for c in df_eng.columns if 'output' in c.lower() or 'yield' in c.lower()), None)
    
    if energy_col:
        # Assuming 1 row = 1 batch
        df_eng['energy_per_batch'] = df_eng[energy_col]
        metadata["new_features"].append('energy_per_batch')
        
        if output_col:
            # Add a small epsilon to avoid division by zero
            df_eng['energy_per_output'] = df_eng[energy_col] / (df_eng[output_col] + 1e-9)
            df_eng['process_efficiency_ratio'] = df_eng[output_col] / (df_eng[energy_col] + 1e-9)
            metadata["new_features"].extend(['energy_per_output', 'process_efficiency_ratio'])
            
    # Temporal indicators
    timestamp_col = next((c for c in df_eng.columns if 'time' in c.lower() or 'date' in c.lower()), None)
    if timestamp_col:
        try:
            df_eng[timestamp_col] = pd.to_datetime(df_eng[timestamp_col])
            df_eng['hour_of_operation'] = df_eng[timestamp_col].dt.hour
            df_eng['day_of_week'] = df_eng[timestamp_col].dt.dayofweek
            metadata["new_features"].extend(['hour_of_operation', 'day_of_week'])
        except Exception:
            pass
            
    # Parameter interactions & Normalized process parameters
    numeric_cols = df_eng.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        # Interaction between first two numeric columns as an example
        df_eng['parameter_interaction_features'] = df_eng[numeric_cols[0]] * df_eng[numeric_cols[1]]
        metadata["new_features"].append('parameter_interaction_features')
        
    # Scaling using StandardScaler
    if numeric_cols:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_eng[numeric_cols])
        
        for i, col in enumerate(numeric_cols):
            df_eng[f'normalized_{col}'] = scaled_features[:, i]
            metadata["new_features"].append(f'normalized_{col}')
        
    return df_eng, metadata
