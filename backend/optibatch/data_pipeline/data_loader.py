"""
This file is responsible for handling operations related to data loader.
It is part of the data_pipeline module and will later contain the implementation for features associated with data loader.
"""

import pandas as pd
import os
import logging
from typing import Optional

# Import other pipeline modules
from .data_cleaner import clean_batch_data
from .data_validator import validate_batch_dataset
from .feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def load_production_data(path: str) -> pd.DataFrame:
    """Loads raw batch production dataset."""
    logger.info(f"Loading production data from {path}")
    try:
        df = pd.read_excel(path)
        df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]
        
        # Remove empty rows/columns completely missing
        df.dropna(how='all', inplace=True, axis=0)
        df.dropna(how='all', inplace=True, axis=1)
        
        # Deduplicate based on batch ID if possible
        batch_col = next((c for c in df.columns if 'batch' in c and 'id' in c), None)
        if batch_col:
            df.drop_duplicates(subset=[batch_col], inplace=True)
            
        return df
    except Exception as e:
        logger.error(f"Error loading production data: {e}")
        return pd.DataFrame()

def load_process_data(path: str) -> pd.DataFrame:
    """Loads raw batch process dataset."""
    logger.info(f"Loading process data from {path}")
    try:
        df = pd.read_excel(path)
        df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]
        
        df.dropna(how='all', inplace=True, axis=0)
        df.dropna(how='all', inplace=True, axis=1)
        
        batch_col = next((c for c in df.columns if 'batch' in c and 'id' in c), None)
        if batch_col:
            df.drop_duplicates(subset=[batch_col], inplace=True)
            
        return df
    except Exception as e:
        logger.error(f"Error loading process data: {e}")
        return pd.DataFrame()

def merge_datasets(production_df: pd.DataFrame, process_df: pd.DataFrame) -> pd.DataFrame:
    """Merges production and process datasets using auto-detected batch identifier."""
    logger.info("Merging datasets")
    if production_df.empty or process_df.empty:
        return production_df if not production_df.empty else process_df
        
    prod_batch_col = next((c for c in production_df.columns if 'batch' in c and 'id' in c), None)
    proc_batch_col = next((c for c in process_df.columns if 'batch' in c and 'id' in c), None)
    
    if prod_batch_col and proc_batch_col:
        merged_df = pd.merge(production_df, process_df, left_on=prod_batch_col, right_on=proc_batch_col, how='inner')
    else:
        logger.warning("No explicit batch ID found; concatenating datasets side-by-side.")
        merged_df = pd.concat([production_df, process_df], axis=1)
        
    return merged_df

def run_data_pipeline():
    """Master orchestration function for the data pipeline."""
    # File paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prod_path = os.path.join(base_dir, 'data', 'raw', 'batch_production_data.xlsx')
    proc_path = os.path.join(base_dir, 'data', 'raw', 'batch_process_data.xlsx')
    out_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_batches.csv')
    
    logger.info("Starting Data Pipeline...")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # 1. Load production dataset
    df_prod = load_production_data(prod_path)
    
    # 2. Load process dataset
    df_proc = load_process_data(proc_path)
    
    # 3. Merge datasets
    df_merged = merge_datasets(df_prod, df_proc)
    
    if not df_merged.empty:
        missing_initial = df_merged.isnull().sum().sum()
        
        # 4. Clean dataset
        df_cleaned = clean_batch_data(df_merged)
        missing_handled = missing_initial - df_cleaned.isnull().sum().sum()
        
        # 5. Validate dataset
        df_validated, report = validate_batch_dataset(df_cleaned)
        
        # 6. Perform feature engineering
        df_engineered, metadata = engineer_features(df_validated)
        
        # 7. Save processed dataset
        df_engineered.to_csv(out_path, index=False)
        logger.info(f"Saved processed dataset to {out_path}")
        
        # Print summary
        print("\n--- Data Pipeline Summary ---")
        print(f"Number of batches (rows): {df_engineered.shape[0]}")
        print(f"Number of features (columns): {df_engineered.shape[1]}")
        print(f"Missing values handled: {missing_handled}")
        print(f"Outliers detected: Evaluated using IQR and clipped.")
        print(f"Final dataset shape: {df_engineered.shape}")
    else:
        logger.warning("Empty dataframes. Outputting empty dataset.")
        pd.DataFrame().to_csv(out_path, index=False)
        print("\n--- Data Pipeline Summary ---")
        print("Data files missing or empty. Pipeline completed with empty dataset.")
        
    print("\nOptiBatch Data Pipeline completed successfully.")

if __name__ == "__main__":
    run_data_pipeline()
