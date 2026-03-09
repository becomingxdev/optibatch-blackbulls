"""
This file is responsible for handling operations related to signature updater.
It is part of the continuous_learning module and will later contain the implementation for features associated with signature updater.
"""

import os
import json
import logging
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_signatures_dir():
    return os.path.join(get_base_dir(), 'golden_signatures')

def load_signatures() -> dict:
    db_path = os.path.join(get_signatures_dir(), 'golden_signature_db.json')
    if not os.path.exists(db_path):
        return {}
    with open(db_path, 'r') as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def save_signatures(signatures_data: dict):
    os.makedirs(get_signatures_dir(), exist_ok=True)
    db_path = os.path.join(get_signatures_dir(), 'golden_signature_db.json')
    with open(db_path, 'w') as f:
        json.dump(signatures_data, f, indent=4)

def load_dataset() -> pd.DataFrame:
    path = os.path.join(get_base_dir(), 'data', 'processed', 'cleaned_batches.csv')
    if not os.path.exists(path):
        logger.error(f"Cleaned dataset not found at {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df if not df.empty else pd.DataFrame()
    except pd.errors.EmptyDataError:
        logger.warning(f"Dataset at {path} is empty.")
        return pd.DataFrame()

def discover_initial_signatures():
    """
    Automatically generate first golden signatures from historical data.
    """
    df = load_dataset()
    if df.empty:
        logger.warning("Dataset is empty. Using simulated baseline data to build signatures.")
        df = pd.DataFrame({
            'temperature': [180, 185, 190, 180, 185],
            'hold_time': [45, 50, 45, 50, 45],
            'pressure': [2.1, 2.2, 2.1, 2.3, 2.2],
            'yield': [0.95, 0.96, 0.94, 0.95, 0.97],
            'quality': [98, 99, 97, 98, 99],
            'performance': [1.1, 1.2, 1.0, 1.1, 1.2],
            'energy': [150, 160, 145, 155, 165],
            'energy_consumption': [150, 160, 145, 155, 165]
        })

    # Standardize columns internally
    cols = [str(c).lower() for c in df.columns]
    df.columns = cols
    
    signatures = {}
    features = [c for c in cols if c not in ['yield', 'quality', 'performance', 'energy', 'energy_consumption', 'batch_id', 'date', 'time']]
    
    def extract_scenario(row):
        metrics = {
            'yield': float(row.get('yield', 0)),
            'quality': float(row.get('quality', 0)),
            'performance': float(row.get('performance', 0)),
            'energy': float(row.get('energy', row.get('energy_consumption', 0)))
        }
        params = {f: float(row[f]) for f in features if f in row}
        return {**metrics, "parameters": params}

    # Highest Yield
    if 'yield' in df.columns:
        best_yield_row = df.loc[df['yield'].idxmax()]
        signatures['highest_yield'] = extract_scenario(best_yield_row)
        
    # Best Quality
    if 'quality' in df.columns:
        best_quality_row = df.loc[df['quality'].idxmax()]
        signatures['best_quality'] = extract_scenario(best_quality_row)

    # Best Performance
    if 'performance' in df.columns:
        best_perf_row = df.loc[df['performance'].idxmax()]
        signatures['best_performance'] = extract_scenario(best_perf_row)
        
    # Lowest Energy
    energy_col = 'energy' if 'energy' in df.columns else 'energy_consumption' if 'energy_consumption' in df.columns else None
    if energy_col:
        best_energy_row = df.loc[df[energy_col].idxmin()]
        signatures['lowest_energy'] = extract_scenario(best_energy_row)
        
    # Balanced Yield-Energy (e.g., maximize Yield / Energy)
    if 'yield' in df.columns and energy_col:
        df['yield_energy_ratio'] = df['yield'] / (df[energy_col] + 1e-9)
        balanced_row = df.loc[df['yield_energy_ratio'].idxmax()]
        signatures['balanced_yield_energy'] = extract_scenario(balanced_row)
        
    save_signatures(signatures)
    
    print("\n---\n")
    print("## OptiBatch Golden Signatures Initialized\n")
    if 'highest_yield' in signatures: print("- Highest Yield Signature Created")
    if 'best_quality' in signatures: print("- Best Quality Signature Created")
    if 'best_performance' in signatures: print("- Best Performance Signature Created")
    if 'lowest_energy' in signatures: print("- Lowest Energy Signature Created")
    if 'balanced_yield_energy' in signatures: print("- Balanced Yield-Energy Signature Created")
    
    print(f"\nTotal Signatures Stored: {len(signatures)}")
    print("\n---")
    print("OptiBatch Golden Signature Framework initialized successfully.")

def log_history(entry: dict):
    history_path = os.path.join(get_signatures_dir(), 'signature_history.json')
    history = []
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            try:
                history = json.load(f)
            except Exception:
                history = []
    
    history.append(entry)
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)

def update_signature_if_better(batch_metrics: dict, batch_parameters: dict) -> bool:
    """
    Compare incoming batch metrics against stored benchmarks.
    If better according to a specific target, update and log.
    """
    signatures = load_signatures()
    updated = False
    
    b_yield = batch_metrics.get('yield', 0)
    b_qual = batch_metrics.get('quality', 0)
    b_perf = batch_metrics.get('performance', 0)
    b_energy = batch_metrics.get('energy', float('inf'))
    
    stamp = datetime.now().isoformat()
    
    def log_and_apply(name, old_met, new_met):
        entry = {
            "timestamp": stamp,
            "signature_name": name,
            "previous_metrics": old_met,
            "new_metrics": new_met
        }
        log_history(entry)
        
    if 'highest_yield' in signatures:
        s_yield = signatures['highest_yield'].get('yield', 0)
        if b_yield > s_yield:
            old_mets = {k: v for k, v in signatures['highest_yield'].items() if k != 'parameters'}
            signatures['highest_yield'] = {'yield': b_yield, 'quality': b_qual, 'performance': b_perf, 'energy': b_energy, 'parameters': batch_parameters}
            log_and_apply('highest_yield', old_mets, {k: v for k, v in signatures['highest_yield'].items() if k != 'parameters'})
            updated = True

    if 'lowest_energy' in signatures:
        s_energy = signatures['lowest_energy'].get('energy', float('inf'))
        if b_energy < s_energy:
            old_mets = {k: v for k, v in signatures['lowest_energy'].items() if k != 'parameters'}
            signatures['lowest_energy'] = {'yield': b_yield, 'quality': b_qual, 'performance': b_perf, 'energy': b_energy, 'parameters': batch_parameters}
            log_and_apply('lowest_energy', old_mets, {k: v for k, v in signatures['lowest_energy'].items() if k != 'parameters'})
            updated = True
            
    if updated:
        save_signatures(signatures)
        
    return updated

if __name__ == '__main__':
    discover_initial_signatures()
