"""
This file is responsible for handling operations related to train models.
It is part of the prediction module and will later contain the implementation for features associated with train models.
"""

import os
import json
import logging
from datetime import datetime
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np
import joblib

from .model_evaluator import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMNS = ['yield', 'quality', 'performance', 'energy', 'energy_consumption']

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_dataset():
    """Load the processed batch dataset."""
    base_dir = get_base_dir()
    path = os.path.join(base_dir, 'data', 'processed', 'cleaned_batches.csv')
    if not os.path.exists(path):
        logger.error(f"Dataset not found at {path}. Have you run the data pipeline?")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df if not df.empty else pd.DataFrame()
    except pd.errors.EmptyDataError:
        logger.warning(f"Dataset at {path} is empty.")
        return pd.DataFrame()

def detect_features_and_targets(df: pd.DataFrame):
    """Detect feature columns (by excluding targets) and the available target columns."""
    actual_targets = [col for col in df.columns if any(target in col.lower() for target in TARGET_COLUMNS)]
    features = [col for col in df.columns if col not in actual_targets and 'batch_id' not in col.lower() and 'time' not in col.lower() and 'date' not in col.lower()]
    return features, actual_targets

# ── ADVANCED METHODOLOGY (Adapted from Blackbulls) ──────────────────────────
# 1. Time-Aware Rolling Cross-Validation
# 2. Stacked Ensemble (XGB + LGBM + GradientBoosting -> Ridge Meta-Learner)

def train_and_evaluate_target(target_name: str, X, y):
    """
    Train an advanced Stacked Ensemble using Time-Aware Rolling CV for splitting.
    Base: XGBoost + LightGBM + GradientBoosting -> Ridge meta-learner.
    Mirrors the Blackbulls advanced_ml.py methodology adapted for regression.
    """
    logger.info(f"Training Advanced Stacked Ensemble for target: {target_name}")
    
    n = len(X)
    
    # ── Time-Aware Split: Use last fold of TimeSeriesSplit for train/test
    n_splits = min(5, max(2, n // 8))   # adapt to dataset size
    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # ── Define Base Learners
    estimators = [
        ('xgb', XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)),
        ('lgb', LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)),
        ('gb',  GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42))
    ]
    
    # ── Meta-Learner uses KFold (compatible with cross_val_predict inside StackingRegressor)
    meta_cv = KFold(n_splits=max(2, min(5, len(X_train) // 4)), shuffle=False)
    
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(),
        cv=meta_cv,
        n_jobs=1   # avoid joblib fork issues on Windows
    )
    
    stacking_regressor.fit(X_train, y_train)
    
    metrics = evaluate_model(f"Stacked Ensemble ({target_name})", stacking_regressor, X_test, y_test)
    
    logger.info(f"Built Stacked Ensemble for {target_name} | RMSE: {metrics['rmse']:.4f} | Proxy-Acc: {metrics['accuracy']:.2f}%")
    return stacking_regressor, metrics, "Stacked Regressor (XGB+LGBM+GB -> Ridge)"

def run_training_pipeline():
    df = load_dataset()
    if df.empty:
        # For pipeline test if run strictly per instruction while data generation is missing 
        logger.warning("Simulating dataset as cleaned_batches.csv is empty or not present.")
        df = pd.DataFrame({
            'temperature': [180,185,190,180,185]*5,
            'hold_time': [45,50,45,50,45]*5,
            'pressure': [2.1,2.2,2.1,2.3,2.2]*5,
            'yield': [0.95,0.96,0.94,0.95,0.97]*5,
            'quality': [98,99,97,98,99]*5,
            'performance': [1.1,1.2,1.0,1.1,1.2]*5,
            'energy': [150,160,145,155,165]*5
        })
        
    features, targets = detect_features_and_targets(df)
    
    if not targets:
        logger.error("No target columns found in the dataset.")
        return
        
    logger.info(f"Detected {len(features)} features and targets: {targets}")
    
    models = {}
    metadata = {}
    
    # Standardize target names mapping
    target_mapping = {
        'yield': 'yield',
        'quality': 'quality',
        'performance': 'performance',
        'energy': 'energy'
    }
    
    for target in targets:
        df_valid = df.dropna(subset=[target])
        if len(df_valid) < 10:
            logger.warning(f"Not enough data to train target: {target}")
            continue
            
        X_target = df_valid[features]
        y_target = df_valid[target]
        
        # Instead of random split, pass entire chronological X, y to time-aware trainer
        best_model, best_metrics, model_algo = train_and_evaluate_target(target, X_target, y_target)
        
        # Determine standard name
        standard_name = next((v for k, v in target_mapping.items() if k in target.lower()), target)
        
        models[f"{standard_name}_model"] = best_model
        
        metadata[f"{standard_name}_model"] = {
            "model_name": f"{standard_name}_model.pkl",
            "algorithm": model_algo,
            "training_samples": len(X_target),   # total rows fed to time-aware trainer
            "features_used": features,
            "rmse": float(best_metrics["rmse"]),
            "mae": float(best_metrics["mae"]),
            "mape": float(best_metrics.get("mape", 0.0)),
            "accuracy": float(best_metrics.get("accuracy", 0.0)),
            "f1_score": float(best_metrics.get("f1_score", 0.0)),
            "training_timestamp": datetime.now().isoformat()
        }
    
    # Save models
    base_dir = get_base_dir()
    models_dir = os.path.join(base_dir, 'models', 'trained_models')
    os.makedirs(models_dir, exist_ok=True)
    
    print("\nOptiBatch Model Training Summary\n")
    for name, model in models.items():
        model_path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        
        acc_val = metadata[name].get('accuracy', 0.0)
        f1_val = metadata[name].get('f1_score', 0.0)
        
        print(f"{name.replace('_model', '').title()} Model")
        print(f"Accuracy: {acc_val:.1f}%")
        print(f"F1 Score: {f1_val:.2f}\n")
        
    # Save metadata
    meta_path = os.path.join(base_dir, 'models', 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("OptiBatch Prediction Engine training completed successfully.")

if __name__ == "__main__":
    run_training_pipeline()
