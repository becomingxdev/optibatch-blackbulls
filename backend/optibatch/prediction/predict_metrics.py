"""
This file is responsible for handling operations related to predict metrics.
It is part of the prediction module and will later contain the implementation for features associated with predict metrics.
"""

import os
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_models() -> dict:
    """Load all trained models from the models directory."""
    base_dir = get_base_dir()
    models_dir = os.path.join(base_dir, 'models', 'trained_models')
    
    models = {}
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found at {models_dir}")
        return models
        
    for file in os.listdir(models_dir):
        if file.endswith('.pkl'):
            model_name = file.replace('.pkl', '')
            model_path = os.path.join(models_dir, file)
            try:
                import joblib  # optional dependency
                models[model_name] = joblib.load(model_path)
            except Exception as e:
                logger.error(f"Error loading model {file}: {e}")
                
    return models


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _compute_model_confidence(meta_path: str) -> float:
    """
    Dynamically compute prediction confidence from stored model validation metrics.
    
    Reads accuracy, F1 score, and MAPE from model_metadata.json and computes a
    weighted aggregate confidence score. When models are highly accurate (e.g. 99%+),
    confidence will exceed 95%.
    
    Returns a float between 0.0 and 1.0.
    """
    if not os.path.exists(meta_path):
        return 0.75  # reasonable default when metadata is missing but models exist

    try:
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, IOError):
        return 0.75

    if not metadata:
        return 0.75

    # Collect per-model confidence scores
    model_confidences = []
    for model_name, model_meta in metadata.items():
        accuracy = model_meta.get("accuracy", 0.0)       # 0-100 scale
        f1 = model_meta.get("f1_score", 0.0)             # 0-1 scale
        mape = model_meta.get("mape", 100.0)              # percentage error

        # Accuracy component (0-100 -> 0-1), weight: 50%
        acc_score = _clip(accuracy / 100.0, 0.0, 1.0)

        # F1 component (already 0-1), weight: 25%
        f1_score_val = _clip(f1, 0.0, 1.0)

        # MAPE component: lower is better. Convert to confidence.
        # MAPE of 0% -> 1.0, MAPE of 10% -> 0.9, MAPE of 50%+ -> ~0.5
        mape_confidence = _clip(1.0 - (mape / 100.0), 0.0, 1.0)
        # Weight: 25%

        model_conf = (acc_score * 0.50) + (f1_score_val * 0.25) + (mape_confidence * 0.25)
        model_confidences.append(model_conf)

    if not model_confidences:
        return 0.75

    # Aggregate: use the average across all models
    avg_confidence = sum(model_confidences) / len(model_confidences)

    # Allow full range up to 0.99
    return round(_clip(avg_confidence, 0.0, 0.99), 4)


def predict_batch_metrics(batch_parameters: dict) -> dict:
    """
    Predict batch outcomes based on a dictionary of process parameters.
    Raises ValueError if trained ML models are unavailable.
    """
    models = load_models()
    if not models:
        logger.error("No trained ML models found.")
        raise ValueError("ML Models are missing. Prediction cannot be performed.")
        
    # Convert input dictionary to dataframe
    try:
        import pandas as pd
    except Exception as e:
        logger.error(f"pandas not available ({e}).")
        raise ValueError("Pandas is required for real ML prediction.")

    df = pd.DataFrame([batch_parameters])
    
    # Get features and metadata
    base_dir = get_base_dir()
    meta_path = os.path.join(base_dir, 'models', 'model_metadata.json')
    features = []
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            if len(metadata) > 0:
                first_model = list(metadata.keys())[0]
                features = metadata[first_model].get('features_used', [])
            
    # Align features: fill missing with 0, drop extra
    if features:
        for f in features:
            if f not in df.columns:
                df[f] = 0.0
        df = df[features]
        
    # Predict
    predictions = {}
    successful_predictions = 0
    total_models = len(models)
    
    for model_name, model in models.items():
        target_name = model_name.split('_model')[0]
        try:
            pred_val = model.predict(df)[0]
            # Convert to standard Python float for serialization
            predictions[f"predicted_{target_name}"] = float(pred_val)
            successful_predictions += 1
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            predictions[f"predicted_{target_name}"] = None

    # Compute dynamic confidence from model validation metrics
    base_confidence = _compute_model_confidence(meta_path)
    
    # Penalize if some models failed during this prediction
    if total_models > 0:
        success_ratio = successful_predictions / total_models
        adjusted_confidence = base_confidence * success_ratio
    else:
        adjusted_confidence = 0.5
    
    predictions["is_heuristic_fallback"] = False
    predictions["prediction_confidence"] = round(_clip(adjusted_confidence, 0.0, 0.99), 4)
    return predictions

def predict_batch_dataframe(df):
    """
    Return predictions for entire dataframe.
    """
    # Kept for offline training/analysis workflows.
    models = load_models()
    if not models:
        return df
        
    # Get features used during training
    base_dir = get_base_dir()
    meta_path = os.path.join(base_dir, 'models', 'model_metadata.json')
    features = []
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            if len(metadata) > 0:
                first_model = list(metadata.keys())[0]
                features = metadata[first_model].get('features_used', [])
            
    # Align features
    df_aligned = df.copy()
    if features:
        for f in features:
            if f not in df_aligned.columns:
                df_aligned[f] = 0.0
        X = df_aligned[features]
    else:
        X = df_aligned
        
    # Predict
    for model_name, model in models.items():
        target_name = model_name.split('_model')[0]
        try:
            df[f"predicted_{target_name}"] = model.predict(X)
        except Exception as e:
            logger.error(f"Batch prediction failed for {model_name}: {e}")
            
    return df
