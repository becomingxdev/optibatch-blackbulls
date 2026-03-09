"""
This file is responsible for prediction API endpoints.
It is part of the api module and handles batch prediction functionality.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

from optibatch.prediction.predict_metrics import predict_batch_metrics
from optibatch.prediction.model_evaluator import convert_to_performance_class

router = APIRouter()

class PredictionRequest(BaseModel):
    temperature: float = 0.0
    pressure: float = 0.0
    hold_time: float = 0.0
    # Allow any other parameters that might be dynamically passed
    class Config:
        extra = "allow"

@router.post("/predict")
def predict(request: PredictionRequest):
    batch_params = request.dict()
    
    # Generate predictions
    predictions = predict_batch_metrics(batch_params)
    
    # Extract trust metadata before stripping prefixes
    is_heuristic = predictions.pop("is_heuristic_fallback", True)
    prediction_confidence = predictions.pop("prediction_confidence", 0.5)

    # Strip prefix for cleaner output
    clean_predictions = {}
    for key, val in predictions.items():
        clean_key = key.replace("predicted_", "")
        clean_predictions[clean_key] = val

    # Normalize yield to percentage for UI consumers
    if clean_predictions.get("yield") is not None:
        try:
            y = float(clean_predictions["yield"])
            clean_predictions["yield"] = (y * 100.0) if y <= 1.0 else y
        except Exception:
            pass

    # Provide alias for energy if present
    if clean_predictions.get("energy") is not None and clean_predictions.get("energy_consumption") is None:
        clean_predictions["energy_consumption"] = clean_predictions["energy"]
        
    # Categorize performance
    if "yield" in clean_predictions and "performance" not in clean_predictions:
        # Yield is normalized to percentage already
        perf_val = clean_predictions["yield"]
    else:
        if clean_predictions.get("performance") is not None:
            perf_val = clean_predictions.get("performance")
        else:
            y = clean_predictions.get("yield", 0)
            try:
                y = float(y)
                perf_val = y * 100.0 if y <= 1.0 else y
            except Exception:
                perf_val = 0
        
    perf_class = "unknown"
    if perf_val is not None:
        perf_class = convert_to_performance_class([perf_val])[0].lower()
        
    return {
        "predicted_metrics": clean_predictions,
        "performance_class": perf_class,
        "is_heuristic_fallback": is_heuristic,
        "prediction_confidence": prediction_confidence,
    }
