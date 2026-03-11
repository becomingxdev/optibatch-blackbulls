"""
This file is responsible for optimization API endpoints.
It is part of the api module and handles optimization requests.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from optibatch.optimization.parameter_optimizer import optimize_batch_parameters
from optibatch.prediction.predict_metrics import predict_batch_metrics

router = APIRouter()

class OptimizationRequest(BaseModel):
    # Supports both:
    # - { "batch_parameters": {...}, "predicted_metrics": {...} } (canonical)
    # - { "batchParameters": {...}, "predictedMetrics": {...} } (frontend/legacy)
    batch_parameters: Optional[Dict[str, Any]] = None
    predicted_metrics: Optional[Dict[str,Any ]] = None
    batchParameters: Optional[Dict[str, Any]] = None
    predictedMetrics: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


def _normalize_predicted_metrics(predicted: Dict[str, Any]) -> Dict[str, Any]:
    predicted = predicted or {}

    raw_yield = predicted.get("predicted_yield", predicted.get("yield", predicted.get("predicted_yield_percentage")))
    raw_energy = predicted.get("predicted_energy_consumption", predicted.get("predicted_energy", predicted.get("energy_consumption", predicted.get("energy"))))
    raw_cost = predicted.get("predicted_production_cost", predicted.get("production_cost"))

    out: Dict[str, Any] = {}
    if raw_yield is not None:
        try:
            y = float(raw_yield)
            out["yield_percentage"] = y * 100.0 if y <= 1.0 else y
        except Exception:
            pass
    if raw_energy is not None:
        try:
            out["energy_consumption"] = float(raw_energy)
        except Exception:
            pass
    if raw_cost is not None:
        try:
            out["production_cost"] = float(raw_cost)
        except Exception:
            pass

    for k, v in predicted.items():
        if k not in out:
            out[k] = v
    return out

@router.post("/optimize")
def optimize(request: OptimizationRequest):
    batch_parameters = request.batch_parameters or request.batchParameters
    predicted_metrics = request.predicted_metrics or request.predictedMetrics

    if not isinstance(batch_parameters, dict) or len(batch_parameters) == 0:
        raise HTTPException(status_code=422, detail="Missing batch_parameters")

    predicted_metrics = predicted_metrics or {}
    
    cleaned_parameters = {}
    for key, value in batch_parameters.items():
        if isinstance(value, (int, float)):
            cleaned_parameters[key] = float(value)
        elif isinstance(value, str):
            try:
                # Try to convert strings that are actually numbers (like "85.5")
                cleaned_parameters[key] = float(value)
            except ValueError:
                # If it's a word like "balanced", we safely ignore it
              pass
    try:
        report = optimize_batch_parameters(cleaned_parameters, predicted_metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization engine error: {e}")

    if not isinstance(report, dict):
        raise HTTPException(status_code=500, detail="Optimization engine returned invalid response")

    if report.get("error"):
        raise HTTPException(status_code=500, detail=report.get("error"))

    optimal_parameters = report.get("optimal_parameters") or batch_parameters

    optimized_predicted_raw = predict_batch_metrics(optimal_parameters)
    optimized_predicted = _normalize_predicted_metrics(optimized_predicted_raw)

    expected_metric_improvement = report.get("expected_metric_improvement", {})

    # Build explainability text
    target_sig = report.get("target_signature", "unknown")
    confidence = report.get("optimization_confidence", 0.0)
    param_recs = report.get("parameter_recommendations", {})
    
    explanation_parts = [f"Optimization targets the '{target_sig}' golden signature."]
    if param_recs:
        changes = [f"{k}: {v}" for k, v in param_recs.items()]
        explanation_parts.append(f"Recommended changes: {', '.join(changes)}.")
    explanation_parts.append(f"Parameters are adjusted incrementally (~20% toward the target) to minimize process disruption.")
    explanation_parts.append(f"Confidence: {confidence:.0%} — {'high' if confidence >= 0.7 else 'moderate' if confidence >= 0.4 else 'low'} reliability.")
    optimization_explanation = " ".join(explanation_parts)

    # Extract heuristic flag from the re-prediction
    is_heuristic = optimized_predicted_raw.get("is_heuristic_fallback", True)

    return {
        "target_signature": target_sig,
        "parameter_recommendations": param_recs,
        "optimal_parameters": optimal_parameters,
        "expected_metric_improvement": expected_metric_improvement,
        # Frontend currently looks for this alias in a couple places
        "expected_improvement": expected_metric_improvement.get("yield") or expected_metric_improvement.get("energy") or "",
        "optimization_confidence": confidence,
        "optimization_explanation": optimization_explanation,
        "is_heuristic_fallback": is_heuristic,
        # Dashboard consumes this for the impact panel
        "predicted_metrics": optimized_predicted,
    }
