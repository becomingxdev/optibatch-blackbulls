"""
This file handles batch simulation logic.
It allows testing hypothetical process parameter configurations.
"""

import logging
import random
from typing import Dict, Any, List

from optibatch.prediction.predict_metrics import predict_batch_metrics
from optibatch.prediction.model_evaluator import convert_to_performance_class
from optibatch.monitoring.batch_comparator import compare_batch_to_signatures
from optibatch.monitoring.drift_detector import detect_metric_drift
from optibatch.monitoring.realtime_monitor import _generate_ml_historical_context
from optibatch.optimization.parameter_optimizer import optimize_batch_parameters

logger = logging.getLogger(__name__)

def simulate_batch(batch_parameters: Dict[str, float]) -> Dict[str, Any]:
    """
    Simulates a batch run using the provided process parameters.
    """
    # 1 & 2. Predict metrics
    predicted_metrics = predict_batch_metrics(batch_parameters)
    if not predicted_metrics:
        return {"error": "Prediction models failed or are missing."}
        
    clean_predictions = {}
    for k, v in predicted_metrics.items():
        clean_predictions[k.replace("predicted_", "")] = v

    # Normalize yield to percentage for UI consumers
    if clean_predictions.get("yield") is not None:
        try:
            y = float(clean_predictions["yield"])
            clean_predictions["yield"] = (y * 100.0) if y <= 1.0 else y
        except Exception:
            pass

    # Provide an alias for energy to match some UI expectations
    if clean_predictions.get("energy") is not None and clean_predictions.get("energy_consumption") is None:
        clean_predictions["energy_consumption"] = clean_predictions["energy"]
        
    # 3. Performance class
    perf_val = clean_predictions.get("performance", clean_predictions.get("yield", 0) * 100)
    perf_class = convert_to_performance_class([perf_val])[0].lower() if perf_val is not None else "unknown"
    
    # 4. Compare to signatures
    comparison_data = {**batch_parameters, **predicted_metrics}
    signature_report = compare_batch_to_signatures(comparison_data)
    closest_sig = signature_report.get("closest_signature", "unknown")
    
    # 5. Drift risk
    hist_dist = _generate_ml_historical_context(batch_parameters)
    drift_report = detect_metric_drift(predicted_metrics, hist_dist)
    drift_risk = drift_report.get("drift_score", 0.0)
    
    # 6. Optimizer suggestions
    opt_report = optimize_batch_parameters(batch_parameters, predicted_metrics)
    opt_suggestions = opt_report.get("parameter_recommendations", {})
    
    return {
        "predicted_metrics": clean_predictions,
        "performance_class": perf_class,
        "closest_signature": closest_sig,
        "drift_risk": drift_risk,
        "optimization_suggestions": opt_suggestions
    }

def run_parameter_sweep(parameter_ranges: Dict[str, List[float]], num_simulations: int = 100) -> Dict[str, Any]:
    """
    Run Monte Carlo style simulated batches over defined parameter ranges.
    Returns the top-performing configurations.
    """
    simulations = []
    any_heuristic = False
    
    for _ in range(num_simulations):
        # Generate random parameters within ranges
        params = {}
        for param, rnge in parameter_ranges.items():
            if len(rnge) == 2:
                params[param] = random.uniform(rnge[0], rnge[1])
            else:
                params[param] = rnge[0] # Fallback if not a range

        predicted = predict_batch_metrics(params)
        if predicted:
            is_heuristic = predicted.pop("is_heuristic_fallback", True)
            pred_confidence = predicted.pop("prediction_confidence", 0.5)
            if is_heuristic:
                any_heuristic = True

            # Reformat to flatten for output
            sim_result = {**params}
            for k, v in predicted.items():
                clean_key = k.replace("predicted_", "")
                sim_result[clean_key] = v
            
            # Normalize yield to 0-100 percentage
            if "yield" in sim_result:
                y = sim_result["yield"]
                if y is not None and y <= 1.0:
                    sim_result["yield"] = round(y * 100.0, 2)
            
            # Ensure energy_consumption alias
            if "energy" in sim_result and "energy_consumption" not in sim_result:
                sim_result["energy_consumption"] = sim_result["energy"]

            sim_result["prediction_confidence"] = pred_confidence
            simulations.append(sim_result)
            
    # Sort simulations to find the best (Assume maximizing yield is priority for now)
    if simulations and "yield" in simulations[0]:
        simulations.sort(key=lambda x: x.get("yield", 0), reverse=True)
        
    return {
        "best_simulated_batches": simulations[:min(5, len(simulations))],
        "total_simulations": len(simulations),
        "is_heuristic_fallback": any_heuristic,
    }
