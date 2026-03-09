"""
This file is responsible for handling parameter optimization.
It is part of the optimization module and will later contain the implementation for generating parameter recommendations.
"""

import logging
from typing import Dict, Any

from optibatch.monitoring.batch_comparator import load_golden_signatures, compare_batch_to_signatures

logger = logging.getLogger(__name__)

# Safety bounds for known process variables
SAFETY_BOUNDS = {
    "temperature": [150.0, 240.0],
    "pressure": [1.0, 10.0],
    "hold_time": [10.0, 120.0],
    "catalyst_ratio": [0.5, 5.0],
    "reaction_time": [30.0, 300.0],
    "mixing_speed": [100.0, 1000.0]
}

def enforce_safety_bounds(param_name: str, value: float) -> float:
    """Clips the proposed parameter value to its safety boundary."""
    if param_name in SAFETY_BOUNDS:
        min_v, max_v = SAFETY_BOUNDS[param_name]
        return max(min(value, max_v), min_v)
    return value

def optimize_batch_parameters(current_params: Dict[str, float], predicted_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Analyzes current batch logic and provides an optimization recommendation map.
    """
    signatures = load_golden_signatures()
    if not signatures:
        return {"error": "Golden signatures not found"}
        
    # Prepare batch data for comparison natively
    batch_data = {**current_params, **predicted_metrics}
    
    # 3. Determine Target Signature
    # Priority defaults to 'highest_yield' unless it's way too expensive.
    comp_result = compare_batch_to_signatures(batch_data)
    closest_sig = comp_result.get("closest_signature", "highest_yield")
    
    # Usually you'd want to move towards a specific goal, we'll use highest_yield as default if finding fails
    target_sig_key = "highest_yield" if "highest_yield" in signatures else closest_sig
    if not target_sig_key:
        return {"error": "Unable to determine a valid target signature."}
        
    target_sig_data = signatures[target_sig_key]
    target_params = target_sig_data.get("parameters", {})
    
    # 4. Generate candidates and use ML to pick the optimal step
    from optibatch.prediction.predict_metrics import predict_batch_metrics
    
    best_candidate = dict(current_params)
    best_performance = predicted_metrics.get("predicted_performance", 0)
    optimal_metrics = dict(predicted_metrics)
    
    candidates = []
    for step_pct in [0.1, 0.25, 0.5, 0.75, 1.0]:  # Up to 100% towards target signature
        candidate_params = dict(current_params)
        for param, current_val in current_params.items():
            if param in target_params:
                target_val = float(target_params[param])
                diff = target_val - float(current_val)
                new_val = current_val + (diff * step_pct)
                new_val = enforce_safety_bounds(param, new_val)
                candidate_params[param] = new_val
        candidates.append(candidate_params)
        
    for candidate in candidates:
        try:
            preds = predict_batch_metrics(candidate)
            perf = preds.get("predicted_performance", 0)
            if perf > best_performance:
                best_performance = perf
                best_candidate = candidate
                optimal_metrics = preds
        except Exception as e:
            logger.error(f"Failed to evaluate candidate in optimizer: {e}")
            
    param_recommendations = {}
    optimal_parameters = dict(best_candidate)
    
    for param, optimal_val in optimal_parameters.items():
        current_val = float(current_params.get(param, optimal_val))
        # Account for floating point issues
        if abs(current_val - optimal_val) > 0.001:
            if current_val != 0:
                pct_change = ((optimal_val - current_val) / current_val) * 100
                prefix = "+" if pct_change > 0 else ""
                param_recommendations[param] = f"{prefix}{pct_change:.1f}%"
            else:
                param_recommendations[param] = f"+{optimal_val:.2f}"
                
    expected_improvements = {}
    for metric in ["yield", "quality", "energy"]:
        p_val = predicted_metrics.get(f"predicted_{metric}", 0)
        o_val = optimal_metrics.get(f"predicted_{metric}", 0)
        
        if p_val and o_val and abs(p_val - o_val) > 0.001:
            pct = ((o_val - p_val) / p_val) * 100
            prefix = "+" if pct > 0 else ""
            expected_improvements[metric] = f"{prefix}{pct:.1f}%"
            
    # 7. Confidence Score (derived from ML prediction directly)
    confidence = optimal_metrics.get("prediction_confidence", 0.75)
    
    return {
        "target_signature": target_sig_key,
        "parameter_recommendations": param_recommendations,
        "optimal_parameters": optimal_parameters,
        "expected_metric_improvement": expected_improvements,
        "optimization_confidence": round(confidence, 2)
    }
