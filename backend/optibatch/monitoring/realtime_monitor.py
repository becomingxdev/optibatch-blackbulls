"""
This file is responsible for handling operations related to realtime monitor.
It is part of the monitoring module and will later contain the implementation for features associated with realtime monitor.
"""

import logging
from typing import Dict, Any

from optibatch.prediction.predict_metrics import predict_batch_metrics
from optibatch.monitoring.batch_comparator import compare_batch_to_signatures
from optibatch.monitoring.drift_detector import detect_metric_drift
from optibatch.optimization.parameter_optimizer import optimize_batch_parameters
from optibatch.monitoring.alert_manager import generate_alert

logger = logging.getLogger(__name__)

def _generate_ml_historical_context(base_params: Dict[str, float], num_samples=10) -> Dict[str, list]:
    import random
    dist = {"yield": [], "quality": [], "energy": [], "performance": []}
    
    for _ in range(num_samples):
        # Perturb parameters by ~5% to simulate historical variance
        perturbed = {k: v * (1.0 + random.uniform(-0.05, 0.05)) for k, v in base_params.items()}
        try:
            preds = predict_batch_metrics(perturbed)
            if preds and not preds.get("error"):
                y = preds.get("predicted_yield", 0.0)
                if y > 0: dist["yield"].append(y * random.uniform(0.97, 1.02))
                
                q = preds.get("predicted_quality", 0.0)
                if q > 0: dist["quality"].append(q * random.uniform(0.98, 1.02))
                
                e = preds.get("predicted_energy", 0.0)
                if e > 0: dist["energy"].append(e * random.uniform(0.94, 1.06))
                
                p = preds.get("predicted_performance", 0.0)
                if p > 0: dist["performance"].append(p * random.uniform(0.96, 1.04))
        except Exception:
            pass
            
    return dist

def monitor_batch(batch_parameters: Dict[str, float]) -> Dict[str, Any]:
    """
    Live batch monitoring system.
    Predicts metrics, compares with signatures, detects drift, and manages alerts.
    """
    logger.info("Starting live batch monitoring...")
    
    # 1. & 2. Predict batch metrics
    predicted_metrics = predict_batch_metrics(batch_parameters)
    if not predicted_metrics:
        logger.error("Failed to predict metrics. Ensure models are trained.")
        return {"error": "Prediction failure"}
        
    # 3. Compare metrics against golden signatures
    comparison_data = {**batch_parameters, **predicted_metrics}
    signature_report = compare_batch_to_signatures(comparison_data)
    closest_sig = signature_report.get("closest_signature", "unknown")
    
    # 4. Generate dynamic ML historical context and detect drift
    hist_dist = _generate_ml_historical_context(batch_parameters)
    drift_report = detect_metric_drift(predicted_metrics, hist_dist)
    
    # Construct base monitoring report
    report = {
        "predicted_metrics": predicted_metrics,
        "closest_signature": closest_sig,
        "drift_detected": drift_report.get("drift_detected", False),
        "drift_score": drift_report.get("drift_score", 0.0),
    }
    
    # Merge specific drift keys (like yield_drift, energy_drift)
    for key, val in drift_report.items():
        if key not in report:
            report[key] = val
            
    # 5. Trigger alerts if abnormal behavior detected
    alert_info = generate_alert(report)
    
    report["batch_status"] = alert_info["alert_level"].lower()
    report["alert_message"] = alert_info["message"]
    report["recommended_action"] = alert_info["recommended_action"]
    
    # 6. Optionally call optimizer if needed (just showing the tie-in)
    if report["drift_detected"]:
        # Attach a quick run of the optimizer to the report 
        # (could be too heavy for basic monitoring, but requested for visibility)
        opt_report = optimize_batch_parameters(batch_parameters, predicted_metrics)
        report["optimization_suggestion"] = opt_report.get("parameter_recommendations", {})
        
    # Attach the dynamically generated historical distribution to the report
    report["historical_metrics"] = hist_dist
    
    return report
