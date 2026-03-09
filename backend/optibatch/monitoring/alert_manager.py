"""
This file is responsible for alerting operations based on batch monitoring thresholds.
It is part of the monitoring module and generates actionable alerts.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def generate_alert(monitoring_report: Dict[str, Any]) -> Dict[str, str]:
    """
    Generates an alert object based on the severity of the monitoring report.
    """
    drift_score = monitoring_report.get("drift_score", 0.0)
    drifted_metrics = []
    
    for key, val in monitoring_report.items():
        if key.endswith("_drift") and val is True:
            drifted_metrics.append(key.replace("_drift", "").title())
            
    # Determine Level
    level = "NORMAL"
    if drift_score > 0.8:
        level = "CRITICAL"
    elif drift_score > 0.6 or len(drifted_metrics) > 0:
        level = "WARNING"
        
    message = "Batch operating within normal bounds."
    action = "No intervention necessary"
    
    if level == "CRITICAL":
        msg_parts = ", ".join(drifted_metrics) if drifted_metrics else "Multiple metrics"
        message = f"Severe drift detected in {msg_parts}."
        action = "Halt operation or trigger emergency parameter optimizer immediately."
    elif level == "WARNING":
        msg_parts = ", ".join(drifted_metrics) if drifted_metrics else "General metrics"
        message = f"{msg_parts} drift detected."
        action = "Run parameter optimizer for corrective adjustments."
        
    alert = {
        "alert_level": level,
        "message": message,
        "recommended_action": action
    }
    
    return alert
