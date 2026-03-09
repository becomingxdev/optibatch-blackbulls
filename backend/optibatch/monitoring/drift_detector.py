"""
This file is responsible for detecting statistical deviations in batch metrics.
It is part of the monitoring module and helps trigger adjustments.
"""

import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def detect_metric_drift(current_metrics: Dict[str, float], historical_distribution: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Identifies when batch behavior deviates from normal production patterns.
    Uses Z-score analysis and standard deviation thresholds.
    """
    drift_report = {}
    drift_score = 0.0
    detected = False
    
    # We'll analyze yield, quality, energy, performance if present
    analyzed_metrics = 0
    total_z = 0.0
    
    for metric_name, current_val in current_metrics.items():
        base_name = metric_name.replace("predicted_", "")
        
        # Look for historical list
        hist_values = historical_distribution.get(base_name, [])
        
        if len(hist_values) < 5:
            # Not enough data to confidently detect drift, skip
            drift_report[f"{base_name}_drift"] = False
            continue
            
        mean_val = np.mean(hist_values)
        std_val = np.std(hist_values)
        
        # Avoid division by zero
        if std_val < 1e-5:
            std_val = 1e-5
            
        # Z-score: how many standard deviations away is the current batch?
        z_score = abs(current_val - mean_val) / std_val
        
        # Consider a z-score > 2.0 as a drift event
        is_drifting = bool(z_score > 2.0)
        drift_report[f"{base_name}_drift"] = is_drifting
        
        if is_drifting:
            detected = True
            
        total_z += z_score
        analyzed_metrics += 1
        
    if analyzed_metrics > 0:
        # Scale drift score roughly between 0-1 (e.g., avg z-score of 3.0 gives high drift)
        avg_z = total_z / analyzed_metrics
        drift_score = round(min(1.0, avg_z / 3.0), 2)
        
    drift_report["drift_score"] = drift_score
    drift_report["drift_detected"] = detected
    
    return drift_report
