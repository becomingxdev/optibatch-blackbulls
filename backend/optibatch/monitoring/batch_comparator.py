"""
This file is responsible for handling operations related to batch comparator.
It is part of the monitoring module and will later contain the implementation for features associated with batch comparator.
"""

import os
import json
import logging
import math

logger = logging.getLogger(__name__)

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_golden_signatures() -> dict:
    """Loads golden signatures from the database."""
    base_dir = get_base_dir()
    db_path = os.path.join(base_dir, 'golden_signatures', 'golden_signature_db.json')
    if not os.path.exists(db_path):
        logger.warning(f"Signature DB not found at {db_path}.")
        return {}
    
    with open(db_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            logger.error("Failed to decode Golden Signature DB.")
            return {}

def compare_batch_to_signatures(batch_data: dict) -> dict:
    """
    Compares a new batch profile against stored Golden Signatures.
    
    Args:
        batch_data: Dictionary containing predicted/actual batch metrics.
        
    Returns:
        Dictionary with comparison results and real-time recommendations.
    """
    signatures = load_golden_signatures()
    if not signatures:
        return {"error": "No Golden Signatures available for comparison."}
        
    closest_signature = None
    min_distance = float('inf')
    best_diffs = {}
    
    # Simple Euclidean distance for numerical metrics to find closest signature context
    for sig_name, sig_data in signatures.items():
        dist = 0.0
        diffs = {}
        
        # Calculate metric differences
        for metric in ['yield', 'energy', 'performance', 'quality']:
            batch_val = batch_data.get(metric) or batch_data.get(f'predicted_{metric}')
            sig_val = sig_data.get(metric)
            if batch_val is not None and sig_val is not None:
                diff = float(batch_val) - float(sig_val)
                diffs[f"{metric}_difference"] = diff
                # Distance could be weighted based on importance (using uniform for now)
                
                # Normalize distances slightly because energy values are larger than yield (e.g. 150 vs 0.95)
                # Using simple percentage distance for generic approach:
                if sig_val != 0:
                    pct_diff = (diff / float(sig_val)) * 100
                    dist += (pct_diff ** 2)
                else:
                    dist += (diff ** 2)
                    
        dist = math.sqrt(dist) if dist > 0 else 0
        if dist < min_distance and len(diffs) > 0:
            min_distance = dist
            closest_signature = sig_name
            best_diffs = diffs
            
    # Formulate dummy recommendation based on metric differences
    recommendation = "Maintain current parameters."
    if best_diffs.get('energy_difference', 0) > 5:
        recommendation = "Consider lowering temperature/pressure to reduce energy consumption."
    elif best_diffs.get('yield_difference', 0) < -0.02:
        recommendation = "Increase hold time or temperature slightly to improve yield."
        
    result = {
        "closest_signature": closest_signature,
        "recommendation": recommendation
    }
    result.update(best_diffs)
    
    return result
