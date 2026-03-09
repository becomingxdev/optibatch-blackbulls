import os
import sys
import json
import logging

# Ensure we can import from optibatch modules if script is run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optibatch.monitoring.realtime_monitor import monitor_batch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def run_cli():
    # Simulated batch parameters (deliberately pushing energy slightly high to trigger drift)
    sample_parameters = {
        "temperature": 182.0,
        "hold_time": 47.0,
        "pressure": 2.15,
        "catalyst_ratio": 1.2,
        "reaction_time": 120.0,
        "mixing_speed": 300.0
    }
    
    report = monitor_batch(sample_parameters)
    
    print("\n--- OptiBatch Live Monitoring Report ---")
    print("\nPredicted Metrics")
    for metric, val in report.get("predicted_metrics", {}).items():
        if val is not None:
             print(f"{metric.replace('predicted_', '').title()}: {val:.2f}")
             
    print(f"\nClosest Signature: {report.get('closest_signature', 'unknown').replace('_', ' ').title()}")
    
    print("\nDrift Status")
    for key, val in report.items():
        if key.endswith("_drift"):
            drift_name = key.replace("_drift", "").title()
            print(f"{drift_name} Drift: {str(val).upper()}")
            
    print(f"\nAlert Level: {report.get('batch_status', 'unknown').upper()}")
    print(f"Recommended Action: {report.get('recommended_action', 'None')}")
    print("---\n")
    
    print("JSON Output:\n")
    print(json.dumps(report, indent=4))
    
if __name__ == "__main__":
    run_cli()
