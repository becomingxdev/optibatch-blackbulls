import os
import sys
import json
import logging

# Ensure we can import from optibatch modules if script is run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optibatch.simulation.batch_simulator import simulate_batch, run_parameter_sweep

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def run_cli():
    # Simulated batch parameters
    sample_parameters = {
        "temperature": 182.0,
        "hold_time": 47.0,
        "pressure": 2.15,
        "catalyst_ratio": 1.2,
        "reaction_time": 120.0,
        "mixing_speed": 300.0
    }
    
    report = simulate_batch(sample_parameters)
    
    if "error" in report:
        logger.error(report["error"])
        return
        
    print("\n--- OptiBatch Batch Simulation Report ---\n")
    
    print("Predicted Metrics")
    for metric, val in report.get("predicted_metrics", {}).items():
         print(f"{metric.title()}: {val:.2f}" if isinstance(val, (int, float)) else f"{metric.title()}: {val}")
             
    print(f"\nClosest Signature: {report.get('closest_signature', 'unknown').replace('_', ' ').title()}")
    print(f"Performance Class: {report.get('performance_class', 'unknown').title()}")
    print(f"Drift Risk: {report.get('drift_risk', 0.0):.2f}\n")
    
    print("Suggested Improvements")
    for param, adj in report.get("optimization_suggestions", {}).items():
        print(f"{param.replace('_', ' ').title()}: {adj}")

    print("---\n")
    
    print("Running Parameter Sweep Simulation (Monte Carlo)...")
    ranges = {
        "temperature": [180.0, 190.0],
        "hold_time": [40.0, 50.0],
        "pressure": [2.0, 2.5]
    }
    sweep_results = run_parameter_sweep(ranges, num_simulations=50)
    best_batches = sweep_results.get("best_simulated_batches", [])
    if best_batches:
        print(f"\nTop configuration found:")
        top = best_batches[0]
        for k, v in top.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print("\nJSON Output:\n")
    print(json.dumps(report, indent=4))
    
if __name__ == "__main__":
    run_cli()
