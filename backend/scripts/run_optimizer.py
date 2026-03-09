import os
import sys
import json
import logging

# Ensure we can import from optibatch modules if script is run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optibatch.optimization.parameter_optimizer import optimize_batch_parameters
from optibatch.prediction.model_evaluator import convert_to_performance_class

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
    
    # Simulated predicted metrics
    sample_metrics = {
        "predicted_yield": 0.94,
        "predicted_quality": 96.5,
        "predicted_performance": 1.05,
        "predicted_energy": 158.0
    }
    
    # We use yield scaled up as an example to categorize performance if needed
    performance_val = sample_metrics["predicted_yield"] * 100
    batch_class = convert_to_performance_class([performance_val])[0].title()
    
    # Generate recommendations
    report = optimize_batch_parameters(sample_parameters, sample_metrics)
    
    if "error" in report:
        logger.error(report["error"])
        return
        
    print("\n--- OptiBatch Optimization Report ---")
    print(f"Current Batch Class: {batch_class}")
    target_sig = report["target_signature"].replace('_', ' ').title()
    print(f"Target Signature: {target_sig}\n")
    
    print("Recommended Adjustments")
    for param, adj in report.get("parameter_recommendations", {}).items():
        print(f"{param.replace('_', ' ').title()}: {adj}")
        
    print("\nExpected Improvements")
    for metric, imp in report.get("expected_metric_improvement", {}).items():
        print(f"{metric.replace('predicted_', '').title()}: {imp}")
        
    conf = report.get("optimization_confidence", 0.0)
    print(f"\nOptimization Confidence: {conf:.2f}")
    print("-------------------------------------\n")
    
    # Print JSON output for completeness
    print("JSON Output:\n")
    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    run_cli()
