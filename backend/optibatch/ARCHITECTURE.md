# OptiBatch Backend Architecture

This document serves as the permanent reference for the "OptiBatch Backend Architecture". It describes the base structure and requirements for the AI-driven manufacturing optimization system.

## Project Context
The backend architecture is designed for an AI-driven manufacturing optimization system that includes:
- Golden Signature optimization
- Multi-objective Pareto optimization
- ML prediction models
- Energy pattern analytics
- Carbon emission tracking
- Continuous learning
- Real-time batch monitoring
- REST APIs
- Data pipeline
- Industrial ROI validation

## Directory Structure
The defined system directory tree is as follows:

```text
optibatch/
│
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
│
├── data/
│   ├── raw/
│   │   ├── batch_production_data.xlsx
│   │   └── batch_process_data.xlsx
│   │
│   ├── processed/
│   │   └── cleaned_batches.csv
│   │
│   └── simulated_stream/
│       └── realtime_batches.csv
│
├── logs/
│   └── system_logs.log
│
├── models/
│   ├── trained_models/
│   │   ├── yield_model.pkl
│   │   ├── quality_model.pkl
│   │   ├── performance_model.pkl
│   │   └── energy_model.pkl
│   │
│   └── model_metadata.json
│
├── golden_signatures/
│   ├── golden_signature_db.json
│   └── signature_history.json
│
├── optimization/
│   ├── pareto_optimizer.py
│   ├── optimization_objective.py
│   └── adaptive_weight_manager.py
│
├── prediction/
│   ├── train_models.py
│   ├── predict_metrics.py
│   └── model_evaluator.py
│
├── energy_analysis/
│   ├── energy_pattern_analysis.py
│   ├── carbon_emission_calculator.py
│   └── savings_projection.py
│
├── anomaly_detection/
│   └── anomaly_detector.py
│
├── continuous_learning/
│   ├── signature_updater.py
│   └── retraining_pipeline.py
│
├── monitoring/
│   ├── batch_comparator.py
│   └── realtime_monitor.py
│
├── explainability/
│   └── shap_explainer.py
│
├── data_pipeline/
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── feature_engineering.py
│   └── data_validator.py
│
├── api/
│   ├── api_server.py
│   ├── routes_optimization.py
│   ├── routes_prediction.py
│   ├── routes_signatures.py
│   └── routes_monitoring.py
│
├── utils/
│   ├── constants.py
│   ├── helper_functions.py
│   └── logger.py
│
└── validation/
    ├── roi_simulator.py
    └── industrial_validation.py
```

## Architecture Rules
1. **Directory Integrity**: The directory structure is exactly preserved and treated as the permanent backend architecture for the OptiBatch project.
2. **Setup Rules**: 
   - Every `.py` file requires ONE SINGLE LONG COMMENT at the top explaining the purpose of that file.
   - Placeholder JSON/CSV/XLSX/PKL/LOG files must be established in their respective directories.
   - Example Python header format:
     ```python
     """
     This file is responsible for <describe the role>.
     It is part of the <module name> module and will later contain the implementation for <feature description>.
     """
     ```
