# config.py

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Subdirectories for organized outputs
RAW_DATA_DIR          = os.path.join(OUTPUT_DIR, "raw_batches")
ADVANCED_ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "advanced_analysis")
PARETO_DIR           = os.path.join(OUTPUT_DIR, "pareto_analysis")
MONITORING_DIR       = os.path.join(OUTPUT_DIR, "monitoring")
ML_MODELS_DIR        = os.path.join(OUTPUT_DIR, "ml_models")

# Dataset paths (exact filenames on disk)
PRODUCTION_FILE = os.path.join(DATA_DIR, "_h_batch_production_data.xlsx")
PROCESS_FILE    = os.path.join(DATA_DIR, "_h_batch_process_data.xlsx")

# ── Scoring Weights ───────────────────────────────────────────────────────────
DEFAULT_WEIGHTS: dict[str, float] = {
    "quality":     0.40,
    "yield":       0.30,
    "performance": 0.20,
    "energy":      0.10,
}

# ── Thresholds ────────────────────────────────────────────────────────────────
Z_THRESHOLD: float = 2.0          # vibration spike detection multiplier
GOLDEN_PERCENTILE: float = 0.90   # top-N% threshold for golden signature