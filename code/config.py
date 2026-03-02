# config.py

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Dataset paths
PRODUCTION_FILE = os.path.join(DATA_DIR, "h_batch_production_data.xlsx")
PROCESS_FILE = os.path.join(DATA_DIR, "h_batch_process_data.xlsx")

# Scoring Weights (default balanced mode)
DEFAULT_WEIGHTS = {
    "quality": 0.40,
    "yield": 0.30,
    "performance": 0.20,
    "energy": 0.10
}

# Deviation threshold
Z_THRESHOLD = 2.0