# config.py

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

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