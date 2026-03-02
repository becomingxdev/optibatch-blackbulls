# data/loader.py

import sys
import os
import pandas as pd

# Allow imports from the code/ root when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PRODUCTION_FILE, PROCESS_FILE

# ── Column contracts ──────────────────────────────────────────────────────────
PRODUCTION_COLS: list[str] = [
    "Batch_ID", "Granulation_Time", "Binder_Amount", "Drying_Temp",
    "Drying_Time", "Compression_Force", "Machine_Speed", "Lubricant_Conc",
    "Moisture_Content", "Tablet_Weight", "Hardness", "Friability",
    "Disintegration_Time", "Dissolution_Rate", "Content_Uniformity",
]

PROCESS_NUMERIC_COLS: list[str] = [
    "Time_Minutes", "Temperature_C", "Pressure_Bar", "Humidity_Percent",
    "Motor_Speed_RPM", "Compression_Force_kN", "Flow_Rate_LPM",
    "Power_Consumption_kW", "Vibration_mm_s",
]


def load_production_data() -> pd.DataFrame:
    """Load the first sheet of the production Excel file.

    Returns
    -------
    pd.DataFrame
        Production data with enforced numeric types on known numeric columns.

    Raises
    ------
    FileNotFoundError
        If the production file does not exist at the configured path.
    """
    if not os.path.exists(PRODUCTION_FILE):
        raise FileNotFoundError(
            f"Production file not found: {PRODUCTION_FILE}"
        )

    df: pd.DataFrame = pd.read_excel(PRODUCTION_FILE, sheet_name=0)

    # Enforce numeric dtypes for all columns except Batch_ID
    numeric_cols = [c for c in PRODUCTION_COLS if c != "Batch_ID" and c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    print(f"[loader] Production data loaded: {df.shape}")
    return df


def load_process_data() -> pd.DataFrame:
    """Load and concatenate all sheets from the process Excel file.

    Returns
    -------
    pd.DataFrame
        Combined process data across all batch sheets with enforced numeric types.

    Raises
    ------
    FileNotFoundError
        If the process file does not exist at the configured path.
    ValueError
        If the process file contains no parseable sheets.
    """
    if not os.path.exists(PROCESS_FILE):
        raise FileNotFoundError(
            f"Process file not found: {PROCESS_FILE}"
        )

    xl = pd.ExcelFile(PROCESS_FILE)
    if not xl.sheet_names:
        raise ValueError(f"Process file has no sheets: {PROCESS_FILE}")

    # List-comprehension concat — no manual loop appending
    frames: list[pd.DataFrame] = [xl.parse(sheet) for sheet in xl.sheet_names]
    process_df: pd.DataFrame = pd.concat(frames, ignore_index=True)

    # Enforce numeric dtypes
    existing_numeric = [c for c in PROCESS_NUMERIC_COLS if c in process_df.columns]
    process_df[existing_numeric] = process_df[existing_numeric].apply(
        pd.to_numeric, errors="coerce"
    )

    print(f"[loader] Process data loaded : {process_df.shape} ({len(xl.sheet_names)} sheets)")
    return process_df