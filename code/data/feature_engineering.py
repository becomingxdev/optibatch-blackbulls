# data/feature_engineering.py

import pandas as pd

# Sensor columns to aggregate
_SENSOR_COLS: list[str] = [
    "Temperature_C",
    "Pressure_Bar",
    "Humidity_Percent",
    "Motor_Speed_RPM",
    "Compression_Force_kN",
    "Flow_Rate_LPM",
    "Power_Consumption_kW",
    "Vibration_mm_s",
]

# Statistics to compute for each sensor column
_AGG_STATS: list[str] = ["mean", "std", "min", "max"]


def generate_process_features(process_df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-batch statistical features from time-series process data.

    Uses a single vectorised groupby aggregation dictionary — no row-wise loops.
    Vibration spike detection is computed via groupby transform to stay vectorised.

    Parameters
    ----------
    process_df : pd.DataFrame
        Raw process data with columns including 'Batch_ID' and the sensor columns.

    Returns
    -------
    pd.DataFrame
        One row per Batch_ID with the following engineered features per sensor:
        ``{col}_mean``, ``{col}_std``, ``{col}_min``, ``{col}_max``,
        plus ``total_energy_kwh`` and ``vibration_spikes``.

    Raises
    ------
    KeyError
        If 'Batch_ID' column is absent from process_df.
    """
    if "Batch_ID" not in process_df.columns:
        raise KeyError("'Batch_ID' column not found in process data.")

    # Only aggregate columns that actually exist in the DataFrame
    present_sensors = [c for c in _SENSOR_COLS if c in process_df.columns]

    # ── 1. Build aggregation dictionary in one pass ───────────────────────────
    agg_dict: dict[str, list[str]] = {col: _AGG_STATS for col in present_sensors}

    features: pd.DataFrame = (
        process_df
        .groupby("Batch_ID", sort=False)
        .agg(agg_dict)
    )

    # Flatten multi-level column index → "Temperature_C_mean", etc.
    features.columns = ["_".join(col) for col in features.columns]
    features = features.reset_index()

    # ── 2. Total energy (sum of Power_Consumption_kW) ─────────────────────────
    if "Power_Consumption_kW" in process_df.columns:
        energy: pd.Series = (
            process_df
            .groupby("Batch_ID", sort=False)["Power_Consumption_kW"]
            .sum()
            .rename("total_energy_kwh")
            .reset_index()
        )
        features = features.merge(energy, on="Batch_ID", how="left")

    # ── 3. Vibration spikes (vectorised via transform) ────────────────────────
    if "Vibration_mm_s" in process_df.columns:
        grp = process_df.groupby("Batch_ID", sort=False)["Vibration_mm_s"]
        vib_mean = grp.transform("mean")
        vib_std  = grp.transform("std").fillna(0)

        spike_mask = process_df["Vibration_mm_s"] > (vib_mean + 2 * vib_std)
        spikes: pd.Series = (
            spike_mask
            .groupby(process_df["Batch_ID"])
            .sum()
            .astype(int)
            .rename("vibration_spikes")
            .reset_index()
        )
        features = features.merge(spikes, on="Batch_ID", how="left")

    return features