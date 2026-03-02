import numpy as np
import pandas as pd

def generate_process_features(process_df):

    batch_features = []

    grouped = process_df.groupby("batch_id")

    for batch_id, df in grouped:

        features = {"batch_id": batch_id}

        numeric_cols = [
            "temperature", "pressure", "humidity",
            "motor_speed", "compression_force",
            "flow_rate", "power_consumption", "vibration_mm_s"
        ]

        for col in numeric_cols:
            if col in df.columns:
                features[f"{col}_mean"] = df[col].mean()
                features[f"{col}_std"] = df[col].std()
                features[f"{col}_max"] = df[col].max()
                features[f"{col}_trend"] = np.polyfit(
                    range(len(df)), df[col].fillna(method="ffill"), 1
                )[0]

        if "power_consumption" in df.columns:
            features["total_energy_kwh"] = df["power_consumption"].sum()

        if "vibration_mm_s" in df.columns:
            mean_v = df["vibration_mm_s"].mean()
            std_v = df["vibration_mm_s"].std()
            features["vibration_spikes"] = (
                (df["vibration_mm_s"] > mean_v + 2 * std_v)
            ).sum()

        batch_features.append(features)

    return pd.DataFrame(batch_features)