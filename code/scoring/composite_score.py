from sklearn.preprocessing import MinMaxScaler

def compute_scores(df, weights):

    scaler = MinMaxScaler()

    quality_cols = ["dissolution_rate", "Content_uniformity", "hardness"]

    if "fribal" in df.columns:
        df["friability_inv"] = 1 - scaler.fit_transform(df[["fribal"]])
        quality_cols.append("friability_inv")

    df["quality_score"] = scaler.fit_transform(df[quality_cols]).mean(axis=1) * 100
    df["yield_score"] = scaler.fit_transform(df[["table_weight"]]).flatten() * 100

    if "vibration_spikes" in df.columns:
        df["performance_score"] = (
            1 - scaler.fit_transform(df[["vibration_spikes"]])
        ).flatten() * 100
    else:
        df["performance_score"] = 80

    if "total_energy_kwh" in df.columns:
        df["energy_score"] = (
            1 - scaler.fit_transform(df[["total_energy_kwh"]])
        ).flatten() * 100
    else:
        df["energy_score"] = 50

    df["composite_score"] = (
        weights["quality"] * df["quality_score"] +
        weights["yield"] * df["yield_score"] +
        weights["performance"] * df["performance_score"] +
        weights["energy"] * df["energy_score"]
    )

    return df