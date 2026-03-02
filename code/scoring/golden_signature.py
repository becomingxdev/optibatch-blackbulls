def generate_golden_signature(df, percentile=0.90):

    threshold = df["composite_score"].quantile(percentile)
    top_batches = df[df["composite_score"] >= threshold]

    feature_cols = [
        c for c in df.columns
        if c not in [
            "batch_id",
            "quality_score",
            "yield_score",
            "performance_score",
            "energy_score",
            "composite_score"
        ]
    ]

    golden_mean = top_batches[feature_cols].mean()
    golden_std = top_batches[feature_cols].std()

    return golden_mean, golden_std