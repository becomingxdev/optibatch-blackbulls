"""
This file is responsible for handling operations related to model evaluator.
It is part of the prediction module and will later contain the implementation for features associated with model evaluator.
"""

import logging

logger = logging.getLogger(__name__)

def convert_to_performance_class(values):
    """Convert continuous values into performance categories."""
    classes = []
    for val in values:
        if val >= 90:
            classes.append("excellent")
        elif val >= 80:
            classes.append("good")
        elif val >= 70:
            classes.append("average")
        else:
            classes.append("poor")
    return classes

def calculate_regression_metrics(y_true, y_pred) -> dict:
    """
    Calculates regression metrics (MAE, RMSE, MAPE) and additional metrics (accuracy, F1 score).
    """
    import numpy as np  # optional dependency for evaluation workflows
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        mean_absolute_percentage_error,
        f1_score,
    )

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Handle zeros in y_true for MAPE
    try:
        mape_frac = mean_absolute_percentage_error(y_true, y_pred)
        mape = mape_frac * 100.0
        accuracy = max(0.0, 100.0 - mape)
    except Exception:
        mape = np.nan
        accuracy = np.nan
        
    y_true_classes = convert_to_performance_class(y_true)
    y_pred_classes = convert_to_performance_class(y_pred)
    f1 = f1_score(y_true_classes, y_pred_classes, average="weighted", zero_division=0)
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "accuracy": float(accuracy),
        "f1_score": float(f1)
    }

def evaluate_model(model_name: str, model, X_test, y_test) -> dict:
    """
    Evaluates a machine learning model and returns a dictionary of metrics.
    Also prints a formatted evaluation report to the console.
    """
    y_pred = model.predict(X_test)
    metrics = calculate_regression_metrics(y_test, y_pred)
    
    print("\n---")
    print(f"## Model Evaluation Report: {model_name}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAPE: {metrics['mape']:.1f}%")
    print()
    print(f"Accuracy: {metrics['accuracy']:.1f}%")
    print(f"F1 Score: {metrics['f1_score']:.2f}")
    print("---\n")
    
    return metrics
