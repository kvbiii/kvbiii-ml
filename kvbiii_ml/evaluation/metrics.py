"""Centralized metrics definitions for machine learning evaluation.

This module provides standardized metric functions and configurations that can be
used across different components of the kvbiii_ml package for consistent evaluation
of machine learning models.
"""

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
    root_mean_squared_log_error,
    r2_score,
)


METRICS_NAMES: dict[str, callable] = {
    "Accuracy": accuracy_score,
    "Balanced Accuracy": balanced_accuracy_score,
    "F1": f1_score,
    "F1 (Micro)": lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro"),
    "F1 (Macro)": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
    "F1 (Weighted)": lambda y_true, y_pred: f1_score(
        y_true, y_pred, average="weighted"
    ),
    "Recall": recall_score,
    "Precision": precision_score,
    "Roc AUC": roc_auc_score,
    "MAE": mean_absolute_error,
    "MAPE": mean_absolute_percentage_error,
    "MSE": mean_squared_error,
    "RMSE": root_mean_squared_error,
    "RMSLE": root_mean_squared_log_error,
    "R2": r2_score,
}

METRICS: dict[str, list] = {
    key: [
        eval_func,
        "preds" if key not in ["Roc AUC", "Mean Average Precision"] else "probs",
        "minimize" if key in ["MAE", "MAPE", "MSE", "RMSE", "RMSLE"] else "maximize",
    ]
    for key, eval_func in METRICS_NAMES.items()
}


def get_metric_function(metric_name: str) -> callable:
    """Gets the metric function by name.

    Args:
        metric_name (str): Name of the metric.

    Returns:
        callable: The metric function.

    Raises:
        ValueError: If metric name is not found.
    """
    if metric_name not in METRICS:
        raise ValueError(
            f"Metric '{metric_name}' not found. Available metrics: {list(METRICS.keys())}"
        )
    return METRICS[metric_name][0]


def get_metric_type(metric_name: str) -> str:
    """Gets the prediction type required for the metric.

    Args:
        metric_name (str): Name of the metric.

    Returns:
        str: Either 'preds' for predictions or 'probs' for probabilities.

    Raises:
        ValueError: If metric name is not found.
    """
    if metric_name not in METRICS:
        raise ValueError(
            f"Metric '{metric_name}' not found. Available metrics: {list(METRICS.keys())}"
        )
    return METRICS[metric_name][1]


def get_metric_direction(metric_name: str) -> str:
    """Gets the optimization direction for the metric.

    Args:
        metric_name (str): Name of the metric.

    Returns:
        str: Either 'minimize' or 'maximize'.

    Raises:
        ValueError: If metric name is not found.
    """
    if metric_name not in METRICS:
        raise ValueError(
            f"Metric '{metric_name}' not found. Available metrics: {list(METRICS.keys())}"
        )
    return METRICS[metric_name][2]


def list_available_metrics() -> list[str]:
    """Lists all available metric names.

    Returns:
        list[str]: List of available metric names.
    """
    return list(METRICS.keys())


if __name__ == "__main__":
    import numpy as np

    print("📊 Available Metrics in kvbiii_ml")
    print("=" * 50)

    classification_metrics = []
    regression_metrics = []

    for metric_name in list_available_metrics():
        if metric_name in ["MAE", "MAPE", "MSE", "RMSE", "RMSLE", "R2"]:
            regression_metrics.append(metric_name)
        else:
            classification_metrics.append(metric_name)

    print("\n🎯 Classification Metrics:")
    for metric in classification_metrics:
        direction = get_metric_direction(metric)
        pred_type = get_metric_type(metric)
        print(f"  • {metric:<20} | {direction:<8} | requires {pred_type}")

    print("\n📈 Regression Metrics:")
    for metric in regression_metrics:
        direction = get_metric_direction(metric)
        pred_type = get_metric_type(metric)
        print(f"  • {metric:<20} | {direction:<8} | requires {pred_type}")

    print(f"\nTotal available metrics: {len(list_available_metrics())}")

    # Example usage
    print("\n🔍 Example Usage:")
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    accuracy_func = get_metric_function("Accuracy")
    accuracy_score_result = accuracy_func(y_true, y_pred)
    print(f"Accuracy score: {accuracy_score_result:.4f}")

    f1_func = get_metric_function("F1")
    f1_score_result = f1_func(y_true, y_pred)
    print(f"F1 score: {f1_score_result:.4f}")
