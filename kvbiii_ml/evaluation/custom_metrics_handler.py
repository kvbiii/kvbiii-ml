from typing import Callable, Any, Dict
import pandas as pd
import numpy as np


class CustomMetricsHandler:
    """Handler for extracting and wrapping custom metrics from configuration dictionaries.

    This class validates custom metric configurations and creates wrapped metric functions
    that properly handle additional data and parameters beyond y_true and y_pred.
    """

    @staticmethod
    def validate_custom_metric_config(custom_metric: Dict[str, Any]) -> None:
        """Validate a custom metric configuration dictionary.

        Args:
            custom_metric (Dict[str, Any]): Custom metric configuration with required keys.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if not isinstance(custom_metric, dict):
            raise ValueError("custom_metric must be a dictionary")

        required_keys = {"name", "function", "metric_type", "direction"}
        missing_keys = required_keys - set(custom_metric.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in custom_metric: {missing_keys}")

        if not callable(custom_metric["function"]):
            raise ValueError("custom_metric['function'] must be callable")

        metric_type = custom_metric["metric_type"]
        if metric_type not in {"preds", "probs"}:
            raise ValueError("metric_type must be one of: 'preds', 'probs'")

        if "kwargs" in custom_metric and not isinstance(custom_metric["kwargs"], dict):
            raise ValueError("kwargs must be a dictionary")

    @staticmethod
    def extract_metric_details(
        custom_metric: Dict[str, Any],
    ) -> tuple[str, Callable, str, str]:
        """Extract and validate metric details from a configuration dictionary.

        This method validates the custom metric configuration and returns a wrapped
        metric function that properly handles parameters.

        Args:
            custom_metric (Dict[str, Any]): Configuration dictionary with keys:
                - name (str): Metric name
                - function (Callable): The metric function
                - metric_type (str): Type of metric ("preds" or "probs")
                - kwargs (Dict[str, Any], optional): Parameters to pass to the metric function

        Returns:
            tuple[str, Callable, str, str]: Tuple of (metric_name, wrapped_metric_function, metric_type, metric_direction)

        Raises:
            ValueError: If the configuration is invalid.
        """
        CustomMetricsHandler.validate_custom_metric_config(custom_metric)

        metric_name = custom_metric["name"]
        metric_function = custom_metric["function"]
        metric_type = custom_metric["metric_type"]
        metric_direction = custom_metric["direction"]
        kwargs = custom_metric.get("kwargs", {})

        def wrapped_metric(
            y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
        ) -> float:
            """Wrapped metric function that injects parameters."""
            return metric_function(y_true, y_pred, **kwargs)

        return metric_name, wrapped_metric, metric_type, metric_direction


def f_beta_selection_score(
    y_true_labels: pd.Series,
    y_pred_probs: pd.Series | np.ndarray,
    y_true_points: pd.Series,
    threshold: float = 0.5,
    beta: float = 0.1,
    normalize_by: float | None = None,
    min_selected: int = 1,
) -> float:
    """
    Computes an F-beta style harmonic mean between normalized average points and selection coverage for binary classification tasks.

    Args:
        y_true_labels (pd.Series): Actual binary labels (0 or 1).
        y_pred_probs (pd.Series | np.ndarray): Predicted probabilities for positive class.
        y_true_points (pd.Series): Actual target values (e.g., FPL points).
        threshold (float): Probability threshold for selection. Default is 0.5.
        beta (float): Weight parameter (beta>1 favors precision/avg points). Default is 0.1.
        normalize_by (float | None): Value to normalize average points (None uses max points). Default is None.
        min_selected (int): Minimum selections required to avoid zero score. Default is 1.

    Returns:
        float: Combined score in range [0, 1] where higher is better
    """
    y_true_points = y_true_points.loc[y_true_labels.index]
    y_true_labels = np.asarray(y_true_labels)
    y_pred_probs = np.asarray(y_pred_probs)
    y_true_points = np.asarray(y_true_points)
    selected_mask = y_pred_probs >= threshold
    n_selected = np.sum(selected_mask)
    if n_selected < min_selected:
        return 0.0
    avg_points = y_true_points[selected_mask].mean()
    coverage = n_selected / len(y_true_points)
    if normalize_by is None:
        norm_base = max(y_true_points.max(), 1.0)
    else:
        norm_base = normalize_by
    norm_avg = min(avg_points / norm_base, 1.0)
    if norm_avg == 0 and coverage == 0:
        return 0.0
    beta_sq = beta * beta
    numerator = (1 + beta_sq) * norm_avg * coverage
    denominator = beta_sq * norm_avg + coverage
    return numerator / denominator


if __name__ == "__main__":
    example_points = pd.Series([2, 5, 8, 1, 12, 3, 7, 9, 4, 6])
    example_labels = pd.Series([1 if p >= 5 else 0 for p in example_points])
    custom_metric_config = {
        "name": "f_beta_selection",
        "function": f_beta_selection_score,
        "metric_type": "probs",
        "direction": "maximize",
        "kwargs": {
            "y_true_points": example_points,
            "threshold": 0.6,
            "beta": 0.2,
            "min_selected": 2,
        },
    }
    metric_name, metric_fn, metric_type = CustomMetricsHandler.extract_metric_details(
        custom_metric_config
    )
    y_pred_example = np.array([0.7, 0.8, 0.3, 0.9, 0.4, 0.6, 0.2, 0.8, 0.5, 0.7])
    score = metric_fn(example_points, y_pred_example)
    print(f"Metric name: {metric_name}")
    print(f"Metric type: {metric_type}")
    print(f"F-beta selection score: {score:.4f}")
