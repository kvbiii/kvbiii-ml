from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


class CustomMetricsHandler:
    """Handler for extracting and wrapping custom metrics from configuration dictionaries.

    This class validates custom metric configurations and creates wrapped metric functions
    that properly handle additional data and parameters beyond y_true and y_pred.
    """

    @staticmethod
    def validate_custom_metric_config(custom_metric: dict[str, Any]) -> None:
        """Validate a custom metric configuration dictionary.

        Args:
            custom_metric (dict[str, Any]): Custom metric configuration with required keys.

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

        metric_direction = custom_metric["direction"]
        if metric_direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be one of: 'minimize', 'maximize'")

        if "kwargs" in custom_metric and not isinstance(custom_metric["kwargs"], dict):
            raise ValueError("kwargs must be a dictionary")

    @staticmethod
    def extract_metric_details(
        custom_metric: dict[str, Any],
    ) -> tuple[str, Callable[..., float], str, str]:
        """Extract and validate metric details from a configuration dictionary.

        This method validates the custom metric configuration and returns a wrapped
        metric function that properly handles parameters.

        Args:
            custom_metric (dict[str, Any]): Configuration dictionary with keys:
                - name (str): Metric name. Required.
                - function (Callable[..., float]): ``f(y_true, y_pred, **kwargs)``. Required.
                - metric_type (str): "preds" or "probs". Required.
                - direction (str): "minimize" or "maximize". Required.
                - kwargs (dict[str, Any], optional): Extra keyword arguments passed to
                  the metric function on every call.

        Returns:
            tuple[str, Callable[..., float], str, str]: Tuple of (metric_name, wrapped_metric_function, metric_type, metric_direction)

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
