import numpy as np
import pytest

from kvbiii_ml.evaluation.custom_metrics_handler import (
    CustomMetricsHandler,
)


def _valid_metric_config() -> dict:
    """Builds a minimal, valid custom metric configuration dictionary.

    Returns:
        dict: A configuration dictionary satisfying validate_custom_metric_config.
    """
    return {
        "name": "custom_sum",
        "function": lambda y_true, y_pred: float(np.sum(y_pred)),
        "metric_type": "preds",
        "direction": "maximize",
    }


def test_custom_metrics_handler_validate_raises_error_for_non_dict_input():
    """Tests that a non-dict custom_metric is rejected.

    Asserts:
        - ValueError is raised when custom_metric is not a dict.
    """
    with pytest.raises(ValueError, match="dictionary"):
        CustomMetricsHandler.validate_custom_metric_config(["not", "a", "dict"])


@pytest.mark.parametrize(
    "missing_key", ["name", "function", "metric_type", "direction"]
)
def test_custom_metrics_handler_validate_raises_error_for_missing_required_key(
    missing_key,
):
    """Tests that a config missing any required key is rejected.

    Args:
        missing_key (str): The required key removed from an otherwise valid config.

    Asserts:
        - ValueError is raised when a required key is absent.
    """
    config = _valid_metric_config()
    del config[missing_key]
    with pytest.raises(ValueError, match="Missing required keys"):
        CustomMetricsHandler.validate_custom_metric_config(config)


def test_custom_metrics_handler_validate_raises_error_for_non_callable_function():
    """Tests that a non-callable 'function' value is rejected.

    Asserts:
        - ValueError is raised when custom_metric['function'] is not callable.
    """
    config = _valid_metric_config()
    config["function"] = "not_callable"
    with pytest.raises(ValueError, match="callable"):
        CustomMetricsHandler.validate_custom_metric_config(config)


def test_custom_metrics_handler_validate_raises_error_for_invalid_metric_type():
    """Tests that an unsupported metric_type is rejected.

    Asserts:
        - ValueError is raised when metric_type is not "preds"/"probs".
    """
    config = _valid_metric_config()
    config["metric_type"] = "scores"
    with pytest.raises(ValueError, match="metric_type"):
        CustomMetricsHandler.validate_custom_metric_config(config)


def test_custom_metrics_handler_validate_raises_error_for_invalid_direction():
    """Tests that an unsupported direction is rejected.

    Asserts:
        - ValueError is raised when direction is not "minimize"/"maximize".
    """
    config = _valid_metric_config()
    config["direction"] = "sideways"
    with pytest.raises(ValueError, match="direction"):
        CustomMetricsHandler.validate_custom_metric_config(config)


def test_custom_metrics_handler_validate_raises_error_for_non_dict_kwargs():
    """Tests that a non-dict 'kwargs' value is rejected.

    Asserts:
        - ValueError is raised when custom_metric['kwargs'] is present but not a dict.
    """
    config = _valid_metric_config()
    config["kwargs"] = ["not", "a", "dict"]
    with pytest.raises(ValueError, match="kwargs"):
        CustomMetricsHandler.validate_custom_metric_config(config)


def test_custom_metrics_handler_validate_accepts_a_well_formed_config():
    """Tests that a fully valid configuration does not raise.

    Asserts:
        - validate_custom_metric_config returns None without raising for a valid config.
    """
    if (
        CustomMetricsHandler.validate_custom_metric_config(_valid_metric_config())
        is not None
    ):
        raise AssertionError(
            "validate_custom_metric_config should return None on success."
        )


def test_custom_metrics_handler_extract_metric_details_returns_expected_tuple():
    """Tests that extract_metric_details unpacks a valid config correctly.

    Asserts:
        - The returned tuple contains the configured name, metric_type, and direction.
        - The returned wrapped function is callable.
    """
    config = _valid_metric_config()
    name, wrapped_fn, metric_type, direction = (
        CustomMetricsHandler.extract_metric_details(config)
    )
    if name != "custom_sum":
        raise AssertionError("Unexpected metric name.")
    if metric_type != "preds":
        raise AssertionError("Unexpected metric_type.")
    if direction != "maximize":
        raise AssertionError("Unexpected direction.")
    if not callable(wrapped_fn):
        raise AssertionError("Wrapped metric function is not callable.")


def test_custom_metrics_handler_extract_metric_details_injects_kwargs():
    """Tests that the wrapped metric function injects configured kwargs.

    Asserts:
        - Calling the wrapped function applies the configured kwargs, changing
          the output relative to calling the raw function with defaults.
    """

    def metric_fn(y_true, y_pred, multiplier=1.0):
        """Sums predictions scaled by a multiplier."""
        return float(np.sum(y_pred) * multiplier)

    config = {
        "name": "scaled_sum",
        "function": metric_fn,
        "metric_type": "preds",
        "direction": "maximize",
        "kwargs": {"multiplier": 2.0},
    }
    _, wrapped_fn, _, _ = CustomMetricsHandler.extract_metric_details(config)
    y_true = np.array([0, 1, 0])
    y_pred = np.array([1, 2, 3])
    result = wrapped_fn(y_true, y_pred)
    if result != pytest.approx(12.0):
        raise AssertionError(f"Expected kwargs-scaled result 12.0, got {result}.")


def test_custom_metrics_handler_extract_metric_details_propagates_validation_error():
    """Tests that extract_metric_details validates before extracting.

    Asserts:
        - ValueError is raised for an invalid config, matching validate's error.
    """
    config = _valid_metric_config()
    config["direction"] = "invalid"
    with pytest.raises(ValueError, match="direction"):
        CustomMetricsHandler.extract_metric_details(config)
