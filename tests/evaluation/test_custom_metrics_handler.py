import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.evaluation.custom_metrics_handler import (
    CustomMetricsHandler,
    FBetaSelectionConfig,
    f_beta_selection_score,
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


def test_fbetaselectionconfig_default_values():
    """Tests the default field values of FBetaSelectionConfig.

    Asserts:
        - threshold, beta, normalize_by, and min_selected match documented defaults.
    """
    config = FBetaSelectionConfig()
    if config.threshold != 0.5:
        raise AssertionError("Unexpected default threshold.")
    if config.beta != 0.1:
        raise AssertionError("Unexpected default beta.")
    if config.normalize_by is not None:
        raise AssertionError("Unexpected default normalize_by.")
    if config.min_selected != 1:
        raise AssertionError("Unexpected default min_selected.")


def test_f_beta_selection_score_returns_zero_when_fewer_than_min_selected():
    """Tests the min_selected guard when nothing clears the probability threshold.

    Asserts:
        - The score is exactly 0.0 when n_selected < config.min_selected.
    """
    y_true_labels = pd.Series([0, 1, 0, 1, 0])
    y_true_points = pd.Series([1, 2, 3, 4, 5])
    y_pred_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    score = f_beta_selection_score(
        y_true_labels, y_pred_probs, y_true_points, FBetaSelectionConfig(min_selected=1)
    )
    if score != 0.0:
        raise AssertionError(f"Expected 0.0 for below-min_selected case, got {score}.")


def test_f_beta_selection_score_returns_zero_when_selected_points_average_zero():
    """Tests that an all-zero selected-points average yields a zero score.

    Asserts:
        - The score is exactly 0.0 when every selected point's true value is 0,
          even though coverage (n_selected / total) is nonzero.
    """
    y_true_labels = pd.Series([0, 1, 0, 1, 0])
    y_true_points = pd.Series([0, 0, 0, 0, 0])
    y_pred_probs = np.array([0.9, 0.1, 0.9, 0.1, 0.9])
    score = f_beta_selection_score(
        y_true_labels,
        y_pred_probs,
        y_true_points,
        FBetaSelectionConfig(threshold=0.5, min_selected=1, normalize_by=1.0),
    )
    if score != 0.0:
        raise AssertionError(f"Expected 0.0 for zero-average selection, got {score}.")


def test_f_beta_selection_score_matches_expected_value_for_well_posed_case():
    """Tests a well-posed case against a manually verified expected score.

    Asserts:
        - The score matches the exact value confirmed by direct computation
          (avg_points=4.0, norm_base=5, coverage=0.6, beta=0.1 => 0.6).
    """
    y_true_labels = pd.Series([0, 1, 0, 1, 0])
    y_true_points = pd.Series([1, 2, 3, 4, 5])
    y_pred_probs = np.array([0.9, 0.1, 0.9, 0.1, 0.9])
    score = f_beta_selection_score(
        y_true_labels, y_pred_probs, y_true_points, FBetaSelectionConfig(threshold=0.5)
    )
    if score != pytest.approx(0.6, abs=1e-9):
        raise AssertionError(f"Expected 0.6, got {score}.")


def test_f_beta_selection_score_aligns_points_by_index():
    """Tests that y_true_points is realigned to y_true_labels' index before use.

    Asserts:
        - Passing y_true_points with extra rows outside y_true_labels' index does
          not affect the result relative to a version with matching rows only.
    """
    y_true_labels = pd.Series([0, 1, 0], index=[10, 11, 12])
    y_true_points_matching = pd.Series([1, 2, 3], index=[10, 11, 12])
    y_true_points_superset = pd.Series([1, 2, 3, 999], index=[10, 11, 12, 13])
    y_pred_probs = np.array([0.9, 0.9, 0.1])
    config = FBetaSelectionConfig(threshold=0.5, min_selected=1)

    score_matching = f_beta_selection_score(
        y_true_labels, y_pred_probs, y_true_points_matching, config
    )
    score_superset = f_beta_selection_score(
        y_true_labels, y_pred_probs, y_true_points_superset, config
    )
    if score_matching != pytest.approx(score_superset):
        raise AssertionError(
            "Extra rows in y_true_points outside the label index changed the score."
        )


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
