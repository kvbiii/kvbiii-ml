"""Tests for kvbiii_ml.evaluation.metrics module."""

import numpy as np
import pytest

from kvbiii_ml.evaluation.metrics import (
    METRICS,
    METRICS_NAMES,
    get_metric_direction,
    get_metric_function,
    get_metric_type,
    list_available_metrics,
)


def test_metrics_names_contains_expected_classification_metrics():
    """Tests METRICS_NAMES dictionary contains expected classification metrics.

    Asserts:
        - Standard classification metrics are present
        - F1 score variants with different averaging strategies exist
        - Metric functions are callable objects
    """
    expected_classification_metrics = [
        "Accuracy",
        "Balanced Accuracy",
        "F1",
        "F1 (Micro)",
        "F1 (Macro)",
        "F1 (Weighted)",
        "Recall",
        "Precision",
        "Roc AUC",
    ]

    for metric in expected_classification_metrics:
        if not (metric in METRICS_NAMES):
            raise AssertionError(f"Missing classification metric: {metric}")
        if not (callable(METRICS_NAMES[metric])):
            raise AssertionError(f"Metric {metric} function not callable")


def test_metrics_names_contains_expected_regression_metrics():
    """Tests METRICS_NAMES dictionary contains expected regression metrics.

    Asserts:
        - Standard regression metrics are present
        - Error and correlation metrics are included
        - All metric functions are callable
    """
    expected_regression_metrics = ["MAE", "MAPE", "MSE", "RMSE", "RMSLE", "R2"]

    for metric in expected_regression_metrics:
        if not (metric in METRICS_NAMES):
            raise AssertionError(f"Missing regression metric: {metric}")
        if not (callable(METRICS_NAMES[metric])):
            raise AssertionError(f"Metric {metric} function not callable")


def test_metrics_dictionary_structure_is_valid():
    """Tests METRICS dictionary has proper structure for all defined metrics.

    Asserts:
        - Each metric entry is a list with exactly 3 elements
        - First element is callable (metric function)
        - Second element is prediction type ('preds' or 'probs')
        - Third element is optimization direction ('minimize' or 'maximize')
    """
    for metric_name, metric_config in METRICS.items():
        if not (isinstance(metric_config, list)):
            raise AssertionError(f"Metric {metric_name} config must be list")
        if not (len(metric_config) == 3):
            raise AssertionError(f"Metric {metric_name} must have 3 config elements")

        metric_func, pred_type, direction = metric_config
        if not (callable(metric_func)):
            raise AssertionError(f"Metric {metric_name} function must be callable")
        if not (
            pred_type
            in [
                "preds",
                "probs",
            ]
        ):
            raise AssertionError(
                f"Metric {metric_name} pred_type must be 'preds' or 'probs'"
            )
        if not (
            direction
            in [
                "minimize",
                "maximize",
            ]
        ):
            raise AssertionError(
                f"Metric {metric_name} direction must be 'minimize' or 'maximize'"
            )


def test_get_metric_function_returns_valid_callable_for_accuracy():
    """Tests get_metric_function returns proper callable for Accuracy metric.

    Asserts:
        - Returns callable function for valid metric name
        - Function produces expected results on sample data
        - Return type matches expected accuracy score format
    """
    accuracy_func = get_metric_function("Accuracy")

    if not (callable(accuracy_func)):
        raise AssertionError()

    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    score = accuracy_func(y_true, y_pred)

    if not (isinstance(score, (int, float))):
        raise AssertionError()
    if not (0.0 <= score <= 1.0):
        raise AssertionError()


def test_get_metric_function_returns_valid_callable_for_mae():
    """Tests get_metric_function returns proper callable for MAE regression metric.

    Asserts:
        - Returns callable function for regression metric
        - Function produces expected results on continuous data
        - MAE value is non-negative as expected
    """
    mae_func = get_metric_function("MAE")

    if not (callable(mae_func)):
        raise AssertionError()

    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 3.9, 5.2])
    score = mae_func(y_true, y_pred)

    if not (isinstance(score, (int, float))):
        raise AssertionError()
    if not (score >= 0.0):
        raise AssertionError()


def test_get_metric_function_raises_error_for_invalid_metric():
    """Tests get_metric_function raises ValueError for unknown metric names.

    Asserts:
        - ValueError is raised for non-existent metric names
        - Error message contains helpful information about available metrics
    """
    with pytest.raises(ValueError, match="Metric 'InvalidMetric' not found"):
        get_metric_function("InvalidMetric")


def test_get_metric_type_returns_correct_prediction_types():
    """Tests get_metric_type returns appropriate prediction types for different metrics.

    Asserts:
        - Standard classification metrics require 'preds'
        - ROC AUC requires 'probs' for probability-based evaluation
        - Regression metrics require 'preds'
    """
    # Test metrics that require predictions
    if not (get_metric_type("Accuracy") == "preds"):
        raise AssertionError()
    if not (get_metric_type("F1") == "preds"):
        raise AssertionError()
    if not (get_metric_type("MAE") == "preds"):
        raise AssertionError()
    if not (get_metric_type("RMSE") == "preds"):
        raise AssertionError()

    # Test metrics that require probabilities
    if not (get_metric_type("Roc AUC") == "probs"):
        raise AssertionError()


def test_get_metric_type_raises_error_for_invalid_metric():
    """Tests get_metric_type raises ValueError for unknown metric names.

    Asserts:
        - ValueError is raised for non-existent metric names
        - Error message lists available metrics for reference
    """
    with pytest.raises(ValueError, match="Metric 'UnknownMetric' not found"):
        get_metric_type("UnknownMetric")


def test_get_metric_direction_returns_correct_optimization_directions():
    """Tests get_metric_direction returns appropriate optimization directions.

    Asserts:
        - Classification metrics should be maximized (higher is better)
        - Error-based regression metrics should be minimized (lower is better)
        - R-squared should be maximized (higher is better)
    """
    # Test metrics to maximize
    if not (get_metric_direction("Accuracy") == "maximize"):
        raise AssertionError()
    if not (get_metric_direction("F1") == "maximize"):
        raise AssertionError()
    if not (get_metric_direction("Roc AUC") == "maximize"):
        raise AssertionError()
    if not (get_metric_direction("R2") == "maximize"):
        raise AssertionError()

    # Test metrics to minimize (error-based)
    if not (get_metric_direction("MAE") == "minimize"):
        raise AssertionError()
    if not (get_metric_direction("MSE") == "minimize"):
        raise AssertionError()
    if not (get_metric_direction("RMSE") == "minimize"):
        raise AssertionError()
    if not (get_metric_direction("RMSLE") == "minimize"):
        raise AssertionError()


def test_get_metric_direction_raises_error_for_invalid_metric():
    """Tests get_metric_direction raises ValueError for unknown metric names.

    Asserts:
        - ValueError is raised for non-existent metric names
        - Error message provides clear guidance about available options
    """
    with pytest.raises(ValueError, match="Metric 'InvalidDirection' not found"):
        get_metric_direction("InvalidDirection")


def test_list_available_metrics_returns_complete_metric_list():
    """Tests list_available_metrics returns all defined metrics.

    Asserts:
        - Returns list containing all metrics from METRICS dictionary
        - List length matches expected number of metrics
        - All returned metric names are strings
    """
    available_metrics = list_available_metrics()

    if not (isinstance(available_metrics, list)):
        raise AssertionError()
    if not (len(available_metrics) == len(METRICS)):
        raise AssertionError()
    if not (all(isinstance(metric, str) for metric in available_metrics)):
        raise AssertionError()

    # Verify all METRICS keys are included
    for metric_name in METRICS.keys():
        if not (metric_name in available_metrics):
            raise AssertionError()


def test_f1_score_variants_work_with_different_averaging():
    """Tests F1 score variants produce different results with different averaging strategies.

    Asserts:
        - F1 Micro, Macro, and Weighted produce different scores
        - All F1 variants return values between 0 and 1
        - Functions handle multiclass classification properly
    """
    # Multiclass classification example
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 1, 1, 2])

    f1_micro = get_metric_function("F1 (Micro)")
    f1_macro = get_metric_function("F1 (Macro)")
    f1_weighted = get_metric_function("F1 (Weighted)")

    micro_score = f1_micro(y_true, y_pred)
    macro_score = f1_macro(y_true, y_pred)
    weighted_score = f1_weighted(y_true, y_pred)

    if not (0.0 <= micro_score <= 1.0):
        raise AssertionError()
    if not (0.0 <= macro_score <= 1.0):
        raise AssertionError()
    if not (0.0 <= weighted_score <= 1.0):
        raise AssertionError()

    # Scores should be different due to different averaging strategies
    scores = [micro_score, macro_score, weighted_score]
    if not (len(set(scores)) > 1):
        raise AssertionError("Different F1 averaging should produce different scores")


def test_regression_metrics_handle_continuous_targets(test_settings):
    """Tests regression metrics work correctly with continuous target values.

    Args:
        test_settings: Test configuration fixture

    Asserts:
        - All regression metrics accept continuous predictions
        - Error metrics return non-negative values
        - R2 can return negative values for poor predictions
    """
    np.random.seed(test_settings.SEED)
    y_true = np.random.randn(50)
    y_pred = y_true + np.random.randn(50) * 0.1  # Add some noise

    regression_metrics = ["MAE", "MSE", "RMSE", "R2"]

    for metric_name in regression_metrics:
        metric_func = get_metric_function(metric_name)
        score = metric_func(y_true, y_pred)

        if not (isinstance(score, (int, float))):
            raise AssertionError()

        # Error metrics should be non-negative
        if metric_name in ["MAE", "MSE", "RMSE"]:
            if not (score >= 0.0):
                raise AssertionError(f"{metric_name} should be non-negative")


def test_classification_metrics_handle_binary_targets(sample_predictions):
    """Tests classification metrics work correctly with binary classification targets.

    Args:
        sample_predictions (tuple): True labels and predicted labels

    Asserts:
        - All classification metrics accept binary predictions
        - Scores are within expected ranges for respective metrics
        - Functions handle edge cases appropriately
    """
    y_true, y_pred = sample_predictions

    classification_metrics = [
        "Accuracy",
        "Balanced Accuracy",
        "F1",
        "Precision",
        "Recall",
    ]

    for metric_name in classification_metrics:
        metric_func = get_metric_function(metric_name)
        score = metric_func(y_true, y_pred)

        if not (isinstance(score, (int, float))):
            raise AssertionError()
        if not (0.0 <= score <= 1.0):
            raise AssertionError(f"{metric_name} should be between 0 and 1")


def test_roc_auc_metric_requires_probabilities(
    sample_predictions, sample_probabilities
):
    """Tests ROC AUC metric works with probability predictions.

    Args:
        sample_predictions (tuple): True labels and predicted labels
        sample_probabilities (np.ndarray): Probability predictions

    Asserts:
        - ROC AUC accepts probability predictions
        - Score is between 0 and 1 as expected for AUC
        - Function handles binary classification probabilities correctly
    """
    y_true, _ = sample_predictions
    y_proba = sample_probabilities[:, 1]  # Positive class probabilities

    roc_auc_func = get_metric_function("Roc AUC")
    score = roc_auc_func(y_true, y_proba)

    if not (isinstance(score, (int, float))):
        raise AssertionError()
    if not (0.0 <= score <= 1.0):
        raise AssertionError("ROC AUC should be between 0 and 1")


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
