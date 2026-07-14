"""Tests for kvbiii_ml.evaluation.error_diagnostics module."""

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.evaluation.error_diagnostics import ErrorDiagnostics


@pytest.fixture
def classification_predictions():
    """Provides target and prediction data for classification error diagnostics.

    Returns:
        tuple: y_true, y_pred, and probabilities for binary classification
    """
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1])
    probas = np.array(
        [
            [0.8, 0.2],  # Correct prediction
            [0.3, 0.7],  # Incorrect prediction
            [0.2, 0.8],  # Correct prediction
            [0.6, 0.4],  # Incorrect prediction
            [0.7, 0.3],  # Correct prediction
            [0.1, 0.9],  # Correct prediction
            [0.4, 0.6],  # Incorrect prediction
            [0.3, 0.7],  # Correct prediction
            [0.9, 0.1],  # Correct prediction
            [0.4, 0.6],  # Incorrect prediction
        ]
    )
    return y_true, y_pred, probas


@pytest.fixture
def regression_predictions():
    """Provides target and prediction data for regression error diagnostics.

    Returns:
        tuple: y_true and y_pred for regression
    """
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    y_pred = np.array([12.0, 18.0, 31.0, 45.0, 49.0, 57.0, 75.0, 77.0, 85.0, 105.0])
    return y_true, y_pred


@pytest.fixture
def feature_data():
    """Provides feature data for error analysis.

    Returns:
        pd.DataFrame: Feature data for diagnostic tests
    """
    return pd.DataFrame(
        {
            "numeric_feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "categorical_feature": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
        }
    )


def test_errorsdiagnostics_init_sets_problem_type():
    """Tests ErrorDiagnostics initialization sets problem type correctly.

    Asserts:
        - Problem type is set correctly
        - Invalid problem types raise error
    """
    # Test classification
    diag_cls = ErrorDiagnostics(problem_type="classification")
    if not (diag_cls.problem_type == "classification"):
        raise AssertionError()

    # Test regression
    diag_reg = ErrorDiagnostics(problem_type="regression")
    if not (diag_reg.problem_type == "regression"):
        raise AssertionError()

    # Test invalid type
    with pytest.raises(ValueError):
        ErrorDiagnostics(problem_type="invalid_type")


def test_errorsdiagnostics_compute_classification_errors(
    classification_predictions, feature_data
):
    """Tests compute_errors method for classification problems.

    Args:
        classification_predictions: Classification predictions fixture
        feature_data: Feature data fixture

    Asserts:
        - Error DataFrame is created with correct structure
        - Errors are identified correctly
        - Probability information is included
        - Feature data is merged correctly
    """
    y_true, y_pred, probas = classification_predictions

    diag = ErrorDiagnostics(problem_type="classification")
    error_df = diag.compute_errors(y_true, y_pred, X=feature_data, probas=probas)

    # Check DataFrame structure
    if not (isinstance(error_df, pd.DataFrame)):
        raise AssertionError()
    if not ("y_true" in error_df.columns):
        raise AssertionError()
    if not ("y_pred" in error_df.columns):
        raise AssertionError()
    if not ("error" in error_df.columns):
        raise AssertionError()
    if not ("proba_true_class" in error_df.columns):
        raise AssertionError()
    if not ("numeric_feature" in error_df.columns):
        raise AssertionError()
    if not ("categorical_feature" in error_df.columns):
        raise AssertionError()

    # Check error identification
    if not (error_df["error"].sum() == 4):
        raise AssertionError()

    # Verify probability calculations
    # For errors with true class 0, proba_true_class should be 1 - proba[:, 1]
    # For errors with true class 1, proba_true_class should be proba[:, 1]
    error_indices = np.where(y_true != y_pred)[0]
    for i in error_indices:
        true_class = y_true[i]
        if true_class == 0:
            expected_proba = probas[i, 0]
        else:
            expected_proba = probas[i, 1]
        if not (abs(error_df.loc[i, "proba_true_class"] - expected_proba) < 1e-10):
            raise AssertionError()


def test_errorsdiagnostics_compute_regression_errors(
    regression_predictions, feature_data
):
    """Tests compute_errors method for regression problems.

    Args:
        regression_predictions: Regression predictions fixture
        feature_data: Feature data fixture

    Asserts:
        - Error DataFrame includes absolute and squared errors
        - Error percentages are calculated correctly
        - Feature data is merged correctly
    """
    y_true, y_pred = regression_predictions

    diag = ErrorDiagnostics(problem_type="regression")
    error_df = diag.compute_errors(y_true, y_pred, X=feature_data)

    # Check DataFrame structure
    if not (isinstance(error_df, pd.DataFrame)):
        raise AssertionError()
    if not ("y_true" in error_df.columns):
        raise AssertionError()
    if not ("y_pred" in error_df.columns):
        raise AssertionError()
    if not ("absolute_error" in error_df.columns):
        raise AssertionError()
    if not ("squared_error" in error_df.columns):
        raise AssertionError()
    if not ("percentage_error" in error_df.columns):
        raise AssertionError()
    if not ("numeric_feature" in error_df.columns):
        raise AssertionError()
    if not ("categorical_feature" in error_df.columns):
        raise AssertionError()

    # Check error calculations
    expected_abs_errors = np.abs(y_true - y_pred)
    expected_squared_errors = (y_true - y_pred) ** 2
    expected_pct_errors = np.abs((y_true - y_pred) / y_true) * 100

    np.testing.assert_array_almost_equal(
        error_df["absolute_error"].values, expected_abs_errors
    )
    np.testing.assert_array_almost_equal(
        error_df["squared_error"].values, expected_squared_errors
    )
    np.testing.assert_array_almost_equal(
        error_df["percentage_error"].values, expected_pct_errors
    )


def test_errorsdiagnostics_compute_errors_without_features(classification_predictions):
    """Tests compute_errors method without feature data.

    Args:
        classification_predictions: Classification predictions fixture

    Asserts:
        - Method works without feature data
        - Error metrics are still calculated correctly
    """
    y_true, y_pred, probas = classification_predictions

    diag = ErrorDiagnostics(problem_type="classification")
    error_df = diag.compute_errors(y_true, y_pred, probas=probas)

    # Basic checks
    if not (isinstance(error_df, pd.DataFrame)):
        raise AssertionError()
    if not ("y_true" in error_df.columns):
        raise AssertionError()
    if not ("y_pred" in error_df.columns):
        raise AssertionError()
    if not ("error" in error_df.columns):
        raise AssertionError()
    if not ("proba_true_class" in error_df.columns):
        raise AssertionError()

    # No feature columns should be present
    if not ("numeric_feature" not in error_df.columns):
        raise AssertionError()
    if not ("categorical_feature" not in error_df.columns):
        raise AssertionError()


def test_errorsdiagnostics_compute_errors_without_probabilities(
    classification_predictions, feature_data
):
    """Tests compute_errors method without probability data for classification.

    Args:
        classification_predictions: Classification predictions fixture
        feature_data: Feature data fixture

    Asserts:
        - Method works without probability data
        - Error identification still works correctly
    """
    y_true, y_pred, _ = classification_predictions

    diag = ErrorDiagnostics(problem_type="classification")
    error_df = diag.compute_errors(y_true, y_pred, X=feature_data)

    # Basic checks
    if not (isinstance(error_df, pd.DataFrame)):
        raise AssertionError()
    if not ("y_true" in error_df.columns):
        raise AssertionError()
    if not ("y_pred" in error_df.columns):
        raise AssertionError()
    if not ("error" in error_df.columns):
        raise AssertionError()

    # No probability column should be present
    if not ("proba_true_class" not in error_df.columns):
        raise AssertionError()

    # Error identification should still work
    if not (error_df["error"].sum() == 4):
        raise AssertionError()


def test_errorsdiagnostics_basic_classification_flow(classification_predictions):
    """Reduced legacy plotting test: ensure compute_errors produces expected columns."""
    y_true, y_pred, probas = classification_predictions
    diag = ErrorDiagnostics("classification")
    error_df = diag.compute_errors(y_true, y_pred, probas=probas)
    if not ({"y_true", "y_pred", "error"} <= set(error_df.columns)):
        raise AssertionError()


def test_errorsdiagnostics_basic_regression_flow(regression_predictions):
    """Tests that compute_errors produces absolute and squared error columns for regression."""
    y_true, y_pred = regression_predictions
    df = ErrorDiagnostics("regression").compute_errors(y_true, y_pred)
    if not ({"absolute_error", "squared_error"} <= set(df.columns)):
        raise AssertionError()


def test_errorsdiagnostics_feature_merge(classification_predictions, feature_data):
    """Tests that compute_errors merges the provided feature columns into the output."""
    y_true, y_pred, probas = classification_predictions
    diag = ErrorDiagnostics("classification")
    df = diag.compute_errors(y_true, y_pred, X=feature_data, probas=probas)
    if not (set(feature_data.columns) <= set(df.columns)):
        raise AssertionError()


def test_errorsdiagnostics_misclassified_subset(
    classification_predictions, feature_data
):
    """Tests that misclassified rows can be filtered from the compute_errors output."""
    y_true, y_pred, _ = classification_predictions
    df = ErrorDiagnostics("classification").compute_errors(
        y_true, y_pred, X=feature_data
    )
    mis = df[df.error == 1]
    if not (mis.shape[0] == 4):
        raise AssertionError()


def test_errorsdiagnostics_regression_errors_have_expected_columns(
    regression_predictions,
):
    """Tests that regression compute_errors output includes all expected error columns."""
    y_true, y_pred = regression_predictions
    df = ErrorDiagnostics("regression").compute_errors(y_true, y_pred)
    if not ({"error", "absolute_error", "squared_error"} <= set(df.columns)):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
