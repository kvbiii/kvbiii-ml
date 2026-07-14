"""Tests for kvbiii_ml.modeling.training.base_trainer module."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from kvbiii_ml.modeling.training.base_trainer import BaseTrainer


def test_basetrainer_fit_estimator_basic_fitting(
    sample_dataframe, sample_series, mock_estimator
):
    """Tests fit_estimator method performs basic estimator fitting.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data
        mock_estimator (Mock): Mock estimator with fit method

    Asserts:
        - Estimator.fit is called with training data
        - Method returns the fitted estimator instance
        - Basic parameters are passed correctly
    """
    fitted = BaseTrainer.fit_estimator(mock_estimator, sample_dataframe, sample_series)

    mock_estimator.fit.assert_called_once_with(sample_dataframe, sample_series)
    assert fitted is mock_estimator


def test_basetrainer_fit_estimator_handles_eval_set_parameter(
    sample_dataframe, sample_series
):
    """Tests fit_estimator method includes eval_set when estimator supports it.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data

    Asserts:
        - eval_set parameter is included when available in estimator signature
        - Validation data is properly formatted as list of tuples
        - Other parameters are still handled correctly
    """
    mock_estimator = Mock()
    mock_estimator.fit.return_value = mock_estimator

    # Mock signature to include eval_set parameter
    with patch("kvbiii_ml.modeling.training.base_trainer.signature") as mock_sig:
        mock_sig.return_value.parameters = {"eval_set": Mock(), "verbose": Mock()}

        X_valid = sample_dataframe.copy()
        y_valid = sample_series.copy()

        BaseTrainer.fit_estimator(
            mock_estimator,
            sample_dataframe,
            sample_series,
            X_valid=X_valid,
            y_valid=y_valid,
        )

        call_args = mock_estimator.fit.call_args
        assert "eval_set" in call_args.kwargs
        assert call_args.kwargs["eval_set"] == [(X_valid, y_valid)]
        assert call_args.kwargs["verbose"] is False


def test_basetrainer_fit_estimator_handles_validation_parameters(
    sample_dataframe, sample_series
):
    """Tests fit_estimator method includes X_val/y_val parameters when supported.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data

    Asserts:
        - X_val and y_val parameters are included when available
        - Validation data is passed directly without transformation
        - Parameter names match expected estimator interface
    """
    mock_estimator = Mock()
    mock_estimator.fit.return_value = mock_estimator

    with patch("kvbiii_ml.modeling.training.base_trainer.signature") as mock_sig:
        mock_sig.return_value.parameters = {"X_val": Mock(), "y_val": Mock()}

        X_valid = sample_dataframe.copy()
        y_valid = sample_series.copy()

        BaseTrainer.fit_estimator(
            mock_estimator,
            sample_dataframe,
            sample_series,
            X_valid=X_valid,
            y_valid=y_valid,
        )

        call_args = mock_estimator.fit.call_args
        assert call_args.kwargs["X_val"] is X_valid
        assert call_args.kwargs["y_val"] is y_valid


def test_basetrainer_fit_estimator_handles_sample_weight_parameter(
    sample_dataframe, sample_series
):
    """Tests fit_estimator method includes sample_weight when supported.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data

    Asserts:
        - sample_weight parameter is included when available in signature
        - Weight values are passed through unchanged
        - Other parameters work alongside sample weights
    """
    mock_estimator = Mock()
    mock_estimator.fit.return_value = mock_estimator
    sample_weights = pd.Series(np.random.rand(len(sample_series)))

    with patch("kvbiii_ml.modeling.training.base_trainer.signature") as mock_sig:
        mock_sig.return_value.parameters = {"sample_weight": Mock()}

        BaseTrainer.fit_estimator(
            mock_estimator,
            sample_dataframe,
            sample_series,
            sample_weight=sample_weights,
        )

        call_args = mock_estimator.fit.call_args
        assert call_args.kwargs["sample_weight"] is sample_weights


def test_basetrainer_fit_and_predict_returns_predictions_for_preds_type(
    sample_dataframe, sample_series, mock_estimator
):
    """Tests fit_and_predict method returns standard predictions for 'preds' metric type.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data
        mock_estimator (Mock): Mock estimator with predict method

    Asserts:
        - Standard predict method is called for all datasets
        - Training, validation, and test predictions are returned
        - Fitted estimator is returned as fourth element
    """
    X_valid = sample_dataframe.iloc[:20]
    y_valid = sample_series.iloc[:20]
    X_test = sample_dataframe.iloc[:10]

    train_pred, valid_pred, test_pred, fitted_est = BaseTrainer.fit_and_predict(
        mock_estimator,
        sample_dataframe,
        sample_series,
        X_valid,
        y_valid,
        metric_type="preds",
        X_test=X_test,
    )

    # Verify predict was called for all datasets
    assert mock_estimator.predict.call_count == 3
    assert train_pred is not None
    assert valid_pred is not None
    assert test_pred is not None
    assert fitted_est is mock_estimator


def test_basetrainer_fit_and_predict_returns_probabilities_for_probs_type(
    sample_dataframe, sample_series, mock_estimator
):
    """Tests fit_and_predict method returns probabilities for 'probs' metric type.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data
        mock_estimator (Mock): Mock estimator with predict_proba method

    Asserts:
        - predict_proba method is called instead of predict
        - Probabilities are returned for all provided datasets
        - Binary classification probabilities are handled correctly
    """
    X_valid = sample_dataframe.iloc[:20]
    y_valid = sample_series.iloc[:20]

    train_pred, valid_pred, test_pred, fitted_est = BaseTrainer.fit_and_predict(
        mock_estimator,
        sample_dataframe,
        sample_series,
        X_valid,
        y_valid,
        metric_type="probs",
        X_test=None,
    )

    # Verify predict_proba was called
    assert mock_estimator.predict_proba.call_count == 2  # No test data provided
    assert train_pred is not None
    assert valid_pred is not None
    assert test_pred is None  # No test data provided


def test_basetrainer_fit_and_predict_raises_error_for_invalid_metric_type(
    sample_dataframe, sample_series, mock_estimator
):
    """Tests fit_and_predict method raises error for invalid metric types.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data
        mock_estimator (Mock): Mock estimator fixture

    Asserts:
        - ValueError is raised for unsupported metric types
        - Error message provides clear guidance about valid options
    """
    X_valid = sample_dataframe.iloc[:20]
    y_valid = sample_series.iloc[:20]

    with pytest.raises(
        ValueError, match="Unknown metric type 'invalid'. Must be 'preds' or 'probs'"
    ):
        BaseTrainer.fit_and_predict(
            mock_estimator,
            sample_dataframe,
            sample_series,
            X_valid,
            y_valid,
            metric_type="invalid",
        )


def test_basetrainer_predict_returns_predictions_when_data_provided(mock_estimator):
    """Tests predict method returns predictions when input data is provided.

    Args:
        mock_estimator (Mock): Mock estimator with predict method

    Asserts:
        - predict method is called on estimator when data provided
        - Return value matches estimator's predict output
        - Input data is passed through unchanged
    """
    X_test = pd.DataFrame({"feature": [1, 2, 3]})
    expected_predictions = np.array([0, 1, 0])
    mock_estimator.predict.return_value = expected_predictions

    predictions = BaseTrainer.predict(mock_estimator, X_test)

    mock_estimator.predict.assert_called_once_with(X_test)
    np.testing.assert_array_equal(predictions, expected_predictions)


def test_basetrainer_predict_returns_none_when_no_data_provided(mock_estimator):
    """Tests predict method returns None when no input data is provided.

    Args:
        mock_estimator (Mock): Mock estimator fixture

    Asserts:
        - None is returned when X is None
        - Estimator predict method is not called
        - Method handles None input gracefully
    """
    predictions = BaseTrainer.predict(mock_estimator, None)

    assert predictions is None
    mock_estimator.predict.assert_not_called()


def test_basetrainer_predict_proba_returns_probabilities_when_data_provided(
    mock_estimator,
):
    """Tests predict_proba method returns probabilities when input data is provided.

    Args:
        mock_estimator (Mock): Mock estimator with predict_proba method

    Asserts:
        - predict_proba method is called on estimator
        - Return value matches estimator's predict_proba output
        - Multiclass probabilities are returned unchanged
    """
    X_test = pd.DataFrame({"feature": [1, 2, 3]})
    # Multiclass probabilities (3 classes)
    expected_proba = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.3, 0.3, 0.4]])
    mock_estimator.predict_proba.return_value = expected_proba

    probabilities = BaseTrainer.predict_proba(mock_estimator, X_test)

    mock_estimator.predict_proba.assert_called_once_with(X_test)
    np.testing.assert_array_equal(probabilities, expected_proba)


def test_basetrainer_predict_proba_returns_positive_class_for_binary_classification(
    mock_estimator,
):
    """Tests predict_proba method returns only positive class probabilities for binary classification.

    Args:
        mock_estimator (Mock): Mock estimator with predict_proba method

    Asserts:
        - Binary classification probabilities are reduced to positive class only
        - Shape is reduced from (n_samples, 2) to (n_samples,)
        - Positive class probabilities (column 1) are returned
    """
    X_test = pd.DataFrame({"feature": [1, 2, 3]})
    # Binary classification probabilities
    binary_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    mock_estimator.predict_proba.return_value = binary_proba

    probabilities = BaseTrainer.predict_proba(mock_estimator, X_test)

    expected = binary_proba[:, 1]  # Positive class only
    np.testing.assert_array_equal(probabilities, expected)
    assert probabilities.shape == (3,)  # 1D array for binary classification


def test_basetrainer_predict_proba_returns_none_when_no_data_provided(mock_estimator):
    """Tests predict_proba method returns None when no input data is provided.

    Args:
        mock_estimator (Mock): Mock estimator fixture

    Asserts:
        - None is returned when X is None
        - Estimator predict_proba method is not called
        - Method handles None input gracefully
    """
    probabilities = BaseTrainer.predict_proba(mock_estimator, None)

    assert probabilities is None
    mock_estimator.predict_proba.assert_not_called()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
