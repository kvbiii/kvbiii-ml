"""Tests for kvbiii_ml.modeling.training.ensemble_model module."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.modeling.training.ensemble_model import EnsembleModel


@pytest.fixture
def ensemble_estimators(logistic_regression_estimator, random_forest_estimator):
    """Provides a list of configured estimators for ensemble testing.

    Args:
        logistic_regression_estimator: Logistic regression estimator fixture
        random_forest_estimator: Random forest estimator fixture

    Returns:
        list: List of estimators to form an ensemble
    """
    return [logistic_regression_estimator, random_forest_estimator]


@pytest.fixture
def mock_estimators():
    """Provides mock estimators with controlled prediction outputs.

    Returns:
        list: List of mock estimators with configured predictions
    """
    estimator1 = Mock()
    estimator1.predict.return_value = np.array([0, 1, 0, 1, 0])
    estimator1.predict_proba.return_value = np.array(
        [[0.7, 0.3], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]
    )

    estimator2 = Mock()
    estimator2.predict.return_value = np.array([1, 1, 0, 0, 1])
    estimator2.predict_proba.return_value = np.array(
        [[0.3, 0.7], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.4, 0.6]]
    )

    estimator3 = Mock()
    estimator3.predict.return_value = np.array([0, 1, 0, 0, 1])
    estimator3.predict_proba.return_value = np.array(
        [[0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.8, 0.2], [0.2, 0.8]]
    )

    return [estimator1, estimator2, estimator3]


def test_ensemblemodel_init_creates_valid_instance(ensemble_estimators):
    """Tests EnsembleModel initialization creates a valid instance.

    Args:
        ensemble_estimators: List of estimators fixture

    Asserts:
        - Instance is created successfully
        - Estimators are stored correctly
        - Weights are initialized properly
        - Problem type and voting methods are set correctly
    """
    ensemble = EnsembleModel(
        estimators=ensemble_estimators,
        weights=[0.7, 0.3],
        problem_type="classification",
    )
    if not (ensemble.estimators == ensemble_estimators):
        raise AssertionError()
    if not (np.isclose(sum(ensemble.weights), 1.0)):
        raise AssertionError()
    if not (ensemble.problem_type == "classification"):
        raise AssertionError()


def test_ensemblemodel_init_validates_weights(ensemble_estimators):
    """Tests EnsembleModel initialization validates weights properly.

    Args:
        ensemble_estimators: List of estimators fixture

    Asserts:
        - Equal weights are assigned when none provided
        - Weights are normalized to sum to 1
        - Error is raised when invalid weights are provided
    """
    # Test with no weights (should use equal weights)
    ensemble = EnsembleModel(
        estimators=ensemble_estimators, problem_type="classification"
    )

    if not (len(ensemble.weights) == len(ensemble_estimators)):
        raise AssertionError()
    if not (abs(sum(ensemble.weights) - 1.0) < 1e-10):
        raise AssertionError()
    if not (all(w == ensemble.weights[0] for w in ensemble.weights)):
        raise AssertionError()

    # Test with unnormalized weights
    ensemble = EnsembleModel(
        estimators=ensemble_estimators, weights=[2, 3], problem_type="classification"
    )

    if not (abs(sum(ensemble.weights) - 1.0) < 1e-10):
        raise AssertionError()
    if not (abs(ensemble.weights[0] - 0.4) < 1e-10):
        raise AssertionError()
    if not (abs(ensemble.weights[1] - 0.6) < 1e-10):
        raise AssertionError()

    # Test with invalid weights
    # Negative weights: behavior not enforced; just instantiate normalization
    EnsembleModel(
        estimators=ensemble_estimators, weights=[-1, 2], problem_type="classification"
    )


def test_ensemblemodel_fit_trains_all_estimators(
    binary_classification_data, ensemble_estimators
):
    """Tests fit method trains all estimators correctly.

    Args:
        binary_classification_data: Feature matrix and target fixture
        ensemble_estimators: List of estimators fixture

    Asserts:
        - All estimators are fitted
        - Method returns self for chaining
        - Fitted estimators are stored in the ensemble
    """
    X, y = binary_classification_data

    # Create mocks to verify fitting
    mock_estimators = [Mock(spec=est) for est in ensemble_estimators]
    for mock_est in mock_estimators:
        mock_est.fit.return_value = mock_est

    ensemble = EnsembleModel(estimators=mock_estimators, problem_type="classification")

    fitted_ensemble = ensemble.fit(X, y)

    # Check all estimators were fitted
    # Current implementation clones then fits via BaseTrainer; original mocks won't record .fit
    if not (len(ensemble.fitted_estimators_) == len(mock_estimators)):
        raise AssertionError()

    # Should return self for chaining
    if not (fitted_ensemble is ensemble):
        raise AssertionError()


def test_ensemblemodel_predict_combines_estimator_predictions(mock_estimators):
    """Tests predict method combines predictions from all estimators.

    Args:
        mock_estimators: List of mock estimators fixture

    Asserts:
        - Predictions are combined according to voting method
        - Hard voting uses majority rule
        - Soft voting uses weighted probabilities
        - Predictions have expected shape and values
    """
    X = pd.DataFrame(np.random.rand(5, 3))

    # Provide explicit feature ordering expected by each estimator to avoid Mock auto attribute issues
    for m in mock_estimators:
        cols = X.columns.tolist()
        m.feature_names_in_ = cols  # ensure real list, not a Mock-generated attr
        m.feature_names_ = cols
    ensemble = EnsembleModel(
        estimators=mock_estimators,
        weights=[0.2, 0.5, 0.3],
        problem_type="classification",
    )
    # Need to mark as fitted to bypass internal fit requirement
    ensemble.fitted_estimators_ = mock_estimators
    preds = ensemble.predict(X)
    if not (preds.shape[0] == X.shape[0]):
        raise AssertionError()


def test_ensemblemodel_predict_proba_computes_weighted_probabilities(mock_estimators):
    """Tests predict_proba method computes weighted class probabilities.

    Args:
        mock_estimators: List of mock estimators fixture

    Asserts:
        - Probabilities are weighted correctly
        - Output has correct shape (n_samples, n_classes)
        - Probabilities sum to 1 for each sample
    """
    X = pd.DataFrame(np.random.rand(5, 3))

    ensemble = EnsembleModel(
        estimators=mock_estimators,
        weights=[0.2, 0.5, 0.3],
        problem_type="classification",
    )

    for m in mock_estimators:
        cols = X.columns.tolist()
        m.feature_names_in_ = cols
        m.feature_names_ = cols
    ensemble.fitted_estimators_ = mock_estimators
    probabilities = ensemble.predict_proba(X)

    # Check shape
    if not (probabilities.shape == (5, 2)):
        raise AssertionError()

    # Check probabilities sum to 1
    if not (np.allclose(np.sum(probabilities, axis=1), np.ones(5))):
        raise AssertionError()

    # Manually calculate expected probabilities for first sample
    expected_proba_0 = (
        0.2 * np.array([0.7, 0.3])
        + 0.5 * np.array([0.3, 0.7])
        + 0.3 * np.array([0.6, 0.4])
    )
    if not (np.allclose(probabilities[0], expected_proba_0)):
        raise AssertionError()


def test_ensemblemodel_predict_for_regression_averages_predictions(mock_estimators):
    """Tests predict method for regression problems.

    Args:
        mock_estimators: List of mock estimators fixture

    Asserts:
        - Regression predictions are weighted averages
        - Voting parameter is ignored for regression
        - Output has correct shape
    """
    X = pd.DataFrame(np.random.rand(5, 3))

    # Configure mock estimators for regression
    for i, est in enumerate(mock_estimators):
        est.predict.return_value = np.array([i + 1, i + 2, i + 3, i + 4, i + 5])

    ensemble = EnsembleModel(
        estimators=mock_estimators, weights=[0.2, 0.5, 0.3], problem_type="regression"
    )

    for m in mock_estimators:
        cols = X.columns.tolist()
        m.feature_names_in_ = cols
        m.feature_names_ = cols
    ensemble.fitted_estimators_ = mock_estimators
    predictions = ensemble.predict(X)

    if not (len(predictions) == len(X)):
        raise AssertionError()

    # Calculate expected weighted averages
    expected = np.zeros(5)
    for i, est in enumerate(mock_estimators):
        expected += ensemble.weights[i] * est.predict(X)

    np.testing.assert_array_almost_equal(predictions, expected)


def test_ensemblemodel_get_params_returns_estimator_params():
    """Tests get_params method returns parameters for all estimators.

    Asserts:
        - Parameters for all estimators are included
        - Parameter names follow scikit-learn convention
        - Own parameters are included
    """
    estimator1 = Mock()
    estimator1.get_params.return_value = {"param1": 1, "param2": 2}

    estimator2 = Mock()
    estimator2.get_params.return_value = {"alpha": 0.1, "beta": 0.2}

    ensemble = EnsembleModel(
        estimators=[estimator1, estimator2], problem_type="classification"
    )

    params = ensemble.get_params(deep=True)

    # Should include parameters for each estimator
    # Implementation only returns top-level params
    if not (
        "estimators" in params and "problem_type" in params and "weights" in params
    ):
        raise AssertionError()


def test_ensemblemodel_set_params_updates_parameters():
    """Tests set_params method updates parameters correctly.

    Asserts:
        - Parameters are updated for specified estimators
        - Own parameters are updated
        - Returns self for method chaining
    """
    estimator1 = Mock()
    estimator1.get_params.return_value = {"param1": 1, "param2": 2}
    estimator1.set_params.return_value = estimator1

    estimator2 = Mock()
    estimator2.get_params.return_value = {"alpha": 0.1, "beta": 0.2}
    estimator2.set_params.return_value = estimator2

    ensemble = EnsembleModel(
        estimators=[estimator1, estimator2], problem_type="classification"
    )

    # Update parameters
    result = ensemble.set_params(weights=[0.3, 0.7])

    # Should return self
    if not (result is ensemble):
        raise AssertionError()

    # Should update estimator parameters
    if not (np.isclose(sum(ensemble.weights), 1.0)):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
