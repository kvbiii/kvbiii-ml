"""Tests for kvbiii_ml.modeling.optimization.hyperparameter_tuning module."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from kvbiii_ml.modeling.optimization.hyperparameter_tuning import (
    evaluate_hyperparameters,
    find_best_hyperparameters,
    optimize_hyperparameters,
)


@pytest.fixture
def tuning_data(test_settings):
    """Provides data for hyperparameter tuning testing.

    Args:
        test_settings: Test configuration fixture

    Returns:
        tuple: X, y data for hyperparameter optimization
    """
    np.random.seed(test_settings.SEED)
    X = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
            "feature3": np.random.normal(0, 1, 100),
        }
    )
    y = (X["feature1"] > 0).astype(int)
    return X, y


@pytest.fixture
def sample_param_grid():
    """Provides a sample parameter grid for hyperparameter tuning.

    Returns:
        dict: Parameter grid for testing
    """
    return {
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7],
        "min_child_weight": [1, 3, 5],
    }


@patch("kvbiii_ml.modeling.optimization.hyperparameter_tuning.optuna")
def test_optimize_hyperparameters_uses_optuna_correctly(
    mock_optuna, tuning_data, mock_estimator, sample_param_grid
):
    """Tests optimize_hyperparameters uses Optuna correctly.

    Args:
        mock_optuna: Mocked Optuna library
        tuning_data: X, y data fixture
        mock_estimator: Mock estimator
        sample_param_grid: Parameter grid fixture

    Asserts:
        - Optuna study is created with correct parameters
        - Objective function is passed to optimize
        - Best parameters and values are returned
    """
    X, y = tuning_data

    # Configure mock study
    mock_study = Mock()
    mock_optuna.create_study.return_value = mock_study
    mock_study.best_params = {"learning_rate": 0.1, "max_depth": 5}
    mock_study.best_value = 0.95

    result = optimize_hyperparameters(
        mock_estimator,
        X,
        y,
        param_distributions=sample_param_grid,
        n_trials=20,
        cv=3,
        metric="accuracy",
        direction="maximize",
    )

    # Check Optuna was used correctly
    mock_optuna.create_study.assert_called_once()
    assert mock_optuna.create_study.call_args[1]["direction"] == "maximize"

    # Check study optimize was called
    mock_study.optimize.assert_called_once()
    assert mock_study.optimize.call_args[1]["n_trials"] == 20

    # Check return value structure
    assert "best_params" in result
    assert "best_score" in result
    assert result["best_params"] == mock_study.best_params
    assert result["best_score"] == mock_study.best_value


@patch("kvbiii_ml.modeling.optimization.hyperparameter_tuning.GridSearchCV")
def test_find_best_hyperparameters_with_grid_search(
    mock_grid_search, tuning_data, mock_estimator, sample_param_grid
):
    """Tests find_best_hyperparameters using GridSearchCV.

    Args:
        mock_grid_search: Mocked GridSearchCV
        tuning_data: X, y data fixture
        mock_estimator: Mock estimator
        sample_param_grid: Parameter grid fixture

    Asserts:
        - GridSearchCV is created with correct parameters
        - Grid search is fitted with data
        - Best parameters and score are returned
    """
    X, y = tuning_data

    # Configure mock grid search
    grid_instance = Mock()
    mock_grid_search.return_value = grid_instance
    grid_instance.best_params_ = {"learning_rate": 0.1, "max_depth": 5}
    grid_instance.best_score_ = 0.95
    grid_instance.cv_results_ = {"mean_test_score": np.array([0.9, 0.95, 0.85])}

    result = find_best_hyperparameters(
        mock_estimator,
        X,
        y,
        param_grid=sample_param_grid,
        cv=3,
        scoring="accuracy",
        method="grid",
    )

    # Check GridSearchCV was created correctly
    mock_grid_search.assert_called_once()
    assert mock_grid_search.call_args[1]["estimator"] is mock_estimator
    assert mock_grid_search.call_args[1]["param_grid"] is sample_param_grid
    assert mock_grid_search.call_args[1]["cv"] == 3
    assert mock_grid_search.call_args[1]["scoring"] == "accuracy"

    # Check fit was called
    grid_instance.fit.assert_called_once_with(X, y)

    # Check return value structure
    assert "best_params" in result
    assert "best_score" in result
    assert "cv_results" in result
    assert result["best_params"] == grid_instance.best_params_
    assert result["best_score"] == grid_instance.best_score_


@patch("kvbiii_ml.modeling.optimization.hyperparameter_tuning.RandomizedSearchCV")
def test_find_best_hyperparameters_with_random_search(
    mock_random_search, tuning_data, mock_estimator, sample_param_grid
):
    """Tests find_best_hyperparameters using RandomizedSearchCV.

    Args:
        mock_random_search: Mocked RandomizedSearchCV
        tuning_data: X, y data fixture
        mock_estimator: Mock estimator
        sample_param_grid: Parameter grid fixture

    Asserts:
        - RandomizedSearchCV is created with correct parameters
        - Random search is fitted with data
        - Best parameters and score are returned
    """
    X, y = tuning_data

    # Configure mock random search
    random_instance = Mock()
    mock_random_search.return_value = random_instance
    random_instance.best_params_ = {"learning_rate": 0.1, "max_depth": 5}
    random_instance.best_score_ = 0.95
    random_instance.cv_results_ = {"mean_test_score": np.array([0.9, 0.95, 0.85])}

    result = find_best_hyperparameters(
        mock_estimator,
        X,
        y,
        param_grid=sample_param_grid,
        cv=3,
        scoring="accuracy",
        method="random",
        n_iter=10,
    )

    # Check RandomizedSearchCV was created correctly
    mock_random_search.assert_called_once()
    assert mock_random_search.call_args[1]["estimator"] is mock_estimator
    assert mock_random_search.call_args[1]["param_distributions"] is sample_param_grid
    assert mock_random_search.call_args[1]["cv"] == 3
    assert mock_random_search.call_args[1]["scoring"] == "accuracy"
    assert mock_random_search.call_args[1]["n_iter"] == 10

    # Check fit was called
    random_instance.fit.assert_called_once_with(X, y)

    # Check return value structure
    assert "best_params" in result
    assert "best_score" in result
    assert result["best_params"] == random_instance.best_params_
    assert result["best_score"] == random_instance.best_score_


@patch("kvbiii_ml.modeling.optimization.hyperparameter_tuning.cross_val_score")
def test_evaluate_hyperparameters_uses_cross_validation(
    mock_cv_score, tuning_data, mock_estimator
):
    """Tests evaluate_hyperparameters uses cross-validation correctly.

    Args:
        mock_cv_score: Mocked cross_val_score function
        tuning_data: X, y data fixture
        mock_estimator: Mock estimator

    Asserts:
        - cross_val_score is called with correct parameters
        - Parameters are applied to estimator
        - Score and statistics are returned
    """
    X, y = tuning_data
    params = {"learning_rate": 0.1, "max_depth": 5}

    # Configure mock
    mock_cv_score.return_value = np.array([0.94, 0.96, 0.95])

    score = evaluate_hyperparameters(
        mock_estimator, X, y, params, cv=3, scoring="accuracy"
    )

    # Check parameters were set
    mock_estimator.set_params.assert_called_once_with(**params)

    # Check cross_val_score was called correctly
    mock_cv_score.assert_called_once()
    call_args = mock_cv_score.call_args[0]
    assert call_args[0] is mock_estimator
    assert call_args[1] is X
    assert call_args[2] is y

    call_kwargs = mock_cv_score.call_args[1]
    assert call_kwargs["cv"] == 3
    assert call_kwargs["scoring"] == "accuracy"

    # Check returned score is mean of CV scores
    assert score == np.mean([0.94, 0.96, 0.95])


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
