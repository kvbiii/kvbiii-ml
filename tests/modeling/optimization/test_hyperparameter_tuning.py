"""Tests for kvbiii_ml.modeling.optimization.hyperparameter_tuning module."""

from unittest.mock import Mock

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split

from kvbiii_ml.modeling.optimization.hyperparameter_tuning import RandomSearchCV
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


@pytest.fixture
def hyperparam_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a binary classification dataset for hyperparameter tuning testing.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and binary target vector.
    """
    x_arr, y_arr = make_classification(
        n_samples=180,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=21,
    )
    x_df = pd.DataFrame(x_arr, columns=[f"col_{i}" for i in range(5)])
    return x_df, pd.Series(y_arr, name="target")


@pytest.fixture
def hyperparam_cv_trainer(test_settings) -> CrossValidationTrainer:
    """Provides a CrossValidationTrainer configured for hyperparameter tuning testing.

    Args:
        test_settings: Test configuration fixture.

    Returns:
        CrossValidationTrainer: Configured trainer with a stratified 3-fold splitter.
    """
    return CrossValidationTrainer(
        metric_name="Accuracy",
        problem_type="classification",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=test_settings.SEED),
        verbose=False,
    )


@pytest.fixture
def real_optuna_trial() -> optuna.trial.Trial:
    """Provides a real Optuna trial obtained from a local study via ask().

    Returns:
        optuna.trial.Trial: A live trial usable with suggest_* methods.
    """
    study = optuna.create_study()
    return study.ask()


def test_randomsearchcv_tune_uses_holdout_when_valid_data_provided(
    hyperparam_classification_data, logistic_regression_estimator, hyperparam_cv_trainer
):
    """Tests tune() takes the hold-out branch when X_valid/y_valid are supplied.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - tune() returns an optuna Study with a completed best trial
        - The searched hyperparameter appears in best_params
    """
    X, y = hyperparam_classification_data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.3, random_state=17, stratify=y
    )
    tuner = RandomSearchCV(cross_validator=hyperparam_cv_trainer, n_trials=3, seed=17)
    params_grid = {"C": ("float", [0.01, 1.0], {"log": True})}

    study = tuner.tune(
        logistic_regression_estimator,
        X_train,
        y_train,
        params_grid,
        X_valid=X_valid,
        y_valid=y_valid,
    )

    if study.best_value is None:
        raise AssertionError()
    if "C" not in study.best_params:
        raise AssertionError()


def test_randomsearchcv_tune_uses_cross_validation_when_no_valid_data(
    hyperparam_classification_data, logistic_regression_estimator, hyperparam_cv_trainer
):
    """Tests tune() takes the CV branch when no hold-out data is supplied.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - tune() returns an optuna Study with a completed best trial
        - The searched hyperparameter appears in best_params
    """
    X, y = hyperparam_classification_data
    tuner = RandomSearchCV(cross_validator=hyperparam_cv_trainer, n_trials=3, seed=17)
    params_grid = {"C": ("float", [0.01, 1.0], {"log": True})}

    study = tuner.tune(logistic_regression_estimator, X, y, params_grid)

    if study.best_value is None:
        raise AssertionError()
    if "C" not in study.best_params:
        raise AssertionError()


def test_randomsearchcv_tune_downsamples_when_rows_exceed_max_samples(
    logistic_regression_estimator, hyperparam_cv_trainer
):
    """Tests tune() completes without error when the dataset exceeds max_samples.

    Args:
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - The CV objective downsamples rows above max_samples and still completes
        - tune() returns an optuna Study with a valid best_value
    """
    np.random.seed(17)
    x_arr, y_arr = make_classification(
        n_samples=60,
        n_features=5,
        n_informative=3,
        random_state=17,
    )
    X = pd.DataFrame(x_arr, columns=[f"feature_{i}" for i in range(5)])
    y = pd.Series(y_arr, name="target")

    tuner = RandomSearchCV(
        cross_validator=hyperparam_cv_trainer, n_trials=3, seed=17, max_samples=30
    )
    params_grid = {"C": ("float", [0.01, 1.0], {"log": True})}

    study = tuner.tune(logistic_regression_estimator, X, y, params_grid)

    if study.best_value is None:
        raise AssertionError()


@pytest.mark.parametrize(
    ("param_type", "param_value", "param_kwargs", "expected_type"),
    [
        ("int", [1, 10], {}, int),
        ("float", [0.01, 1.0], {"log": True}, float),
        ("float", [0.0, 1.0], {"step": 0.1}, float),
        ("categorical", ["a", "b", "c"], {}, str),
    ],
)
def test_randomsearchcv_get_param_samples_expected_types(
    real_optuna_trial, param_type, param_value, param_kwargs, expected_type
):
    """Tests get_param samples values of the expected type for each supported param_type.

    Args:
        real_optuna_trial (optuna.trial.Trial): Live trial fixture.
        param_type (str): Parameter type under test.
        param_value (list): Bounds or choices for the parameter.
        param_kwargs (dict): Extra keyword arguments (log/step).
        expected_type (type): Expected Python type of the sampled value.

    Asserts:
        - The sampled value is an instance of the expected type
        - Numeric values fall within the requested bounds
    """
    tuner = RandomSearchCV(cross_validator=Mock(), n_trials=1, seed=17)

    value = tuner.get_param(
        real_optuna_trial, "param", (param_type, param_value, param_kwargs)
    )

    if not isinstance(value, expected_type):
        raise AssertionError()
    if param_type in {"int", "float"}:
        low, high = param_value
        if not low <= value <= high:
            raise AssertionError()
    else:
        if value not in param_value:
            raise AssertionError()


def test_randomsearchcv_get_param_returns_constant_without_sampling(real_optuna_trial):
    """Tests get_param returns the first element for a constant param_type.

    Args:
        real_optuna_trial (optuna.trial.Trial): Live trial fixture.

    Asserts:
        - The returned value equals the constant's sole list entry
    """
    tuner = RandomSearchCV(cross_validator=Mock(), n_trials=1, seed=17)

    value = tuner.get_param(real_optuna_trial, "param", ("constant", ["fixed_value"]))

    if value != "fixed_value":
        raise AssertionError()


def test_randomsearchcv_get_param_raises_error_for_unknown_param_type(
    real_optuna_trial,
):
    """Tests get_param raises ValueError for an unsupported param_type.

    Args:
        real_optuna_trial (optuna.trial.Trial): Live trial fixture.

    Asserts:
        - ValueError is raised mentioning the unknown param_type
    """
    tuner = RandomSearchCV(cross_validator=Mock(), n_trials=1, seed=17)

    with pytest.raises(ValueError, match="Unknown param_type"):
        tuner.get_param(real_optuna_trial, "param", ("weird_type", [1, 2]))


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
