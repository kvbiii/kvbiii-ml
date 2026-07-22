"""Tests for kvbiii_ml.modeling.optimization.cutoff_tuning module."""

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from kvbiii_ml.modeling.optimization.cutoff_tuning import CutoffTunerCV
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


@pytest.fixture
def binary_cutoff_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a binary classification dataset for cutoff tuning testing.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and binary target vector.
    """
    x_arr, y_arr = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        random_state=17,
    )
    x_df = pd.DataFrame(x_arr, columns=[f"feature_{i}" for i in range(6)])
    return x_df, pd.Series(y_arr, name="target")


@pytest.fixture
def multiclass_cutoff_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a multiclass classification dataset for cutoff tuning testing.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and multiclass target vector.
    """
    x_arr, y_arr = make_classification(
        n_samples=240,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_classes=3,
        random_state=17,
    )
    x_df = pd.DataFrame(x_arr, columns=[f"feature_{i}" for i in range(8)])
    return x_df, pd.Series(y_arr, name="target")


@pytest.fixture
def multiclass_logistic_regression_estimator(test_settings) -> LogisticRegression:
    """Provides a LogisticRegression estimator compatible with multiclass targets.

    The shared ``logistic_regression_estimator`` fixture uses the liblinear solver,
    which does not support more than two classes, so multiclass cutoff tests use
    this lbfgs-solver variant instead.

    Args:
        test_settings: Test configuration fixture.

    Returns:
        LogisticRegression: Configured multiclass-capable estimator.
    """
    return LogisticRegression(
        max_iter=200, random_state=test_settings.SEED, solver="lbfgs"
    )


@pytest.fixture
def cutoff_cv_trainer(test_settings) -> CrossValidationTrainer:
    """Provides a CrossValidationTrainer configured for cutoff tuning testing.

    Args:
        test_settings: Test configuration fixture.

    Returns:
        CrossValidationTrainer: Configured trainer with a stratified 3-fold splitter.
    """
    return CrossValidationTrainer(
        metric_name="Balanced Accuracy",
        problem_type="classification",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=test_settings.SEED),
        verbose=False,
    )


def test_cutofftunercv_tune_binary_produces_single_best_cutoff(
    binary_cutoff_data, logistic_regression_estimator, cutoff_cv_trainer
):
    """Tests tune() on binary data returns a study and a single best cutoff.

    Args:
        binary_cutoff_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        cutoff_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - tune() returns an optuna Study
        - best_cutoffs has shape (1,)
        - best_cutoffs value falls within the searched (0, 1) range
    """
    X, y = binary_cutoff_data
    tuner = CutoffTunerCV(
        estimator=logistic_regression_estimator,
        cross_validator=cutoff_cv_trainer,
        n_trials=5,
        seed=17,
    )

    study = tuner.tune(X, y)

    if not isinstance(study, optuna.study.Study):
        raise AssertionError()
    if tuner.best_cutoffs.shape != (1,):
        raise AssertionError()
    if not 0.0 <= tuner.best_cutoffs[0] <= 1.0:
        raise AssertionError()


def test_cutofftunercv_tune_multiclass_produces_per_class_best_cutoffs(
    multiclass_cutoff_data, multiclass_logistic_regression_estimator, cutoff_cv_trainer
):
    """Tests tune() on multiclass data returns one cutoff per class.

    Args:
        multiclass_cutoff_data (tuple): Feature matrix and multiclass target vector.
        multiclass_logistic_regression_estimator (LogisticRegression): Configured
            multiclass-capable estimator.
        cutoff_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - best_cutoffs has shape (n_classes,)
        - every cutoff falls within the searched (0, 1) range
    """
    X, y = multiclass_cutoff_data
    tuner = CutoffTunerCV(
        estimator=multiclass_logistic_regression_estimator,
        cross_validator=cutoff_cv_trainer,
        n_trials=5,
        seed=17,
    )

    tuner.tune(X, y)

    n_classes = y.nunique()
    if tuner.best_cutoffs.shape != (n_classes,):
        raise AssertionError()
    if not np.all((tuner.best_cutoffs >= 0.0) & (tuner.best_cutoffs <= 1.0)):
        raise AssertionError()


def test_cutofftunercv_predict_raises_runtimeerror_before_tune(
    logistic_regression_estimator, cutoff_cv_trainer
):
    """Tests predict() raises before tune() has been called.

    Args:
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        cutoff_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - RuntimeError is raised instructing the caller to tune first
    """
    tuner = CutoffTunerCV(
        estimator=logistic_regression_estimator, cross_validator=cutoff_cv_trainer
    )

    with pytest.raises(RuntimeError, match=r"Call tune\(\) before predict\(\)\."):
        tuner.predict(pd.DataFrame({"feature_0": [0.1, 0.2]}))


def test_cutofftunercv_predict_after_tune_returns_expected_predictions(
    binary_cutoff_data, logistic_regression_estimator, cutoff_cv_trainer
):
    """Tests predict() after tune() returns integer class predictions of expected length.

    Args:
        binary_cutoff_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        cutoff_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - Predictions length matches the input sample count
        - Predictions are integer-valued and drawn from the observed class set
    """
    X, y = binary_cutoff_data
    tuner = CutoffTunerCV(
        estimator=logistic_regression_estimator,
        cross_validator=cutoff_cv_trainer,
        n_trials=5,
        seed=17,
    )
    tuner.tune(X, y)

    predictions = tuner.predict(X)

    if len(predictions) != len(X):
        raise AssertionError()
    if not np.issubdtype(predictions.dtype, np.integer):
        raise AssertionError()
    if not set(np.unique(predictions)).issubset(set(np.unique(y))):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
