"""Tests for kvbiii_ml.modeling.optimization.ensemble_weights_tuner.EnsembleWeightTunerCV (new version).

Focus: weight normalization, negative weights path, blending shapes, and integration with CrossValidationTrainer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

from kvbiii_ml.modeling.optimization.ensemble_weights_tuner import EnsembleWeightTunerCV
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


@pytest.fixture
def small_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a small non-linearly-separable binary classification dataset."""
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=[f"f{i}" for i in range(4)])
    # Non-linearly separable target to avoid trivial perfect scores
    y = pd.Series(
        ((X["f0"] + X["f1"] * 0.5 + rng.normal(scale=0.3, size=60)) > 0).astype(int),
        name="target",
    )
    return X, y


@pytest.fixture
def small_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a small linear regression dataset."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(50, 5)), columns=[f"r{i}" for i in range(5)])
    y = pd.Series(
        2 * X["r0"] - 1.5 * X["r1"] + rng.normal(scale=0.5, size=50), name="target"
    )
    return X, y


def _build_cv(metric: str, problem: str) -> CrossValidationTrainer:
    return CrossValidationTrainer(
        metric_name=metric,
        problem_type=problem,
        cv=KFold(n_splits=3, shuffle=True, random_state=17),
        processors=None,
        verbose=False,
    )


def test_tuner_classification_positive_weights(small_classification_data):
    """Tests tuning with non-negative weights (default)."""
    X, y = small_classification_data
    estimators = [
        LogisticRegression(max_iter=200, solver="liblinear", random_state=0),
        LogisticRegression(C=0.5, max_iter=200, solver="liblinear", random_state=1),
    ]
    cv_trainer = _build_cv("Roc AUC", "classification")
    tuner = EnsembleWeightTunerCV(
        estimators=estimators, cross_validator=cv_trainer, n_trials=5, seed=17
    )

    study = tuner.tune(X, y)

    if study.best_value is None:
        raise AssertionError()
    if tuner.best_weights is None:
        raise AssertionError()
    if not np.isclose(tuner.best_weights.sum(), 1.0, atol=1e-6):
        raise AssertionError()
    if not np.all(tuner.best_weights >= 0):
        raise AssertionError()


def test_tuner_classification_negative_weights_path(small_classification_data):
    """Tests tuning with allow_negative_weights=True (L1 normalization)."""
    X, y = small_classification_data
    estimators = [
        LogisticRegression(max_iter=200, solver="liblinear", random_state=2),
        LogisticRegression(C=0.3, max_iter=200, solver="liblinear", random_state=3),
        LogisticRegression(C=2.0, max_iter=200, solver="liblinear", random_state=4),
    ]
    cv_trainer = _build_cv("Roc AUC", "classification")
    tuner = EnsembleWeightTunerCV(
        estimators=estimators,
        cross_validator=cv_trainer,
        n_trials=5,
        seed=13,
        allow_negative_weights=True,
    )
    tuner.tune(X, y)
    if tuner.best_weights is None:
        raise AssertionError()
    l1 = np.sum(np.abs(tuner.best_weights))
    if not np.isclose(l1, 1.0, atol=1e-6):
        raise AssertionError()
    # Some negative weight likely (not guaranteed) – so only check range
    if not np.all(np.abs(tuner.best_weights) <= 1.0 + 1e-6):
        raise AssertionError()


def test_tuner_regression_weights(small_regression_data):
    """Tests regression branch with MSE minimization and weight normalization."""
    X, y = small_regression_data
    estimators = [
        LinearRegression(),
        LinearRegression(),
    ]  # identical models OK for smoke
    cv_trainer = _build_cv("MSE", "regression")
    tuner = EnsembleWeightTunerCV(
        estimators=estimators, cross_validator=cv_trainer, n_trials=3, seed=5
    )
    tuner.tune(X, y)
    if tuner.best_weights is None:
        raise AssertionError()
    if not np.isclose(tuner.best_weights.sum(), 1.0, atol=1e-6):
        raise AssertionError()


def test_blend_predictions_shapes_classification_logits():
    """Directly tests _blend_predictions for binary prob case with negative weights (logit averaging)."""

    class DummyCV:  # minimal stub
        metric_name = "Roc AUC"
        problem_type = "classification"

    dummy = DummyCV()
    t = EnsembleWeightTunerCV([], dummy, n_trials=1, allow_negative_weights=True)
    preds_list = [np.full(10, 0.7), np.full(10, 0.2)]
    weights = np.array([0.4, -0.6])
    blended = t._blend_predictions(preds_list, weights)
    if blended.shape != (10,):
        raise AssertionError()
    if not np.all((blended >= 0.0) & (blended <= 1.0)):
        raise AssertionError()


def test_blend_predictions_multiclass_probability_normalization():
    """Checks probability rows sum to 1 after blending multiclass probs."""

    class DummyCV:
        metric_name = "Accuracy"  # preds type but we want probs path -> simulate classification probs metric
        problem_type = "classification"

    t = EnsembleWeightTunerCV([], DummyCV(), n_trials=1)
    # Force metric_type to 'probs' to exercise path
    t.metric_type = "probs"
    preds_list = [
        np.tile([[0.2, 0.3, 0.5]], (6, 1)),
        np.tile([[0.1, 0.6, 0.3]], (6, 1)),
    ]
    weights = np.array([0.3, 0.7])
    blended = t._blend_predictions(preds_list, weights)
    if blended.shape != (6, 3):
        raise AssertionError()
    row_sums = blended.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
