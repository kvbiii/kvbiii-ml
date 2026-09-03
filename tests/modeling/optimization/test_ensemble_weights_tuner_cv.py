from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from kvbiii_ml.evaluation.metrics import (
    get_metric_direction,
    get_metric_function,
    get_metric_type,
)
from kvbiii_ml.modeling.optimization.ensemble_weights_tuner import EnsembleWeightTunerCV
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


def _build_dummy_cv(metric_name: str, problem_type: str) -> Mock:
    """Build a minimal cross-validator stub exposing the attributes EnsembleWeightTunerCV reads.

    Args:
        metric_name (str): Metric name registered in METRICS.
        problem_type (str): Problem type, either "classification" or "regression".

    Returns:
        Mock: Stub with metric_name, problem_type, metric_fn, metric_type, and
            metric_direction attributes populated from METRICS.
    """
    dummy = Mock()
    dummy.metric_name = metric_name
    dummy.problem_type = problem_type
    dummy.metric_fn = get_metric_function(metric_name)
    dummy.metric_type = get_metric_type(metric_name)
    dummy.metric_direction = get_metric_direction(metric_name)
    dummy.verbose = False
    return dummy


@pytest.fixture
def small_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a small non-linearly-separable binary classification dataset."""
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=[f"f{i}" for i in range(4)])
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
    """Build a CrossValidationTrainer with a fixed 3-fold splitter.

    Args:
        metric (str): Metric name registered in METRICS.
        problem (str): Problem type, either "classification" or "regression".

    Returns:
        CrossValidationTrainer: Configured trainer.
    """
    return CrossValidationTrainer(
        metric_name=metric,
        problem_type=problem,
        cv=KFold(n_splits=3, shuffle=True, random_state=17),
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
    """Tests tuning with allow_negative_weights=True normalizes by signed sum, not L1 norm."""
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
    if not np.isclose(tuner.best_weights.sum(), 1.0, atol=1e-6):
        raise AssertionError()


def test_tuner_regression_weights(small_regression_data):
    """Tests regression branch with MSE minimization and weight normalization."""
    X, y = small_regression_data
    estimators = [
        LinearRegression(),
        LinearRegression(),
    ]
    cv_trainer = _build_cv("MSE", "regression")
    tuner = EnsembleWeightTunerCV(
        estimators=estimators, cross_validator=cv_trainer, n_trials=3, seed=5
    )
    tuner.tune(X, y)
    if tuner.best_weights is None:
        raise AssertionError()
    if not np.isclose(tuner.best_weights.sum(), 1.0, atol=1e-6):
        raise AssertionError()


def test_tune_never_selects_weights_worse_than_uniform(small_classification_data):
    """The guardrail must never return weights that score worse than uniform averaging."""
    X, y = small_classification_data
    estimators = [
        LogisticRegression(max_iter=200, solver="liblinear", random_state=0),
        LogisticRegression(C=0.5, max_iter=200, solver="liblinear", random_state=1),
        LogisticRegression(C=5.0, max_iter=200, solver="liblinear", random_state=2),
    ]
    cv_trainer = _build_cv("Roc AUC", "classification")
    tuner = EnsembleWeightTunerCV(
        estimators=estimators, cross_validator=cv_trainer, n_trials=10, seed=17
    )
    tuner.tune(X, y)

    y_true, preds_list, fold_boundaries = tuner._perform_cv(X, y)
    n = len(estimators)
    uniform_score = tuner._score_weights(
        np.full(n, 1.0 / n), y_true, preds_list, fold_boundaries
    )

    if tuner.best_score_ < uniform_score - 1e-9:
        raise AssertionError()


def test_normalize_weights_uses_signed_sum_not_l1():
    """Negative-weight normalization must divide by the signed sum, not the L1 norm."""
    dummy_cv = _build_dummy_cv("MSE", "regression")
    tuner = EnsembleWeightTunerCV([], dummy_cv, n_trials=1, allow_negative_weights=True)

    normalized = tuner._normalize_weights(np.array([1.5, -0.5]))

    if normalized is None:
        raise AssertionError()
    if not np.isclose(normalized.sum(), 1.0, atol=1e-9):
        raise AssertionError()
    if np.isclose(np.abs(normalized).sum(), 1.0, atol=1e-9):
        raise AssertionError("L1-normalized result should no longer be produced.")


def test_normalize_weights_degenerate_signed_sum_returns_none():
    """Weight vectors whose signed sum is near zero cannot be safely normalized."""
    dummy_cv = _build_dummy_cv("MSE", "regression")
    tuner = EnsembleWeightTunerCV([], dummy_cv, n_trials=1, allow_negative_weights=True)

    normalized = tuner._normalize_weights(np.array([0.5, -0.5]))

    if normalized is not None:
        raise AssertionError()


def test_objective_prunes_degenerate_weights():
    """The objective must prune trials whose sampled weights cancel out."""
    dummy_cv = _build_dummy_cv("MSE", "regression")
    tuner = EnsembleWeightTunerCV([], dummy_cv, n_trials=1, allow_negative_weights=True)
    tuner.estimators = [object(), object()]

    mock_trial = Mock()
    mock_trial.suggest_float.side_effect = [0.5, -0.5]

    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    preds_list = [pd.Series([1.0, 2.0, 3.0, 4.0]), pd.Series([1.0, 2.0, 3.0, 4.0])]
    fold_boundaries = [0, 2, 4]

    import optuna

    with pytest.raises(optuna.TrialPruned):
        tuner._objective(mock_trial, y_true, preds_list, fold_boundaries)


def test_score_weights_aggregates_per_fold_not_pooled():
    """The score must be a mean-minus-std across folds, not a single pooled metric."""
    dummy_cv = _build_dummy_cv("Roc AUC", "classification")
    tuner = EnsembleWeightTunerCV([], dummy_cv, n_trials=1)

    y_true = pd.Series([0, 1, 0, 1, 1, 0, 1, 0])
    preds = pd.Series([0.1, 0.9, 0.2, 0.8, 0.6, 0.4, 0.55, 0.45])
    fold_boundaries = [0, 4, 8]

    score = tuner._score_weights(np.array([1.0]), y_true, [preds], fold_boundaries)

    fold_1_auc = roc_auc_score(y_true.iloc[0:4], preds.iloc[0:4])
    fold_2_auc = roc_auc_score(y_true.iloc[4:8], preds.iloc[4:8])
    expected_mean = float(np.mean([fold_1_auc, fold_2_auc]))
    expected_std = float(np.std([fold_1_auc, fold_2_auc]))
    expected = expected_mean - tuner.fold_std_penalty * expected_std

    if not np.isclose(score, expected, atol=1e-9):
        raise AssertionError()


def test_blend_predictions_shapes_classification_logits():
    """Directly tests _blend_predictions for binary prob case with negative weights (logit averaging).

    Asserts:
        - Blended output preserves the (n_samples,) shape
        - Blended values remain valid probabilities in [0, 1]
    """
    dummy_cv = _build_dummy_cv("Roc AUC", "classification")
    t = EnsembleWeightTunerCV([], dummy_cv, n_trials=1, allow_negative_weights=True)
    preds_list = [pd.Series(np.full(10, 0.7)), pd.Series(np.full(10, 0.2))]
    weights = np.array([0.4, -0.6])
    blended = t._blend_predictions(preds_list, weights)
    if blended.shape != (10,):
        raise AssertionError()
    if not np.all((blended >= 0.0) & (blended <= 1.0)):
        raise AssertionError()


def test_blend_predictions_multiclass_probability_normalization():
    """Checks probability rows sum to 1 after blending multiclass probs.

    Asserts:
        - Blended output preserves the (n_samples, n_classes) shape
        - Every row of blended probabilities sums to 1
    """
    dummy_cv = _build_dummy_cv("Accuracy", "classification")
    t = EnsembleWeightTunerCV([], dummy_cv, n_trials=1)
    t.metric_type = "probs"
    columns = ["class_0", "class_1", "class_2"]
    preds_list = [
        pd.DataFrame(np.tile([[0.2, 0.3, 0.5]], (6, 1)), columns=columns),
        pd.DataFrame(np.tile([[0.1, 0.6, 0.3]], (6, 1)), columns=columns),
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
