"""Tests for ShapRecursiveFeatureElimination with patched SHAP computations."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from kvbiii_ml.data_processing.feature_selection import shap_rfe as shap_rfe_module
from kvbiii_ml.data_processing.feature_selection.shap_rfe import (
    ShapRecursiveFeatureElimination,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


@pytest.fixture
def shap_rfe_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a synthetic classification dataset for SHAP-based RFE tests.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and binary target vector.
    """
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(50, 5)), columns=[f"s{i}" for i in range(5)])
    y = pd.Series(
        ((X["s0"] - 0.7 * X["s1"] + rng.normal(scale=0.5, size=50)) > 0).astype(int)
    )
    return X, y


class FakeShapExplanation:
    """Deterministic stand-in for a shap.Explanation object used to patch SHAP computation."""

    def __init__(self, values: np.ndarray) -> None:
        """Initialize the fake explanation with (n_samples, n_features) SHAP values.

        Args:
            values (np.ndarray): Pseudo SHAP values of shape (n_samples, n_features).
        """
        self.values = values
        self.base_values = np.zeros(values.shape[0])


@pytest.fixture(autouse=True)
def patch_compute_shap_values(monkeypatch: pytest.MonkeyPatch):
    """Replaces compute_shap_values with a deterministic stub for all tests in this module.

    Args:
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture.

    Yields:
        None: Control back to the test after patching is applied.
    """

    def _fake_compute(
        _model: BaseEstimator, X: pd.DataFrame, _feature_names: list[str]
    ) -> FakeShapExplanation:
        """Returns deterministic pseudo-importance SHAP values for the given samples."""
        n_samples, n_features = X.shape
        vals = np.tile(np.linspace(0.01, 0.02 * n_features, n_features), (n_samples, 1))
        return FakeShapExplanation(vals)

    monkeypatch.setenv("SHAP_DISABLE_PROG_BAR", "1")
    monkeypatch.setattr(shap_rfe_module, "compute_shap_values", _fake_compute)
    yield


def _build_cv() -> CrossValidationTrainer:
    """Builds a CrossValidationTrainer configured for accuracy-based classification CV."""
    return CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=KFold(n_splits=3, shuffle=True, random_state=17),
        preprocessing_pipeline=None,
        verbose=False,
    )


def test_removal_schedule_shap():
    """Tests that the removal schedule has one entry per step and never removes all features.

    Asserts:
        - Schedule length always equals the configured number of steps
        - All scheduled removal counts are non-negative
        - Total scheduled removals leave at least one feature remaining
    """
    selector = ShapRecursiveFeatureElimination(
        estimator=LogisticRegression(max_iter=100, solver="liblinear"),
        cross_validator=_build_cv(),
        steps=4,
        verbose=False,
    )
    sched = selector.compute_removal_schedule(9)
    if len(sched) != 4:
        raise AssertionError()
    if not all(n >= 0 for n in sched):
        raise AssertionError()
    if not sum(sched) <= 8:
        raise AssertionError()


def test_run_shap_rfe(shap_rfe_data):
    """Tests that run() produces a non-empty history and a valid selected feature set.

    Uses a low alpha so the weighted score favors steps with fewer remaining features,
    ensuring the deterministic stub SHAP values (which drive a monotonic metric drop
    after the base step) do not select the base step as best.

    Args:
        shap_rfe_data (tuple[pd.DataFrame, pd.Series]): Synthetic classification data.

    Asserts:
        - Result dict exposes selected_features, selected_features_names, and history
        - History is non-empty and starts at step 0
        - Selected features are no more numerous than the original feature set
    """
    X, y = shap_rfe_data
    selector = ShapRecursiveFeatureElimination(
        estimator=LogisticRegression(max_iter=150, solver="liblinear"),
        cross_validator=_build_cv(),
        steps=3,
        alpha=0.3,
        verbose=False,
    )
    result = selector.run(X, y)
    if set(result.keys()) != {
        "selected_features",
        "selected_features_names",
        "history",
    }:
        raise AssertionError()
    history = result["history"]
    if history.empty:
        raise AssertionError()
    if history.iloc[0]["step"] != 0:
        raise AssertionError()
    if not len(result["selected_features"]) <= X.shape[1]:
        raise AssertionError()


def test_select_features_weighted_score_logic():
    """Tests that select_features_weighted_score returns a feature list and a metric.

    Asserts:
        - Selected features are returned as a list
        - Best metric value is not None
    """
    selector = ShapRecursiveFeatureElimination(
        estimator=LogisticRegression(max_iter=50, solver="liblinear"),
        cross_validator=_build_cv(),
        steps=2,
        verbose=False,
    )
    history = pd.DataFrame(
        [
            {
                "step": 0,
                "n_features_removed": 0,
                "n_features_remaining": 4,
                "removed_feature_name": None,
                "metric_value": 0.70,
                "metric_change": 0.0,
            },
            {
                "step": 1,
                "n_features_removed": 1,
                "n_features_remaining": 3,
                "removed_feature_name": "s0",
                "metric_value": 0.71,
                "metric_change": 0.01,
            },
            {
                "step": 2,
                "n_features_removed": 2,
                "n_features_remaining": 2,
                "removed_feature_name": "s1",
                "metric_value": 0.69,
                "metric_change": -0.02,
            },
        ]
    )
    selected, best_metric = selector.select_features_weighted_score(history, alpha=0.8)
    if not isinstance(selected, list):
        raise AssertionError()
    if best_metric is None:
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
