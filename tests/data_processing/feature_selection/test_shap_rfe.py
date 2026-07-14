"""Tests for ShapRecursiveFeatureElimination with patched SHAP computations."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from kvbiii_ml.data_processing.feature_selection.shap_rfe import (
    ShapRecursiveFeatureElimination,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


@pytest.fixture
def shap_rfe_data():
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(50, 5)), columns=[f"s{i}" for i in range(5)])
    y = pd.Series(
        ((X["s0"] - 0.7 * X["s1"] + rng.normal(scale=0.5, size=50)) > 0).astype(int)
    )
    return X, y


class FakeShapExplanation:
    def __init__(self, values):
        self.values = values  # (n_samples, n_features)
        # base_values[0] indexing pattern used – provide array
        self.base_values = np.zeros(values.shape[0])


@pytest.fixture(autouse=True)
def patch_compute_shap_values(monkeypatch):
    def _fake_compute(model, X, feature_names):  # pragma: no cover - patch logic
        n_samples, n_features = X.shape
        # Deterministic pseudo "importance"
        vals = np.tile(np.linspace(0.01, 0.02 * n_features, n_features), (n_samples, 1))
        return FakeShapExplanation(vals)

    monkeypatch.setenv("SHAP_DISABLE_PROG_BAR", "1")
    from kvbiii_ml.data_processing.feature_selection import shap_rfe as module

    monkeypatch.setattr(module, "compute_shap_values", _fake_compute)
    yield


def _build_cv():
    return CrossValidationTrainer(
        metric_name="Accuracy",
        problem_type="classification",
        cv=KFold(n_splits=3, shuffle=True, random_state=17),
        processors=None,
        verbose=False,
    )


def test_removal_schedule_shap():
    rng = ShapRecursiveFeatureElimination(
        estimator=LogisticRegression(max_iter=100, solver="liblinear"),
        cross_validator=_build_cv(),
        problem_type="classification",
        steps=4,
        verbose=False,
    )
    sched = rng.compute_removal_schedule(9)
    assert all(n > 0 for n in sched)
    assert sum(sched) <= 9


def test_run_shap_rfe(shap_rfe_data):
    X, y = shap_rfe_data
    selector = ShapRecursiveFeatureElimination(
        estimator=LogisticRegression(max_iter=150, solver="liblinear"),
        cross_validator=_build_cv(),
        problem_type="classification",
        steps=3,
        alpha=0.9,
        verbose=False,
    )
    result = selector.run(X, y)
    assert set(result.keys()) == {
        "selected_features",
        "selected_features_names",
        "history",
    }
    history = result["history"]
    assert not history.empty
    assert history.iloc[0]["step"] == 0
    assert len(result["selected_features"]) <= X.shape[1]


def test_select_features_weighted_score_logic():
    selector = ShapRecursiveFeatureElimination(
        estimator=LogisticRegression(max_iter=50, solver="liblinear"),
        cross_validator=_build_cv(),
        problem_type="classification",
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
    assert isinstance(selected, list)
    assert best_metric is not None


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
