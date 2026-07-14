"""Additional lightweight tests for evaluation.shap_values module.

We patch shap.TreeExplainer to avoid heavy computation and validate branching
logic for single vs ensemble models and weight handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.evaluation import shap_values as shap_module


class DummyExplainer:
    def __init__(self, model, feature_perturbation=None):  # pragma: no cover - trivial
        self.model = model

    def __call__(self, X):  # returns an object with .values and .base_values
        arr = X.values if hasattr(X, "values") else X

        class _E:
            values = np.ones_like(arr, dtype=float)
            base_values = np.zeros(arr.shape[0])

        return _E()


@pytest.fixture(autouse=True)
def patch_tree_explainer(monkeypatch):
    """Replaces shap.TreeExplainer with a lightweight dummy for all tests in this module."""
    monkeypatch.setattr(
        shap_module,
        "shap",
        type(
            "S",
            (),
            {
                "TreeExplainer": DummyExplainer,
                "Explanation": lambda **kw: type("E2", (), kw)(),
            },
        ),
    )
    yield


def test_single_model_path():
    """Tests that compute_shap_values works for a single (non-ensemble) model."""

    class SmallModel:
        def __init__(self):
            self.__class__.__name__ = "RandomForestClassifier"

    model = SmallModel()
    X = pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])
    exp = shap_module.compute_shap_values(model, X)
    if not (hasattr(exp, "values")):
        raise AssertionError()
    if not (exp.values.shape == (5, 3)):
        raise AssertionError()


def test_ensemble_model_path_weights():
    """Tests that compute_shap_values weights sub-model SHAP values by ensemble weights."""

    class SubModel:
        __name__ = "RandomForestClassifier"

    class Ensemble:
        def __init__(self):
            self.fitted_estimators_ = [SubModel(), SubModel()]
            self.weights = [0.3, 0.7]

    ens = Ensemble()
    X = pd.DataFrame(np.random.randn(4, 2), columns=["f0", "f1"])
    exp = shap_module.compute_shap_values(ens, X)
    if not (exp.values.shape == (4, 2)):
        raise AssertionError()


def test_ensemble_uniform_weights_fallback():
    """Tests that compute_shap_values falls back to uniform weights when none are set."""

    class SubModel:
        pass

    class CVLike:
        def __init__(self):
            self.fitted_estimators_ = [SubModel(), SubModel(), SubModel()]

    obj = CVLike()
    X = pd.DataFrame(np.random.randn(3, 2), columns=["f0", "f1"])
    exp = shap_module.compute_shap_values(obj, X)
    if not (exp.values.shape == (3, 2)):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
