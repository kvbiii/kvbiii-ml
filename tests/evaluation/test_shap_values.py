"""Tests for kvbiii_ml.evaluation.shap_values module."""

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.evaluation import shap_values as shap_module


class _DummyExplanation:
    """Minimal explanation object exposing `.values` and `.base_values`."""

    def __init__(self, values: np.ndarray, base_values: np.ndarray) -> None:
        """Store SHAP-like values and base values.

        Args:
            values (np.ndarray): Pseudo SHAP values.
            base_values (np.ndarray): Pseudo base values.
        """
        self.values = values
        self.base_values = base_values


class DummyExplainer:
    """Lightweight stand-in for shap.TreeExplainer avoiding real SHAP computation."""

    def __init__(self, model: object, feature_perturbation: str | None = None) -> None:
        """Store the wrapped model; feature_perturbation is accepted but unused.

        Args:
            model (object): Model instance the explainer wraps.
            feature_perturbation (str | None): Ignored, kept for signature compatibility.
        """
        self.model = model

    def __call__(self, X: pd.DataFrame | np.ndarray) -> _DummyExplanation:
        """Returns a deterministic explanation object of ones with zero base values.

        Args:
            X (pd.DataFrame | np.ndarray): Input data to explain.

        Returns:
            _DummyExplanation: Explanation with `.values` of ones and zero `.base_values`.
        """
        arr = X.values if hasattr(X, "values") else X
        return _DummyExplanation(np.ones_like(arr, dtype=float), np.zeros(arr.shape[0]))


@pytest.fixture(autouse=True)
def patch_tree_explainer(monkeypatch: pytest.MonkeyPatch):
    """Replaces shap.TreeExplainer and shap.Explanation with lightweight dummies.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Yields:
        None: Control back to the test after patching is applied.
    """
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


def test_computeshapvalues_single_model_path():
    """Tests that compute_shap_values works for a single (non-ensemble) model.

    Asserts:
        - Returned explanation exposes a `.values` attribute
        - Values shape matches (n_samples, n_features)
    """

    class SmallModel:
        def __init__(self):
            self.__class__.__name__ = "RandomForestClassifier"

    model = SmallModel()
    X = pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])
    exp = shap_module.compute_shap_values(model, X)

    if not hasattr(exp, "values"):
        raise AssertionError()
    if exp.values.shape != (5, 3):
        raise AssertionError()


def test_computeshapvalues_ensemble_model_path_weights():
    """Tests that compute_shap_values weights sub-model SHAP values by ensemble weights.

    Asserts:
        - Weighted ensemble SHAP values have the expected shape
    """

    class SubModel:
        __name__ = "RandomForestClassifier"

    class Ensemble:
        def __init__(self):
            self.fitted_estimators_ = [SubModel(), SubModel()]
            self.weights = [0.3, 0.7]

    ens = Ensemble()
    X = pd.DataFrame(np.random.randn(4, 2), columns=["f0", "f1"])
    exp = shap_module.compute_shap_values(ens, X)

    if exp.values.shape != (4, 2):
        raise AssertionError()


def test_computeshapvalues_ensemble_uniform_weights_fallback():
    """Tests that compute_shap_values falls back to uniform weights when none are set.

    Asserts:
        - Ensemble SHAP computation succeeds without an explicit weights attribute
        - Values shape matches (n_samples, n_features)
    """

    class SubModel:
        pass

    class CVLike:
        def __init__(self):
            self.fitted_estimators_ = [SubModel(), SubModel(), SubModel()]

    obj = CVLike()
    X = pd.DataFrame(np.random.randn(3, 2), columns=["f0", "f1"])
    exp = shap_module.compute_shap_values(obj, X)

    if exp.values.shape != (3, 2):
        raise AssertionError()


def test_computeshapvalues_raises_for_histgradientboosting_single_model():
    """Tests that a HistGradientBoosting model raises ValueError on the single-model path.

    Asserts:
        - ValueError is raised mentioning HistGradientBoosting
    """
    model = type("HistGradientBoostingClassifier", (), {})()
    X = pd.DataFrame(np.random.randn(4, 2), columns=["f0", "f1"])

    with pytest.raises(ValueError, match="HistGradientBoosting is not supported"):
        shap_module.compute_shap_values(model, X)


def test_computeshapvalues_ensemble_skips_histgradientboosting_submodel():
    """Tests that a HistGradientBoosting sub-model is silently skipped within an ensemble.

    Asserts:
        - Ensemble SHAP computation succeeds despite one failing sub-model
        - Only the surviving sub-model contributes to the output shape
    """

    class GoodSubModel:
        pass

    class Ensemble:
        def __init__(self):
            self.fitted_estimators_ = [
                GoodSubModel(),
                type("HistGradientBoostingRegressor", (), {})(),
            ]
            self.weights = [0.5, 0.5]

    ens = Ensemble()
    X = pd.DataFrame(np.random.randn(4, 2), columns=["f0", "f1"])
    exp = shap_module.compute_shap_values(ens, X)

    if exp.values.shape != (4, 2):
        raise AssertionError()


def test_computeshapvalues_ensemble_raises_when_exceeds_max_estimators():
    """Tests that an ensemble with more than 50 estimators raises ValueError.

    Asserts:
        - ValueError is raised mentioning the ensemble size limit
    """

    class Ensemble:
        def __init__(self):
            self.fitted_estimators_ = [object() for _ in range(51)]
            self.weights = [1.0] * 51

    ens = Ensemble()
    X = pd.DataFrame(np.random.randn(3, 2), columns=["f0", "f1"])

    with pytest.raises(ValueError, match="Ensemble too large"):
        shap_module.compute_shap_values(ens, X)


def test_computeshapvalues_ensemble_skips_nested_ensemble_with_too_many_estimators():
    """Tests that a nested ensemble sub-model with more than 10 sub-estimators is skipped.

    Asserts:
        - Ensemble SHAP computation succeeds despite the oversized nested ensemble
        - Only the surviving sub-model contributes to the output shape
    """

    class GoodSubModel:
        pass

    class NestedEnsemble:
        def __init__(self):
            self.fitted_estimators_ = [object() for _ in range(11)]

    class OuterEnsemble:
        def __init__(self):
            self.fitted_estimators_ = [NestedEnsemble(), GoodSubModel()]
            self.weights = [0.5, 0.5]

    ens = OuterEnsemble()
    X = pd.DataFrame(np.random.randn(4, 2), columns=["f0", "f1"])
    exp = shap_module.compute_shap_values(ens, X)

    if exp.values.shape != (4, 2):
        raise AssertionError()


def test_computeshapvalues_ensemble_raises_when_no_valid_models_found():
    """Tests that an ensemble with every sub-model failing raises ValueError.

    Asserts:
        - ValueError is raised mentioning no valid models were found
    """

    class Ensemble:
        def __init__(self):
            self.fitted_estimators_ = [
                type("HistGradientBoostingClassifier", (), {})(),
                type("HistGradientBoostingRegressor", (), {})(),
            ]
            self.weights = [0.5, 0.5]

    ens = Ensemble()
    X = pd.DataFrame(np.random.randn(4, 2), columns=["f0", "f1"])

    with pytest.raises(ValueError, match="No valid models found in ensemble"):
        shap_module.compute_shap_values(ens, X)


def test_computeshapvalues_ensemble_raises_when_all_weights_zero_or_nonfinite():
    """Tests that an ensemble with all-zero/non-finite weights raises ValueError.

    Asserts:
        - ValueError is raised mentioning invalid weights
    """

    class SubModel:
        pass

    class Ensemble:
        def __init__(self):
            self.fitted_estimators_ = [SubModel(), SubModel()]
            self.weights = [0.0, np.nan]

    ens = Ensemble()
    X = pd.DataFrame(np.random.randn(4, 2), columns=["f0", "f1"])

    with pytest.raises(ValueError, match="All weights are zero or invalid"):
        shap_module.compute_shap_values(ens, X)


def test_computesinglemodelshap_falls_back_to_values_on_attributeerror(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tests that _compute_single_model_shap retries with a raw ndarray on AttributeError.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Asserts:
        - The ndarray fallback path succeeds when the DataFrame path raises AttributeError
        - The resulting explanation values keep the expected shape
    """

    class FlakyExplainer:
        def __init__(
            self, model: object, feature_perturbation: str | None = None
        ) -> None:
            """Store the wrapped model; feature_perturbation is accepted but unused."""
            self.model = model

        def __call__(self, X: pd.DataFrame | np.ndarray) -> _DummyExplanation:
            """Raises AttributeError for DataFrame input, succeeds for ndarray input."""
            if hasattr(X, "columns"):
                raise AttributeError("DataFrame path not supported")
            return _DummyExplanation(np.ones_like(X, dtype=float), np.zeros(X.shape[0]))

    monkeypatch.setattr(shap_module.shap, "TreeExplainer", FlakyExplainer)

    class GoodModel:
        pass

    X = pd.DataFrame(np.random.randn(4, 2), columns=["f0", "f1"])
    exp = shap_module._compute_single_model_shap(GoodModel(), X)

    if exp.values.shape != (4, 2):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
