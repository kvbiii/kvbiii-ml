"""Tests for FeatureGenerationPipeline."""

import pandas as pd
import pytest

from kvbiii_ml.data_processing.feature_engineering.cross_encoding import (
    CrossFeatureGenerator,
)
from kvbiii_ml.data_processing.feature_engineering.feature_generation_pipeline import (
    FeatureGenerationPipeline,
)


class DummyAdder:
    """Minimal fit/transform stub that appends a constant ADDED column."""

    def __init__(self):
        self.fitted = False

    def fit(self, _x, _y=None):
        """Marks the stub as fitted without learning anything."""
        self.fitted = True
        return self

    def transform(self, X):
        """Returns a copy of X with a constant ADDED column appended."""
        if not self.fitted:
            raise AssertionError()
        X = X.copy()
        X["ADDED"] = 1
        return X


def test_pipeline_basic_flow():
    """Tests the pipeline's end-to-end fit_transform flow with custom preprocessing."""
    df = pd.DataFrame({"A": ["a", "b", "a", "b"], "B": ["x", "y", "x", "y"]})

    def before(df_in: pd.DataFrame) -> pd.DataFrame:
        df_in = df_in.copy()
        df_in["A2"] = df_in["A"].astype(str) + "2"
        return df_in

    steps = [CrossFeatureGenerator(features_names=[], degree=2), DummyAdder()]
    pipe = FeatureGenerationPipeline(
        features_names=["A", "B"],
        preprocessing_steps=steps,
        custom_features_generation_before_preprocessing=before,
    )
    transformed = pipe.fit_transform(df, y=pd.Series([0, 1, 0, 1]))
    cross_cols = steps[0].get_feature_names()
    if not any(col in transformed.columns for col in cross_cols):
        raise AssertionError()
    if "ADDED" not in transformed.columns:
        raise AssertionError()


def test_pipeline_transform_reuses_fitted_steps():
    """Tests that transform reuses steps already fitted during fit."""
    df = pd.DataFrame({"A": ["a", "b"], "B": ["x", "y"]})
    steps = [CrossFeatureGenerator(features_names=[], degree=2)]
    pipe = FeatureGenerationPipeline(
        features_names=["A", "B"], preprocessing_steps=steps
    )
    pipe.fit(df, y=pd.Series([0, 1]))
    out = pipe.transform(df)
    if not set(out.columns) >= set(df.columns):
        raise AssertionError()


def test_pipeline_error_on_missing_fit_method():
    """Tests that the pipeline raises ValueError when a step lacks a fit method."""

    class NoFit:
        """Preprocessing step stub deliberately missing a fit method."""

        def transform(self, X):  # pragma: no cover - negative path
            """Returns X unchanged."""
            return X

    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    pipe = FeatureGenerationPipeline(
        features_names=["A", "B"], preprocessing_steps=[NoFit()]
    )
    with pytest.raises(ValueError, match="must implement a fit"):
        pipe.fit(df, y=pd.Series([0, 1]))


def test_pipeline_supports_sklearn_preprocessing_transformer():
    """Tests that the pipeline works with a real sklearn preprocessing transformer."""
    sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")
    target_encoder = getattr(sklearn_preprocessing, "TargetEncoder", None)
    if target_encoder is None:
        pytest.skip("TargetEncoder is not available in this scikit-learn version.")

    df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": ["x", "y", "x", "z"]})
    y = pd.Series([1, 0, 1, 0])

    pipe = FeatureGenerationPipeline(
        features_names=["A", "B"],
        preprocessing_steps=[target_encoder(random_state=17)],
    )

    transformed = pipe.fit_transform(df, y=y)
    if not isinstance(transformed, pd.DataFrame):
        raise AssertionError()
    if list(transformed.columns) != ["A", "B"]:
        raise AssertionError()
    if transformed.shape != df.shape:
        raise AssertionError()

    transformed_test = pipe.transform(df)
    if not isinstance(transformed_test, pd.DataFrame):
        raise AssertionError()
    if list(transformed_test.columns) != ["A", "B"]:
        raise AssertionError()
    if transformed_test.shape != df.shape:
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
