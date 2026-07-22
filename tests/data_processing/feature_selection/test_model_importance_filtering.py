"""Tests for kvbiii_ml.data_processing.feature_selection.model_importance_filtering module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from kvbiii_ml.data_processing.feature_selection.model_importance_filtering import (
    ModelImportanceFiltering,
)
from kvbiii_ml.data_processing.preprocessing.discretisation.equal_width_discretiser import (
    EqualWidthDiscretiserWithOriginal,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

N_SAMPLES = 16
N_SPLITS = 2
RANDOM_STATE = 17


@pytest.fixture
def small_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a tiny synthetic binary classification dataset.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and binary target vector.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    X = pd.DataFrame(
        {
            "f0": rng.normal(size=N_SAMPLES),
            "f1": rng.normal(size=N_SAMPLES),
            "f2": rng.normal(size=N_SAMPLES),
            "f3": rng.normal(size=N_SAMPLES),
            "noise": rng.normal(size=N_SAMPLES),
        }
    )
    y = pd.Series(((X["f0"] + X["f1"]) > 0).astype(int), name="target")
    return X, y


def _build_cv(pipeline: Pipeline | None = None) -> CrossValidationTrainer:
    """Builds a fast CrossValidationTrainer for classification tests.

    Args:
        pipeline (Pipeline | None): Optional preprocessing pipeline. Defaults to None.

    Returns:
        CrossValidationTrainer: Configured trainer with a tiny 2-fold KFold splitter.
    """
    return CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        preprocessing_pipeline=pipeline,
        verbose=False,
    )


def _build_estimator() -> RandomForestClassifier:
    """Builds a fast RandomForestClassifier estimator for tests.

    Returns:
        RandomForestClassifier: Small, fast-fitting estimator with feature_importances_.
    """
    return RandomForestClassifier(n_estimators=5, max_depth=2, random_state=RANDOM_STATE)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"threshold": "not_a_number"}, "threshold must be a numeric value"),
        ({"max_steps": 0}, "max_steps must be a positive integer"),
        ({"max_steps": "3"}, "max_steps must be a positive integer"),
        ({"verbose": "yes"}, "verbose must be a boolean"),
        ({"protected_features": "f0"}, "protected_features must be a list of strings"),
        ({"protected_features": [1, 2]}, "protected_features must be a list of strings"),
    ],
)
def test_modelimportancefiltering_init_raises_valueerror_for_invalid_params(kwargs, match):
    """Tests init validation rejects invalid parameter types and values.

    Args:
        kwargs (dict): Invalid keyword argument overriding a valid default.
        match (str): Expected substring of the ValueError message.

    Asserts:
        - ValueError is raised with a message matching the invalid parameter
    """
    with pytest.raises(ValueError, match=match):
        ModelImportanceFiltering(
            estimator=_build_estimator(), cross_validator=_build_cv(), **kwargs
        )


def test_modelimportancefiltering_run_raises_valueerror_for_unknown_protected_feature(
    small_classification_data,
):
    """Tests run raises ValueError when a protected feature is absent from processed columns.

    Args:
        small_classification_data (tuple): Feature matrix and target fixture.

    Asserts:
        - ValueError is raised mentioning the missing protected feature
    """
    X, y = small_classification_data
    selector = ModelImportanceFiltering(
        estimator=_build_estimator(),
        cross_validator=_build_cv(),
        protected_features=["does_not_exist"],
        max_steps=1,
    )

    with pytest.raises(ValueError, match="Protected features not found"):
        selector.run(X, y)


def test_modelimportancefiltering_run_raises_attributeerror_when_estimator_lacks_importances(
    small_classification_data,
):
    """Tests run raises AttributeError when the estimator lacks feature_importances_.

    Args:
        small_classification_data (tuple): Feature matrix and target fixture.

    Asserts:
        - AttributeError is raised when using LogisticRegression as the estimator
    """
    X, y = small_classification_data
    selector = ModelImportanceFiltering(
        estimator=LogisticRegression(max_iter=100, solver="liblinear"),
        cross_validator=_build_cv(),
        max_steps=1,
    )

    with pytest.raises(AttributeError):
        selector.run(X, y)


def test_modelimportancefiltering_run_stops_when_only_protected_features_remain(
    small_classification_data,
):
    """Tests the elimination loop stops early once only protected features remain.

    Args:
        small_classification_data (tuple): Feature matrix and target fixture.

    Asserts:
        - All non-protected features are removed within a single step
        - The final selected_features contain exactly the protected features
        - The loop does not exceed max_steps
    """
    X, y = small_classification_data
    selector = ModelImportanceFiltering(
        estimator=_build_estimator(),
        cross_validator=_build_cv(),
        threshold=1.1,
        protected_features=["f0"],
        max_steps=5,
        verbose=False,
    )

    result = selector.run(X, y)

    if result["selected_features"] != ["f0"]:
        raise AssertionError()
    if int(result["history"]["step"].max()) > 5:
        raise AssertionError()


def test_modelimportancefiltering_history_schema_matches_documented_columns(
    small_classification_data,
):
    """Tests the returned history DataFrame exposes exactly the documented columns.

    Args:
        small_classification_data (tuple): Feature matrix and target fixture.

    Asserts:
        - history columns exactly match ModelImportanceFiltering.history_schema keys
    """
    X, y = small_classification_data
    selector = ModelImportanceFiltering(
        estimator=_build_estimator(),
        cross_validator=_build_cv(),
        threshold=0.0,
        max_steps=2,
        verbose=False,
    )

    result = selector.run(X, y)

    if set(result["history"].columns) != set(ModelImportanceFiltering.history_schema.keys()):
        raise AssertionError()


def test_modelimportancefiltering_history_covers_every_feature_via_trailing_block(
    small_classification_data,
):
    """Tests every non-protected processed feature appears exactly once in history.

    With a default threshold of 0.0 no RandomForest feature typically qualifies for
    removal, so all features are expected to land in the trailing "never removed" block.

    Args:
        small_classification_data (tuple): Feature matrix and target fixture.

    Asserts:
        - Every non-protected processed feature appears exactly once as a removed_feature_name
        - No feature appears more than once in the removal history
    """
    X, y = small_classification_data
    selector = ModelImportanceFiltering(
        estimator=_build_estimator(),
        cross_validator=_build_cv(),
        threshold=0.0,
        max_steps=2,
        verbose=False,
    )

    result = selector.run(X, y)

    removed_names = result["history"]["removed_feature_name"].dropna().tolist()
    if sorted(removed_names) != sorted(X.columns):
        raise AssertionError()
    if len(removed_names) != len(set(removed_names)):
        raise AssertionError()


def test_modelimportancefiltering_fit_wraps_run_and_caches_selected_features(
    small_classification_data,
):
    """Tests fit is a thin wrapper around run that caches selected_features_ and returns self.

    Args:
        small_classification_data (tuple): Feature matrix and target fixture.

    Asserts:
        - fit returns the selector instance itself
        - selected_features_ matches the features selected by an equivalent run() call
    """
    X, y = small_classification_data
    fit_selector = ModelImportanceFiltering(
        estimator=_build_estimator(),
        cross_validator=_build_cv(),
        threshold=0.0,
        max_steps=1,
        verbose=False,
    )
    run_selector = ModelImportanceFiltering(
        estimator=_build_estimator(),
        cross_validator=_build_cv(),
        threshold=0.0,
        max_steps=1,
        verbose=False,
    )

    fitted = fit_selector.fit(X, y)
    run_result = run_selector.run(X, y)

    if fitted is not fit_selector:
        raise AssertionError()
    if fit_selector.selected_features_ != run_result["selected_features"]:
        raise AssertionError()


def test_modelimportancefiltering_run_with_pipeline_expansion_discovers_derived_columns(
    small_classification_data,
):
    """Tests the elimination loop works in processed feature space with an expansion pipeline.

    Args:
        small_classification_data (tuple): Feature matrix and target fixture.

    Asserts:
        - all_processed_features includes derived discretised columns
        - selected_features is a subset of all_processed_features
    """
    X, y = small_classification_data
    pipeline = Pipeline(
        [
            (
                "eq_width",
                EqualWidthDiscretiserWithOriginal(variables=["f0", "f1"], bins=3),
            ),
        ]
    )
    selector = ModelImportanceFiltering(
        estimator=_build_estimator(),
        cross_validator=_build_cv(pipeline=pipeline),
        threshold=0.0,
        max_steps=1,
        verbose=False,
    )

    result = selector.run(X, y)

    if not any("_PREPROCESS_EQ_WIDTH" in f for f in selector.all_processed_features):
        raise AssertionError()
    if not set(result["selected_features"]).issubset(set(selector.all_processed_features)):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
