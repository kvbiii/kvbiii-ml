"""Tests for kvbiii_ml.data_processing.feature_selection.permutation_feature_importance module."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from feature_engine.encoding import MeanEncoder
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from kvbiii_ml.data_processing.feature_selection import (
    permutation_feature_importance as permutation_module,
)
from kvbiii_ml.data_processing.feature_selection.model_importance_rfe import (
    ModelImportanceRecursiveFeatureElimination,
)
from kvbiii_ml.data_processing.feature_selection.permutation_feature_importance import (
    PermutationRecursiveFeatureElimination,
)
from kvbiii_ml.data_processing.preprocessing.categorical_encoding.string_similarity_encoder import (
    StringSimilarityEncoderWithOriginal,
)
from kvbiii_ml.data_processing.preprocessing.discretisation.equal_width_discretiser import (
    EqualWidthDiscretiserWithOriginal,
)

from ._test_support import (
    HISTORY_COLUMNS,
    RANDOM_STATE,
    assert_empty_history_returns_no_selection,
    assert_rfe_run_summary_starts_with_full_feature_set,
    build_cross_validator,
    build_fast_estimator,
)
from ._test_support import (
    build_small_classification_data as small_classification_data_factory,
)

_build_estimator = build_fast_estimator


class _FakeImportanceResult:
    """Deterministic stand-in for sklearn's permutation_importance return value."""

    def __init__(self, importances_mean: np.ndarray) -> None:
        """Initializes the fake result with a fixed per-feature mean importance array.

        Args:
            importances_mean (np.ndarray): Mean importance value per feature.
        """
        self.importances_mean = importances_mean


@pytest.fixture
def small_classification_data():
    """Provides a tiny synthetic binary classification dataset.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and binary target vector.
    """
    return small_classification_data_factory()


def _build_cv(n_jobs_scoring: str = "Accuracy"):
    """Builds a fast CrossValidationTrainer for permutation RFE tests.

    Args:
        n_jobs_scoring (str): Metric name to optimize. Defaults to "Accuracy".

    Returns:
        CrossValidationTrainer: Configured trainer with a tiny 2-fold KFold splitter.
    """
    return build_cross_validator(metric_name=n_jobs_scoring)


def test_permutationrfe_run_raises_valueerror_for_unknown_protected_feature(
    small_classification_data,
):
    """Tests run raises ValueError when a protected feature is absent from processed columns.

    Args:
        small_classification_data (tuple): Feature matrix and target fixture.

    Asserts:
        - ValueError is raised mentioning the missing protected feature
    """
    X, y = small_classification_data
    selector = PermutationRecursiveFeatureElimination(
        estimator=_build_estimator(),
        cross_validator=_build_cv(),
        steps=2,
        n_repeats=1,
        n_jobs=1,
        protected_features=["does_not_exist"],
        verbose=False,
    )

    with pytest.raises(ValueError, match="Protected features not found"):
        selector.run(X, y)


def test_permutationrfe_compute_fold_importances_averages_across_folds():
    """Tests _compute_fold_importances averages the per-fold importances_mean arrays.

    Asserts:
        - The returned importance per feature equals the mean across fold-level results
    """
    selector = PermutationRecursiveFeatureElimination(
        estimator=_build_estimator(),
        cross_validator=_build_cv(),
        steps=2,
        n_repeats=1,
        n_jobs=1,
        verbose=False,
    )
    columns = ["f0", "f1", "f2"]
    fold_results = iter([np.array([0.1, 0.2, 0.3]), np.array([0.3, 0.0, 0.3])])

    def _fake_permutation_importance(_estimator, _x, _y, **_kwargs):
        """Returns the next queued deterministic fake importance result."""
        return _FakeImportanceResult(next(fold_results))

    original = permutation_module.permutation_importance
    permutation_module.permutation_importance = _fake_permutation_importance
    try:
        fold_data = [
            (
                Mock(),
                pd.DataFrame(np.zeros((4, 3)), columns=columns),
                pd.Series([0, 1, 0, 1]),
            ),
            (
                Mock(),
                pd.DataFrame(np.zeros((4, 3)), columns=columns),
                pd.Series([1, 0, 1, 0]),
            ),
        ]
        importances = selector._compute_fold_importances(fold_data)
    finally:
        permutation_module.permutation_importance = original

    expected = {"f0": 0.2, "f1": 0.1, "f2": 0.3}
    for feature, value in expected.items():
        if importances[feature] != pytest.approx(value):
            raise AssertionError()


def test_permutationrfe_run_removes_lowest_importance_features_first(
    small_classification_data,
):
    """Tests run feeds averaged permutation importance into the removal order.

    A deterministic fake permutation_importance always reports strictly increasing
    scores by column index, so the least important feature should always be removed
    first regardless of which fold called it.

    Args:
        small_classification_data (tuple): Feature matrix and target fixture.

    Asserts:
        - The first step removes features in ascending importance order
        - The removed feature with the lowest importance is removed before the rest
    """
    X, y = small_classification_data

    def _fake_permutation_importance(_estimator, x_val, _y, **_kwargs):
        """Returns a fixed increasing-by-column importance array for every call."""
        return _FakeImportanceResult(np.arange(x_val.shape[1], dtype=float))

    original = permutation_module.permutation_importance
    permutation_module.permutation_importance = _fake_permutation_importance
    try:
        selector = PermutationRecursiveFeatureElimination(
            estimator=_build_estimator(),
            cross_validator=_build_cv(),
            steps=2,
            n_repeats=1,
            n_jobs=1,
            verbose=False,
        )
        result = selector.run(X, y)
    finally:
        permutation_module.permutation_importance = original

    history = result["history"]
    step_1_removed = history[history["step"] == 1]["removed_feature_name"].tolist()

    if step_1_removed != ["f0", "f1", "f2"]:
        raise AssertionError()


def test_permutationrfe_select_features_weighted_score_aggregates_per_step_not_per_row():
    """Tests select_features_weighted_score aggregates by step, unlike per-row RFE variants.

    Builds a history with two removed-feature rows within the same step and confirms
    the permutation-based selector treats them as a single step-level state, while
    ModelImportanceRecursiveFeatureElimination's per-row scoring yields a different,
    narrower selection on the identical history.

    Asserts:
        - The permutation selector returns all features not removed strictly before
          the best step (step-aggregated semantics)
        - The row-level selector returns only the single best-scoring row's feature
        - The two selectors disagree, proving the aggregation granularity differs
    """
    all_features = ["f0", "f1", "f2", "f3", "f4", "f5"]
    history = pd.DataFrame(
        [
            {
                "step": 0,
                "n_features_removed": 0,
                "n_features_remaining": 6,
                "removed_feature_name": None,
                "metric_value": 0.70,
                "metric_change": 0.0,
                "importance_score": np.nan,
            },
            {
                "step": 1,
                "n_features_removed": 1,
                "n_features_remaining": 5,
                "removed_feature_name": "f4",
                "metric_value": 0.75,
                "metric_change": 0.05,
                "importance_score": 0.02,
            },
            {
                "step": 1,
                "n_features_removed": 2,
                "n_features_remaining": 4,
                "removed_feature_name": "f5",
                "metric_value": 0.75,
                "metric_change": 0.05,
                "importance_score": 0.01,
            },
            {
                "step": 2,
                "n_features_removed": 3,
                "n_features_remaining": 3,
                "removed_feature_name": "f3",
                "metric_value": 0.80,
                "metric_change": 0.05,
                "importance_score": 0.015,
            },
        ]
    )

    perm_selector = PermutationRecursiveFeatureElimination(
        estimator=_build_estimator(),
        cross_validator=_build_cv(),
        steps=2,
        alpha=1.0,
        n_repeats=1,
        n_jobs=1,
        verbose=False,
    )
    perm_selector.all_processed_features = all_features

    row_selector = ModelImportanceRecursiveFeatureElimination(
        estimator=_build_estimator(), cross_validator=_build_cv(), steps=2, alpha=1.0
    )

    perm_selected, perm_metric = perm_selector.select_features_weighted_score(
        history, alpha=1.0
    )
    row_selected, _ = row_selector.select_features_weighted_score(history, alpha=1.0)

    if perm_selected != ["f0", "f1", "f2", "f3"]:
        raise AssertionError()
    if perm_metric != pytest.approx(0.80):
        raise AssertionError()
    if row_selected != ["f3"]:
        raise AssertionError()
    if perm_selected == row_selected:
        raise AssertionError()


def test_permutationrfe_select_features_weighted_score_empty_history_returns_none():
    """Tests select_features_weighted_score returns ([], None) for an empty history.

    Asserts:
        - An empty selected-features list is returned
        - The metric value is None
    """
    selector = PermutationRecursiveFeatureElimination(
        estimator=_build_estimator(), cross_validator=_build_cv(), steps=2, n_repeats=1
    )
    empty_history = pd.DataFrame(columns=HISTORY_COLUMNS)

    selected, best_metric = selector.select_features_weighted_score(empty_history)
    assert_empty_history_returns_no_selection(selected, best_metric)


def test_permutationrfe_run_returns_valid_summary_end_to_end(small_classification_data):
    """Tests a real (non-mocked) run() call produces a coherent selection summary.

    Uses n_repeats=1, n_jobs=1, and a tiny 2-fold split to keep runtime low while
    exercising the true sklearn permutation_importance code path.

    Args:
        small_classification_data (tuple): Feature matrix and target fixture.

    Asserts:
        - The result exposes selected_features, selected_features_names, and history
        - history starts at step 0 with the full feature set
        - selected_features is a non-empty subset of the original feature columns
        - selected_features_names aliases selected_features
    """
    X, y = small_classification_data
    selector = PermutationRecursiveFeatureElimination(
        estimator=_build_estimator(),
        cross_validator=_build_cv(),
        steps=2,
        alpha=0.95,
        n_repeats=1,
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbose=False,
    )

    assert_rfe_run_summary_starts_with_full_feature_set(selector.run(X, y), X)


def test_permutationrfe_run_with_pipeline_expansion_discovers_derived_columns(
    small_classification_data,
):
    """Tests the elimination loop works in processed feature space with an expansion pipeline.

    This selector had zero pipeline-restriction coverage before the
    pipeline_dependency redesign - it silently ignored preprocessing_pipeline.

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
    selector = PermutationRecursiveFeatureElimination(
        estimator=_build_estimator(),
        cross_validator=build_cross_validator(pipeline=pipeline),
        steps=2,
        n_repeats=1,
        n_jobs=1,
        verbose=False,
    )

    result = selector.run(X, y)

    if not any("_PREPROCESS_EQ_WIDTH" in f for f in selector.all_processed_features):
        raise AssertionError()
    if not set(result["selected_features"]).issubset(
        set(selector.all_processed_features)
    ):
        raise AssertionError()


def test_permutationrfe_run_with_string_similarity_encoder_survives_raw_column_removal():
    """Regression test for the confirmed suffix-matching bug in the old mechanism.

    StringSimilarityEncoderWithOriginal is a 1-to-many expansion transformer whose
    derived columns don't follow the `{var}_{suffix}` naming convention, so the old
    suffix-matching logic never discovered them as depending on their raw source.
    permutation_importance is faked to guarantee "product" is always the least
    important column; the real dependency graph and a real LGBMClassifier fit still
    run at every step.

    Asserts:
        - The raw "product" pass-through column is removed
        - The protected derived "product_apple" column survives to the final selection
    """
    products = ["apple", "orange", "grape", "melon"] * 10
    rng = np.random.default_rng(17)
    X = pd.DataFrame(
        {"product": pd.Categorical(products), "num": rng.normal(size=len(products))}
    )
    y = pd.Series((X["num"] > 0).astype(int), name="target")

    pipeline = Pipeline(
        [("sse", StringSimilarityEncoderWithOriginal(variables=["product"]))]
    )
    estimator = LGBMClassifier(n_estimators=5, max_depth=2, verbose=-1, random_state=17)

    def _fake_permutation_importance(_estimator, x_val, _y, **_kwargs):
        """Forces "product" to always be the least important column in x_val."""
        return _FakeImportanceResult(
            np.array([-10.0 if c == "product" else 1.0 for c in x_val.columns])
        )

    original = permutation_module.permutation_importance
    permutation_module.permutation_importance = _fake_permutation_importance
    try:
        selector = PermutationRecursiveFeatureElimination(
            estimator=estimator,
            cross_validator=build_cross_validator(pipeline=pipeline),
            steps=2,
            n_repeats=1,
            n_jobs=1,
            protected_features=["product_apple"],
            verbose=False,
        )
        result = selector.run(X, y)
    finally:
        permutation_module.permutation_importance = original

    if "product" not in result["history"]["removed_feature_name"].tolist():
        raise AssertionError("expected the raw pass-through column to be removed")
    if "product_apple" not in result["selected_features"]:
        raise AssertionError("expected the protected derived column to survive")


def test_permutationrfe_run_bootstraps_catboost_cat_features_from_graph_dtypes():
    """Tests CatBoost cat_features narrows correctly using the dependency graph's dtypes.

    A MeanEncoder converts the raw categorical column to float in place; cat_features
    must be narrowed away from that now-numeric column, otherwise CatBoost raises on
    a column declared categorical that is actually numeric.

    Asserts:
        - run() completes without CatBoost raising a cat-features/dtype error
        - A non-empty final selection is produced
    """
    rng = np.random.default_rng(17)
    n = 40
    X = pd.DataFrame(
        {
            "cat_col": rng.choice(["a", "b", "c"], size=n).astype(str),
            "num": rng.normal(size=n),
        }
    )
    y = pd.Series((X["num"] > 0).astype(int), name="target")

    pipeline = Pipeline([("mean_enc", MeanEncoder(variables=["cat_col"]))])
    estimator = CatBoostClassifier(
        n_estimators=10, verbose=0, random_state=17, cat_features=["cat_col"]
    )
    selector = PermutationRecursiveFeatureElimination(
        estimator=estimator,
        cross_validator=build_cross_validator(pipeline=pipeline),
        steps=1,
        n_repeats=1,
        n_jobs=1,
        verbose=False,
    )

    result = selector.run(X, y)

    if not len(result["selected_features"]) > 0:
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
