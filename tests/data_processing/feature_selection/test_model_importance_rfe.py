"""Tests for kvbiii_ml.data_processing.feature_selection.model_importance_rfe module."""

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from feature_engine.encoding import MeanEncoder
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from kvbiii_ml.data_processing.feature_selection.model_importance_rfe import (
    ModelImportanceRecursiveFeatureElimination,
)
from kvbiii_ml.data_processing.preprocessing.categorical_encoding.string_similarity_encoder import (
    StringSimilarityEncoderWithOriginal,
)
from kvbiii_ml.data_processing.preprocessing.discretisation.equal_width_discretiser import (
    EqualWidthDiscretiserWithOriginal,
)

from ._test_support import (
    HISTORY_COLUMNS,
    assert_empty_history_returns_no_selection,
    assert_rfe_run_summary_starts_with_full_feature_set,
    build_cross_validator,
    build_fast_estimator,
)
from ._test_support import (
    build_small_classification_data as small_classification_data_factory,
)

_build_estimator = build_fast_estimator


@pytest.fixture
def small_classification_data():
    """Provides a tiny synthetic binary classification dataset.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and binary target vector.
    """
    return small_classification_data_factory()


def _build_cv(metric_name: str = "Accuracy", problem_type: str = "classification"):
    """Builds a fast CrossValidationTrainer for RFE tests.

    Args:
        metric_name (str): Metric to optimize. Defaults to "Accuracy".
        problem_type (str): Either "classification" or "regression". Defaults to
            "classification".

    Returns:
        CrossValidationTrainer: Configured trainer with a tiny 2-fold KFold splitter.
    """
    return build_cross_validator(metric_name=metric_name, problem_type=problem_type)


@pytest.mark.parametrize(
    "total_removable,steps,expected",
    [
        (9, 4, [3, 3, 2, 1]),
        (20, 5, [7, 5, 4, 3, 1]),
        (3, 10, [1, 1, 1]),
        (1, 5, [1]),
        (0, 5, []),
    ],
)
def test_modelimportancerfe_compute_removal_schedule_matches_expected_decay(
    total_removable, steps, expected
):
    """Tests compute_removal_schedule produces the exact linear-decay schedule.

    Args:
        total_removable (int): Number of removable features passed to the schedule.
        steps (int): Number of elimination steps configured on the selector.
        expected (list[int]): Exact expected per-step removal counts.

    Asserts:
        - The computed schedule exactly matches the expected linear-decay schedule
        - The schedule sums to at most total_removable (trailing zero-steps trimmed)
    """
    selector = ModelImportanceRecursiveFeatureElimination(
        estimator=_build_estimator(), cross_validator=_build_cv(), steps=steps
    )

    schedule = selector.compute_removal_schedule(total_removable)

    if schedule != expected:
        raise AssertionError()
    if sum(schedule) > total_removable:
        raise AssertionError()


def test_modelimportancerfe_compute_removal_schedule_returns_empty_for_zero_steps():
    """Tests compute_removal_schedule returns an empty list when steps is zero.

    Asserts:
        - An empty list is returned regardless of total_removable_features
    """
    selector = ModelImportanceRecursiveFeatureElimination(
        estimator=_build_estimator(), cross_validator=_build_cv(), steps=0
    )

    if selector.compute_removal_schedule(10) != []:
        raise AssertionError()


def _build_history(
    metric_values: list[float], feature_names: list[str | None]
) -> pd.DataFrame:
    """Builds a hand-crafted step-wise history DataFrame for weighted-score tests.

    Args:
        metric_values (list[float]): Metric value recorded at each step.
        feature_names (list[str | None]): Removed feature name recorded at each step,
            with None for the baseline step-0 row.

    Returns:
        pd.DataFrame: History DataFrame with one row per step.
    """
    n_steps = len(metric_values)
    return pd.DataFrame(
        {
            "step": list(range(n_steps)),
            "n_features_removed": list(range(n_steps)),
            "n_features_remaining": [n_steps - i for i in range(n_steps)],
            "removed_feature_name": feature_names,
            "metric_value": metric_values,
            "metric_change": [0.0] * n_steps,
            "importance_score": [np.nan] + [0.1] * (n_steps - 1),
        }
    )


def test_modelimportancerfe_select_features_weighted_score_operates_per_removed_row():
    """Tests select_features_weighted_score picks the single best-scoring row (maximize).

    With alpha=1.0 only the metric matters, so the best step is the one with the
    highest metric_value, and only that row's removed_feature_name is selected.

    Asserts:
        - The selected feature list contains only the best step's removed feature
        - The returned metric equals the best step's metric_value
    """
    selector = ModelImportanceRecursiveFeatureElimination(
        estimator=_build_estimator(), cross_validator=_build_cv(), steps=2, alpha=1.0
    )
    history = _build_history(
        metric_values=[0.70, 0.72, 0.75], feature_names=[None, "f_a", "f_b"]
    )

    selected, best_metric = selector.select_features_weighted_score(history, alpha=1.0)

    if selected != ["f_b"]:
        raise AssertionError()
    if best_metric != pytest.approx(0.75):
        raise AssertionError()


def test_modelimportancerfe_select_features_weighted_score_direction_aware_for_minimize():
    """Tests select_features_weighted_score favors the lowest metric for minimize direction.

    Asserts:
        - The selector with a minimize-direction metric selects the row with the
          smallest metric_value when alpha=1.0
    """
    selector = ModelImportanceRecursiveFeatureElimination(
        estimator=_build_estimator(),
        cross_validator=_build_cv(metric_name="RMSE", problem_type="regression"),
        steps=2,
        alpha=1.0,
    )
    history = _build_history(
        metric_values=[10.0, 8.0, 5.0], feature_names=[None, "f_a", "f_b"]
    )

    selected, best_metric = selector.select_features_weighted_score(history, alpha=1.0)

    if selected != ["f_b"]:
        raise AssertionError()
    if best_metric != pytest.approx(5.0):
        raise AssertionError()


def test_modelimportancerfe_select_features_weighted_score_includes_protected_features():
    """Tests select_features_weighted_score always includes protected_features in output.

    Asserts:
        - Protected features are present in the returned selection even though they
          never appear as a removed_feature_name in the history
    """
    selector = ModelImportanceRecursiveFeatureElimination(
        estimator=_build_estimator(),
        cross_validator=_build_cv(),
        steps=2,
        alpha=1.0,
        protected_features=["f_protected"],
    )
    history = _build_history(
        metric_values=[0.70, 0.72, 0.75], feature_names=[None, "f_a", "f_b"]
    )

    selected, _ = selector.select_features_weighted_score(history, alpha=1.0)

    if "f_protected" not in selected:
        raise AssertionError()


def test_modelimportancerfe_select_features_weighted_score_empty_history_returns_none():
    """Tests select_features_weighted_score returns ([], None) for an empty history.

    Asserts:
        - An empty selected-features list is returned
        - The metric value is None
    """
    selector = ModelImportanceRecursiveFeatureElimination(
        estimator=_build_estimator(), cross_validator=_build_cv(), steps=2
    )
    empty_history = pd.DataFrame(columns=HISTORY_COLUMNS)

    selected, best_metric = selector.select_features_weighted_score(empty_history)

    assert_empty_history_returns_no_selection(selected, best_metric)


def test_modelimportancerfe_run_raises_valueerror_for_unknown_protected_feature(
    small_classification_data,
):
    """Tests run raises ValueError when a protected feature is absent from X.columns.

    Args:
        small_classification_data (tuple): Feature matrix and target fixture.

    Asserts:
        - ValueError is raised mentioning the missing protected feature
    """
    X, y = small_classification_data
    selector = ModelImportanceRecursiveFeatureElimination(
        estimator=_build_estimator(),
        cross_validator=_build_cv(),
        steps=2,
        protected_features=["does_not_exist"],
    )

    with pytest.raises(ValueError, match="Protected features not found"):
        selector.run(X, y)


def test_modelimportancerfe_run_returns_valid_summary_end_to_end(
    small_classification_data,
):
    """Tests run() produces a coherent selection summary on small synthetic data.

    Args:
        small_classification_data (tuple): Feature matrix and target fixture.

    Asserts:
        - The result exposes selected_features, selected_features_names, and history
        - history starts at step 0 with the full feature set
        - selected_features is a non-empty subset of the original feature columns
        - selected_features_names aliases selected_features
    """
    X, y = small_classification_data
    selector = ModelImportanceRecursiveFeatureElimination(
        estimator=_build_estimator(),
        cross_validator=_build_cv(),
        steps=2,
        alpha=0.95,
        verbose=False,
    )

    result = selector.run(X, y)

    assert_rfe_run_summary_starts_with_full_feature_set(result, X)


def test_modelimportancerfe_run_with_pipeline_expansion_discovers_derived_columns(
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
    selector = ModelImportanceRecursiveFeatureElimination(
        estimator=_build_estimator(),
        cross_validator=build_cross_validator(pipeline=pipeline),
        steps=1,
        verbose=False,
    )

    result = selector.run(X, y)

    if not any("_PREPROCESS_EQ_WIDTH" in f for f in selector.all_processed_features):
        raise AssertionError()
    if not set(result["selected_features"]).issubset(
        set(selector.all_processed_features)
    ):
        raise AssertionError()


def test_modelimportancerfe_run_with_string_similarity_encoder_survives_raw_column_removal():
    """Regression test proving the raw-vs-processed feature-space fix for RFE elimination.

    StringSimilarityEncoderWithOriginal is a 1-to-many expansion transformer, so this
    exercises the same dependency-graph resolution proven for ModelImportanceFiltering.
    The real dependency graph and a real LGBMClassifier fit still run at every step;
    only the raw "product" column's importance score is patched to the global minimum
    (after the real fit), guaranteeing it is always the first candidate removed
    whenever present. This proves the restricted pipeline can still regenerate the
    protected derived column from just the raw "product" column once "product" itself
    is no longer selected.

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
    selector = ModelImportanceRecursiveFeatureElimination(
        estimator=estimator,
        cross_validator=build_cross_validator(pipeline=pipeline),
        steps=3,
        protected_features=["product_apple"],
        verbose=False,
    )
    original_importance = selector._cross_val_model_importance

    def _forced_importance(*args, **kwargs):
        """Forces "product"'s score to the global minimum whenever it is present."""
        importance_map, fold_metric, fold_std = original_importance(*args, **kwargs)
        if "product" in importance_map:
            importance_map["product"] = min(importance_map.values()) - 1.0
        return importance_map, fold_metric, fold_std

    selector._cross_val_model_importance = _forced_importance

    result = selector.run(X, y)

    if "product" not in result["history"]["removed_feature_name"].tolist():
        raise AssertionError("expected the raw pass-through column to be removed")
    if "product_apple" not in result["selected_features"]:
        raise AssertionError("expected the protected derived column to survive")


def test_modelimportancerfe_run_bootstraps_catboost_cat_features_from_graph_dtypes():
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
    selector = ModelImportanceRecursiveFeatureElimination(
        estimator=estimator,
        cross_validator=build_cross_validator(pipeline=pipeline),
        steps=1,
        verbose=False,
    )

    result = selector.run(X, y)

    if not len(result["selected_features"]) > 0:
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
