"""Tests for kvbiii_ml.data_processing.feature_selection.model_importance_rfe module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from kvbiii_ml.data_processing.feature_selection.model_importance_rfe import (
    ModelImportanceRecursiveFeatureElimination,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

N_SAMPLES = 16
N_SPLITS = 2
RANDOM_STATE = 17

HISTORY_COLUMNS = [
    "step",
    "n_features_removed",
    "n_features_remaining",
    "removed_feature_name",
    "metric_value",
    "metric_change",
    "importance_score",
]


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
        }
    )
    y = pd.Series(((X["f0"] + X["f1"]) > 0).astype(int), name="target")
    return X, y


def _build_cv(metric_name: str = "Accuracy", problem_type: str = "classification"):
    """Builds a fast CrossValidationTrainer for RFE tests.

    Args:
        metric_name (str): Metric to optimize. Defaults to "Accuracy".
        problem_type (str): Either "classification" or "regression". Defaults to
            "classification".

    Returns:
        CrossValidationTrainer: Configured trainer with a tiny 2-fold KFold splitter.
    """
    return CrossValidationTrainer(
        problem_type=problem_type,
        metric_name=metric_name,
        cv=KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        preprocessing_pipeline=None,
        verbose=False,
    )


def _build_estimator() -> RandomForestClassifier:
    """Builds a fast RandomForestClassifier estimator for tests.

    Returns:
        RandomForestClassifier: Small, fast-fitting estimator with feature_importances_.
    """
    return RandomForestClassifier(
        n_estimators=5, max_depth=2, random_state=RANDOM_STATE
    )


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

    if selected != []:
        raise AssertionError()
    if best_metric is not None:
        raise AssertionError()


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

    if set(result.keys()) != {
        "selected_features",
        "selected_features_names",
        "history",
    }:
        raise AssertionError()
    history = result["history"]
    if int(history.iloc[0]["step"]) != 0:
        raise AssertionError()
    if int(history.iloc[0]["n_features_remaining"]) != X.shape[1]:
        raise AssertionError()
    if not (0 < len(result["selected_features"]) <= X.shape[1]):
        raise AssertionError()
    if not set(result["selected_features"]).issubset(set(X.columns)):
        raise AssertionError()
    if result["selected_features_names"] != result["selected_features"]:
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
