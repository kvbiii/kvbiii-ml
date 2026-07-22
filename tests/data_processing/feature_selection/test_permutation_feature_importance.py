"""Tests for kvbiii_ml.data_processing.feature_selection.permutation_feature_importance module."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from kvbiii_ml.data_processing.feature_selection import (
    permutation_feature_importance as permutation_module,
)
from kvbiii_ml.data_processing.feature_selection.model_importance_rfe import (
    ModelImportanceRecursiveFeatureElimination,
)
from kvbiii_ml.data_processing.feature_selection.permutation_feature_importance import (
    PermutationRecursiveFeatureElimination,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

N_SAMPLES = 16
N_SPLITS = 2
RANDOM_STATE = 17


class _FakeImportanceResult:
    """Deterministic stand-in for sklearn's permutation_importance return value."""

    def __init__(self, importances_mean: np.ndarray) -> None:
        """Initializes the fake result with a fixed per-feature mean importance array.

        Args:
            importances_mean (np.ndarray): Mean importance value per feature.
        """
        self.importances_mean = importances_mean


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


def _build_cv(n_jobs_scoring: str = "Accuracy") -> CrossValidationTrainer:
    """Builds a fast CrossValidationTrainer for permutation RFE tests.

    Args:
        n_jobs_scoring (str): Metric name to optimize. Defaults to "Accuracy".

    Returns:
        CrossValidationTrainer: Configured trainer with a tiny 2-fold KFold splitter.
    """
    return CrossValidationTrainer(
        problem_type="classification",
        metric_name=n_jobs_scoring,
        cv=KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        preprocessing_pipeline=None,
        verbose=False,
    )


def _build_estimator() -> RandomForestClassifier:
    """Builds a fast RandomForestClassifier estimator for tests.

    Returns:
        RandomForestClassifier: Small, fast-fitting estimator.
    """
    return RandomForestClassifier(n_estimators=5, max_depth=2, random_state=RANDOM_STATE)


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

    def _fake_permutation_importance(_estimator, _X, _y, **_kwargs):
        """Returns the next queued deterministic fake importance result."""
        return _FakeImportanceResult(next(fold_results))

    original = permutation_module.permutation_importance
    permutation_module.permutation_importance = _fake_permutation_importance
    try:
        fold_data = [
            (Mock(), pd.DataFrame(np.zeros((4, 3)), columns=columns), pd.Series([0, 1, 0, 1])),
            (Mock(), pd.DataFrame(np.zeros((4, 3)), columns=columns), pd.Series([1, 0, 1, 0])),
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
    empty_history = pd.DataFrame(
        columns=[
            "step",
            "n_features_removed",
            "n_features_remaining",
            "removed_feature_name",
            "metric_value",
            "metric_change",
            "importance_score",
        ]
    )

    selected, best_metric = selector.select_features_weighted_score(empty_history)

    if selected != []:
        raise AssertionError()
    if best_metric is not None:
        raise AssertionError()


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

    result = selector.run(X, y)

    if set(result.keys()) != {"selected_features", "selected_features_names", "history"}:
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
