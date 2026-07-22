"""Tests for MutualInformationRecursiveFeatureElimination."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from kvbiii_ml.data_processing.feature_selection.mutual_information_rfe import (
    MutualInformationRecursiveFeatureElimination,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


@pytest.fixture
def mi_rfe_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a synthetic classification dataset for MI-based RFE tests.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and binary target vector.
    """
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(80, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(
        ((X["f0"] * 0.8 + X["f1"] * -0.5 + rng.normal(size=80)) > 0).astype(int)
    )
    return X, y


def _build_cv() -> CrossValidationTrainer:
    """Builds a CrossValidationTrainer configured for accuracy-based classification CV.

    Returns:
        CrossValidationTrainer: Trainer wired for 3-fold classification CV.
    """
    return CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=KFold(n_splits=3, shuffle=True, random_state=17),
        preprocessing_pipeline=None,
        verbose=False,
    )


def _build_selector(
    steps: int = 4,
    alpha: float = 0.95,
    protected_features: list[str] | None = None,
) -> MutualInformationRecursiveFeatureElimination:
    """Builds a MutualInformationRecursiveFeatureElimination wired to a real CV trainer.

    Args:
        steps (int, optional): Number of elimination iterations. Defaults to 4.
        alpha (float, optional): Weighted score alpha. Defaults to 0.95.
        protected_features (list[str] | None, optional): Features never removed.
            Defaults to None.

    Returns:
        MutualInformationRecursiveFeatureElimination: Configured selector instance.
    """
    return MutualInformationRecursiveFeatureElimination(
        estimator=LogisticRegression(max_iter=100, solver="liblinear"),
        cross_validator=_build_cv(),
        steps=steps,
        alpha=alpha,
        verbose=False,
        protected_features=protected_features,
    )


def test_mutual_information_rfe_init_derives_metric_from_cross_validator():
    """Tests that constructor derives metric/problem-type attributes from cross_validator.

    Asserts:
        - problem_type, metric_fn, metric_type, and metric_direction mirror the
          injected CrossValidationTrainer
        - protected_features defaults to an empty list when not provided
    """
    cross_validator = _build_cv()
    selector = MutualInformationRecursiveFeatureElimination(
        estimator=LogisticRegression(max_iter=100, solver="liblinear"),
        cross_validator=cross_validator,
        steps=3,
        verbose=False,
    )
    if selector.problem_type != cross_validator.problem_type:
        raise AssertionError()
    if selector.metric_fn is not cross_validator.metric_fn:
        raise AssertionError()
    if selector.metric_type != cross_validator.metric_type:
        raise AssertionError()
    if selector.metric_direction != cross_validator.metric_direction:
        raise AssertionError()
    if selector.protected_features != []:
        raise AssertionError()


@pytest.mark.parametrize(
    ("total_removable_features", "steps", "expected"),
    [
        (10, 4, [4, 3, 2, 1]),
        (9, 4, [3, 3, 2, 1]),
        (5, 4, [2, 2, 1]),
        (1, 4, [1]),
        (0, 4, []),
    ],
)
def test_compute_removal_schedule_linear_decay(
    total_removable_features: int, steps: int, expected: list[int]
):
    """Tests that compute_removal_schedule follows a linear-decay weighting scheme.

    Args:
        total_removable_features (int): Number of features eligible for removal.
        steps (int): Number of elimination steps configured on the selector.
        expected (list[int]): Expected per-step removal counts.

    Asserts:
        - Removal counts match the expected linspace(1, 0.2, steps)-derived schedule
        - Trailing zero-count steps are trimmed from the schedule
        - The schedule sums to exactly total_removable_features when non-empty
    """
    selector = _build_selector(steps=steps)
    schedule = selector.compute_removal_schedule(total_removable_features)
    if schedule != expected:
        raise AssertionError()
    if schedule and sum(schedule) != total_removable_features:
        raise AssertionError()
    if schedule and schedule[-1] == 0:
        raise AssertionError()


def test_compute_removal_schedule_zero_steps_returns_empty():
    """Tests that compute_removal_schedule returns an empty list when steps <= 0.

    Asserts:
        - Empty list is returned regardless of the number of removable features
    """
    selector = _build_selector(steps=0)
    if selector.compute_removal_schedule(10) != []:
        raise AssertionError()


def test_select_features_weighted_score_empty_history_returns_none():
    """Tests select_features_weighted_score on an empty history DataFrame.

    Asserts:
        - Selected features is an empty list
        - Best metric is None
    """
    selector = _build_selector()
    selected, best_metric = selector.select_features_weighted_score(pd.DataFrame())
    if selected != []:
        raise AssertionError()
    if best_metric is not None:
        raise AssertionError()


def test_select_features_weighted_score_edge():
    """Tests select_features_weighted_score on a history with improvement then a drop.

    Asserts:
        - Selected features is a list
        - Best metric is not None when history is non-empty
    """
    selector = _build_selector(steps=2, alpha=0.8)
    history = pd.DataFrame(
        [
            {
                "step": 0,
                "n_features_removed": 0,
                "n_features_remaining": 5,
                "removed_feature_name": None,
                "metric_value": 0.6,
                "metric_change": 0.0,
                "mi_score": np.nan,
            },
            {
                "step": 1,
                "n_features_removed": 1,
                "n_features_remaining": 4,
                "removed_feature_name": "f0",
                "metric_value": 0.62,
                "metric_change": 0.02,
                "mi_score": 0.1,
            },
            {
                "step": 2,
                "n_features_removed": 2,
                "n_features_remaining": 3,
                "removed_feature_name": "f1",
                "metric_value": 0.58,
                "metric_change": -0.04,
                "mi_score": 0.05,
            },
        ]
    )
    selected, best_metric = selector.select_features_weighted_score(history, alpha=0.8)
    if not isinstance(selected, list):
        raise AssertionError()
    if best_metric is None:
        raise AssertionError()


def test_select_features_weighted_score_prefers_best_metric_step():
    """Tests that the row maximizing the weighted metric/features score is selected.

    Asserts:
        - With alpha=1.0 the step with the maximum metric value alone determines
          which removed features are included in the selection
    """
    selector = _build_selector(steps=2, alpha=1.0)
    history = pd.DataFrame(
        [
            {
                "step": 1,
                "n_features_removed": 1,
                "n_features_remaining": 3,
                "removed_feature_name": "f0",
                "metric_value": 0.60,
                "metric_change": 0.0,
                "mi_score": 0.1,
            },
            {
                "step": 2,
                "n_features_removed": 2,
                "n_features_remaining": 2,
                "removed_feature_name": "f1",
                "metric_value": 0.75,
                "metric_change": 0.15,
                "mi_score": 0.05,
            },
        ]
    )
    selected, best_metric = selector.select_features_weighted_score(history, alpha=1.0)
    if selected != ["f1"]:
        raise AssertionError()
    if best_metric != pytest.approx(0.75):
        raise AssertionError()


def test_prepare_x_y_categorical_mi_rfe():
    """Tests that _prepare_x_y_for_mi encodes categorical columns and preserves y.

    Asserts:
        - discrete_features is populated in mi_kwargs when categorical columns exist
        - Categorical column is converted to a numeric dtype
        - y is returned unchanged
    """
    selector = _build_selector(steps=2)
    X = pd.DataFrame(
        {
            "a": [1, 2, 1, 2],
            "b": pd.Categorical(["x", "y", "x", "y"]),
        }
    )
    y = pd.Series([0, 1, 0, 1])
    prepared_x, prepared_y = selector._prepare_x_y_for_mi(X, y)
    if "discrete_features" not in selector.mi_kwargs:
        raise AssertionError()
    if not pd.api.types.is_numeric_dtype(prepared_x["b"]):
        raise AssertionError()
    if not prepared_y.equals(y):
        raise AssertionError()


def test_prepare_x_y_for_mi_rejects_invalid_y_type():
    """Tests that _prepare_x_y_for_mi raises ValueError for a non-Series/ndarray target.

    Asserts:
        - ValueError is raised when y is a plain Python list
    """
    selector = _build_selector(steps=2)
    X = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="y must be a pandas Series or numpy array"):
        selector._prepare_x_y_for_mi(X, [0, 1, 0])


def test_convert_categorical_to_codes():
    """Tests that _convert_categorical_to_codes converts category columns to codes.

    Asserts:
        - Categorical column becomes a category dtype containing integer-like codes
        - Non-categorical columns are left untouched
    """
    selector = _build_selector(steps=2)
    X = pd.DataFrame(
        {
            "num": [1, 2, 3],
            "cat": pd.Categorical(["x", "y", "x"]),
        }
    )
    converted = selector._convert_categorical_to_codes(X)
    if not isinstance(converted["cat"].dtype, pd.CategoricalDtype):
        raise AssertionError()
    if not converted["num"].equals(X["num"]):
        raise AssertionError()


def test_run_returns_selected_features(mi_rfe_data: tuple[pd.DataFrame, pd.Series]):
    """Tests MutualInformationRecursiveFeatureElimination run method returns expected output.

    Args:
        mi_rfe_data (tuple[pd.DataFrame, pd.Series]): Synthetic MI-RFE dataset fixture.

    Asserts:
        - Result contains the expected keys
        - Selected features are a subset of the original features
        - History is a non-empty DataFrame recorded from step 1 onward
    """
    X, y = mi_rfe_data
    selector = MutualInformationRecursiveFeatureElimination(
        estimator=LogisticRegression(max_iter=150, solver="liblinear"),
        cross_validator=_build_cv(),
        steps=3,
        alpha=0.9,
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
    if history.iloc[0]["step"] != 1:
        raise AssertionError()
    if not len(result["selected_features"]) <= X.shape[1]:
        raise AssertionError()
    if not set(result["selected_features"]).issubset(set(X.columns)):
        raise AssertionError()


def test_run_respects_protected_features(mi_rfe_data: tuple[pd.DataFrame, pd.Series]):
    """Tests that protected features are never removed and remain selected.

    Args:
        mi_rfe_data (tuple[pd.DataFrame, pd.Series]): Synthetic MI-RFE dataset fixture.

    Asserts:
        - Protected features are present in the final selected feature set
        - Protected feature names never appear as removed_feature_name in history
    """
    X, y = mi_rfe_data
    protected = ["f0", "f1"]
    selector = MutualInformationRecursiveFeatureElimination(
        estimator=LogisticRegression(max_iter=150, solver="liblinear"),
        cross_validator=_build_cv(),
        steps=3,
        alpha=0.9,
        verbose=False,
        protected_features=protected,
    )
    result = selector.run(X, y)
    if not set(protected).issubset(set(result["selected_features"])):
        raise AssertionError()
    if any(
        feat in protected
        for feat in result["history"]["removed_feature_name"].tolist()
    ):
        raise AssertionError()


def test_run_raises_for_unknown_protected_feature(
    mi_rfe_data: tuple[pd.DataFrame, pd.Series]
):
    """Tests that run() raises ValueError when protected_features references a missing column.

    Args:
        mi_rfe_data (tuple[pd.DataFrame, pd.Series]): Synthetic MI-RFE dataset fixture.

    Asserts:
        - ValueError is raised mentioning the missing protected feature
    """
    X, y = mi_rfe_data
    selector = MutualInformationRecursiveFeatureElimination(
        estimator=LogisticRegression(max_iter=100, solver="liblinear"),
        cross_validator=_build_cv(),
        steps=2,
        verbose=False,
        protected_features=["does_not_exist"],
    )
    with pytest.raises(ValueError, match="Protected features not found in dataset"):
        selector.run(X, y)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
