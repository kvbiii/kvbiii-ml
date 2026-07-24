import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

RANDOM_STATE = 17
N_SAMPLES = 16
N_SPLITS = 2

HISTORY_COLUMNS = [
    "step",
    "n_features_removed",
    "n_features_remaining",
    "removed_feature_name",
    "metric_value",
    "metric_change",
    "importance_score",
]


def build_small_classification_data(
    include_noise: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Builds a tiny synthetic binary classification dataset for RFE-style tests.

    Shared across the feature_selection test modules to avoid near-duplicate
    fixtures; f0..f3 are generated in the same rng order regardless of
    include_noise, so values match whether or not the noise column is added.

    Args:
        include_noise (bool): Whether to append an extra "noise" column that is
            not used by the target. Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and binary target vector.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    columns = {
        "f0": rng.normal(size=N_SAMPLES),
        "f1": rng.normal(size=N_SAMPLES),
        "f2": rng.normal(size=N_SAMPLES),
        "f3": rng.normal(size=N_SAMPLES),
    }
    if include_noise:
        columns["noise"] = rng.normal(size=N_SAMPLES)
    X = pd.DataFrame(columns)
    y = pd.Series(((X["f0"] + X["f1"]) > 0).astype(int), name="target")
    return X, y


def build_cross_validator(
    metric_name: str = "Accuracy",
    problem_type: str = "classification",
    pipeline: Pipeline | None = None,
    n_splits: int = N_SPLITS,
) -> CrossValidationTrainer:
    """Builds a fast CrossValidationTrainer shared across feature_selection RFE tests.

    Args:
        metric_name (str): Metric name to optimize. Defaults to "Accuracy".
        problem_type (str): Either "classification" or "regression". Defaults to
            "classification".
        pipeline (Pipeline | None): Optional preprocessing pipeline. Defaults to None.
        n_splits (int): Number of KFold splits. Defaults to N_SPLITS.

    Returns:
        CrossValidationTrainer: Configured trainer with a tiny KFold splitter.
    """
    return CrossValidationTrainer(
        problem_type=problem_type,
        metric_name=metric_name,
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE),
        preprocessing_pipeline=pipeline,
        verbose=False,
    )


def build_fast_estimator() -> RandomForestClassifier:
    """Builds a fast RandomForestClassifier estimator shared across feature_selection tests.

    Returns:
        RandomForestClassifier: Small, fast-fitting estimator with feature_importances_.
    """
    return RandomForestClassifier(
        n_estimators=5, max_depth=2, random_state=RANDOM_STATE
    )


def assert_valid_rfe_run_result_keys(result: dict) -> None:
    """Asserts an RFE-style run() result exposes exactly the documented keys.

    Args:
        result (dict): Return value of an RFE selector's run(X, y) call.

    Raises:
        AssertionError: If a documented key is missing or an extra key is present.
    """
    if set(result.keys()) != {
        "selected_features",
        "selected_features_names",
        "history",
    }:
        raise AssertionError("run() result is missing an expected key.")


def assert_selected_features_subset_of_columns(result: dict, X: pd.DataFrame) -> None:
    """Asserts selected_features is a non-larger subset of X's columns.

    Args:
        result (dict): Return value of an RFE selector's run(X, y) call.
        X (pd.DataFrame): The feature matrix passed into run().

    Raises:
        AssertionError: If selected_features exceeds X's column count or contains
            a name absent from X.columns.
    """
    if not len(result["selected_features"]) <= X.shape[1]:
        raise AssertionError("selected_features exceeds the original feature count.")
    if not set(result["selected_features"]).issubset(set(X.columns)):
        raise AssertionError("selected_features is not a subset of X.columns.")


def assert_empty_history_returns_no_selection(
    selected: list, best_metric: float | None
) -> None:
    """Asserts select_features_weighted_score returns ([], None) for an empty history.

    Args:
        selected (list): The selected-features list returned by the selector.
        best_metric (float | None): The best-step metric value returned by the selector.

    Raises:
        AssertionError: If the selection is non-empty or best_metric is not None.
    """
    if selected != []:
        raise AssertionError("Expected an empty selection for an empty history.")
    if best_metric is not None:
        raise AssertionError("Expected best_metric to be None for an empty history.")


def assert_rfe_run_summary_starts_with_full_feature_set(
    result: dict, X: pd.DataFrame
) -> None:
    """Asserts the run() summary starts at step 0 with the full feature set selected.

    Shared by the RFE variants (model importance, permutation) that report step 0
    as a baseline row covering every input feature, require a non-empty final
    selection, and alias selected_features_names to selected_features.

    Args:
        result (dict): Return value of an RFE selector's run(X, y) call.
        X (pd.DataFrame): The feature matrix passed into run().

    Raises:
        AssertionError: If any documented invariant is violated.
    """
    assert_valid_rfe_run_result_keys(result)
    history = result["history"]
    if int(history.iloc[0]["step"]) != 0:
        raise AssertionError("Expected the first history row to be step 0.")
    if int(history.iloc[0]["n_features_remaining"]) != X.shape[1]:
        raise AssertionError("Expected step 0 to cover the full feature set.")
    if not len(result["selected_features"]) > 0:
        raise AssertionError("Expected a non-empty final selection.")
    assert_selected_features_subset_of_columns(result, X)
    if result["selected_features_names"] != result["selected_features"]:
        raise AssertionError("selected_features_names should alias selected_features.")


def assert_rfe_run_result_has_nonempty_history(result: dict) -> pd.DataFrame:
    """Asserts a run() result exposes the documented keys and a non-empty history.

    Shared by the RFE variants (mutual information, SHAP) whose history-start
    step and remaining-feature checks differ enough per file to stay local.

    Args:
        result (dict): Return value of an RFE selector's run(X, y) call.

    Returns:
        pd.DataFrame: The result's history DataFrame, for further local assertions.

    Raises:
        AssertionError: If a documented key is missing or history is empty.
    """
    assert_valid_rfe_run_result_keys(result)
    history = result["history"]
    if history.empty:
        raise AssertionError("history should not be empty.")
    return history


def assert_selection_is_nonnull_list_with_metric(
    selected: list, best_metric: float | None
) -> None:
    """Asserts select_features_weighted_score returned a list selection with a real metric.

    Args:
        selected (list): The selected-features list returned by the selector.
        best_metric (float | None): The best-step metric value returned by the selector.

    Raises:
        AssertionError: If selected isn't a list, or best_metric is None.
    """
    if not isinstance(selected, list):
        raise AssertionError("selected should be a list.")
    if best_metric is None:
        raise AssertionError("best_metric should not be None for a non-empty history.")


if __name__ == "__main__":
    print("This module provides shared test support, not tests to run directly.")
