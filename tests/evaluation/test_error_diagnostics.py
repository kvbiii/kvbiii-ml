"""Tests for kvbiii_ml.evaluation.error_diagnostics module."""

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.evaluation.error_diagnostics import (
    display_classification_errors,
    display_regression_under_over_errors,
    get_top_classification_errors,
    get_top_regression_under_over_errors,
)


def test_get_top_regression_under_over_errors_raises_typeerror_for_non_series_y_true():
    """Tests get_top_regression_under_over_errors rejects a non-Series y_true.

    Asserts:
        - TypeError is raised when y_true is a plain numpy array.
    """
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.2, 2.8])

    with pytest.raises(TypeError, match="y_true must be a pandas Series"):
        get_top_regression_under_over_errors(y_true, y_pred)


def test_get_top_regression_under_over_errors_protects_against_zero_true_values():
    """Tests the epsilon guard prevents division by zero when y_true contains zeros.

    Asserts:
        - No infinite or NaN percentage errors are produced for the zero-valued row.
        - The zero-valued row is classified as overestimated since its prediction is positive.
    """
    y_true = pd.Series([10.0, 20.0, 0.0, 40.0, 50.0])
    y_pred = pd.Series([12.0, 15.0, 5.0, 35.0, 55.0])

    _underestimated, overestimated = get_top_regression_under_over_errors(
        y_true, y_pred, epsilon=1e-9
    )

    if 2 not in overestimated.index:
        raise AssertionError()
    if not np.isfinite(overestimated.loc[2, "Percentage Error (%)"]):
        raise AssertionError()


def test_get_top_regression_under_over_errors_truncates_to_top_n():
    """Tests top_n truncation for the direction with more errors than requested.

    Asserts:
        - underestimated_df is truncated to top_n rows, largest magnitude first.
        - overestimated_df returns fewer than top_n rows without padding when
          fewer errors exist in that direction.
    """
    y_true = pd.Series([10.0] * 5, index=range(5))
    y_pred = pd.Series([8.0, 7.0, 6.0, 12.0, 10.0], index=range(5))

    underestimated, overestimated = get_top_regression_under_over_errors(
        y_true, y_pred, top_n=2
    )

    if len(underestimated) != 2:
        raise AssertionError()
    if list(underestimated.index) != [2, 1]:
        raise AssertionError()
    if len(overestimated) != 1:
        raise AssertionError()
    if list(overestimated.index) != [3]:
        raise AssertionError()


def test_get_top_regression_under_over_errors_adds_exponentiated_columns_when_log():
    """Tests log=True adds exponentiated true/predicted/error columns.

    Asserts:
        - True Value (exp), Predicted Value (exp), Error (exp), and
          Percentage Error (exp) (%) columns are present in both outputs.
    """
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([0.8, 2.5, 2.7])

    underestimated, overestimated = get_top_regression_under_over_errors(
        y_true, y_pred, log=True
    )

    expected_cols = {
        "True Value (exp)",
        "Predicted Value (exp)",
        "Error (exp)",
        "Percentage Error (exp) (%)",
    }
    if not expected_cols <= set(underestimated.columns):
        raise AssertionError()
    if not expected_cols <= set(overestimated.columns):
        raise AssertionError()


def test_display_regression_under_over_errors_runs_without_raising():
    """Tests display_regression_under_over_errors renders without raising.

    Asserts:
        - The function completes without raising for valid DataFrame inputs.
    """
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = pd.Series([8.0, 25.0, 27.0])
    underestimated, overestimated = get_top_regression_under_over_errors(y_true, y_pred)

    display_regression_under_over_errors(underestimated, overestimated)


def test_get_top_classification_errors_raises_typeerror_for_non_series_y_true():
    """Tests get_top_classification_errors rejects a non-Series y_true.

    Asserts:
        - TypeError is raised when y_true is a plain numpy array.
    """
    y_true = np.array([0, 1, 0])
    y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])

    with pytest.raises(TypeError, match="y_true must be a pandas Series"):
        get_top_classification_errors(
            y_true, y_pred_proba, class_id=0, id2label={0: "neg", 1: "pos"}
        )


def test_get_top_classification_errors_raises_valueerror_for_shape_mismatch():
    """Tests get_top_classification_errors rejects mismatched sample counts.

    Asserts:
        - ValueError is raised when y_pred_proba.shape[0] != len(y_true).
    """
    y_true = pd.Series([0, 1, 0])
    y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8]])

    with pytest.raises(ValueError, match="same number of samples"):
        get_top_classification_errors(
            y_true, y_pred_proba, class_id=0, id2label={0: "neg", 1: "pos"}
        )


def test_get_top_classification_errors_raises_valueerror_for_out_of_bounds_class_id():
    """Tests get_top_classification_errors rejects an out-of-range class_id.

    Asserts:
        - ValueError is raised when class_id is outside [0, n_classes).
    """
    y_true = pd.Series([0, 1, 0])
    y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])

    with pytest.raises(ValueError, match="out of bounds"):
        get_top_classification_errors(
            y_true, y_pred_proba, class_id=5, id2label={0: "neg", 1: "pos"}
        )


def test_get_top_classification_errors_raises_valueerror_for_cutoff_length_mismatch():
    """Tests get_top_classification_errors rejects a multiclass cutoff of the wrong length.

    Asserts:
        - ValueError is raised when the cutoff list length does not match n_classes.
    """
    y_true = pd.Series([0, 1, 2])
    y_pred_proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    id2label = {0: "a", 1: "b", 2: "c"}

    with pytest.raises(ValueError, match="Number of cutoffs"):
        get_top_classification_errors(
            y_true, y_pred_proba, class_id=0, id2label=id2label, cutoff=[0.5, 0.5]
        )


def test_get_top_classification_errors_stacks_1d_binary_probabilities():
    """Tests a 1-D binary y_pred_proba array is stacked into a 2-D matrix internally.

    Asserts:
        - The function runs without raising for a 1-D probability array.
        - Only the false-negative row for the target class is returned.
    """
    y_true = pd.Series([0, 1, 0, 1, 0], index=[10, 11, 12, 13, 14])
    y_pred_proba_1d = np.array([0.2, 0.9, 0.7, 0.3, 0.4])
    id2label = {0: "neg", 1: "pos"}

    result = get_top_classification_errors(
        y_true, y_pred_proba_1d, class_id=1, id2label=id2label
    )

    if len(result) != 1:
        raise AssertionError()
    if list(result.index) != [13]:
        raise AssertionError()


def test_get_top_classification_errors_only_considers_false_negatives():
    """Tests only false negatives for the target class are returned, not false positives.

    A false positive for class 1 exists at index 12 (true=0, predicted=1) but must
    not appear in the class_id=1 result, which should contain only the false
    negative at index 13 (true=1, predicted=0).

    Asserts:
        - The false-positive row is excluded from the class_id=1 result.
        - The false-negative row is included in the class_id=1 result.
    """
    y_true = pd.Series([0, 1, 0, 1, 0], index=[10, 11, 12, 13, 14])
    y_pred_proba = np.array(
        [
            [0.8, 0.2],
            [0.1, 0.9],
            [0.3, 0.7],
            [0.7, 0.3],
            [0.6, 0.4],
        ]
    )
    id2label = {0: "neg", 1: "pos"}

    result = get_top_classification_errors(
        y_true, y_pred_proba, class_id=1, id2label=id2label
    )

    if 12 in result.index:
        raise AssertionError()
    if 13 not in result.index:
        raise AssertionError()


def test_get_top_classification_errors_returns_empty_dataframe_for_zero_false_negatives():
    """Tests an empty DataFrame is returned when a class has zero false negatives.

    Asserts:
        - The result is an empty DataFrame when every instance of class 0 is
          predicted correctly.
    """
    y_true = pd.Series([0, 1])
    y_pred_proba = np.array([[0.9, 0.1], [0.1, 0.9]])
    id2label = {0: "neg", 1: "pos"}

    result = get_top_classification_errors(
        y_true, y_pred_proba, class_id=0, id2label=id2label
    )

    if not result.empty:
        raise AssertionError()


def test_get_top_classification_errors_truncates_to_top_n_by_error_magnitude():
    """Tests top_n truncation sorted by descending error magnitude for false negatives.

    Asserts:
        - Result length is capped at top_n.
        - Rows are ordered by descending (predicted_confidence - true_class_probability).
    """
    y_true = pd.Series([1, 1, 1, 1, 0], index=[0, 1, 2, 3, 4])
    y_pred_proba = np.array(
        [
            [0.9, 0.1],
            [0.6, 0.4],
            [0.55, 0.45],
            [0.51, 0.49],
            [0.9, 0.1],
        ]
    )
    id2label = {0: "neg", 1: "pos"}

    result = get_top_classification_errors(
        y_true, y_pred_proba, class_id=1, id2label=id2label, top_n=2
    )

    if len(result) != 2:
        raise AssertionError()
    if list(result.index) != [0, 1]:
        raise AssertionError()


def test_display_classification_errors_runs_without_raising():
    """Tests display_classification_errors renders without raising, including empty results.

    Asserts:
        - The function completes without raising for a mix of non-empty and
          empty per-class error DataFrames.
    """
    y_true = pd.Series([0, 1, 0, 1, 0], index=[10, 11, 12, 13, 14])
    y_pred_proba = np.array(
        [
            [0.8, 0.2],
            [0.1, 0.9],
            [0.3, 0.7],
            [0.7, 0.3],
            [0.6, 0.4],
        ]
    )
    id2label = {0: "neg", 1: "pos"}
    errors_by_class = {
        class_id: get_top_classification_errors(
            y_true, y_pred_proba, class_id=class_id, id2label=id2label
        )
        for class_id in id2label
    }

    display_classification_errors(errors_by_class, id2label)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
