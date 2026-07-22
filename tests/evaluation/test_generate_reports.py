"""Tests for kvbiii_ml.evaluation.generate_reports module."""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from kvbiii_ml.evaluation.generate_reports import (
    classification_results,
    regression_results,
)


def test_regression_results_computes_metrics_matching_sklearn_equivalents():
    """Tests regression_results computes MAE/MSE/RMSE/R2/Explained Var correctly.

    Asserts:
        - Each metric column matches the equivalent sklearn function computed
          directly on the same train/test arrays.
    """
    y_train_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_train_pred = np.array([1.1, 2.2, 2.8, 3.9, 5.2])
    y_test_true = np.array([10.0, 20.0, 30.0])
    y_test_pred = np.array([12.0, 18.0, 33.0])

    styler = regression_results(y_train_true, y_train_pred, y_test_true, y_test_pred)
    df = styler.data

    if df.loc["Train", "MAE"] != pytest.approx(
        mean_absolute_error(y_train_true, y_train_pred)
    ):
        raise AssertionError()
    if df.loc["Test", "MAE"] != pytest.approx(
        mean_absolute_error(y_test_true, y_test_pred)
    ):
        raise AssertionError()
    if df.loc["Train", "MSE"] != pytest.approx(
        mean_squared_error(y_train_true, y_train_pred)
    ):
        raise AssertionError()
    if df.loc["Test", "RMSE"] != pytest.approx(
        root_mean_squared_error(y_test_true, y_test_pred)
    ):
        raise AssertionError()
    if df.loc["Train", "R²"] != pytest.approx(r2_score(y_train_true, y_train_pred)):
        raise AssertionError()
    if df.loc["Test", "Explained Var"] != pytest.approx(
        explained_variance_score(y_test_true, y_test_pred)
    ):
        raise AssertionError()


def test_regression_results_mape_is_nan_for_all_zero_target():
    """Tests the internal _safe_mape helper returns NaN when a target is all zeros.

    Asserts:
        - MAPE is NaN for both train and test rows when the true values are
          entirely zero, avoiding a division-by-zero crash.
    """
    y_zero = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    styler = regression_results(y_zero, y_pred, y_zero, y_pred)
    df = styler.data

    if not np.isnan(df.loc["Train", "MAPE"]):
        raise AssertionError()
    if not np.isnan(df.loc["Test", "MAPE"]):
        raise AssertionError()


@pytest.fixture
def binary_classification_report_data():
    """Provides binary classification data with label-space predictions and probabilities.

    Returns:
        dict: Train/test true labels, predictions, probabilities, and id2label mapping,
            all consistent with the same label space required by classification_results.
    """
    return {
        "y_train_true": np.array(["neg", "neg", "pos", "pos", "neg", "pos"]),
        "y_train_pred": np.array(["neg", "pos", "pos", "pos", "neg", "neg"]),
        "y_train_proba": np.array(
            [[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4], [0.55, 0.45]]
        ),
        "y_test_true": np.array(["neg", "pos", "neg", "pos"]),
        "y_test_pred": np.array(["neg", "pos", "pos", "neg"]),
        "y_test_proba": np.array([[0.7, 0.3], [0.3, 0.7], [0.4, 0.6], [0.6, 0.4]]),
        "id2label": {0: "neg", 1: "pos"},
    }


def test_classification_results_returns_two_stylers_without_cutoffs(
    binary_classification_report_data,
):
    """Tests classification_results returns exactly 2 Stylers when cutoffs is None.

    Args:
        binary_classification_report_data (dict): Consistent binary classification fixture data.

    Asserts:
        - The return value has exactly 2 elements.
        - Both elements expose a .data DataFrame.
    """
    data = binary_classification_report_data
    result = classification_results(
        data["y_train_true"],
        data["y_train_pred"],
        data["y_test_true"],
        data["y_test_pred"],
    )

    if len(result) != 2:
        raise AssertionError()
    if not all(hasattr(styler, "data") for styler in result):
        raise AssertionError()


def test_classification_results_returns_four_stylers_with_cutoffs(
    binary_classification_report_data,
):
    """Tests classification_results returns exactly 4 Stylers when cutoffs is given.

    Args:
        binary_classification_report_data (dict): Consistent binary classification fixture data.

    Asserts:
        - The return value has exactly 4 elements when cutoffs, probabilities, and
          id2label are all supplied.
    """
    data = binary_classification_report_data
    result = classification_results(
        data["y_train_true"],
        data["y_train_pred"],
        data["y_test_true"],
        data["y_test_pred"],
        y_train_proba=data["y_train_proba"],
        y_test_proba=data["y_test_proba"],
        id2label=data["id2label"],
        cutoffs=0.6,
    )

    if len(result) != 4:
        raise AssertionError()


def test_classification_results_raises_valueerror_when_cutoffs_missing_dependencies(
    binary_classification_report_data,
):
    """Tests classification_results raises ValueError when cutoffs lacks required data.

    Args:
        binary_classification_report_data (dict): Consistent binary classification fixture data.

    Asserts:
        - ValueError is raised when cutoffs is provided without probabilities and id2label.
    """
    data = binary_classification_report_data

    with pytest.raises(ValueError, match="Probabilities and class labels are required"):
        classification_results(
            data["y_train_true"],
            data["y_train_pred"],
            data["y_test_true"],
            data["y_test_pred"],
            cutoffs=0.5,
        )


def test_classification_results_overall_accuracy_matches_sklearn(
    binary_classification_report_data,
):
    """Tests the overall metrics table's Accuracy column matches sklearn's accuracy_score.

    Args:
        binary_classification_report_data (dict): Consistent binary classification fixture data.

    Asserts:
        - Train and Test Accuracy values match accuracy_score computed directly.
    """
    data = binary_classification_report_data
    overall_default, _per_class_default = classification_results(
        data["y_train_true"],
        data["y_train_pred"],
        data["y_test_true"],
        data["y_test_pred"],
    )
    df = overall_default.data

    if df.loc["Train", "Accuracy"] != pytest.approx(
        accuracy_score(data["y_train_true"], data["y_train_pred"])
    ):
        raise AssertionError()
    if df.loc["Test", "Accuracy"] != pytest.approx(
        accuracy_score(data["y_test_true"], data["y_test_pred"])
    ):
        raise AssertionError()


def test_classification_results_degrades_roc_auc_and_log_loss_to_nan_for_single_class_split():
    """Tests ROC-AUC and Log Loss degrade to NaN for an ill-posed single-class test split.

    Asserts:
        - Test-row ROC-AUC and Log Loss are NaN when y_test_true contains a single class.
        - Train-row ROC-AUC and Log Loss remain finite for a well-posed split.
    """
    y_train_true = np.array(["neg", "neg", "pos", "pos", "neg", "pos"])
    y_train_pred = np.array(["neg", "pos", "pos", "pos", "neg", "neg"])
    y_train_proba = np.array(
        [[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4], [0.55, 0.45]]
    )
    y_test_true = np.array(["neg", "neg", "neg"])
    y_test_pred = np.array(["neg", "pos", "neg"])
    y_test_proba = np.array([[0.7, 0.3], [0.3, 0.7], [0.6, 0.4]])
    id2label = {0: "neg", 1: "pos"}

    overall_default, _per_class_default = classification_results(
        y_train_true,
        y_train_pred,
        y_test_true,
        y_test_pred,
        y_train_proba=y_train_proba,
        y_test_proba=y_test_proba,
        id2label=id2label,
    )
    df = overall_default.data

    if not np.isnan(df.loc["Test", "ROC-AUC"]):
        raise AssertionError()
    if not np.isnan(df.loc["Test", "Log Loss"]):
        raise AssertionError()
    if not np.isfinite(df.loc["Train", "ROC-AUC"]):
        raise AssertionError()
    if not np.isfinite(df.loc["Train", "Log Loss"]):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
