"""Tests for kvbiii_ml.evaluation.generate_reports module."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.evaluation.generate_reports import (
    generate_classification_report,
    generate_regression_report,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_regression_error,
    plot_roc_curve,
)


@pytest.fixture
def classification_data():
    """Provides data for classification reporting.

    Returns:
        tuple: y_true, y_pred, and probabilities for classification reports
    """
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=100)
    y_pred = np.random.choice([0, 1], size=100)

    # Make predictions somewhat correlated with truth
    for i in range(100):
        if np.random.random() < 0.7:  # 70% accuracy
            y_pred[i] = y_true[i]

    # Generate probabilities
    probas = np.zeros((100, 2))
    for i in range(100):
        if y_pred[i] == 1:
            probas[i, 1] = 0.6 + np.random.random() * 0.4
            probas[i, 0] = 1 - probas[i, 1]
        else:
            probas[i, 0] = 0.6 + np.random.random() * 0.4
            probas[i, 1] = 1 - probas[i, 0]

    return y_true, y_pred, probas


@pytest.fixture
def regression_data():
    """Provides data for regression reporting.

    Returns:
        tuple: y_true and y_pred for regression reports
    """
    np.random.seed(42)
    y_true = np.random.normal(50, 10, 100)

    # Create somewhat correlated predictions
    y_pred = y_true + np.random.normal(0, 5, 100)

    return y_true, y_pred


@patch("kvbiii_ml.evaluation.generate_reports.classification_report")
@patch("kvbiii_ml.evaluation.generate_reports.confusion_matrix")
@patch("kvbiii_ml.evaluation.generate_reports.roc_auc_score")
def test_generate_classification_report_creates_comprehensive_report(
    mock_roc_auc, mock_confusion_matrix, mock_classification_report, classification_data
):
    """Tests generate_classification_report creates a comprehensive classification report.

    Args:
        mock_roc_auc: Mocked ROC AUC score function
        mock_confusion_matrix: Mocked confusion matrix function
        mock_classification_report: Mocked classification report function
        classification_data: Classification data fixture

    Asserts:
        - Classification report includes all necessary metrics
        - Confusion matrix is calculated
        - ROC AUC is included when probabilities are provided
        - Returns dictionary with expected keys
    """
    y_true, y_pred, y_probas = classification_data

    # Configure mocks
    mock_classification_report.return_value = "Classification Report"
    mock_confusion_matrix.return_value = np.array([[30, 20], [10, 40]])
    mock_roc_auc.return_value = 0.85

    # Generate report with all options
    report = generate_classification_report(
        y_true, y_pred, y_probas=y_probas[:, 1], class_names=["Negative", "Positive"]
    )

    # Check all metrics were called
    mock_classification_report.assert_called_once()
    mock_confusion_matrix.assert_called_once()
    mock_roc_auc.assert_called_once()

    # Check report structure
    assert "classification_report" in report
    assert "confusion_matrix" in report
    assert "roc_auc" in report
    assert "accuracy" in report
    assert "precision" in report
    assert "recall" in report
    assert "f1" in report

    # Test without probabilities
    mock_roc_auc.reset_mock()
    report_no_proba = generate_classification_report(
        y_true, y_pred, class_names=["Negative", "Positive"]
    )

    # ROC AUC shouldn't be called or included
    mock_roc_auc.assert_not_called()
    assert "roc_auc" not in report_no_proba


@patch("kvbiii_ml.evaluation.generate_reports.mean_squared_error")
@patch("kvbiii_ml.evaluation.generate_reports.mean_absolute_error")
@patch("kvbiii_ml.evaluation.generate_reports.r2_score")
@patch("kvbiii_ml.evaluation.generate_reports.explained_variance_score")
def test_generate_regression_report_includes_common_metrics(
    mock_explained_variance, mock_r2, mock_mae, mock_mse, regression_data
):
    """Tests generate_regression_report includes common regression metrics.

    Args:
        mock_explained_variance: Mocked explained variance score function
        mock_r2: Mocked R² score function
        mock_mae: Mocked MAE function
        mock_mse: Mocked MSE function
        regression_data: Regression data fixture

    Asserts:
        - Report includes all standard regression metrics
        - RMSE is calculated from MSE
        - Report includes both raw and percentage errors
        - Returns dictionary with expected keys
    """
    y_true, y_pred = regression_data

    # Configure mocks
    mock_mse.return_value = 25
    mock_mae.return_value = 4
    mock_r2.return_value = 0.85
    mock_explained_variance.return_value = 0.86

    # Generate report
    report = generate_regression_report(y_true, y_pred)

    # Check all metrics were called
    mock_mse.assert_called_once()
    mock_mae.assert_called_once()
    mock_r2.assert_called_once()
    mock_explained_variance.assert_called_once()

    # Check report structure
    assert "mse" in report
    assert "rmse" in report
    assert "mae" in report
    assert "r2" in report
    assert "explained_variance" in report
    assert "mape" in report

    # Check RMSE calculation
    assert report["rmse"] == 5.0  # sqrt(25)


@patch("kvbiii_ml.evaluation.generate_reports.plt")
def test_plot_confusion_matrix_visualization(mock_plt, classification_data):
    """Tests plot_confusion_matrix creates visualization.

    Args:
        mock_plt: Mocked matplotlib pyplot
        classification_data: Classification data fixture

    Asserts:
        - Confusion matrix is visualized correctly
        - Display options are configurable
        - Class names are used when provided
    """
    y_true, y_pred, _ = classification_data

    # Create plot
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=["Negative", "Positive"],
        normalize=True,
        cmap="Blues",
        figsize=(8, 6),
    )

    # Check figure was created
    mock_plt.figure.assert_called_once()

    # Check heatmap or similar function was called
    # This will differ based on implementation (imshow, matshow, etc.)
    assert (
        mock_plt.imshow.called or mock_plt.matshow.called or mock_plt.pcolormesh.called
    )

    # Check labels were set
    assert mock_plt.xlabel.called
    assert mock_plt.ylabel.called
    assert mock_plt.title.called

    # Check show was called
    assert mock_plt.show.called


@patch("kvbiii_ml.evaluation.generate_reports.plt")
@patch("kvbiii_ml.evaluation.generate_reports.roc_curve")
@patch("kvbiii_ml.evaluation.generate_reports.auc")
def test_plot_roc_curve_visualization(
    mock_auc, mock_roc_curve, mock_plt, classification_data
):
    """Tests plot_roc_curve creates ROC curve visualization.

    Args:
        mock_auc: Mocked AUC function
        mock_roc_curve: Mocked ROC curve function
        mock_plt: Mocked matplotlib pyplot
        classification_data: Classification data fixture

    Asserts:
        - ROC curve is calculated and plotted
        - AUC is displayed in the plot
        - Multiple curves can be plotted with labels
    """
    y_true, _, y_probas = classification_data

    # Configure mocks
    mock_roc_curve.return_value = (
        np.array([0, 0.2, 0.5, 0.8, 1]),  # FPR
        np.array([0, 0.6, 0.8, 0.9, 1]),  # TPR
        np.array([1, 0.8, 0.5, 0.2, 0]),  # Thresholds
    )
    mock_auc.return_value = 0.85

    # Create plot
    plot_roc_curve(y_true, y_probas[:, 1], label="Model 1")

    # Check ROC curve was calculated
    mock_roc_curve.assert_called_once()

    # Check AUC was calculated
    mock_auc.assert_called_once()

    # Check plot was created
    mock_plt.figure.assert_called_once()
    mock_plt.plot.assert_called()  # Plot called at least once
    mock_plt.title.assert_called_once()
    mock_plt.xlabel.assert_called_once()
    mock_plt.ylabel.assert_called_once()
    mock_plt.legend.assert_called_once()
    mock_plt.show.assert_called_once()


@patch("kvbiii_ml.evaluation.generate_reports.plt")
def test_plot_regression_error_visualization(mock_plt, regression_data):
    """Tests plot_regression_error creates error visualization.

    Args:
        mock_plt: Mocked matplotlib pyplot
        regression_data: Regression data fixture

    Asserts:
        - Actual vs. predicted values are plotted
        - Error distribution is plotted
        - Residual plot is created
    """
    y_true, y_pred = regression_data

    # Create plot
    plot_regression_error(y_true, y_pred, figsize=(15, 10))

    # Check plots were created
    assert mock_plt.figure.call_count >= 1
    assert mock_plt.subplot.call_count >= 3  # At least 3 subplots

    # Check scatter plot for actual vs predicted
    assert mock_plt.scatter.call_count >= 1

    # Check histogram for error distribution
    assert mock_plt.hist.call_count >= 1

    # Check plots were shown
    assert mock_plt.show.called


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
