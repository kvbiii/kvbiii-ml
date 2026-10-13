"""Tests for kvbiii_ml.modeling.optimization.classification_calibration module."""

import os
from unittest.mock import Mock, patch

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import pytest
from pandas.io.formats.style import Styler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from kvbiii_ml.modeling.optimization.classification_calibration import (
    ClassificationCalibrator,
    _show_figure,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

_VALID_SELECTION_METRIC_NAMES = [None, "ECE", "Brier Score", "Accuracy"]


@pytest.fixture
def binary_calibration_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a binary classification dataset for calibration testing.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and binary target vector.
    """
    x_arr, y_arr = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=17,
    )
    x_df = pd.DataFrame(x_arr, columns=[f"feature_{i}" for i in range(8)])
    return x_df, pd.Series(y_arr, name="target")


@pytest.fixture
def multiclass_calibration_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a multiclass classification dataset for calibration testing.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and multiclass target vector.
    """
    x_arr, y_arr = make_classification(
        n_samples=360,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=3,
        random_state=17,
    )
    x_df = pd.DataFrame(x_arr, columns=[f"feature_{i}" for i in range(10)])
    return x_df, pd.Series(y_arr, name="target")


@pytest.fixture
def multiclass_logistic_regression_estimator(test_settings) -> LogisticRegression:
    """Provides a LogisticRegression estimator compatible with multiclass targets.

    The shared ``logistic_regression_estimator`` fixture uses the liblinear solver,
    which does not support more than two classes, so multiclass calibration tests
    use this lbfgs-solver variant instead.

    Args:
        test_settings: Test configuration fixture.

    Returns:
        LogisticRegression: Configured multiclass-capable estimator.
    """
    return LogisticRegression(
        max_iter=200, random_state=test_settings.SEED, solver="lbfgs"
    )


@pytest.fixture
def calibration_cv_trainer(test_settings) -> CrossValidationTrainer:
    """Provides a CrossValidationTrainer configured for calibration testing.

    Args:
        test_settings: Test configuration fixture.

    Returns:
        CrossValidationTrainer: Configured trainer with a stratified 3-fold splitter.
    """
    return CrossValidationTrainer(
        metric_name="Log Loss",
        problem_type="classification",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=test_settings.SEED),
        verbose=False,
    )


@pytest.mark.parametrize("selection_metric_name", _VALID_SELECTION_METRIC_NAMES)
def test_classificationcalibrator_init_accepts_valid_selection_metric_names(
    selection_metric_name, logistic_regression_estimator, calibration_cv_trainer
):
    """Tests initialization accepts None, built-in, and METRICS-key selection names.

    Args:
        selection_metric_name (str | None): Selection metric under test.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        calibration_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - No exception is raised for any recognised selection metric name
        - The stored selection_metric_name matches the constructor argument
    """
    calibrator = ClassificationCalibrator(
        estimator=logistic_regression_estimator,
        cross_validator=calibration_cv_trainer,
        selection_metric_name=selection_metric_name,
    )

    if calibrator.selection_metric_name != selection_metric_name:
        raise AssertionError()


def test_classificationcalibrator_init_raises_error_for_invalid_selection_metric_name(
    logistic_regression_estimator, calibration_cv_trainer
):
    """Tests initialization rejects unrecognised selection metric names.

    Args:
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        calibration_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - ValueError is raised for a name that is neither None, a built-in, nor a METRICS key
    """
    with pytest.raises(ValueError, match="is not recognised"):
        ClassificationCalibrator(
            estimator=logistic_regression_estimator,
            cross_validator=calibration_cv_trainer,
            selection_metric_name="NotARealMetric",
        )


def test_classificationcalibrator_fit_binary_produces_scores_and_best_method(
    binary_calibration_data, logistic_regression_estimator, calibration_cv_trainer
):
    """Tests fit on binary data populates calibration scores and best method.

    Args:
        binary_calibration_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        calibration_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - fit() returns self for chaining
        - calibration_scores_df_ is populated with an Uncalibrated row
        - best_method_ is one of the evaluated calibration strategies
    """
    X, y = binary_calibration_data
    calibrator = ClassificationCalibrator(
        estimator=logistic_regression_estimator, cross_validator=calibration_cv_trainer
    )

    fitted = calibrator.fit(X, y)

    if fitted is not calibrator:
        raise AssertionError()
    if calibrator.calibration_scores_df_.empty:
        raise AssertionError()
    if "Uncalibrated" not in calibrator.calibration_scores_df_.index:
        raise AssertionError()
    if calibrator.best_method_ not in {"Uncalibrated", "Isotonic", "Sigmoid"}:
        raise AssertionError()


def test_classificationcalibrator_fit_multiclass_produces_scores_and_best_method(
    multiclass_calibration_data,
    multiclass_logistic_regression_estimator,
    calibration_cv_trainer,
):
    """Tests fit on multiclass data populates calibration scores and best method.

    Args:
        multiclass_calibration_data (tuple): Feature matrix and multiclass target vector.
        multiclass_logistic_regression_estimator (LogisticRegression): Configured
            multiclass-capable estimator.
        calibration_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - fit() returns self for chaining
        - calibration_scores_df_ is populated with an Uncalibrated row
        - best_method_ is one of the evaluated calibration strategies
    """
    X, y = multiclass_calibration_data
    calibrator = ClassificationCalibrator(
        estimator=multiclass_logistic_regression_estimator,
        cross_validator=calibration_cv_trainer,
    )

    fitted = calibrator.fit(X, y)

    if fitted is not calibrator:
        raise AssertionError()
    if calibrator.calibration_scores_df_.empty:
        raise AssertionError()
    if "Uncalibrated" not in calibrator.calibration_scores_df_.index:
        raise AssertionError()
    if calibrator.best_method_ not in {"Uncalibrated", "Isotonic", "Sigmoid"}:
        raise AssertionError()


def test_classificationcalibrator_predict_proba_raises_runtimeerror_before_fit(
    logistic_regression_estimator, calibration_cv_trainer
):
    """Tests predict_proba raises before fit() has been called.

    Args:
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        calibration_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - RuntimeError is raised instructing the caller to fit first
    """
    calibrator = ClassificationCalibrator(
        estimator=logistic_regression_estimator, cross_validator=calibration_cv_trainer
    )

    with pytest.raises(RuntimeError, match=r"Call fit\(\) before predict_proba\(\)\."):
        calibrator.predict_proba(pd.DataFrame({"feature_0": [0.1, 0.2]}))


def test_classificationcalibrator_predict_raises_runtimeerror_before_fit(
    logistic_regression_estimator, calibration_cv_trainer
):
    """Tests predict raises before fit() has been called.

    Args:
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        calibration_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - RuntimeError is raised instructing the caller to fit first
    """
    calibrator = ClassificationCalibrator(
        estimator=logistic_regression_estimator, cross_validator=calibration_cv_trainer
    )

    with pytest.raises(RuntimeError, match=r"Call fit\(\) before predict_proba\(\)\."):
        calibrator.predict(pd.DataFrame({"feature_0": [0.1, 0.2]}))


def test_classificationcalibrator_plot_calibration_curves_raises_runtimeerror_before_fit(
    logistic_regression_estimator, calibration_cv_trainer
):
    """Tests plot_calibration_curves raises before fit() has been called.

    Args:
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        calibration_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - RuntimeError is raised instructing the caller to fit first
    """
    calibrator = ClassificationCalibrator(
        estimator=logistic_regression_estimator, cross_validator=calibration_cv_trainer
    )

    with pytest.raises(
        RuntimeError, match=r"Call fit\(\) before plot_calibration_curves\(\)\."
    ):
        calibrator.plot_calibration_curves()


def test_classificationcalibrator_predict_proba_and_predict_work_after_fit_binary(
    binary_calibration_data, logistic_regression_estimator, calibration_cv_trainer
):
    """Tests predict_proba and predict succeed after fit() on binary data.

    Args:
        binary_calibration_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        calibration_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - predict_proba returns a (n_samples, 2) matrix with rows summing to 1
        - predict returns labels drawn from the observed class set
    """
    X, y = binary_calibration_data
    calibrator = ClassificationCalibrator(
        estimator=logistic_regression_estimator, cross_validator=calibration_cv_trainer
    )
    calibrator.fit(X, y)

    proba = calibrator.predict_proba(X.iloc[:20])
    preds = calibrator.predict(X.iloc[:20])

    if proba.shape != (20, 2):
        raise AssertionError()
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError()
    if not (np.all(proba >= 0.0) and np.all(proba <= 1.0)):
        raise AssertionError()
    if preds.shape != (20,):
        raise AssertionError()
    if not set(np.unique(preds)).issubset(set(np.unique(y))):
        raise AssertionError()


def test_classificationcalibrator_predict_proba_and_predict_work_after_fit_multiclass(
    multiclass_calibration_data,
    multiclass_logistic_regression_estimator,
    calibration_cv_trainer,
):
    """Tests predict_proba and predict succeed after fit() on multiclass data.

    Args:
        multiclass_calibration_data (tuple): Feature matrix and multiclass target vector.
        multiclass_logistic_regression_estimator (LogisticRegression): Configured
            multiclass-capable estimator.
        calibration_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - predict_proba returns a (n_samples, 3) matrix with rows summing to 1
        - predict returns labels drawn from the observed class set
    """
    X, y = multiclass_calibration_data
    calibrator = ClassificationCalibrator(
        estimator=multiclass_logistic_regression_estimator,
        cross_validator=calibration_cv_trainer,
    )
    calibrator.fit(X, y)

    proba = calibrator.predict_proba(X.iloc[:20])
    preds = calibrator.predict(X.iloc[:20])

    if proba.shape != (20, 3):
        raise AssertionError()
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError()
    if not (np.all(proba >= 0.0) and np.all(proba <= 1.0)):
        raise AssertionError()
    if preds.shape != (20,):
        raise AssertionError()
    if not set(np.unique(preds)).issubset(set(np.unique(y))):
        raise AssertionError()


def test_classificationcalibrator_plot_calibration_curves_returns_none(
    binary_calibration_data, logistic_regression_estimator, calibration_cv_trainer
):
    """Tests plot_calibration_curves returns None, only side-effecting a display call.

    Args:
        binary_calibration_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        calibration_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - plot_calibration_curves() returns None.
        - The module's _show_figure helper is invoked exactly once.
    """
    X, y = binary_calibration_data
    calibrator = ClassificationCalibrator(
        estimator=logistic_regression_estimator, cross_validator=calibration_cv_trainer
    )
    calibrator.fit(X, y)

    with patch(
        "kvbiii_ml.modeling.optimization.classification_calibration._show_figure"
    ) as mock_show_figure:
        result = calibrator.plot_calibration_curves()

    if result is not None:
        raise AssertionError()
    mock_show_figure.assert_called_once()


def test_show_figure_uses_plain_show_outside_notebook():
    """Tests _show_figure falls back to plt.show() when get_ipython() is None.

    Asserts:
        - plt.show() is called and IPython's display() is not.
    """
    fig = Mock()
    with (
        patch(
            "kvbiii_ml.modeling.optimization.classification_calibration.get_ipython",
            return_value=None,
        ),
        patch(
            "kvbiii_ml.modeling.optimization.classification_calibration.plt.show"
        ) as mock_show,
        patch(
            "kvbiii_ml.modeling.optimization.classification_calibration.display"
        ) as mock_display,
    ):
        _show_figure(fig)

    mock_show.assert_called_once()
    mock_display.assert_not_called()


def test_show_figure_displays_inline_image_inside_notebook():
    """Tests _show_figure rasterizes and displays fig via IPython inside a notebook.

    Asserts:
        - IPython's display() is invoked instead of plt.show().
        - The figure is rasterized to PNG and closed afterward.
    """
    fig = Mock()
    fig.dpi = 100
    fig.get_facecolor.return_value = "white"

    with (
        patch(
            "kvbiii_ml.modeling.optimization.classification_calibration.get_ipython",
            return_value=Mock(),
        ),
        patch(
            "kvbiii_ml.modeling.optimization.classification_calibration.plt.show"
        ) as mock_show,
        patch(
            "kvbiii_ml.modeling.optimization.classification_calibration.plt.close"
        ) as mock_close,
        patch(
            "kvbiii_ml.modeling.optimization.classification_calibration.display"
        ) as mock_display,
    ):
        _show_figure(fig)

    mock_show.assert_not_called()
    mock_display.assert_called_once()
    mock_close.assert_called_once_with(fig)
    fig.savefig.assert_called_once()


def test_classificationcalibrator_style_scores_raises_runtimeerror_before_fit(
    logistic_regression_estimator, calibration_cv_trainer
):
    """Tests style_scores raises before fit() has been called.

    Args:
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        calibration_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - RuntimeError is raised instructing the caller to fit first.
    """
    calibrator = ClassificationCalibrator(
        estimator=logistic_regression_estimator, cross_validator=calibration_cv_trainer
    )

    with pytest.raises(RuntimeError, match=r"Call fit\(\) before style_scores\(\)\."):
        calibrator.style_scores()


def test_classificationcalibrator_style_scores_returns_styler_with_best_row_highlighted(
    binary_calibration_data, logistic_regression_estimator, calibration_cv_trainer
):
    """Tests style_scores returns a Styler that bolds the winning method's row.

    Args:
        binary_calibration_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        calibration_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - style_scores() returns a pandas Styler.
        - The rendered HTML contains a bold-font rule (the best method's row).
    """
    X, y = binary_calibration_data
    calibrator = ClassificationCalibrator(
        estimator=logistic_regression_estimator, cross_validator=calibration_cv_trainer
    )
    calibrator.fit(X, y)

    styled = calibrator.style_scores()

    if not isinstance(styled, Styler):
        raise AssertionError()
    html = styled.to_html()
    if "font-weight: 800" not in html:
        raise AssertionError(
            "Expected the best method's row to be bolded in the rendered table."
        )


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
