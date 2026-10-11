"""Tests for kvbiii_ml.modeling.training.base_trainer module."""

from unittest.mock import Mock, patch

import lightgbm
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

from kvbiii_ml.modeling.training.base_trainer import VERBOSE_LOG_PERIOD, BaseTrainer


def test_basetrainer_fit_estimator_basic_fitting(
    sample_dataframe, sample_series, mock_estimator
):
    """Tests fit_estimator method performs basic estimator fitting.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data
        mock_estimator (Mock): Mock estimator with fit method

    Asserts:
        - Estimator.fit is called with training data
        - Method returns the fitted estimator instance
        - Basic parameters are passed correctly
    """
    fitted = BaseTrainer.fit_estimator(mock_estimator, sample_dataframe, sample_series)

    mock_estimator.fit.assert_called_once_with(sample_dataframe, sample_series)
    if fitted is not mock_estimator:
        raise AssertionError()


def test_basetrainer_fit_estimator_handles_eval_set_parameter(
    sample_dataframe, sample_series
):
    """Tests fit_estimator method includes eval_set when estimator supports it.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data

    Asserts:
        - eval_set parameter is included when available in estimator signature
        - Validation data is formatted as [(X_train, y_train), (X_valid, y_valid)]
        - Other parameters are still handled correctly
    """
    mock_estimator = Mock()
    mock_estimator.fit.return_value = mock_estimator

    with patch("kvbiii_ml.modeling.training.base_trainer.signature") as mock_sig:
        mock_sig.return_value.parameters = {"eval_set": Mock(), "verbose": Mock()}

        X_valid = sample_dataframe.copy()
        y_valid = sample_series.copy()

        BaseTrainer.fit_estimator(
            mock_estimator,
            sample_dataframe,
            sample_series,
            X_valid=X_valid,
            y_valid=y_valid,
            verbose=False,
        )

        call_args = mock_estimator.fit.call_args
        if "eval_set" not in call_args.kwargs:
            raise AssertionError()
        if call_args.kwargs["eval_set"] != [
            (sample_dataframe, sample_series),
            (X_valid, y_valid),
        ]:
            raise AssertionError()
        if call_args.kwargs["verbose"] is not False:
            raise AssertionError()


def test_basetrainer_fit_estimator_handles_validation_parameters(
    sample_dataframe, sample_series
):
    """Tests fit_estimator method includes X_val/y_val parameters when supported.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data

    Asserts:
        - X_val and y_val parameters are included when available
        - Validation data is passed directly without transformation
        - Parameter names match expected estimator interface
    """
    mock_estimator = Mock()
    mock_estimator.fit.return_value = mock_estimator

    with patch("kvbiii_ml.modeling.training.base_trainer.signature") as mock_sig:
        mock_sig.return_value.parameters = {"X_val": Mock(), "y_val": Mock()}

        X_valid = sample_dataframe.copy()
        y_valid = sample_series.copy()

        BaseTrainer.fit_estimator(
            mock_estimator,
            sample_dataframe,
            sample_series,
            X_valid=X_valid,
            y_valid=y_valid,
        )

        call_args = mock_estimator.fit.call_args
        if call_args.kwargs["X_val"] is not X_valid:
            raise AssertionError()
        if call_args.kwargs["y_val"] is not y_valid:
            raise AssertionError()


def test_basetrainer_fit_estimator_handles_sample_weight_parameter(
    sample_dataframe, sample_series
):
    """Tests fit_estimator method includes sample_weight when supported.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data

    Asserts:
        - sample_weight parameter is included when available in signature
        - Weight values are passed through unchanged
        - Other parameters work alongside sample weights
    """
    mock_estimator = Mock()
    mock_estimator.fit.return_value = mock_estimator
    sample_weights = pd.Series(np.random.rand(len(sample_series)))

    with patch("kvbiii_ml.modeling.training.base_trainer.signature") as mock_sig:
        mock_sig.return_value.parameters = {"sample_weight": Mock()}

        BaseTrainer.fit_estimator(
            mock_estimator,
            sample_dataframe,
            sample_series,
            sample_weight=sample_weights,
        )

        call_args = mock_estimator.fit.call_args
        if call_args.kwargs["sample_weight"] is not sample_weights:
            raise AssertionError()


def test_basetrainer_fit_and_predict_returns_predictions_for_preds_type(
    sample_dataframe, sample_series, mock_estimator
):
    """Tests fit_and_predict method returns standard predictions for 'preds' metric type.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data
        mock_estimator (Mock): Mock estimator with predict method

    Asserts:
        - Standard predict method is called for all datasets
        - Training, validation, and test predictions are returned
        - Fitted estimator is returned as fourth element
    """
    X_valid = sample_dataframe.iloc[:20]
    y_valid = sample_series.iloc[:20]
    X_test = sample_dataframe.iloc[:10]

    train_pred, valid_pred, test_pred, fitted_est = BaseTrainer.fit_and_predict(
        mock_estimator,
        sample_dataframe,
        sample_series,
        X_valid,
        y_valid,
        metric_type="preds",
        X_test=X_test,
    )

    # Verify predict was called for all datasets
    if mock_estimator.predict.call_count != 3:
        raise AssertionError()
    if train_pred is None:
        raise AssertionError()
    if valid_pred is None:
        raise AssertionError()
    if test_pred is None:
        raise AssertionError()
    if fitted_est is not mock_estimator:
        raise AssertionError()


def test_basetrainer_fit_and_predict_returns_probabilities_for_probs_type(
    sample_dataframe, sample_series, mock_estimator
):
    """Tests fit_and_predict method returns probabilities for 'probs' metric type.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data
        mock_estimator (Mock): Mock estimator with predict_proba method

    Asserts:
        - predict_proba method is called instead of predict
        - Probabilities are returned for all provided datasets
        - Binary classification probabilities are handled correctly
    """
    X_valid = sample_dataframe.iloc[:20]
    y_valid = sample_series.iloc[:20]

    train_pred, valid_pred, test_pred, _fitted_est = BaseTrainer.fit_and_predict(
        mock_estimator,
        sample_dataframe,
        sample_series,
        X_valid,
        y_valid,
        metric_type="probs",
        X_test=None,
    )

    # Verify predict_proba was called
    if mock_estimator.predict_proba.call_count != 2:
        raise AssertionError()
    if train_pred is None:
        raise AssertionError()
    if valid_pred is None:
        raise AssertionError()
    if test_pred is not None:
        raise AssertionError()


def test_basetrainer_fit_and_predict_raises_error_for_invalid_metric_type(
    sample_dataframe, sample_series, mock_estimator
):
    """Tests fit_and_predict method raises error for invalid metric types.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data
        mock_estimator (Mock): Mock estimator fixture

    Asserts:
        - ValueError is raised for unsupported metric types
        - Error message provides clear guidance about valid options
    """
    X_valid = sample_dataframe.iloc[:20]
    y_valid = sample_series.iloc[:20]

    with pytest.raises(
        ValueError, match="Unknown metric type 'invalid'. Must be 'preds' or 'probs'"
    ):
        BaseTrainer.fit_and_predict(
            mock_estimator,
            sample_dataframe,
            sample_series,
            X_valid,
            y_valid,
            metric_type="invalid",
        )


def test_basetrainer_predict_returns_predictions_when_data_provided(mock_estimator):
    """Tests predict method returns predictions when input data is provided.

    Args:
        mock_estimator (Mock): Mock estimator with predict method

    Asserts:
        - predict method is called on estimator when data provided
        - Return value matches estimator's predict output
        - Input data is passed through unchanged
    """
    X_test = pd.DataFrame({"feature": [1, 2, 3]})
    expected_predictions = np.array([0, 1, 0])
    mock_estimator.predict.return_value = expected_predictions

    predictions = BaseTrainer.predict(mock_estimator, X_test)

    mock_estimator.predict.assert_called_once_with(X_test)
    np.testing.assert_array_equal(predictions, expected_predictions)


def test_basetrainer_predict_returns_none_when_no_data_provided(mock_estimator):
    """Tests predict method returns None when no input data is provided.

    Args:
        mock_estimator (Mock): Mock estimator fixture

    Asserts:
        - None is returned when X is None
        - Estimator predict method is not called
        - Method handles None input gracefully
    """
    predictions = BaseTrainer.predict(mock_estimator, None)

    if predictions is not None:
        raise AssertionError()
    mock_estimator.predict.assert_not_called()


def test_basetrainer_predict_proba_returns_probabilities_when_data_provided(
    mock_estimator,
):
    """Tests predict_proba method returns probabilities when input data is provided.

    Args:
        mock_estimator (Mock): Mock estimator with predict_proba method

    Asserts:
        - predict_proba method is called on estimator
        - Return value matches estimator's predict_proba output
        - Multiclass probabilities are returned unchanged
    """
    X_test = pd.DataFrame({"feature": [1, 2, 3]})
    # Multiclass probabilities (3 classes)
    expected_proba = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.3, 0.3, 0.4]])
    mock_estimator.predict_proba.return_value = expected_proba

    probabilities = BaseTrainer.predict_proba(mock_estimator, X_test)

    mock_estimator.predict_proba.assert_called_once_with(X_test)
    np.testing.assert_array_equal(probabilities, expected_proba)


def test_basetrainer_predict_proba_returns_full_matrix_for_binary_classification(
    mock_estimator,
):
    """Tests predict_proba method returns the full probability matrix for binary classification.

    Args:
        mock_estimator (Mock): Mock estimator with predict_proba method

    Asserts:
        - Binary classification probabilities are returned unchanged
        - Shape stays (n_samples, 2), not reduced to (n_samples,)
        - Both class probability columns are returned
    """
    X_test = pd.DataFrame({"feature": [1, 2, 3]})
    binary_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    mock_estimator.predict_proba.return_value = binary_proba

    probabilities = BaseTrainer.predict_proba(mock_estimator, X_test)

    np.testing.assert_array_equal(probabilities, binary_proba)
    if probabilities.shape != (3, 2):
        raise AssertionError()


def test_basetrainer_predict_proba_returns_none_when_no_data_provided(mock_estimator):
    """Tests predict_proba method returns None when no input data is provided.

    Args:
        mock_estimator (Mock): Mock estimator fixture

    Asserts:
        - None is returned when X is None
        - Estimator predict_proba method is not called
        - Method handles None input gracefully
    """
    probabilities = BaseTrainer.predict_proba(mock_estimator, None)

    if probabilities is not None:
        raise AssertionError()
    mock_estimator.predict_proba.assert_not_called()


def test_basetrainer_resolve_verbose_period_caps_truthy_values():
    """Tests _resolve_verbose_period maps any truthy verbose to VERBOSE_LOG_PERIOD.

    Asserts:
        - True and arbitrary truthy ints all resolve to VERBOSE_LOG_PERIOD.
        - False, 0, and None all resolve to False.
    """
    for truthy in (True, 1, 5, 200):
        if BaseTrainer._resolve_verbose_period(truthy) != VERBOSE_LOG_PERIOD:
            raise AssertionError(f"Expected VERBOSE_LOG_PERIOD for {truthy!r}.")
    for falsy in (False, 0, None):
        if BaseTrainer._resolve_verbose_period(falsy) is not False:
            raise AssertionError(f"Expected False for {falsy!r}.")


def test_basetrainer_fit_estimator_lgbm_caps_verbose_to_log_period(
    binary_classification_data,
):
    """Tests fit_estimator throttles LightGBM's log_evaluation callback period.

    Args:
        binary_classification_data: Numeric feature matrix and binary target.

    Asserts:
        - The callbacks fit kwarg carries a log_evaluation callback whose
          period is VERBOSE_LOG_PERIOD, not every single iteration.
    """
    X, y = binary_classification_data
    estimator = lightgbm.LGBMClassifier(n_estimators=2, verbose=-1)

    with patch.object(type(estimator), "fit", autospec=True) as mock_fit:
        BaseTrainer.fit_estimator(estimator, X, y, verbose=True)
        callbacks = mock_fit.call_args.kwargs["callbacks"]

    periods = [cb.period for cb in callbacks if hasattr(cb, "period")]
    if VERBOSE_LOG_PERIOD not in periods:
        raise AssertionError(
            f"Expected a log_evaluation callback with period={VERBOSE_LOG_PERIOD}, "
            f"got periods={periods}."
        )


def test_basetrainer_fit_estimator_xgb_caps_verbose_to_log_period(
    binary_classification_data,
):
    """Tests fit_estimator throttles XGBoost's verbose fit kwarg to VERBOSE_LOG_PERIOD.

    Args:
        binary_classification_data: Numeric feature matrix and binary target.

    Asserts:
        - verbose is passed as VERBOSE_LOG_PERIOD, not True (every iteration).
    """
    X, y = binary_classification_data
    estimator = XGBClassifier(n_estimators=2)

    with patch.object(type(estimator), "fit", autospec=True) as mock_fit:
        BaseTrainer.fit_estimator(estimator, X, y, verbose=True)
        verbose_kwarg = mock_fit.call_args.kwargs["verbose"]

    if verbose_kwarg != VERBOSE_LOG_PERIOD:
        raise AssertionError(f"Expected verbose={VERBOSE_LOG_PERIOD}, got {verbose_kwarg!r}.")


def test_basetrainer_fit_estimator_catboost_caps_verbose_to_log_period(
    binary_classification_data,
):
    """Tests fit_estimator throttles CatBoost's verbose fit kwarg to VERBOSE_LOG_PERIOD.

    Args:
        binary_classification_data: Numeric feature matrix and binary target.

    Asserts:
        - verbose is passed as VERBOSE_LOG_PERIOD, not True (every iteration).
    """
    X, y = binary_classification_data
    estimator = CatBoostClassifier(n_estimators=2, allow_writing_files=False)

    with patch.object(type(estimator), "fit", autospec=True) as mock_fit:
        BaseTrainer.fit_estimator(estimator, X, y, verbose=True)
        verbose_kwarg = mock_fit.call_args.kwargs["verbose"]

    if verbose_kwarg != VERBOSE_LOG_PERIOD:
        raise AssertionError(f"Expected verbose={VERBOSE_LOG_PERIOD}, got {verbose_kwarg!r}.")


def test_basetrainer_fit_estimator_histgb_enables_plain_verbose(
    binary_classification_data,
):
    """Tests fit_estimator toggles HistGradientBoosting's constructor-level verbose.

    scikit-learn's HistGradientBoosting has no fit-time verbose parameter and no
    periodic-logging support (only a coarse 0/1/2 level set at construction), so
    a truthy `verbose` request sets its constructor `verbose=1` rather than the
    (inapplicable) VERBOSE_LOG_PERIOD.

    Args:
        binary_classification_data: Numeric feature matrix and binary target.

    Asserts:
        - The estimator handed to fit() has verbose=1 in its own get_params().
    """
    X, y = binary_classification_data
    estimator = HistGradientBoostingClassifier(max_iter=2)

    with patch.object(type(estimator), "fit", autospec=True) as mock_fit:
        BaseTrainer.fit_estimator(estimator, X, y, verbose=True)
        fitted_self = mock_fit.call_args.args[0]

    if fitted_self.get_params()["verbose"] != 1:
        raise AssertionError(
            f"Expected verbose=1, got {fitted_self.get_params()['verbose']!r}."
        )


def test_basetrainer_fit_estimator_realigns_cuda_xgb_device_to_cpu(
    binary_classification_data,
):
    """Tests fit_estimator switches a CUDA-configured XGBoost estimator to CPU inference.

    Prevents regressions of the "Falling back to prediction using DMatrix due to
    mismatched devices" UserWarning that XGBoost raises when a booster trained
    with device="cuda" is later asked to predict on CPU-resident pandas data.

    Args:
        binary_classification_data: Numeric feature matrix and binary target.

    Asserts:
        - The estimator handed to fit() still has device="cuda" (GPU training
          is left untouched).
        - The returned fitted estimator has device="cpu" so later predict calls
          on CPU data match without warning.
    """
    X, y = binary_classification_data
    estimator = XGBClassifier(n_estimators=2, device="cuda")
    devices_at_fit_call: list[str | None] = []

    def _record_device_and_fit(self, *_args, **_kwargs):
        """Record the device configured on self at call time, skipping real fit."""
        devices_at_fit_call.append(self.get_params()["device"])

    with patch.object(type(estimator), "fit", autospec=True, side_effect=_record_device_and_fit):
        fitted = BaseTrainer.fit_estimator(estimator, X, y)
    fit_time_device = devices_at_fit_call[0]

    if fit_time_device != "cuda":
        raise AssertionError(f"Expected fit-time device='cuda', got {fit_time_device!r}.")
    if fitted.get_params()["device"] != "cpu":
        raise AssertionError(
            f"Expected post-fit device='cpu', got {fitted.get_params()['device']!r}."
        )


def test_basetrainer_fit_estimator_leaves_cpu_xgb_device_untouched(
    binary_classification_data,
):
    """Tests fit_estimator does not alter an XGBoost estimator already targeting CPU.

    Args:
        binary_classification_data: Numeric feature matrix and binary target.

    Asserts:
        - device stays None (XGBoost's own default) after fit_estimator runs.
    """
    X, y = binary_classification_data
    estimator = XGBClassifier(n_estimators=2)

    with patch.object(type(estimator), "fit", autospec=True):
        fitted = BaseTrainer.fit_estimator(estimator, X, y)

    if fitted.get_params()["device"] is not None:
        raise AssertionError(f"Expected device=None, got {fitted.get_params()['device']!r}.")


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
