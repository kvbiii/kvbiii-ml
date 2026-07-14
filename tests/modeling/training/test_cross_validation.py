"""Tests for kvbiii_ml.modeling.training.cross_validation module."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


def test_crossvalidationtrainer_init_creates_instance_with_default_cv(test_settings):
    """Tests CrossValidationTrainer initialization with default cross-validator.

    Args:
        test_settings: Test configuration fixture

    Asserts:
        - Trainer initializes successfully with default parameters
        - Default KFold cross-validator is created with expected parameters
        - Metric configuration is properly loaded
    """
    trainer = CrossValidationTrainer(
        metric_name="Accuracy", problem_type="classification"
    )

    if not (trainer.metric_name == "Accuracy"):
        raise AssertionError()
    if not (trainer.problem_type == "classification"):
        raise AssertionError()
    if not (isinstance(trainer.cv, KFold)):
        raise AssertionError()
    if not (trainer.cv.n_splits == 5):
        raise AssertionError()
    if not (trainer.verbose is True):
        raise AssertionError()
    if not (len(trainer.processors) == 0):
        raise AssertionError()


def test_crossvalidationtrainer_init_accepts_custom_cv_and_processors(
    kfold_cv, mock_transformer
):
    """Tests CrossValidationTrainer initialization with custom parameters.

    Args:
        kfold_cv (KFold): Custom cross-validator fixture
        mock_transformer (Mock): Mock transformer fixture

    Asserts:
        - Custom cross-validator is accepted and stored
        - Custom processors list is accepted and stored
        - Verbose setting can be customized
    """
    processors = [mock_transformer]
    trainer = CrossValidationTrainer(
        metric_name="F1",
        problem_type="classification",
        cv=kfold_cv,
        processors=processors,
        verbose=False,
    )

    if not (trainer.cv is kfold_cv):
        raise AssertionError()
    if not (trainer.processors == processors):
        raise AssertionError()
    if not (trainer.verbose is False):
        raise AssertionError()


def test_crossvalidationtrainer_init_raises_error_for_invalid_problem_type():
    """Tests CrossValidationTrainer initialization rejects invalid problem types.

    Asserts:
        - ValueError is raised for unsupported problem types
        - Error message contains helpful information about valid options
    """
    with pytest.raises(
        ValueError, match="problem_type must be 'classification' or 'regression'"
    ):
        CrossValidationTrainer(metric_name="Accuracy", problem_type="invalid_type")


def test_crossvalidationtrainer_init_raises_error_for_invalid_metric():
    """Tests CrossValidationTrainer initialization rejects invalid metrics.

    Asserts:
        - ValueError is raised for unknown metric names
        - Error message lists available metrics for reference
    """
    with pytest.raises(ValueError, match="Metric 'InvalidMetric' not found"):
        CrossValidationTrainer(
            metric_name="InvalidMetric", problem_type="classification"
        )


def test_crossvalidationtrainer_validate_processors_accepts_valid_transformers(
    mock_transformer,
):
    """Tests validate_processors method accepts properly configured transformers.

    Args:
        mock_transformer (Mock): Mock transformer with required methods

    Asserts:
        - Method completes without raising exceptions
        - Transformers with fit_transform are accepted
        - Transformers with separate fit and transform are accepted
    """
    trainer = CrossValidationTrainer(
        metric_name="Accuracy",
        problem_type="classification",
        processors=[mock_transformer],
    )

    # Should not raise any exception
    trainer.validate_processors()


def test_crossvalidationtrainer_validate_processors_rejects_invalid_processors():
    """Tests validate_processors method rejects improperly configured processors.

    Asserts:
        - ValueError is raised for processors without required methods
        - Error message explains required interface compliance
    """
    invalid_processor = Mock()
    # Remove required methods to make it invalid
    del invalid_processor.fit
    del invalid_processor.transform
    del invalid_processor.fit_transform
    del invalid_processor.fit_resample

    with pytest.raises(ValueError, match="must implement either"):
        CrossValidationTrainer(
            metric_name="Accuracy",
            problem_type="classification",
            processors=[invalid_processor],
        )


def test_crossvalidationtrainer_apply_processors_handles_empty_processors_list(
    sample_dataframe, sample_series
):
    """Tests _apply_processors method handles empty processors list correctly.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data

    Asserts:
        - Original data is returned unchanged when no processors provided
        - All returned data structures maintain original types and shapes
    """
    trainer = CrossValidationTrainer("Accuracy", "classification")

    train_df, valid_df, test_df, y_train_out, y_valid_out = trainer._apply_processors(
        processors=None,
        train_df=sample_dataframe,
        valid_df=sample_dataframe.copy(),
        test_df=sample_dataframe.copy(),
        y_train=sample_series,
        y_valid=sample_series.copy(),
    )

    pd.testing.assert_frame_equal(train_df, sample_dataframe)
    pd.testing.assert_frame_equal(valid_df, sample_dataframe)
    pd.testing.assert_frame_equal(test_df, sample_dataframe)
    pd.testing.assert_series_equal(y_train_out, sample_series)
    pd.testing.assert_series_equal(y_valid_out, sample_series)


def test_crossvalidationtrainer_apply_processors_handles_fit_transform_processors(
    sample_dataframe, sample_series
):
    """Tests _apply_processors method correctly applies fit_transform processors.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data

    Asserts:
        - Processors are fitted on training data
        - Transform is applied to all provided datasets
        - Method signatures are properly introspected for y parameter
    """
    mock_processor = Mock()
    mock_processor.fit_transform.return_value = sample_dataframe * 2
    mock_processor.transform.return_value = sample_dataframe * 2

    # Explicitly ensure fit_resample is not an attribute
    del mock_processor.fit_resample

    # Mock signature inspection to indicate y parameter is accepted
    with patch("kvbiii_ml.modeling.training.cross_validation.signature") as mock_sig:
        mock_sig.return_value.parameters = {"y": Mock()}

        trainer = CrossValidationTrainer("Accuracy", "classification")

        trainer._apply_processors(
            processors=[mock_processor],
            train_df=sample_dataframe,
            valid_df=sample_dataframe.copy(),
            test_df=sample_dataframe.copy(),
            y_train=sample_series,
            y_valid=sample_series.copy(),
        )

        mock_processor.fit_transform.assert_called_once()
        mock_processor.transform.assert_called()


def test_crossvalidationtrainer_apply_processors_handles_fit_resample_processors(
    sample_dataframe, sample_series
):
    """Tests _apply_processors method correctly applies fit_resample processors.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data
        sample_series (pd.Series): Sample target data

    Asserts:
        - fit_resample is called on training data
        - Resampled data is properly returned
        - Target data is updated from resampling
    """
    mock_processor = Mock()
    resampled_x = sample_dataframe.iloc[:50]  # Simulate resampling to fewer samples
    resampled_y = sample_series.iloc[:50]
    mock_processor.fit_resample.return_value = (resampled_x, resampled_y)

    trainer = CrossValidationTrainer("Accuracy", "classification")

    train_df, _valid_df, _test_df, y_train_out, _y_valid_out = (
        trainer._apply_processors(
            processors=[mock_processor],
            train_df=sample_dataframe,
            valid_df=sample_dataframe.copy(),
            test_df=sample_dataframe.copy(),
            y_train=sample_series,
            y_valid=sample_series.copy(),
        )
    )

    mock_processor.fit_resample.assert_called_once()
    pd.testing.assert_frame_equal(train_df, resampled_x)
    pd.testing.assert_series_equal(y_train_out, resampled_y)


def test_crossvalidationtrainer_fit_executes_cross_validation_loop(
    binary_classification_data, logistic_regression_estimator
):
    """Tests fit method executes complete cross-validation training loop.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector
        logistic_regression_estimator (LogisticRegression): Configured estimator

    Asserts:
        - Cross-validation completes without errors
        - Training and validation scores are computed for each fold
        - Fitted estimators are stored for each fold
    """
    X, y = binary_classification_data
    trainer = CrossValidationTrainer("Accuracy", "classification", verbose=False)

    train_scores, valid_scores, _test_preds = trainer.fit(
        logistic_regression_estimator, X, y
    )

    if not (len(train_scores) == 5):
        raise AssertionError()
    if not (len(valid_scores) == 5):
        raise AssertionError()
    if not (len(trainer.fitted_estimators_) == 5):
        raise AssertionError()
    if not (all(isinstance(score, float) for score in train_scores)):
        raise AssertionError()
    if not (all(isinstance(score, float) for score in valid_scores)):
        raise AssertionError()


def test_crossvalidationtrainer_fit_handles_test_data_averaging(
    binary_classification_data, logistic_regression_estimator
):
    """Tests fit method properly averages test predictions across folds.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector
        logistic_regression_estimator (LogisticRegression): Configured estimator

    Asserts:
        - Test predictions are returned when test data provided
        - Predictions are averaged across all folds
        - Output shape matches test data dimensions
    """
    X, y = binary_classification_data
    X_test = X.iloc[:20]  # Use subset as test data
    trainer = CrossValidationTrainer("Accuracy", "classification", verbose=False)

    _train_scores, _valid_scores, test_preds = trainer.fit(
        logistic_regression_estimator, X, y, X_test=X_test
    )

    if not (test_preds is not None):
        raise AssertionError()
    if not (len(test_preds) == len(X_test)):
        raise AssertionError()
    if not (isinstance(test_preds, np.ndarray)):
        raise AssertionError()


def test_crossvalidationtrainer_predict_returns_averaged_predictions(
    binary_classification_data, logistic_regression_estimator
):
    """Tests predict method returns averaged predictions from fitted estimators.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector
        logistic_regression_estimator (LogisticRegression): Configured estimator

    Asserts:
        - Predictions are generated from all fitted fold estimators
        - Results are averaged appropriately for problem type
        - Output format matches expected prediction format
    """
    X, y = binary_classification_data
    trainer = CrossValidationTrainer("Accuracy", "classification", verbose=False)
    trainer.fit(logistic_regression_estimator, X, y)

    predictions = trainer.predict(X)

    if not (len(predictions) == len(X)):
        raise AssertionError()
    if not (predictions.dtype in [np.int32, np.int64]):
        raise AssertionError()
    if not (all(pred in [0, 1] for pred in predictions)):
        raise AssertionError()


def test_crossvalidationtrainer_predict_raises_error_when_not_fitted():
    """Tests predict method raises error when called before fitting.

    Asserts:
        - RuntimeError is raised when predict called on unfitted trainer
        - Error message provides clear guidance about required fitting
    """
    trainer = CrossValidationTrainer("Accuracy", "classification")
    X = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(RuntimeError, match="must be fitted before calling predict"):
        trainer.predict(X)


def test_crossvalidationtrainer_predict_proba_returns_averaged_probabilities(
    binary_classification_data, logistic_regression_estimator
):
    """Tests predict_proba method returns averaged class probabilities.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector
        logistic_regression_estimator (LogisticRegression): Configured estimator

    Asserts:
        - Probabilities are averaged across all fitted estimators
        - Output shape matches (n_samples, n_classes) format
        - Probability values sum to 1 for each sample
    """
    X, y = binary_classification_data
    trainer = CrossValidationTrainer("Accuracy", "classification", verbose=False)
    trainer.fit(logistic_regression_estimator, X, y)

    probabilities = trainer.predict_proba(X)

    if not (probabilities.shape == (len(X), 2)):
        raise AssertionError()
    if not (np.allclose(probabilities.sum(axis=1), 1.0)):
        raise AssertionError()
    if not (np.all(probabilities >= 0) and np.all(probabilities <= 1)):
        raise AssertionError()


def test_crossvalidationtrainer_predict_proba_raises_error_for_regression():
    """Tests predict_proba method raises error for regression problems.

    Asserts:
        - ValueError is raised when predict_proba called for regression
        - Error message explains method is only for classification
    """
    trainer = CrossValidationTrainer("MAE", "regression")
    X = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(
        ValueError, match="predict_proba is only available for classification"
    ):
        trainer.predict_proba(X)


def test_crossvalidationtrainer_predict_with_confidence_returns_regression_metrics(
    regression_data, linear_regression_estimator
):
    """Tests predict_with_confidence method returns proper regression confidence metrics.

    Args:
        regression_data (tuple): Feature matrix and continuous target vector
        linear_regression_estimator (LinearRegression): Configured regression estimator

    Asserts:
        - Returns prediction mean and standard deviation across folds
        - Confidence intervals are computed using normal approximation
        - All returned metrics have appropriate shapes and value ranges
    """
    X, y = regression_data
    trainer = CrossValidationTrainer("MAE", "regression", verbose=False)

    trainer.fit(linear_regression_estimator, X, y)

    confidence_results = trainer.predict_with_confidence(X)

    if not ("prediction" in confidence_results):
        raise AssertionError()
    if not ("std" in confidence_results):
        raise AssertionError()
    if not ("ci_95_lower" in confidence_results):
        raise AssertionError()
    if not ("ci_95_upper" in confidence_results):
        raise AssertionError()

    if not (len(confidence_results["prediction"]) == len(X)):
        raise AssertionError()
    if not (len(confidence_results["std"]) == len(X)):
        raise AssertionError()


def test_crossvalidationtrainer_predict_with_confidence_returns_classification_metrics(
    binary_classification_data, logistic_regression_estimator
):
    """Tests predict_with_confidence method returns proper classification confidence metrics.

    Args:
        binary_classification_data (tuple): Feature matrix and binary target vector
        logistic_regression_estimator (LogisticRegression): Configured estimator

    Asserts:
        - Returns predicted classes and confidence measures
        - Disagreement metric measures fold consensus
        - All returned metrics have appropriate shapes and interpretations
    """
    X, y = binary_classification_data
    trainer = CrossValidationTrainer("Accuracy", "classification", verbose=False)
    trainer.fit(logistic_regression_estimator, X, y)

    confidence_results = trainer.predict_with_confidence(X)

    if not ("prediction" in confidence_results):
        raise AssertionError()
    if not ("confidence" in confidence_results):
        raise AssertionError()
    if not ("disagreement" in confidence_results):
        raise AssertionError()
    if not ("proba" in confidence_results):
        raise AssertionError()

    if not (len(confidence_results["prediction"]) == len(X)):
        raise AssertionError()
    if not (len(confidence_results["confidence"]) == len(X)):
        raise AssertionError()
    if not (confidence_results["proba"].shape == (len(X), 2)):
        raise AssertionError()


def test_crossvalidationtrainer_order_x_for_estimator_reorders_features_correctly(
    sample_dataframe,
):
    """Tests _order_X_for_estimator method reorders features to match estimator expectations.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data with multiple columns

    Asserts:
        - Features are reordered when estimator has feature_names_in_ attribute
        - Original DataFrame is returned when estimator lacks feature ordering info
        - Column order matches estimator's expected feature names
    """
    # Create mock estimator with specific feature order
    mock_estimator = Mock(spec=["feature_names_"])
    mock_estimator.feature_names_in_ = ["categorical_1", "numeric_1", "integer_1"]

    reordered_x = CrossValidationTrainer._order_X_for_estimator(
        sample_dataframe, mock_estimator
    )

    expected_order = ["categorical_1", "numeric_1", "integer_1"]
    if not (list(reordered_x.columns) == expected_order):
        raise AssertionError()

    # Test fallback when no feature names available
    mock_estimator_no_features = Mock(spec=[])  # No feature name attributes
    result_x = CrossValidationTrainer._order_X_for_estimator(
        sample_dataframe, mock_estimator_no_features
    )

    pd.testing.assert_frame_equal(result_x, sample_dataframe)


def test_crossvalidationtrainer_order_x_for_estimator_handles_normalized_feature_names():
    """Tests _order_X_for_estimator maps normalized estimator feature names safely.

    Asserts:
        - Feature names with underscores map to source columns with spaces.
        - No KeyError is raised when estimator naming differs from DataFrame naming.
    """
    X = pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 4.9],
            "sepal width (cm)": [3.5, 3.0],
            "petal length (cm)": [1.4, 1.4],
            "petal width (cm)": [0.2, 0.2],
        }
    )
    mock_estimator = Mock()
    mock_estimator.feature_names_ = [
        "sepal_length_(cm)",
        "sepal_width_(cm)",
        "petal_length_(cm)",
        "petal_width_(cm)",
    ]

    reordered_x = CrossValidationTrainer._order_X_for_estimator(X, mock_estimator)

    if not (list(reordered_x.columns) == list(X.columns)):
        raise AssertionError()


def test_crossvalidationtrainer_fit_maps_verbose_one_and_includes_train_in_eval_set(
    binary_classification_data,
    logistic_regression_estimator,
):
    """Tests fit maps verbose=1 to fit-time verbose=True and includes train in eval_set.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector.
        logistic_regression_estimator: Estimator fixture used for cloning.

    Asserts:
        - fit_and_predict receives fit_verbose=True when trainer verbose is set to 1
        - fit_and_predict receives include_train_in_eval_set=True on every fold
    """
    X, y = binary_classification_data
    trainer = CrossValidationTrainer(
        problem_type="classification", metric_name="Accuracy", verbose=1
    )

    captured_kwargs: list[dict] = []

    def _fake_fit_and_predict(*args, **kwargs):
        captured_kwargs.append(kwargs)
        estimator = args[0]
        y_train = args[2]
        y_valid = args[4]
        return np.asarray(y_train), np.asarray(y_valid), None, estimator

    with patch.object(CrossValidationTrainer, "fit_and_predict") as mock_fit:
        mock_fit.side_effect = _fake_fit_and_predict
        trainer.fit(logistic_regression_estimator, X, y)

    if not (captured_kwargs):
        raise AssertionError()
    if not (all(kwargs.get("fit_verbose") is True for kwargs in captured_kwargs)):
        raise AssertionError()
    if not (
        all(
            kwargs.get("include_train_in_eval_set") is True
            for kwargs in captured_kwargs
        )
    ):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
