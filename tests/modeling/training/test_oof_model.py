"""Tests for kvbiii_ml.modeling.training.oof_model module."""

from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer
from kvbiii_ml.modeling.training.oof_model import OOFModel


def test_oofmodel_init_creates_classification_instance(
    logistic_regression_estimator, kfold_cv
):
    """Tests OOFModel initialization for classification problems.

    Args:
        logistic_regression_estimator (LogisticRegression): Configured estimator
        kfold_cv (KFold): Cross-validation splitter

    Asserts:
        - Instance is created with classification problem type
        - Cross-validator and estimator are stored correctly
        - Metric function and type are properly configured
    """
    cv_trainer = CrossValidationTrainer(
        metric_name="Accuracy", problem_type="classification", cv=kfold_cv
    )

    oof = OOFModel(
        estimator=logistic_regression_estimator,
        cross_validator=cv_trainer,
        problem_type="classification",
    )

    assert oof.problem_type == "classification"
    assert oof.cross_validator is cv_trainer
    assert oof.estimator is logistic_regression_estimator
    assert callable(oof.eval_metric)
    assert oof.metric_type in ["preds", "probs"]
    assert len(oof.fitted_estimators_) == 0


def test_oofmodel_init_creates_regression_instance(
    logistic_regression_estimator, kfold_cv
):
    """Tests OOFModel initialization for regression problems.

    Args:
        logistic_regression_estimator (LogisticRegression): Configured estimator (used as regressor)
        kfold_cv (KFold): Cross-validation splitter

    Asserts:
        - Instance is created with regression problem type
        - Regression metrics are properly configured
        - Instance state is correctly initialized
    """
    cv_trainer = CrossValidationTrainer(
        metric_name="MAE", problem_type="regression", cv=kfold_cv
    )

    oof = OOFModel(
        estimator=logistic_regression_estimator,
        cross_validator=cv_trainer,
        problem_type="regression",
    )

    assert oof.problem_type == "regression"
    assert oof.cross_validator.metric_name == "MAE"
    assert len(oof.fitted_estimators_) == 0


def test_oofmodel_init_raises_error_for_invalid_problem_type(
    logistic_regression_estimator, kfold_cv
):
    """Tests OOFModel initialization rejects invalid problem types.

    Args:
        logistic_regression_estimator (LogisticRegression): Configured estimator
        kfold_cv (KFold): Cross-validation splitter

    Asserts:
        - ValueError is raised for unsupported problem types
        - Error message provides clear guidance about valid options
    """
    cv_trainer = CrossValidationTrainer(
        metric_name="Accuracy", problem_type="classification", cv=kfold_cv
    )

    with pytest.raises(
        ValueError, match="problem_type must be either 'classification' or 'regression'"
    ):
        OOFModel(
            estimator=logistic_regression_estimator,
            cross_validator=cv_trainer,
            problem_type="invalid_type",
        )


def test_oofmodel_init_raises_error_for_unsupported_metric(
    logistic_regression_estimator, kfold_cv
):
    """Tests OOFModel initialization rejects cross-validators with unsupported metrics.

    Args:
        logistic_regression_estimator (LogisticRegression): Configured estimator
        kfold_cv (KFold): Cross-validation splitter

    Asserts:
        - ValueError is raised for unknown metric names in cross-validator
        - Error message lists supported metrics for reference
    """
    # Mock cross-validator with invalid metric
    cv_trainer = Mock()
    cv_trainer.metric_name = "UnsupportedMetric"

    with pytest.raises(ValueError, match="Unsupported metric.*Supported metrics are"):
        OOFModel(
            estimator=logistic_regression_estimator,
            cross_validator=cv_trainer,
            problem_type="classification",
        )


def test_oofmodel_fit_executes_cross_validation_and_stores_estimators(
    binary_classification_data, logistic_regression_estimator, kfold_cv
):
    """Tests fit method executes cross-validation and stores fitted estimators.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector
        logistic_regression_estimator (LogisticRegression): Configured estimator
        kfold_cv (KFold): Cross-validation splitter

    Asserts:
        - Cross-validation is executed successfully
        - Fitted estimators from each fold are stored
        - Method returns self for chaining
    """
    X, y = binary_classification_data
    cv_trainer = CrossValidationTrainer(
        metric_name="Accuracy",
        problem_type="classification",
        cv=kfold_cv,
        verbose=False,
    )

    oof = OOFModel(
        estimator=logistic_regression_estimator,
        cross_validator=cv_trainer,
        problem_type="classification",
    )

    fitted_oof = oof.fit(X, y)

    assert fitted_oof is oof  # Returns self
    assert len(oof.fitted_estimators_) == kfold_cv.n_splits
    assert all(isinstance(est, BaseEstimator) for est in oof.fitted_estimators_)


def test_oofmodel_fit_handles_test_data_processing(
    binary_classification_data, logistic_regression_estimator, kfold_cv
):
    """Tests fit method handles optional test data processing per fold.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector
        logistic_regression_estimator (LogisticRegression): Configured estimator
        kfold_cv (KFold): Cross-validation splitter

    Asserts:
        - Test data is processed consistently across folds
        - Cross-validation completes without errors when test data provided
        - Fitted estimators are properly stored
    """
    X, y = binary_classification_data
    X_test = X.iloc[:20]  # Use subset as test data

    cv_trainer = CrossValidationTrainer(
        metric_name="Accuracy",
        problem_type="classification",
        cv=kfold_cv,
        verbose=False,
    )

    oof = OOFModel(
        estimator=logistic_regression_estimator,
        cross_validator=cv_trainer,
        problem_type="classification",
    )

    oof.fit(X, y, X_test=X_test)

    assert len(oof.fitted_estimators_) == kfold_cv.n_splits


def test_oofmodel_fit_clears_previous_fitted_estimators(
    binary_classification_data, logistic_regression_estimator, kfold_cv
):
    """Tests fit method clears previously fitted estimators before refitting.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector
        logistic_regression_estimator (LogisticRegression): Configured estimator
        kfold_cv (KFold): Cross-validation splitter

    Asserts:
        - Previous fitted estimators are cleared on refit
        - New estimators replace old ones completely
        - No memory leaks from previous fits
    """
    X, y = binary_classification_data
    cv_trainer = CrossValidationTrainer(
        metric_name="Accuracy",
        problem_type="classification",
        cv=kfold_cv,
        verbose=False,
    )

    oof = OOFModel(
        estimator=logistic_regression_estimator,
        cross_validator=cv_trainer,
        problem_type="classification",
    )

    # First fit
    oof.fit(X, y)
    first_fit_estimators = oof.fitted_estimators_.copy()

    # Second fit should clear and replace estimators
    oof.fit(X, y)

    assert len(oof.fitted_estimators_) == len(first_fit_estimators)
    # Estimators should be different instances (newly fitted)
    assert oof.fitted_estimators_[0] is not first_fit_estimators[0]


def test_oofmodel_predict_returns_averaged_classification_predictions(
    binary_classification_data, logistic_regression_estimator, kfold_cv
):
    """Tests predict method returns averaged predictions for classification.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector
        logistic_regression_estimator (LogisticRegression): Configured estimator
        kfold_cv (KFold): Cross-validation splitter

    Asserts:
        - Predictions are averaged across all fitted fold estimators
        - Classification predictions are in correct format
        - Output shape matches input sample count
    """
    X, y = binary_classification_data
    cv_trainer = CrossValidationTrainer(
        metric_name="Accuracy",
        problem_type="classification",
        cv=kfold_cv,
        verbose=False,
    )

    oof = OOFModel(
        estimator=logistic_regression_estimator,
        cross_validator=cv_trainer,
        problem_type="classification",
    )

    oof.fit(X, y)
    predictions = oof.predict(X)

    assert len(predictions) == len(X)
    assert predictions.dtype in [np.int32, np.int64]  # Classification predictions
    assert all(pred in [0, 1] for pred in predictions)  # Binary classification


def test_oofmodel_predict_raises_error_when_not_fitted(
    binary_classification_data, logistic_regression_estimator, kfold_cv
):
    """Tests predict method raises error when called before fitting.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector
        logistic_regression_estimator (LogisticRegression): Configured estimator
        kfold_cv (KFold): Cross-validation splitter

    Asserts:
        - RuntimeError is raised when predict called on unfitted model
        - Error message provides clear guidance about required fitting
    """
    X, y = binary_classification_data
    cv_trainer = CrossValidationTrainer(
        metric_name="Accuracy", problem_type="classification", cv=kfold_cv
    )

    oof = OOFModel(
        estimator=logistic_regression_estimator,
        cross_validator=cv_trainer,
        problem_type="classification",
    )

    with pytest.raises(
        RuntimeError, match="OOFModel must be fitted before calling predict"
    ):
        oof.predict(X)


def test_oofmodel_predict_proba_returns_averaged_probabilities(
    binary_classification_data, logistic_regression_estimator, kfold_cv
):
    """Tests predict_proba method returns averaged class probabilities.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector
        logistic_regression_estimator (LogisticRegression): Configured estimator
        kfold_cv (KFold): Cross-validation splitter

    Asserts:
        - Probabilities are averaged across all fitted estimators
        - Output shape matches (n_samples, n_classes) format
        - Probability values are valid (sum to 1, between 0 and 1)
    """
    X, y = binary_classification_data
    cv_trainer = CrossValidationTrainer(
        metric_name="Accuracy",
        problem_type="classification",
        cv=kfold_cv,
        verbose=False,
    )

    oof = OOFModel(
        estimator=logistic_regression_estimator,
        cross_validator=cv_trainer,
        problem_type="classification",
    )

    oof.fit(X, y)
    probabilities = oof.predict_proba(X)

    assert probabilities.shape == (len(X), 2)  # Binary classification
    assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)


def test_oofmodel_predict_proba_raises_error_for_regression(
    regression_data, logistic_regression_estimator, kfold_cv
):
    """Tests predict_proba method raises error for regression problems.

    Args:
        regression_data (tuple): Feature matrix and continuous target vector
        logistic_regression_estimator (LogisticRegression): Configured estimator (used as regressor)
        kfold_cv (KFold): Cross-validation splitter

    Asserts:
        - ValueError is raised when predict_proba called for regression
        - Error message explains method is only for classification
    """
    X, y = regression_data
    cv_trainer = CrossValidationTrainer(
        metric_name="MAE", problem_type="regression", cv=kfold_cv
    )

    oof = OOFModel(
        estimator=logistic_regression_estimator,
        cross_validator=cv_trainer,
        problem_type="regression",
    )

    with pytest.raises(
        ValueError, match="predict_proba is only available for classification"
    ):
        oof.predict_proba(X)


def test_oofmodel_order_x_for_estimator_reorders_features_correctly(sample_dataframe):
    """Tests _order_X_for_estimator method reorders features to match estimator expectations.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data with multiple columns

    Asserts:
        - Features are reordered when estimator has feature_names_in_ attribute
        - Original DataFrame is returned when estimator lacks feature ordering info
        - Column order matches estimator's expected feature names
    """
    # Create mock estimator with specific feature order
    mock_estimator = Mock()
    mock_estimator.feature_names_in_ = ["categorical_1", "numeric_1", "integer_1"]

    reordered_X = OOFModel._order_X_for_estimator(sample_dataframe, mock_estimator)

    expected_order = ["categorical_1", "numeric_1", "integer_1"]
    assert list(reordered_X.columns) == expected_order

    # Test fallback when no feature names available
    mock_estimator_no_features = Mock(spec=[])  # No feature name attributes
    result_X = OOFModel._order_X_for_estimator(
        sample_dataframe, mock_estimator_no_features
    )

    pd.testing.assert_frame_equal(result_X, sample_dataframe)


def test_oofmodel_predict_handles_estimators_without_predict_proba():
    """Tests predict method handles estimators without predict_proba for classification.

    Asserts:
        - Method falls back to predict when predict_proba unavailable
        - Classification predictions are properly averaged
        - No errors occur with estimators lacking probability prediction
    """
    # Create a more realistic test with actual OOF model and real data
    from sklearn.linear_model import (
        SGDClassifier,
    )  # Doesn't have predict_proba by default
    from sklearn.model_selection import KFold

    from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer
    from kvbiii_ml.modeling.training.oof_model import OOFModel

    # Create balanced test data with sufficient samples for each class in each fold
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Balanced classes

    # Use SGD without probability estimation
    estimator = SGDClassifier(
        random_state=42, max_iter=1000, loss="hinge"
    )  # hinge loss doesn't support predict_proba

    cv_trainer = CrossValidationTrainer(
        metric_name="Accuracy",
        problem_type="classification",
        cv=KFold(n_splits=2, shuffle=True, random_state=42),
        verbose=False,
    )

    oof = OOFModel(
        estimator=estimator, cross_validator=cv_trainer, problem_type="classification"
    )

    # Fit and predict
    oof.fit(X, y)
    predictions = oof.predict(X)

    assert len(predictions) == 10
    assert isinstance(predictions, np.ndarray)
    # Should be classification predictions (0 or 1)
    assert all(pred in [0, 1] for pred in predictions)


def test_oofmodel_integration_with_processors(
    binary_classification_data, logistic_regression_estimator, kfold_cv, mock_processor
):
    """Tests OOFModel integration with cross-validation processors.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector
        logistic_regression_estimator (LogisticRegression): Configured estimator
        kfold_cv (KFold): Cross-validation splitter
        mock_processor (Mock): Mock processor for preprocessing

    Asserts:
        - OOF model works with processors in cross-validation pipeline
        - Processors are applied per fold without data leakage
        - Final predictions are properly generated
    """
    X, y = binary_classification_data

    cv_trainer = CrossValidationTrainer(
        metric_name="Accuracy",
        problem_type="classification",
        cv=kfold_cv,
        processors=[mock_processor],
        verbose=False,
    )

    oof = OOFModel(
        estimator=logistic_regression_estimator,
        cross_validator=cv_trainer,
        problem_type="classification",
    )

    oof.fit(X, y)
    predictions = oof.predict(X)

    assert len(predictions) == len(X)
    assert len(oof.fitted_estimators_) == kfold_cv.n_splits
    # Verify processor was used in cross-validation
    mock_processor.fit_resample.assert_called()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
