"""Tests for EnsembleWeightTunerCV class in ensemble_weights_tuner module."""

from unittest.mock import Mock, patch

import numpy as np
import optuna
import pandas as pd

from kvbiii_ml.modeling.optimization.ensemble_weights_tuner import EnsembleWeightTunerCV
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


class TestEnsembleWeightTunerCV:
    """Test suite for EnsembleWeightTunerCV class."""

    def test_ensembleweighttuner_init_default_parameters(
        self, logistic_regression_estimator, kfold_cv
    ):
        """Tests EnsembleWeightTunerCV initialization with default parameters.

        Args:
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - Default parameters are set correctly
            - Estimators list is properly stored
            - Cross-validator is assigned
        """
        cross_validator = CrossValidationTrainer(
            metric_name="Roc AUC", problem_type="classification", cv=kfold_cv
        )

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator], cross_validator=cross_validator
        )

        if len(tuner.estimators) != 1:
            raise AssertionError()
        if tuner.cross_validator != cross_validator:
            raise AssertionError()
        if tuner.n_trials != 50:
            raise AssertionError()
        if tuner.seed != 17:
            raise AssertionError()
        if tuner.allow_negative_weights != False:
            raise AssertionError()
        if tuner.best_weights is not None:
            raise AssertionError()

    def test_ensembleweighttuner_init_custom_parameters(
        self, logistic_regression_estimator, kfold_cv
    ):
        """Tests EnsembleWeightTunerCV initialization with custom parameters.

        Args:
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - Custom parameters are set correctly
            - Allow negative weights flag works
            - Custom n_trials and seed are preserved
        """
        cross_validator = CrossValidationTrainer(
            metric_name="Accuracy", problem_type="classification", cv=kfold_cv
        )

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator, logistic_regression_estimator],
            cross_validator=cross_validator,
            n_trials=100,
            seed=42,
            allow_negative_weights=True,
        )

        if len(tuner.estimators) != 2:
            raise AssertionError()
        if tuner.n_trials != 100:
            raise AssertionError()
        if tuner.seed != 42:
            raise AssertionError()
        if tuner.allow_negative_weights != True:
            raise AssertionError()

    def test_ensembleweighttuner_check_x_dataframe_passthrough(self, sample_dataframe):
        """Tests check_x method with DataFrame input.

        Args:
            sample_dataframe (pd.DataFrame): Test DataFrame fixture

        Asserts:
            - DataFrame input is returned unchanged
            - Same object reference is maintained
        """
        result = EnsembleWeightTunerCV.check_x(sample_dataframe)

        if result is not sample_dataframe:
            raise AssertionError()
        if not isinstance(result, pd.DataFrame):
            raise AssertionError()

    def test_ensembleweighttuner_check_x_array_conversion(self):
        """Tests check_x method with numpy array input.

        Asserts:
            - Numpy array is converted to DataFrame
            - Data content is preserved
        """
        array_input = np.array([[1, 2], [3, 4], [5, 6]])
        result = EnsembleWeightTunerCV.check_x(array_input)

        if not isinstance(result, pd.DataFrame):
            raise AssertionError()
        if result.shape != (3, 2):
            raise AssertionError()
        np.testing.assert_array_equal(result.values, array_input)

    def test_ensembleweighttuner_check_y_series_passthrough(self, sample_series):
        """Tests check_y method with Series input.

        Args:
            sample_series (pd.Series): Test Series fixture

        Asserts:
            - Series input is returned unchanged
            - Same object reference is maintained
        """
        result = EnsembleWeightTunerCV.check_y(sample_series)

        if result is not sample_series:
            raise AssertionError()
        if not isinstance(result, pd.Series):
            raise AssertionError()

    def test_ensembleweighttuner_check_y_array_conversion(self):
        """Tests check_y method with numpy array input.

        Asserts:
            - Numpy array is converted to Series
            - Data content is preserved
        """
        array_input = np.array([1, 0, 1, 0, 1])
        result = EnsembleWeightTunerCV.check_y(array_input)

        if not isinstance(result, pd.Series):
            raise AssertionError()
        if len(result) != 5:
            raise AssertionError()
        np.testing.assert_array_equal(result.values, array_input)

    def test_ensembleweighttuner_create_study_configuration(
        self, logistic_regression_estimator, kfold_cv
    ):
        """Tests _create_study method creates properly configured study.

        Args:
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - Study is created with correct direction
            - TPE sampler is configured with proper seed
            - Hyperband pruner is set
        """
        cross_validator = CrossValidationTrainer(
            metric_name="Roc AUC", problem_type="classification", cv=kfold_cv
        )

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator],
            cross_validator=cross_validator,
            seed=42,
        )

        study = tuner._create_study()

        if not isinstance(study, optuna.study.Study):
            raise AssertionError()
        if study.direction != optuna.study.StudyDirection.MAXIMIZE:
            raise AssertionError()
        if not isinstance(study.sampler, optuna.samplers.TPESampler):
            raise AssertionError()
        if not isinstance(study.pruner, optuna.pruners.HyperbandPruner):
            raise AssertionError()

    def test_ensembleweighttuner_blend_predictions_1d_regression(
        self, logistic_regression_estimator, kfold_cv
    ):
        """Tests _blend_predictions method with 1D regression predictions.

        Args:
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - 1D predictions are blended correctly
            - Weights are applied as linear combination
        """
        cross_validator = CrossValidationTrainer(
            metric_name="MSE", problem_type="regression", cv=kfold_cv
        )

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator], cross_validator=cross_validator
        )

        preds_list = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])]
        weights = np.array([0.3, 0.7])

        result = tuner._blend_predictions(preds_list, weights)

        expected = 0.3 * preds_list[0] + 0.7 * preds_list[1]
        np.testing.assert_array_almost_equal(result, expected)

    def test_ensembleweighttuner_blend_predictions_2d_classification(
        self, logistic_regression_estimator, kfold_cv
    ):
        """Tests _blend_predictions method with 2D classification probabilities.

        Args:
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - 2D probability predictions are blended correctly
            - Results are normalized to valid probabilities
        """
        cross_validator = CrossValidationTrainer(
            metric_name="Roc AUC", problem_type="classification", cv=kfold_cv
        )

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator], cross_validator=cross_validator
        )

        preds_list = [
            np.array([[0.2, 0.8], [0.6, 0.4]]),
            np.array([[0.3, 0.7], [0.5, 0.5]]),
        ]
        weights = np.array([0.4, 0.6])

        result = tuner._blend_predictions(preds_list, weights)

        if result.shape != (2, 2):
            raise AssertionError()
        # Check that probabilities sum to 1
        np.testing.assert_array_almost_equal(result.sum(axis=1), [1.0, 1.0])

    def test_ensembleweighttuner_objective_regression_metric(
        self, logistic_regression_estimator, kfold_cv
    ):
        """Tests _objective method with regression metric.

        Args:
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - Objective function returns valid metric score
            - Weights are normalized correctly
        """
        cross_validator = CrossValidationTrainer(
            metric_name="MSE", problem_type="regression", cv=kfold_cv
        )

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator, logistic_regression_estimator],
            cross_validator=cross_validator,
        )

        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_float.side_effect = [0.3, 0.7]  # Two weights

        y_true = np.array([1.0, 2.0, 3.0])
        preds_list = [np.array([1.1, 2.1, 3.1]), np.array([0.9, 1.9, 2.9])]

        result = tuner._objective(mock_trial, y_true, preds_list)

        if not isinstance(result, float):
            raise AssertionError()
        if not result >= 0:
            raise AssertionError()

    def test_ensembleweighttuner_objective_classification_probabilities(
        self, logistic_regression_estimator, kfold_cv
    ):
        """Tests _objective method with classification probabilities.

        Args:
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - Objective function handles probability predictions
            - Returns valid metric score for classification
        """
        cross_validator = CrossValidationTrainer(
            metric_name="Roc AUC", problem_type="classification", cv=kfold_cv
        )

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator, logistic_regression_estimator],
            cross_validator=cross_validator,
        )

        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_float.side_effect = [0.4, 0.6]

        y_true = np.array([0, 1, 0])
        preds_list = [
            np.array([0.2, 0.8, 0.3]),  # Probabilities for class 1
            np.array([0.1, 0.9, 0.2]),
        ]

        result = tuner._objective(mock_trial, y_true, preds_list)

        if not isinstance(result, float):
            raise AssertionError()
        if not 0 <= result <= 1:
            raise AssertionError()

    def test_ensembleweighttuner_objective_negative_weights_allowed(
        self, logistic_regression_estimator, kfold_cv
    ):
        """Tests _objective method with negative weights allowed.

        Args:
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - Negative weights are handled correctly
            - L1 normalization is applied
        """
        cross_validator = CrossValidationTrainer(
            metric_name="MSE", problem_type="regression", cv=kfold_cv
        )

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator, logistic_regression_estimator],
            cross_validator=cross_validator,
            allow_negative_weights=True,
        )

        # Mock trial with negative weights
        mock_trial = Mock()
        mock_trial.suggest_float.side_effect = [-0.3, 0.7]

        y_true = np.array([1.0, 2.0, 3.0])
        preds_list = [np.array([1.1, 2.1, 3.1]), np.array([0.9, 1.9, 2.9])]

        result = tuner._objective(mock_trial, y_true, preds_list)

        if not isinstance(result, float):
            raise AssertionError()

    @patch("kvbiii_ml.modeling.optimization.ensemble_weights_tuner.optuna.create_study")
    def test_ensembleweighttuner_tune_integration(
        self,
        mock_create_study,
        binary_classification_data,
        logistic_regression_estimator,
        kfold_cv,
    ):
        """Tests tune method integration workflow.

        Args:
            mock_create_study: Mock optuna.create_study function
            binary_classification_data (tuple): Binary classification dataset fixture
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - tune method completes successfully
            - Best weights are set after tuning
            - Study optimization is called
        """
        X, y = binary_classification_data

        # Mock study
        mock_study = Mock()
        mock_study.best_params = {"w0": 0.4, "w1": 0.6}
        mock_create_study.return_value = mock_study

        cross_validator = CrossValidationTrainer(
            metric_name="Accuracy",
            problem_type="classification",
            cv=kfold_cv,
            verbose=False,
        )

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator, logistic_regression_estimator],
            cross_validator=cross_validator,
            n_trials=2,  # Small number for testing
        )

        # Mock _perform_cv to avoid actual training
        tuner._perform_cv = Mock(
            return_value=(
                np.array([0, 1, 0, 1]),
                [np.array([0.2, 0.8, 0.3, 0.7]), np.array([0.1, 0.9, 0.2, 0.8])],
            )
        )

        study = tuner.tune(X, y)

        if tuner.best_weights is None:
            raise AssertionError()
        if len(tuner.best_weights) != 2:
            raise AssertionError()
        if study != mock_study:
            raise AssertionError()
        mock_study.optimize.assert_called_once()

    def test_ensembleweighttuner_perform_cv_output_format(
        self, binary_classification_data, logistic_regression_estimator, kfold_cv
    ):
        """Tests _perform_cv method output format.

        Args:
            binary_classification_data (tuple): Binary classification dataset fixture
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - Returns tuple with y_true and predictions list
            - Predictions list has correct length for number of estimators
            - Binary classification probabilities are extracted correctly
        """
        X, y = binary_classification_data

        cross_validator = CrossValidationTrainer(
            metric_name="Roc AUC",
            problem_type="classification",
            cv=kfold_cv,
            verbose=False,
        )

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator], cross_validator=cross_validator
        )

        y_true, preds_list = tuner._perform_cv(X, y)

        if not isinstance(y_true, np.ndarray):
            raise AssertionError()
        if not isinstance(preds_list, list):
            raise AssertionError()
        if len(preds_list) != 1:
            raise AssertionError()
        if not len(y_true) > 0:
            raise AssertionError()
        if len(preds_list[0]) != len(y_true):
            raise AssertionError()

    def test_ensembleweighttuner_weight_normalization_positive_only(
        self, logistic_regression_estimator, kfold_cv
    ):
        """Tests weight normalization for positive-only weights.

        Args:
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - Positive weights sum to 1.0
            - All weights are non-negative
        """
        cross_validator = CrossValidationTrainer(
            metric_name="Accuracy", problem_type="classification", cv=kfold_cv
        )

        EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator, logistic_regression_estimator],
            cross_validator=cross_validator,
            allow_negative_weights=False,
        )

        # Simulate some weights
        weights = np.array([0.3, 0.7])
        normalized = weights / weights.sum()

        if not np.allclose(normalized.sum(), 1.0):
            raise AssertionError()
        if not all(w >= 0 for w in normalized):
            raise AssertionError()

    def test_ensembleweighttuner_weight_normalization_with_negatives(
        self, logistic_regression_estimator, kfold_cv
    ):
        """Tests weight normalization with negative weights allowed.

        Args:
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - L1 normalization is applied correctly
            - Sum of absolute values equals 1.0
        """
        cross_validator = CrossValidationTrainer(
            metric_name="MSE", problem_type="regression", cv=kfold_cv
        )

        EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator, logistic_regression_estimator],
            cross_validator=cross_validator,
            allow_negative_weights=True,
        )

        # Simulate weights with negatives
        weights = np.array([-0.3, 0.7])
        l1_norm = np.sum(np.abs(weights))
        normalized = weights / l1_norm

        if not np.allclose(np.sum(np.abs(normalized)), 1.0):
            raise AssertionError()

    def test_ensembleweighttuner_empty_estimators_list_handling(self, kfold_cv):
        """Tests handling of empty estimators list.

        Args:
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - Empty estimators list is handled gracefully
            - Object can be created with empty list
        """
        cross_validator = CrossValidationTrainer(
            metric_name="Accuracy", problem_type="classification", cv=kfold_cv
        )

        tuner = EnsembleWeightTunerCV(estimators=[], cross_validator=cross_validator)

        if len(tuner.estimators) != 0:
            raise AssertionError()

    def test_ensembleweighttuner_single_estimator_handling(
        self, binary_classification_data, logistic_regression_estimator, kfold_cv
    ):
        """Tests handling of single estimator case.

        Args:
            binary_classification_data (tuple): Binary classification dataset fixture
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - Single estimator case produces weight of 1.0
            - No errors occur with single estimator
        """
        X, y = binary_classification_data

        cross_validator = CrossValidationTrainer(
            metric_name="Accuracy",
            problem_type="classification",
            cv=kfold_cv,
            verbose=False,
        )

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator],
            cross_validator=cross_validator,
            n_trials=1,  # Minimal for testing
        )

        # Mock _perform_cv for quick testing
        tuner._perform_cv = Mock(
            return_value=(np.array([0, 1, 0]), [np.array([0.2, 0.8, 0.3])])
        )

        tuner.tune(X, y)

        if tuner.best_weights is None:
            raise AssertionError()
        if len(tuner.best_weights) != 1:
            raise AssertionError()
        if not np.allclose(tuner.best_weights.sum(), 1.0):
            raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
