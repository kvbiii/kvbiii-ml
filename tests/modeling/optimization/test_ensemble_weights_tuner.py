"""Tests for EnsembleWeightTunerCV class in ensemble_weights_tuner module."""

from unittest.mock import Mock

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from kvbiii_ml.modeling.optimization.ensemble_weights_tuner import EnsembleWeightTunerCV
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


def _mock_trial(*suggested_weights: float) -> Mock:
    """Builds a Mock Optuna trial that suggests fixed weights and never prunes.

    Args:
        *suggested_weights (float): Weight values returned in order by suggest_float.

    Returns:
        Mock: Trial stub compatible with EnsembleWeightTunerCV._objective/_score_weights.
    """
    trial = Mock()
    trial.suggest_float.side_effect = list(suggested_weights)
    trial.should_prune.return_value = False
    return trial


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
        if tuner.best_score_ is not None:
            raise AssertionError()
        if tuner.selection_source_ is not None:
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
            - No pruning is configured (objective is cheap and non-iterative)
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
        if not isinstance(study.pruner, optuna.pruners.NopPruner):
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

        preds_list = [pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 4.0, 6.0])]
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
            pd.DataFrame([[0.2, 0.8], [0.6, 0.4]]),
            pd.DataFrame([[0.3, 0.7], [0.5, 0.5]]),
        ]
        weights = np.array([0.4, 0.6])

        result = tuner._blend_predictions(preds_list, weights)

        if result.shape != (2, 2):
            raise AssertionError()
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

        mock_trial = _mock_trial(0.3, 0.7)

        y_true = pd.Series([1.0, 2.0, 3.0])
        preds_list = [pd.Series([1.1, 2.1, 3.1]), pd.Series([0.9, 1.9, 2.9])]
        fold_boundaries = [0, 3]

        result = tuner._objective(mock_trial, y_true, preds_list, fold_boundaries)

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

        mock_trial = _mock_trial(0.4, 0.6)

        y_true = pd.Series([0, 1, 0])
        preds_list = [
            pd.Series([0.2, 0.8, 0.3]),
            pd.Series([0.1, 0.9, 0.2]),
        ]
        fold_boundaries = [0, 3]

        result = tuner._objective(mock_trial, y_true, preds_list, fold_boundaries)

        if not isinstance(result, float):
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
            - Weights are normalized to sum to 1 (signed sum, not L1 norm)
        """
        cross_validator = CrossValidationTrainer(
            metric_name="MSE", problem_type="regression", cv=kfold_cv
        )

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator, logistic_regression_estimator],
            cross_validator=cross_validator,
            allow_negative_weights=True,
        )

        mock_trial = _mock_trial(-0.3, 0.7)

        y_true = pd.Series([1.0, 2.0, 3.0])
        preds_list = [pd.Series([1.1, 2.1, 3.1]), pd.Series([0.9, 1.9, 2.9])]
        fold_boundaries = [0, 3]

        result = tuner._objective(mock_trial, y_true, preds_list, fold_boundaries)

        if not isinstance(result, float):
            raise AssertionError()

    def test_ensembleweighttuner_objective_prunes_near_zero_weight_sum(
        self, logistic_regression_estimator, kfold_cv
    ):
        """Tests _objective prunes trials whose weights sum too close to zero.

        Args:
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - optuna.TrialPruned is raised for a cancelling weight vector
        """
        cross_validator = CrossValidationTrainer(
            metric_name="MSE", problem_type="regression", cv=kfold_cv
        )

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator, logistic_regression_estimator],
            cross_validator=cross_validator,
            allow_negative_weights=True,
        )

        mock_trial = _mock_trial(0.5, -0.5)

        y_true = pd.Series([1.0, 2.0, 3.0])
        preds_list = [pd.Series([1.1, 2.1, 3.1]), pd.Series([0.9, 1.9, 2.9])]
        fold_boundaries = [0, 3]

        try:
            tuner._objective(mock_trial, y_true, preds_list, fold_boundaries)
        except optuna.TrialPruned:
            return
        raise AssertionError(
            "Expected optuna.TrialPruned for a cancelling weight vector."
        )

    def test_ensembleweighttuner_tune_integration(
        self,
        binary_classification_data,
        logistic_regression_estimator,
        kfold_cv,
    ):
        """Tests tune method integration workflow end to end with a small trial budget.

        Args:
            binary_classification_data (tuple): Binary classification dataset fixture
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - tune method completes successfully
            - Best weights, score, and selection source are set after tuning
        """
        X, y = binary_classification_data

        cross_validator = CrossValidationTrainer(
            metric_name="Accuracy",
            problem_type="classification",
            cv=kfold_cv,
            verbose=False,
        )

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator, logistic_regression_estimator],
            cross_validator=cross_validator,
            n_trials=3,
        )

        tuner._perform_cv = Mock(
            return_value=(
                pd.Series([0, 1, 0, 1]),
                [pd.Series([0.2, 0.8, 0.3, 0.7]), pd.Series([0.1, 0.9, 0.2, 0.8])],
                [0, 2, 4],
            )
        )

        study = tuner.tune(X, y)

        if tuner.best_weights is None:
            raise AssertionError()
        if len(tuner.best_weights) != 2:
            raise AssertionError()
        if tuner.best_score_ is None:
            raise AssertionError()
        if tuner.selection_source_ is None:
            raise AssertionError()
        if not isinstance(study, optuna.study.Study):
            raise AssertionError()

    def test_ensembleweighttuner_perform_cv_output_format(
        self, binary_classification_data, logistic_regression_estimator, kfold_cv
    ):
        """Tests _perform_cv method output format.

        Args:
            binary_classification_data (tuple): Binary classification dataset fixture
            logistic_regression_estimator (LogisticRegression): Test estimator fixture
            kfold_cv (KFold): Test cross-validator fixture

        Asserts:
            - Returns a 3-tuple with y_true, predictions list, and fold boundaries
            - Predictions list has correct length for number of estimators
            - Binary classification probabilities are extracted correctly
            - Fold boundaries span the full OOF set and match the CV fold count
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

        y_true, preds_list, fold_boundaries = tuner._perform_cv(X, y)

        if not isinstance(y_true, pd.Series):
            raise AssertionError()
        if not isinstance(preds_list, list):
            raise AssertionError()
        if len(preds_list) != 1:
            raise AssertionError()
        if not len(y_true) > 0:
            raise AssertionError()
        if len(preds_list[0]) != len(y_true):
            raise AssertionError()
        if fold_boundaries[0] != 0:
            raise AssertionError()
        if fold_boundaries[-1] != len(y_true):
            raise AssertionError()
        if len(fold_boundaries) - 1 != kfold_cv.n_splits:
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

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator, logistic_regression_estimator],
            cross_validator=cross_validator,
            allow_negative_weights=False,
        )

        normalized = tuner._normalize_weights(np.array([0.3, 0.7]))

        if normalized is None:
            raise AssertionError()
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
            - Signed-sum normalization is applied (not L1 normalization)
            - Sum of the (signed) values equals 1.0
        """
        cross_validator = CrossValidationTrainer(
            metric_name="MSE", problem_type="regression", cv=kfold_cv
        )

        tuner = EnsembleWeightTunerCV(
            estimators=[logistic_regression_estimator, logistic_regression_estimator],
            cross_validator=cross_validator,
            allow_negative_weights=True,
        )

        normalized = tuner._normalize_weights(np.array([-0.3, 0.7]))

        if normalized is None:
            raise AssertionError()
        if not np.allclose(np.sum(normalized), 1.0):
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
            n_trials=1,
        )

        tuner._perform_cv = Mock(
            return_value=(
                pd.Series([0, 1, 0]),
                [pd.Series([0.2, 0.8, 0.3])],
                [0, 1, 2, 3],
            )
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


class _ReverseColumnsAfterSecondTransform(BaseEstimator, TransformerMixin):
    """Pass-through transformer that reverses column order from its third transform on.

    The trainer transforms each fold twice (train, validation); the tuner's own
    re-transform of the validation split is the third call, so the estimator
    only scores correctly when the tuner reorders columns for it.
    """

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "_ReverseColumnsAfterSecondTransform":
        """Reset the transform counter."""
        self.n_transforms_ = 0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X, with reversed columns from the third call on."""
        self.n_transforms_ += 1
        return X[X.columns[::-1]] if self.n_transforms_ >= 3 else X


def _classification_frame(n_samples: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    """Small separable binary classification frame."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(n_samples, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n_samples) > 0).astype(int))
    return X, y


def test_tuner_and_trainer_pass_a_custom_probs_metric_the_same_shapes():
    """Tests a custom "probs" metric sees identical y_pred shapes under trainer and tuner.

    Asserts:
        - Every shape the metric received is 1-D (binary positive-class column)
        - The set of shapes recorded under the trainer equals the set under the tuner
    """
    X, y = _classification_frame()
    shapes: list[tuple[int, ...]] = []

    def recording_log_loss(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Record the prediction shape, then score a clipped log loss."""
        shapes.append(np.asarray(y_pred).shape)
        p = np.clip(np.asarray(y_pred, dtype=float), 1e-9, 1 - 1e-9)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    def make_trainer() -> CrossValidationTrainer:
        """Build a trainer using the recording metric."""
        return CrossValidationTrainer(
            problem_type="classification",
            custom_metric={
                "name": "recording_log_loss",
                "function": recording_log_loss,
                "metric_type": "probs",
                "direction": "minimize",
            },
            cv=KFold(n_splits=3, shuffle=True, random_state=0),
            verbose=False,
        )

    make_trainer().fit(LogisticRegression(), X, y)
    trainer_shapes = {len(shape) for shape in shapes}
    shapes.clear()
    EnsembleWeightTunerCV(
        [LogisticRegression(), LogisticRegression(C=0.1)],
        make_trainer(),
        n_trials=3,
        verbose=False,
    ).tune(X, y)
    tuner_shapes = {len(shape) for shape in shapes}

    if trainer_shapes != {1} or tuner_shapes != {1}:
        raise AssertionError((trainer_shapes, tuner_shapes))


def test_tuner_reorders_columns_per_estimator_like_the_trainer():
    """Tests the tuner reorders transformed validation columns to each estimator's order.

    Asserts:
        - tune() succeeds although the pipeline hands back reversed columns, which
          sklearn estimators reject without reordering
    """
    X, _ = _classification_frame()
    y_reg = X["a"] * 2.0 + X["b"]
    trainer = CrossValidationTrainer(
        problem_type="regression",
        metric_name="RMSE",
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        preprocessing_pipeline=Pipeline(
            [("reverse", _ReverseColumnsAfterSecondTransform())]
        ),
        verbose=False,
    )
    tuner = EnsembleWeightTunerCV(
        [LinearRegression(), LinearRegression()], trainer, n_trials=2, verbose=False
    )
    tuner.tune(X, y_reg)
    if tuner.best_score_ is None or tuner.best_score_ > 1e-6:
        raise AssertionError(tuner.best_score_)


def test_tuner_leaves_the_trainer_it_receives_untouched():
    """Tests tune() has no side effects on the passed CrossValidationTrainer.

    Asserts:
        - fitted_estimators_ and valid_scores_ are those of the caller's own earlier fit
        - verbose is never changed
    """
    X, y = _classification_frame()
    trainer = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Roc AUC",
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        verbose=1,
    )
    trainer.fit(LogisticRegression(), X, y)
    estimators_before = list(trainer.fitted_estimators_)
    scores_before = list(trainer.valid_scores_)

    EnsembleWeightTunerCV(
        [LogisticRegression(C=0.1), LogisticRegression(C=10.0)],
        trainer,
        n_trials=2,
        verbose=False,
    ).tune(X, y)

    if (
        trainer.fitted_estimators_ != estimators_before
        or trainer.valid_scores_ != scores_before
    ):
        raise AssertionError("tuner overwrote the trainer's fitted state")
    if trainer.verbose != 1:
        raise AssertionError()


def test_tuner_best_score_describes_selected_weights_even_when_a_baseline_wins():
    """Tests best_score_ always scores best_weights, which the study may not describe.

    Asserts:
        - best_score_ equals the penalised score of best_weights
        - When a baseline is selected, best_weights is that baseline vector
    """
    X, y = _classification_frame()
    trainer = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Log Loss",
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        verbose=False,
    )
    tuner = EnsembleWeightTunerCV(
        [LogisticRegression(), LogisticRegression(C=0.01)],
        trainer,
        n_trials=2,
        verbose=False,
    )
    tuner.tune(X, y)
    y_true, preds, bounds = tuner._perform_cv(X, y)
    if not np.isclose(
        tuner._score_weights(tuner.best_weights, y_true, preds, bounds),
        tuner.best_score_,
    ):
        raise AssertionError()
    if tuner.selection_source_ != "optimized":
        expected = tuner._baseline_candidates(2)[tuner.selection_source_]
        if not np.allclose(tuner.best_weights, expected):
            raise AssertionError()


def test_tuner_supports_offset_column_of_the_trainer():
    """Tests the tuner drops the offset column and applies it like the trainer.

    Asserts:
        - tune() succeeds with a trainer configured with offset_column
        - Tuning two copies of one model gives the same score as the trainer's own
          mean validation score minus the fold-std penalty
    """
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"x0": rng.normal(size=300)})
    X["log_exposure"] = np.log(rng.uniform(0.2, 3.0, size=300))
    y = pd.Series(rng.poisson(np.exp(0.6 * X["x0"] + X["log_exposure"])))
    trainer = CrossValidationTrainer(
        problem_type="regression",
        metric_name="Poisson Deviance",
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        verbose=False,
        offset_column="log_exposure",
    )
    model = LGBMRegressor(objective="poisson", n_estimators=20, verbose=-1)
    _, valid_scores, _ = trainer.fit(model, X, y)

    tuner = EnsembleWeightTunerCV([model, model], trainer, n_trials=2, verbose=False)
    tuner.tune(X, y)
    expected = np.mean(valid_scores) + tuner.fold_std_penalty * np.std(valid_scores)
    if not np.isclose(tuner.best_score_, expected, rtol=1e-6):
        raise AssertionError((tuner.best_score_, expected))
