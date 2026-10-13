"""Tests for kvbiii_ml.modeling.training.cross_validation module."""

import importlib
import sys
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kvbiii_ml.modeling.training import cross_validation
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

from ._test_support import assert_valid_binary_proba_matrix


def test_crossvalidationtrainer_init_creates_instance_with_default_cv():
    """Tests CrossValidationTrainer initialization with the default cross-validator.

    Asserts:
        - Default KFold cross-validator is created with 5 splits.
        - Default KFold shuffles with the fixed random_state=17.
        - Metric configuration is loaded from metric_name.
    """
    trainer = CrossValidationTrainer(
        problem_type="classification", metric_name="Accuracy"
    )

    if not isinstance(trainer.cv, KFold):
        raise AssertionError()
    if trainer.cv.n_splits != 5:
        raise AssertionError()
    if trainer.cv.shuffle is not True:
        raise AssertionError()
    if trainer.cv.random_state != 17:
        raise AssertionError()
    if trainer.metric_name != "Accuracy":
        raise AssertionError()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"problem_type": "invalid_type", "metric_name": "Accuracy"}, "problem_type"),
        ({"problem_type": "classification"}, "Either metric_name or custom_metric"),
        (
            {"problem_type": "classification", "metric_name": "NotAMetric"},
            "not found in available metrics",
        ),
    ],
)
def test_crossvalidationtrainer_init_raises_valueerror_for_invalid_configuration(
    kwargs, match
):
    """Tests CrossValidationTrainer initialization rejects invalid configurations.

    Args:
        kwargs (dict): Constructor keyword arguments under test.
        match (str): Expected substring of the raised ValueError message.

    Asserts:
        - ValueError is raised for an unsupported problem_type.
        - ValueError is raised when neither metric_name nor custom_metric is given.
        - ValueError is raised for a metric_name absent from the metrics registry.
    """
    with pytest.raises(ValueError, match=match):
        CrossValidationTrainer(**kwargs)


def test_crossvalidationtrainer_init_raises_typeerror_for_non_pipeline_preprocessing():
    """Tests CrossValidationTrainer rejects a preprocessing_pipeline that is not a Pipeline.

    Asserts:
        - TypeError is raised when preprocessing_pipeline is neither None nor a Pipeline.
    """
    with pytest.raises(TypeError, match="must be an sklearn Pipeline or None"):
        CrossValidationTrainer(
            problem_type="classification",
            metric_name="Accuracy",
            preprocessing_pipeline=StandardScaler(),
        )


def test_crossvalidationtrainer_init_accepts_none_or_pipeline_preprocessing():
    """Tests CrossValidationTrainer accepts None and a real Pipeline for preprocessing_pipeline.

    Asserts:
        - None is stored unchanged.
        - A Pipeline instance is stored unchanged.
    """
    trainer_none = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        preprocessing_pipeline=None,
    )
    if trainer_none.preprocessing_pipeline is not None:
        raise AssertionError()

    pipeline = Pipeline([("scaler", StandardScaler())])
    trainer_pipeline = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        preprocessing_pipeline=pipeline,
    )
    if trainer_pipeline.preprocessing_pipeline is not pipeline:
        raise AssertionError()


def test_crossvalidationtrainer_validate_pipeline_raises_for_invalid_state():
    """Tests validate_pipeline raises TypeError when the stored pipeline is invalid.

    Asserts:
        - Calling validate_pipeline directly after mutating preprocessing_pipeline
          to a non-Pipeline value raises TypeError.
    """
    trainer = CrossValidationTrainer(
        problem_type="classification", metric_name="Accuracy"
    )
    trainer.preprocessing_pipeline = "not_a_pipeline"

    with pytest.raises(TypeError, match="must be an sklearn Pipeline or None"):
        trainer.validate_pipeline()


def test_crossvalidationtrainer_init_accepts_custom_metric():
    """Tests CrossValidationTrainer initialization with a custom_metric configuration.

    Asserts:
        - metric_name, metric_type, and metric_direction come from the custom_metric dict.
        - metric_name argument is not required when custom_metric is provided.
    """
    custom_metric = {
        "name": "custom_mae",
        "function": lambda y_true, y_pred: float(
            np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))
        ),
        "metric_type": "preds",
        "direction": "minimize",
    }
    trainer = CrossValidationTrainer(
        problem_type="regression", custom_metric=custom_metric
    )

    if trainer.metric_name != "custom_mae":
        raise AssertionError()
    if trainer.metric_type != "preds":
        raise AssertionError()
    if trainer.metric_direction != "minimize":
        raise AssertionError()


def test_crossvalidationtrainer_fit_returns_scores_matching_cv_splits(
    binary_classification_data, logistic_regression_estimator, kfold_cv
):
    """Tests fit returns per-fold train/valid scores whose lengths match cv.n_splits.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        kfold_cv (KFold): Cross-validation splitter fixture with 3 splits.

    Asserts:
        - train_scores has length equal to cv.n_splits.
        - valid_scores has length equal to cv.n_splits.
        - avg_test_pred is None when X_test is not provided.
        - fitted_estimators_ has one entry per fold.
    """
    X, y = binary_classification_data
    trainer = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=kfold_cv,
        verbose=False,
    )

    train_scores, valid_scores, avg_test_pred = trainer.fit(
        logistic_regression_estimator, X, y
    )

    if len(train_scores) != kfold_cv.n_splits:
        raise AssertionError()
    if len(valid_scores) != kfold_cv.n_splits:
        raise AssertionError()
    if avg_test_pred is not None:
        raise AssertionError()
    if len(trainer.fitted_estimators_) != kfold_cv.n_splits:
        raise AssertionError()


def test_crossvalidationtrainer_fit_averages_test_predictions_across_folds(
    binary_classification_data, logistic_regression_estimator, kfold_cv
):
    """Tests fit averages test-set predictions across folds when X_test is provided.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        kfold_cv (KFold): Cross-validation splitter fixture with 3 splits.

    Asserts:
        - avg_test_pred is not None when X_test is passed.
        - avg_test_pred has one row per test sample.
    """
    X, y = binary_classification_data
    X_test = X.iloc[:10]
    trainer = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=kfold_cv,
        verbose=False,
    )

    _train_scores, _valid_scores, avg_test_pred = trainer.fit(
        logistic_regression_estimator, X, y, X_test=X_test
    )

    if avg_test_pred is None:
        raise AssertionError()
    if len(avg_test_pred) != len(X_test):
        raise AssertionError()


def test_crossvalidationtrainer_fit_with_preprocessing_pipeline_round_trips(
    binary_classification_data, logistic_regression_estimator, kfold_cv
):
    """Tests fit clones and applies a preprocessing_pipeline independently per fold.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        kfold_cv (KFold): Cross-validation splitter fixture with 3 splits.

    Asserts:
        - fit completes without raising when a Pipeline is supplied.
        - One fitted pipeline is stored per fold.
        - predict returns one prediction per input row after fitting.
    """
    X, y = binary_classification_data
    pipeline = Pipeline([("scaler", StandardScaler())]).set_output(transform="pandas")
    trainer = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=kfold_cv,
        preprocessing_pipeline=pipeline,
        verbose=False,
    )

    trainer.fit(logistic_regression_estimator, X, y)

    if len(trainer.fitted_pipelines_) != kfold_cv.n_splits:
        raise AssertionError()
    if not all(fp is not None for fp in trainer.fitted_pipelines_):
        raise AssertionError()

    predictions = trainer.predict(X)
    if len(predictions) != len(X):
        raise AssertionError()


@pytest.mark.parametrize(
    "method_name", ["predict", "predict_proba", "predict_with_confidence"]
)
def test_crossvalidationtrainer_prediction_methods_raise_before_fit(method_name):
    """Tests predict, predict_proba, and predict_with_confidence raise before fit().

    Args:
        method_name (str): Name of the prediction method under test.

    Asserts:
        - RuntimeError is raised when the method is called before fit().
    """
    trainer = CrossValidationTrainer(
        problem_type="classification", metric_name="Accuracy"
    )
    X = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(RuntimeError, match="must be fitted before calling"):
        getattr(trainer, method_name)(X)


def test_crossvalidationtrainer_predict_proba_raises_valueerror_for_regression():
    """Tests predict_proba raises ValueError when problem_type is regression.

    Asserts:
        - ValueError is raised regardless of fitted state for regression trainers.
    """
    trainer = CrossValidationTrainer(problem_type="regression", metric_name="MAE")
    X = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(
        ValueError, match="predict_proba is only available for classification"
    ):
        trainer.predict_proba(X)


def test_crossvalidationtrainer_predict_with_confidence_returns_regression_summary(
    regression_data, linear_regression_estimator, kfold_cv
):
    """Tests predict_with_confidence returns mean/std/CI for regression problems.

    Args:
        regression_data (tuple): Feature matrix and continuous target vector.
        linear_regression_estimator (LinearRegression): Configured regression estimator.
        kfold_cv (KFold): Cross-validation splitter fixture with 3 splits.

    Asserts:
        - Returned dict contains prediction, std, ci_95_lower, and ci_95_upper keys.
        - Each array has one entry per input row.
    """
    X, y = regression_data
    trainer = CrossValidationTrainer(
        problem_type="regression", metric_name="MAE", cv=kfold_cv, verbose=False
    )
    trainer.fit(linear_regression_estimator, X, y)

    result = trainer.predict_with_confidence(X)

    for key in ("prediction", "std", "ci_95_lower", "ci_95_upper"):
        if key not in result:
            raise AssertionError(f"Missing key: {key}")
        if len(result[key]) != len(X):
            raise AssertionError()


def test_crossvalidationtrainer_predict_with_confidence_raises_without_predict_proba(
    binary_classification_data, kfold_cv
):
    """Tests predict_with_confidence raises ValueError when a fold estimator lacks predict_proba.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector.
        kfold_cv (KFold): Cross-validation splitter fixture with 3 splits.

    Asserts:
        - ValueError is raised naming predict_proba as the missing requirement.
    """
    X, y = binary_classification_data
    estimator = SGDClassifier(random_state=17, max_iter=1000, loss="hinge")
    trainer = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=kfold_cv,
        verbose=False,
    )
    trainer.fit(estimator, X, y)

    with pytest.raises(ValueError, match="must implement predict_proba"):
        trainer.predict_with_confidence(X)


def test_crossvalidationtrainer_predict_proba_returns_averaged_probabilities(
    binary_classification_data, logistic_regression_estimator, kfold_cv
):
    """Tests predict_proba returns averaged, well-formed class probabilities.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        kfold_cv (KFold): Cross-validation splitter fixture with 3 splits.

    Asserts:
        - Output shape is (n_samples, n_classes).
        - Each row sums to 1.
        - All probability values lie within [0, 1].
    """
    X, y = binary_classification_data
    trainer = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=kfold_cv,
        verbose=False,
    )
    trainer.fit(logistic_regression_estimator, X, y)

    probabilities = trainer.predict_proba(X)
    assert_valid_binary_proba_matrix(probabilities, len(X))


def test_crossvalidationtrainer_order_x_for_estimator_reorders_exact_names(
    sample_dataframe,
):
    """Tests _order_x_for_estimator reorders columns for an exact feature-name match.

    Args:
        sample_dataframe (pd.DataFrame): Sample feature data with multiple columns.

    Asserts:
        - Columns are reordered to match estimator.feature_names_in_ exactly.
        - The original DataFrame is returned unchanged when the estimator exposes
          no feature-name attribute at all.
    """
    mock_estimator = Mock(spec=["feature_names_in_"])
    mock_estimator.feature_names_in_ = ["categorical_1", "numeric_1", "integer_1"]

    reordered_x = CrossValidationTrainer._order_x_for_estimator(
        sample_dataframe, mock_estimator
    )
    if list(reordered_x.columns) != ["categorical_1", "numeric_1", "integer_1"]:
        raise AssertionError()

    mock_estimator_no_features = Mock(spec=[])
    result_x = CrossValidationTrainer._order_x_for_estimator(
        sample_dataframe, mock_estimator_no_features
    )
    pd.testing.assert_frame_equal(result_x, sample_dataframe)


def test_crossvalidationtrainer_order_x_for_estimator_falls_back_to_normalized_names():
    """Tests _order_x_for_estimator's tolerant normalized-name matching fallback.

    Args:
        None.

    Asserts:
        - Estimator feature names differing only by whitespace/underscore
          normalization still resolve to the matching DataFrame columns.
        - No KeyError is raised despite the naming mismatch.
    """
    X = pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 4.9],
            "sepal width (cm)": [3.5, 3.0],
            "petal length (cm)": [1.4, 1.4],
            "petal width (cm)": [0.2, 0.2],
        }
    )
    mock_estimator = Mock(spec=["feature_names_"])
    mock_estimator.feature_names_ = [
        "sepal_length_(cm)",
        "sepal_width_(cm)",
        "petal_length_(cm)",
        "petal_width_(cm)",
    ]

    reordered_x = CrossValidationTrainer._order_x_for_estimator(X, mock_estimator)

    if list(reordered_x.columns) != list(X.columns):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")


def test_crossvalidationtrainer_fit_in_parallel_matches_sequential_fit(
    binary_classification_data, random_forest_estimator, kfold_cv
):
    """Tests fold-parallel fit reproduces a sequential fit exactly and in fold order.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector.
        random_forest_estimator (RandomForestClassifier): Seeded estimator.
        kfold_cv (KFold): Cross-validation splitter fixture with 3 splits.

    Asserts:
        - Train and validation scores are identical, fold by fold.
        - Averaged test predictions are identical.
        - Every fitted pipeline and estimator predicts identically to its sequential twin.
    """
    X, y = binary_classification_data
    X_test = X.iloc[:15]
    pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    results = {}
    for n_jobs in (1, 3):
        trainer = CrossValidationTrainer(
            problem_type="classification",
            metric_name="Log Loss",
            cv=kfold_cv,
            preprocessing_pipeline=pipeline,
            verbose=False,
            n_jobs=n_jobs,
        )
        train_scores, valid_scores, avg_test_pred = trainer.fit(
            random_forest_estimator, X, y, X_test=X_test
        )
        fold_probas = [
            estimator.predict_proba(fitted_pipeline.transform(X_test))
            for estimator, fitted_pipeline in zip(
                trainer.fitted_estimators_, trainer.fitted_pipelines_
            )
        ]
        results[n_jobs] = (
            list(train_scores),
            list(valid_scores),
            avg_test_pred,
            fold_probas,
        )

    sequential, parallel = results[1], results[3]
    if sequential[0] != parallel[0] or sequential[1] != parallel[1]:
        raise AssertionError()
    if not np.array_equal(sequential[2], parallel[2]):
        raise AssertionError()
    for sequential_proba, parallel_proba in zip(sequential[3], parallel[3]):
        if not np.array_equal(sequential_proba, parallel_proba):
            raise AssertionError()


def test_crossvalidationtrainer_init_rejects_non_positive_n_jobs():
    """Tests the constructor rejects a fold concurrency below one.

    Asserts:
        - n_jobs=0 raises ValueError naming n_jobs.
        - The default n_jobs is 1.
    """
    with pytest.raises(ValueError, match="n_jobs"):
        CrossValidationTrainer(
            problem_type="classification", metric_name="Accuracy", n_jobs=0
        )
    trainer = CrossValidationTrainer(
        problem_type="classification", metric_name="Accuracy"
    )
    if trainer.n_jobs != 1:
        raise AssertionError()


def test_crossvalidationtrainer_fit_computes_sample_weights_per_fold(
    binary_classification_data, kfold_cv
):
    """Tests sample_weight_fn is evaluated on each fold's training rows and reaches fit.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector.
        kfold_cv (KFold): Cross-validation splitter fixture with 3 splits.

    Asserts:
        - sample_weight_fn is called once per fold with exactly that fold's training
          rows, in their original order, and never sees a validation row.
        - The returned weights reach the estimator: zero-weighting class 0 makes the
          weighted class prior of every fold estimator put all mass on class 1.
    """
    X, y = binary_classification_data
    seen_indices = []

    def positive_class_only(X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        """Record the fold's training index and weight only the positive class."""
        seen_indices.append(X_train.index)
        return (y_train == 1).to_numpy(dtype=float)

    trainer = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=kfold_cv,
        verbose=False,
        sample_weight_fn=positive_class_only,
    )
    trainer.fit(DummyClassifier(strategy="prior"), X, y)

    expected_indices = [X.index[train_idx] for train_idx, _ in kfold_cv.split(X, y)]
    if len(seen_indices) != len(expected_indices):
        raise AssertionError()
    for seen, expected in zip(seen_indices, expected_indices):
        if not seen.equals(expected):
            raise AssertionError()
    for estimator in trainer.fitted_estimators_:
        if not np.allclose(estimator.class_prior_, [0.0, 1.0]):
            raise AssertionError()


def test_crossvalidationtrainer_fit_rejects_misaligned_sample_weights(
    binary_classification_data, kfold_cv
):
    """Tests fit raises when sample_weight_fn returns the wrong number of weights.

    Args:
        binary_classification_data (tuple): Feature matrix and target vector.
        kfold_cv (KFold): Cross-validation splitter fixture with 3 splits.

    Asserts:
        - ValueError naming sample_weight_fn is raised for a too-short weight vector.
    """
    X, y = binary_classification_data
    trainer = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=kfold_cv,
        verbose=False,
        sample_weight_fn=lambda X_train, _y_train: np.ones(len(X_train) - 1),
    )
    with pytest.raises(ValueError, match="sample_weight_fn"):
        trainer.fit(DummyClassifier(), X, y)


def test_crossvalidationtrainer_init_rejects_non_callable_sample_weight_fn():
    """Tests the constructor rejects a sample_weight_fn that is not callable.

    Asserts:
        - Passing a precomputed weight array raises TypeError naming sample_weight_fn.
    """
    with pytest.raises(TypeError, match="sample_weight_fn"):
        CrossValidationTrainer(
            problem_type="classification",
            metric_name="Accuracy",
            sample_weight_fn=np.ones(10),
        )


class _EvalSetRecordingRegressor(RegressorMixin, BaseEstimator):
    """Mean regressor that records the index of the eval_set it was fitted with."""

    def fit(
        self, X: pd.DataFrame, y: pd.Series, eval_set: list | None = None
    ) -> "_EvalSetRecordingRegressor":
        """Store the training mean, training index and eval_set index."""
        self.mean_ = float(np.mean(y))
        self.fit_index_ = list(X.index)
        self.eval_index_ = list(eval_set[-1][0].index) if eval_set else []
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the training mean."""
        return np.full(len(X), self.mean_)


def test_crossvalidationtrainer_binary_preds_metric_averages_test_probabilities(
    binary_classification_data,
):
    """Tests X_test predictions are averaged probabilities, not averaged hard labels.

    Asserts:
        - With a "preds" metric (Balanced Accuracy) test_pred is (n, 2) and rows sum to 1
        - Values are genuine probabilities, not label fractions like 0.33 / 0.67
    """
    X, y = binary_classification_data
    trainer = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Balanced Accuracy",
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        verbose=False,
    )
    _, _, test_pred = trainer.fit(LogisticRegression(), X, y, X_test=X.iloc[:20])

    assert_valid_binary_proba_matrix(test_pred, 20)
    fold_label_fractions = {0.0, 1 / 3, 2 / 3, 1.0}
    if all(
        round(v, 6) in {round(f, 6) for f in fold_label_fractions}
        for v in test_pred[:, 1]
    ):
        raise AssertionError("test_pred looks like averaged labels")


def test_crossvalidationtrainer_custom_metric_receives_y_true_with_original_index(
    binary_classification_data,
):
    """Tests a custom metric gets y_true as a pd.Series keeping X's original index.

    Asserts:
        - Every call receives a Series whose index is a subset of X.index, so kwargs
          data can be aligned with .loc[y_true.index]
    """
    X, y = binary_classification_data
    X, y = X.set_axis(X.index + 1000), y.set_axis(y.index + 1000)
    weights = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    seen: list[pd.Index] = []

    def weighted_error(
        y_true: pd.Series, y_pred: np.ndarray, sample_weights: pd.Series
    ) -> float:
        """Weighted share of wrong labels, aligning weights by y_true's index."""
        seen.append(y_true.index)
        w = sample_weights.loc[y_true.index].to_numpy()
        return float(np.sum(w * (y_true.to_numpy() != y_pred)) / w.sum())

    trainer = CrossValidationTrainer(
        problem_type="classification",
        custom_metric={
            "name": "weighted_error",
            "function": weighted_error,
            "metric_type": "preds",
            "direction": "minimize",
            "kwargs": {"sample_weights": weights},
        },
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        verbose=False,
    )
    trainer.fit(LogisticRegression(), X, y)

    if not seen or not all(index.isin(X.index).all() for index in seen):
        raise AssertionError(seen)


def test_crossvalidationtrainer_early_stopping_holdout_keeps_validation_fold_out_of_eval_set(
    regression_data,
):
    """Tests early_stopping_holdout takes the eval set from the end of the training fold.

    Asserts:
        - The eval set never contains validation-fold rows
        - It is the last 25% of the fold's training rows and is not trained on
        - Default None keeps the validation fold as the eval set
    """
    X, y = regression_data
    cv = KFold(n_splits=3, shuffle=False)
    splits = list(cv.split(X, y))

    trainer = CrossValidationTrainer(
        problem_type="regression",
        metric_name="RMSE",
        cv=cv,
        verbose=False,
        early_stopping_holdout=0.25,
    )
    trainer.fit(_EvalSetRecordingRegressor(), X, y)
    for (train_idx, valid_idx), est in zip(splits, trainer.fitted_estimators_):
        train_index = list(X.index[train_idx])
        n_holdout = round(len(train_index) * 0.25)
        if est.eval_index_ != train_index[-n_holdout:]:
            raise AssertionError("eval set is not the tail of the training fold")
        if set(est.eval_index_) & set(X.index[valid_idx]) or set(est.eval_index_) & set(
            est.fit_index_
        ):
            raise AssertionError("eval set overlaps validation or fit rows")

    default = CrossValidationTrainer(
        problem_type="regression", metric_name="RMSE", cv=cv, verbose=False
    )
    default.fit(_EvalSetRecordingRegressor(), X, y)
    if default.fitted_estimators_[0].eval_index_ != list(X.index[splits[0][1]]):
        raise AssertionError()

    with pytest.raises(ValueError, match="early_stopping_holdout"):
        CrossValidationTrainer(
            problem_type="regression", metric_name="RMSE", early_stopping_holdout=1.0
        )


def test_cross_validation_import_does_not_modify_sys_path():
    """Tests (re)importing the module leaves sys.path untouched.

    Asserts:
        - sys.path is identical before and after reloading the module
    """
    before = list(sys.path)
    importlib.reload(cross_validation)
    if sys.path != before:
        raise AssertionError("cross_validation modified sys.path")
