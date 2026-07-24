import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from kvbiii_ml.modeling.training.calibrated_model import CalibratedModel


def test_calibratedmodel_init_raises_error_for_invalid_problem_type(
    logistic_regression_estimator,
):
    """Tests that a non-classification problem_type is rejected at construction.

    Args:
        logistic_regression_estimator (LogisticRegression): Estimator to wrap.

    Asserts:
        - ValueError is raised when problem_type is not "classification".
    """
    with pytest.raises(ValueError, match="classification"):
        CalibratedModel(
            estimator=logistic_regression_estimator, problem_type="regression"
        )


def test_calibratedmodel_init_raises_error_for_invalid_calibration_method(
    logistic_regression_estimator,
):
    """Tests that an unsupported calibration_method is rejected at construction.

    Args:
        logistic_regression_estimator (LogisticRegression): Estimator to wrap.

    Asserts:
        - ValueError is raised when calibration_method is not "isotonic"/"sigmoid".
    """
    with pytest.raises(ValueError, match="isotonic"):
        CalibratedModel(
            estimator=logistic_regression_estimator, calibration_method="platt"
        )


@pytest.mark.parametrize("calibration_size", [0.0, 1.0, -0.1, 1.5])
def test_calibratedmodel_init_raises_error_for_invalid_calibration_size(
    logistic_regression_estimator, calibration_size
):
    """Tests that calibration_size outside the open interval (0, 1) is rejected.

    Args:
        logistic_regression_estimator (LogisticRegression): Estimator to wrap.
        calibration_size (float): Boundary or out-of-range value under test.

    Asserts:
        - ValueError is raised for calibration_size at or beyond the (0, 1) bounds.
    """
    with pytest.raises(ValueError, match="calibration_size"):
        CalibratedModel(
            estimator=logistic_regression_estimator, calibration_size=calibration_size
        )


def test_calibratedmodel_init_accepts_valid_calibration_size(
    logistic_regression_estimator,
):
    """Tests that a calibration_size strictly inside (0, 1) is accepted.

    Args:
        logistic_regression_estimator (LogisticRegression): Estimator to wrap.

    Asserts:
        - Construction succeeds and calibration_size is stored unchanged.
    """
    model = CalibratedModel(
        estimator=logistic_regression_estimator, calibration_size=0.5
    )
    if model.calibration_size != 0.5:
        raise AssertionError("calibration_size was not stored correctly.")


def test_calibratedmodel_predict_proba_raises_before_fit(logistic_regression_estimator):
    """Tests that predict_proba raises before fit() has been called.

    Args:
        logistic_regression_estimator (LogisticRegression): Estimator to wrap.

    Asserts:
        - RuntimeError is raised when predict_proba is called on an unfitted instance.
    """
    model = CalibratedModel(estimator=logistic_regression_estimator)
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict_proba(pd.DataFrame({"feature_0": [0.1, 0.2]}))


def test_calibratedmodel_predict_raises_before_fit(logistic_regression_estimator):
    """Tests that predict raises before fit() has been called.

    Args:
        logistic_regression_estimator (LogisticRegression): Estimator to wrap.

    Asserts:
        - RuntimeError is raised when predict is called on an unfitted instance,
          propagated from the underlying predict_proba() guard.
    """
    model = CalibratedModel(estimator=logistic_regression_estimator)
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(pd.DataFrame({"feature_0": [0.1, 0.2]}))


def test_calibratedmodel_fit_and_predict_proba_returns_valid_probability_matrix(
    binary_classification_data, test_settings
):
    """Tests fit() followed by predict_proba() on real binary classification data.

    Args:
        binary_classification_data (tuple[pd.DataFrame, pd.Series]): Synthetic
            binary classification data.
        test_settings (TestSettings): Shared test configuration for the seed.

    Asserts:
        - predict_proba() returns a (n_samples, n_classes) matrix.
        - Every row sums to approximately 1.0.
        - All probabilities lie within [0, 1].
    """
    X, y = binary_classification_data
    estimator = LogisticRegression(max_iter=200, random_state=test_settings.SEED)
    model = CalibratedModel(estimator=estimator, seed=test_settings.SEED)
    model.fit(X, y)
    probas = model.predict_proba(X)

    if probas.shape != (len(X), 2):
        raise AssertionError(f"Unexpected predict_proba shape: {probas.shape}")
    if not np.allclose(probas.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("Probability rows do not sum to 1.")
    if np.any(probas < 0.0) or np.any(probas > 1.0):
        raise AssertionError("Probabilities fall outside [0, 1].")


def test_calibratedmodel_fit_and_predict_returns_known_class_labels(
    binary_classification_data, test_settings
):
    """Tests that predict() returns labels drawn from classes_ after fit().

    Args:
        binary_classification_data (tuple[pd.DataFrame, pd.Series]): Synthetic
            binary classification data.
        test_settings (TestSettings): Shared test configuration for the seed.

    Asserts:
        - predict() returns an array of the same length as the input.
        - Every predicted label is present in the fitted classes_ array.
    """
    X, y = binary_classification_data
    estimator = LogisticRegression(max_iter=200, random_state=test_settings.SEED)
    model = CalibratedModel(estimator=estimator, seed=test_settings.SEED)
    model.fit(X, y)
    predictions = model.predict(X)

    if len(predictions) != len(X):
        raise AssertionError("predict() length mismatch with input.")
    if not set(np.unique(predictions)).issubset(set(model.classes_.tolist())):
        raise AssertionError("predict() returned labels outside classes_.")


@pytest.mark.parametrize(
    "probas_1d, expected_columns",
    [
        (np.array([0.2, 0.8, 0.5]), 2),
    ],
)
def test_calibratedmodel_ensure_proba_matrix_coerces_1d_to_2d(
    probas_1d, expected_columns
):
    """Tests that _ensure_proba_matrix coerces a 1-D positive-class array to 2-D.

    Args:
        probas_1d (np.ndarray): 1-D array of positive-class probabilities.
        expected_columns (int): Expected number of columns after coercion.

    Asserts:
        - The output is 2-D with the expected number of columns.
        - Column 0 is 1 - column 1 (negative-class complement).
    """
    coerced = CalibratedModel._ensure_proba_matrix(probas_1d)
    if coerced.ndim != 2 or coerced.shape[1] != expected_columns:
        raise AssertionError(f"Unexpected coerced shape: {coerced.shape}")
    if not np.allclose(coerced[:, 0], 1.0 - probas_1d):
        raise AssertionError(
            "Negative-class column is not the complement of the input."
        )


def test_calibratedmodel_ensure_proba_matrix_passthrough_for_2d():
    """Tests that _ensure_proba_matrix leaves an already-2D array unchanged.

    Asserts:
        - A 2-D input array is returned unchanged (same values).
    """
    probas_2d = np.array([[0.3, 0.7], [0.6, 0.4]])
    coerced = CalibratedModel._ensure_proba_matrix(probas_2d)
    if not np.allclose(coerced, probas_2d):
        raise AssertionError("2-D input was altered by _ensure_proba_matrix.")


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
