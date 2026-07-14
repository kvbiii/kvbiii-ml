import importlib.util
import os

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedKFold,
    train_test_split,
)

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

from kvbiii_ml.modeling.optimization.classification_calibration import (
    ClassificationCalibrator,
)


class SimpleCVProvider:
    """Minimal CV provider used in tests."""

    def __init__(self, cv: BaseCrossValidator) -> None:
        self.cv = cv


def _binary_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Create a binary classification dataset."""
    x_arr, y_arr = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=17,
    )
    return pd.DataFrame(x_arr), pd.Series(y_arr)


def _multiclass_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Create a multiclass classification dataset."""
    x_arr, y_arr = make_classification(
        n_samples=360,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=3,
        random_state=17,
    )
    return pd.DataFrame(x_arr), pd.Series(y_arr)


def _to_binary_proba(proba: np.ndarray) -> np.ndarray:
    """Normalize probability output to a binary vector."""
    if proba.ndim == 2 and proba.shape[1] == 2:
        return proba[:, 1]
    return proba


def test_calibrator_binary_returns_estimator() -> None:
    """Ensure the calibrator returns a fitted estimator for binary targets."""
    x_df, y_ser = _binary_dataset()
    cv_trainer = SimpleCVProvider(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=17),
    )
    model = LogisticRegression(max_iter=500, solver="lbfgs")
    calibrator = ClassificationCalibrator(
        estimator=model,
        cross_validator=cv_trainer,
        metric_name="Log Loss",
    )
    best_estimator = calibrator.fit(x_df, y_ser)
    proba = best_estimator.predict_proba(x_df.iloc[:12])
    if proba.shape[0] != 12:
        raise AssertionError()

    plot_proba = _to_binary_proba(best_estimator.predict_proba(x_df))
    if HAS_MATPLOTLIB:
        calibrator.plot_calibration_curves(
            y_true=y_ser,
            proba_by_name={"Best": plot_proba},
            model_name="Binary",
        )


def test_calibrator_multiclass_returns_estimator() -> None:
    """Ensure the calibrator returns a fitted estimator for multiclass targets."""
    x_df, y_ser = _multiclass_dataset()
    cv_trainer = SimpleCVProvider(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=17),
    )
    model = LogisticRegression(max_iter=500, solver="lbfgs")
    calibrator = ClassificationCalibrator(
        estimator=model,
        cross_validator=cv_trainer,
        metric_name="Log Loss",
        id2label={0: "A", 1: "B", 2: "C"},
    )
    best_estimator = calibrator.fit(x_df, y_ser)
    proba = best_estimator.predict_proba(x_df.iloc[:12])
    if proba.shape[0] != 12:
        raise AssertionError()

    plot_proba = best_estimator.predict_proba(x_df)
    if HAS_MATPLOTLIB:
        calibrator.plot_calibration_curves(
            y_true=y_ser,
            proba_by_name={"Best": plot_proba},
            model_name="Multiclass",
        )


def test_calibrator_supports_explicit_validation_split() -> None:
    """Ensure explicit validation inputs are honored."""
    x_df, y_ser = _binary_dataset()
    X_train, X_valid, y_train, y_valid = train_test_split(
        x_df,
        y_ser,
        test_size=0.25,
        stratify=y_ser,
        random_state=17,
    )
    cv_trainer = SimpleCVProvider(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=17),
    )
    model = LogisticRegression(max_iter=500, solver="lbfgs")
    calibrator = ClassificationCalibrator(
        estimator=model,
        cross_validator=cv_trainer,
        metric_name="Log Loss",
    )
    plot_flag = HAS_MATPLOTLIB
    best_estimator = calibrator.fit(
        X_train,
        y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        plot=plot_flag,
        model_name="Binary",
    )
    proba = best_estimator.predict_proba(X_valid.iloc[:12])
    if proba.shape[0] != 12:
        raise AssertionError()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
