from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from kvbiii_ml.modeling.training.base_trainer import BaseTrainer
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer
from kvbiii_ml.modeling.training.oof_model import (
    OOFModel,
    _apply_calibrator,
    _disable_early_stopping,
    _fit_calibrator,
    _get_n_iterations,
    _set_n_iterations,
)

_DEFAULT_SEED = 17
_DEFAULT_CALIBRATION_SIZE = 0.2


class CalibratedModel:
    """Wrap any classifier with a post-hoc probability calibration layer.

    A lightweight, single-split alternative to ``OOFModel(calibrate=True)``. On
    ``fit`` it carves out a single stratified hold-out, trains the base estimator on
    the remaining data, fits a probability calibrator on the hold-out predictions,
    then refits the base estimator on the full training data (reusing the hold-out
    model's iteration count, or early stopping when an eval_set is supplied). At
    inference it applies the calibrator on top of the full-data estimator's
    probabilities.

    Unlike ``OOFModel``, fitting performs no inner cross-validation. When fed into
    ``CrossValidationTrainer.fit`` this keeps the total cost linear in the number of
    outer folds instead of quadratic, which matters for large datasets and for
    black-box estimators such as ``EnsembleModel``. The base estimator is treated as
    a black box via ``fit``/``predict_proba``, so single models and ensembles are
    supported identically.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        problem_type: str = "classification",
        calibration_method: str = "isotonic",
        calibration_size: float = _DEFAULT_CALIBRATION_SIZE,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        """Initialize the calibrated wrapper.

        Args:
            estimator (BaseEstimator): Unfitted estimator or EnsembleModel to calibrate.
            problem_type (str): Problem type. Only "classification" is supported.
                Defaults to "classification".
            calibration_method (str): Calibration method, "isotonic" or "sigmoid".
                Defaults to "isotonic".
            calibration_size (float): Fraction of the training data held out to fit the
                calibrator. Defaults to 0.2.
            seed (int): Random seed for the hold-out split. Defaults to 17.

        Raises:
            ValueError: If problem_type is not "classification", calibration_method is
                not "isotonic"/"sigmoid", or calibration_size is not in (0, 1).
        """
        if problem_type != "classification":
            raise ValueError(
                "CalibratedModel only supports problem_type='classification'."
            )
        if calibration_method not in {"isotonic", "sigmoid"}:
            raise ValueError("calibration_method must be 'isotonic' or 'sigmoid'.")
        if not 0.0 < calibration_size < 1.0:
            raise ValueError("calibration_size must lie in the open interval (0, 1).")

        self.estimator = estimator
        self.problem_type = problem_type
        self.calibration_method = calibration_method
        self.calibration_size = calibration_size
        self.seed = seed

        self.classes_: np.ndarray | None = None
        self.calibrator_: IsotonicRegression | LogisticRegression | list | None = None
        self.fitted_estimator_: BaseEstimator | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
    ) -> "CalibratedModel":
        """Fit the calibrator on a hold-out split and refit the base estimator fully.

        Args:
            X (pd.DataFrame): Training features. Expected already preprocessed when used
                inside CrossValidationTrainer (which applies its pipeline per fold).
            y (pd.Series): Training target.
            eval_set (list[tuple[pd.DataFrame, pd.Series]] | None): Validation data for
                early stopping during the full-data refit, in EnsembleModel format
                [(X_train, y_train), (X_valid, y_valid)] or [(X_valid, y_valid)]. When
                used inside CrossValidationTrainer the fold's validation split is injected
                automatically. Defaults to None.

        Returns:
            CalibratedModel: Fitted instance.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(np.asarray(y).ravel())

        self.classes_ = np.unique(y.to_numpy())

        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X,
            y,
            test_size=self.calibration_size,
            random_state=self.seed,
            stratify=y,
        )

        holdout_estimator = BaseTrainer.fit_estimator(
            deepcopy(self.estimator), X_fit, y_fit, X_cal, y_cal
        )
        cal_probas = self._ensure_proba_matrix(holdout_estimator.predict_proba(X_cal))
        self.calibrator_ = _fit_calibrator(
            cal_probas, y_cal.to_numpy(), self.calibration_method, self.classes_
        )

        self._fit_full_estimator(X, y, holdout_estimator, eval_set)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated class predictions.

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated class probabilities from the full-data estimator.

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Calibrated probabilities of shape (n_samples, n_classes).

        Raises:
            RuntimeError: If called before fit().
        """
        if self.fitted_estimator_ is None:
            raise RuntimeError(
                "CalibratedModel must be fitted before calling predict_proba()."
            )

        X_ord = CrossValidationTrainer._order_X_for_estimator(X, self.fitted_estimator_)
        raw = self._ensure_proba_matrix(self.fitted_estimator_.predict_proba(X_ord))
        return _apply_calibrator(
            self.calibrator_, raw, self.classes_, self.calibration_method
        )

    @staticmethod
    def _ensure_proba_matrix(probas: np.ndarray) -> np.ndarray:
        """Coerce a probability output to a 2-D (n_samples, n_classes) matrix."""
        probas = np.asarray(probas, dtype=float)
        if probas.ndim == 1:
            return np.column_stack([1.0 - probas, probas])
        return probas

    def _fit_full_estimator(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        holdout_estimator: BaseEstimator,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None,
    ) -> None:
        """Refit the base estimator on the full data for inference.

        Reuses the iteration count discovered by the hold-out estimator and preserves
        early stopping when an eval_set is supplied; otherwise early stopping is
        disabled and the fixed iteration count is used.

        Args:
            X (pd.DataFrame): Full training features.
            y (pd.Series): Full training target.
            holdout_estimator (BaseEstimator): Estimator fitted on the hold-out split.
            eval_set (list[tuple] | None): Validation data for early stopping, or None.
        """
        X_valid, y_valid = OOFModel._parse_eval_set(eval_set)
        n_iter = _get_n_iterations(holdout_estimator)
        final_est = deepcopy(self.estimator)
        if n_iter is not None:
            _set_n_iterations(final_est, n_iter)
        if X_valid is None:
            _disable_early_stopping(final_est)
        self.fitted_estimator_ = BaseTrainer.fit_estimator(
            final_est, X, y, X_valid, y_valid
        )


if __name__ == "__main__":
    import time

    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedKFold

    from kvbiii_ml.modeling.training.ensemble_model import EnsembleModel

    RANDOM_STATE = 42
    N_SAMPLES = 3_000
    N_FEATURES = 10
    N_FOLDS = 3
    ES = 30
    FEATURE_NAMES = [f"feature_{i}" for i in range(N_FEATURES)]

    X_arr, y_arr = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=RANDOM_STATE,
    )
    X_df = pd.DataFrame(X_arr, columns=FEATURE_NAMES)
    y_ser = pd.Series(y_arr, name="target")

    estimators = [
        LGBMClassifier(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=-1,
            random_state=RANDOM_STATE,
        ),
        CatBoostClassifier(
            iterations=200,
            early_stopping_rounds=ES,
            verbose=0,
            random_state=RANDOM_STATE + 2,
        ),
        CatBoostClassifier(
            iterations=200,
            early_stopping_rounds=ES,
            verbose=0,
            random_state=RANDOM_STATE + 3,
        ),
    ]
    ensemble = EnsembleModel(estimators=estimators, problem_type="classification")
    cv = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Log Loss",
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        verbose=False,
    )

    calibrated = CalibratedModel(
        estimator=ensemble, calibration_method="isotonic", seed=RANDOM_STATE
    )

    print("=== CalibratedModel(EnsembleModel) inside CrossValidationTrainer ===")
    start = time.perf_counter()
    _, valid_scores, _ = cv.fit(calibrated, X_df, y_ser)
    elapsed = time.perf_counter() - start
    print(
        f"  Calibrated CV valid: {np.mean(valid_scores):.4f} ± {np.std(valid_scores):.4f}"
    )
    print(f"  Time for CV fit: {elapsed:.2f} seconds")

    calibrated.fit(X_df, y_ser)
    print(f"  predict_proba shape: {calibrated.predict_proba(X_df).shape}")
    print(f"  calibrator_ type: {type(calibrated.calibrator_).__name__}")
    print(f"  fitted_estimator_ type: {type(calibrated.fitted_estimator_).__name__}")
