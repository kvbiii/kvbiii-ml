from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from kvbiii_ml._typing import Estimator
from kvbiii_ml._validation import check_fitted
from kvbiii_ml.modeling.training.base_trainer import BaseTrainer, FitRows
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

    classes_: np.ndarray
    calibrator_: IsotonicRegression | LogisticRegression | list
    fitted_estimator_: Estimator

    def __init__(
        self,
        estimator: Estimator,
        problem_type: str = "classification",
        calibration_method: str = "isotonic",
        calibration_size: float = _DEFAULT_CALIBRATION_SIZE,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        """Initialize the calibrated wrapper.

        Args:
            estimator (Estimator): Unfitted estimator or EnsembleModel to calibrate.
            problem_type (str): Only "classification" is supported. Defaults to "classification".
            calibration_method (str): "isotonic" or "sigmoid". Defaults to "isotonic".
            calibration_size (float): Held-out share for the calibrator. Defaults to 0.2.
            seed (int): Random seed for the hold-out split. Defaults to 17.

        Raises:
            ValueError: If problem_type, calibration_method or calibration_size is invalid.
        """
        if problem_type != "classification":
            raise ValueError("CalibratedModel only supports problem_type='classification'.")
        if calibration_method not in {"isotonic", "sigmoid"}:
            raise ValueError("calibration_method must be 'isotonic' or 'sigmoid'.")
        if not 0.0 < calibration_size < 1.0:
            raise ValueError("calibration_size must lie in the open interval (0, 1).")

        self.estimator = estimator
        self.problem_type = problem_type
        self.calibration_method = calibration_method
        self.calibration_size = calibration_size
        self.seed = seed

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
    ) -> CalibratedModel:
        """Fit the calibrator on a hold-out split and refit the base estimator fully.

        Notes:
        - eval_set: Validation data for early stopping during the full-data refit, in EnsembleModel
          format [(X_train, y_train), (X_valid, y_valid)] or [(X_valid, y_valid)]. When used inside
          CrossValidationTrainer the fold's validation split is injected automatically. Defaults to
          None.

        Args:
            X (pd.DataFrame): Training features; already preprocessed inside CrossValidationTrainer.
            y (pd.Series): Training target.
            eval_set (list[tuple[pd.DataFrame, pd.Series]] | None): Data for early stopping.

        Returns:
            CalibratedModel: Fitted instance.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(np.asarray(y).ravel())

        self.classes_ = np.unique(y.to_numpy())

        x_fit, x_cal, y_fit, y_cal = train_test_split(
            X,
            y,
            test_size=self.calibration_size,
            random_state=self.seed,
            stratify=y,
        )

        holdout_estimator = BaseTrainer.fit_estimator(
            deepcopy(self.estimator), FitRows(x_fit, y_fit), FitRows(x_cal, y_cal)
        )
        cal_probas = self._ensure_proba_matrix(holdout_estimator.predict_proba(x_cal))
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
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probabilities of the full-data estimator; NotFittedError before fit.

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Calibrated probabilities of shape (n_samples, n_classes).
        """
        check_fitted(self, "fitted_estimator_")

        x_ord = CrossValidationTrainer._order_x_for_estimator(X, self.fitted_estimator_)  # noqa: SLF001
        raw = self._ensure_proba_matrix(self.fitted_estimator_.predict_proba(x_ord))
        return _apply_calibrator(self.calibrator_, raw, self.classes_, self.calibration_method)

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
        holdout_estimator: Estimator,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None,
    ) -> None:
        """Refit the base estimator on the full data for inference.

        Reuses the iteration count discovered by the hold-out estimator and preserves
        early stopping when an eval_set is supplied; otherwise early stopping is
        disabled and the fixed iteration count is used.

        Args:
            X (pd.DataFrame): Full training features.
            y (pd.Series): Full training target.
            holdout_estimator (Estimator): Estimator fitted on the hold-out split.
            eval_set (list[tuple] | None): Validation data for early stopping, or None.
        """
        X_valid, y_valid = OOFModel._parse_eval_set(eval_set)  # noqa: SLF001
        n_iter = _get_n_iterations(holdout_estimator)
        final_est = deepcopy(self.estimator)
        if n_iter is not None:
            _set_n_iterations(final_est, n_iter)
        if X_valid is None:
            _disable_early_stopping(final_est)
        self.fitted_estimator_ = BaseTrainer.fit_estimator(
            final_est, FitRows(X, y), FitRows.optional(X_valid, y_valid)
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

    def _run_demo() -> None:
        """Run CalibratedModel(EnsembleModel) inside a CrossValidationTrainer."""
        x_arr, y_arr = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=6,
            n_redundant=2,
            n_classes=2,
            n_clusters_per_class=1,
            random_state=RANDOM_STATE,
        )
        x_df = pd.DataFrame(x_arr, columns=FEATURE_NAMES)
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
                allow_writing_files=False,
            ),
            CatBoostClassifier(
                iterations=200,
                early_stopping_rounds=ES,
                verbose=0,
                random_state=RANDOM_STATE + 3,
                allow_writing_files=False,
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
        _, valid_scores, _ = cv.fit(calibrated, x_df, y_ser)  # ty: ignore[invalid-argument-type]
        elapsed = time.perf_counter() - start
        print(f"  Calibrated CV valid: {np.mean(valid_scores):.4f} ± {np.std(valid_scores):.4f}")
        print(f"  Time for CV fit: {elapsed:.2f} seconds")

        calibrated.fit(x_df, y_ser)
        print(f"  predict_proba shape: {calibrated.predict_proba(x_df).shape}")
        print(f"  calibrator_ type: {type(calibrated.calibrator_).__name__}")
        print(f"  fitted_estimator_ type: {type(calibrated.fitted_estimator_).__name__}")

    _run_demo()
