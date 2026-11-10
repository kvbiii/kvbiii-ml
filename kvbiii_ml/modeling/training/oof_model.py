from __future__ import annotations

from copy import deepcopy
from typing import cast

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from kvbiii_ml._typing import Estimator
from kvbiii_ml._validation import check_fitted
from kvbiii_ml.modeling.training.base_trainer import (
    BINARY_N_CLASSES,
    CATBOOST_PREFIX,
    LGBM_PREFIX,
    PROBA_MATRIX_NDIM,
    PROBABILITY_THRESHOLD,
    TRAIN_VALID_PAIR,
    XGB_PREFIX,
    BaseTrainer,
    FitRows,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

_CALIBRATION_PROBA_EPS = 1e-6
_PROBA_SUM_FLOOR = 1e-9


def _get_n_iterations(estimator: Estimator) -> int | None:
    """Extract the number of boosting iterations actually used by a fitted GBM estimator.

    Args:
        estimator (Estimator): A fitted boosting estimator.

    Returns:
        int | None: Number of trees/rounds used, or None if not determinable.
    """
    class_name = type(estimator).__name__
    if class_name.startswith(LGBM_PREFIX):
        best_iter = getattr(estimator, "best_iteration_", None)
        return (
            int(best_iter)
            if best_iter is not None and best_iter > 0
            else int(getattr(estimator, "n_estimators", 100))
        )
    if class_name.startswith(CATBOOST_PREFIX):
        tree_count = getattr(estimator, "tree_count_", None)
        return int(tree_count) if tree_count is not None else None
    if class_name.startswith(XGB_PREFIX):
        best_iter = getattr(estimator, "best_iteration", None)
        return (
            int(best_iter) + 1
            if best_iter is not None
            else int(getattr(estimator, "n_estimators", 100))
        )
    n_est = getattr(estimator, "n_estimators", None)
    return int(n_est) if n_est is not None else None


def _set_n_iterations(estimator: Estimator, n: int) -> None:
    """Set the number of boosting iterations on an unfitted estimator.

    Args:
        estimator (Estimator): Unfitted estimator to configure.
        n (int): Number of iterations to set.
    """
    class_name = type(estimator).__name__
    if class_name.startswith(CATBOOST_PREFIX):
        estimator.set_params(iterations=n)
    elif class_name.startswith((LGBM_PREFIX, XGB_PREFIX)) or hasattr(estimator, "n_estimators"):
        estimator.set_params(n_estimators=n)


def _disable_early_stopping(estimator: Estimator) -> Estimator:
    """Remove early-stopping configuration so an estimator can train without eval_set.

    Called before final refits on the full dataset where no held-out eval_set is
    available. The number of iterations should already be fixed via _set_n_iterations
    before calling this function. Recurses into EnsembleModel._estimators_list so
    every sub-estimator is also patched.

    Args:
        estimator (Estimator): Estimator to modify in-place (already a deepcopy).

    Returns:
        Estimator: Same estimator with early-stopping disabled.
    """
    if hasattr(estimator, "_estimators_list"):
        for sub in estimator._estimators_list:  # noqa: SLF001  # ty: ignore[not-iterable]
            _disable_early_stopping(sub)
        return estimator
    params = estimator.get_params() if hasattr(estimator, "get_params") else {}
    if "early_stopping_rounds" in params:
        estimator.set_params(early_stopping_rounds=None)
    if "early_stopping_round" in params:
        estimator.set_params(early_stopping_round=None)
    if "early_stopping" in params:
        if type(estimator).__name__.startswith(LGBM_PREFIX):
            estimator.set_params(early_stopping=None)
        else:
            estimator.set_params(early_stopping=False)
    return estimator


def _fit_calibrator(
    oof_probas: np.ndarray,
    y_oof: np.ndarray,
    method: str,
    classes: np.ndarray,
) -> IsotonicRegression | LogisticRegression | list:
    """Fit a calibrator on out-of-fold probabilities.

    Binary: fits a single calibrator on the positive-class column against a
    0/1 indicator for ``classes[1]`` - labels are binarized rather than passed
    through as-is, since IsotonicRegression/LogisticRegression require a
    numeric target and class labels are not guaranteed to already be 0/1.
    Multiclass: fits one calibrator per class (OVR), each against its own
    per-class 0/1 indicator.

    Args:
        oof_probas (np.ndarray): OOF probabilities of shape (n_samples, n_classes).
        y_oof (np.ndarray): OOF true labels of shape (n_samples,).
        method (str): Calibration method, either "isotonic" or "sigmoid".
        classes (np.ndarray): Unique class labels.

    Returns:
        IsotonicRegression | LogisticRegression | list: Fitted calibrator for binary,
            or list of fitted calibrators (one per class) for multiclass.

    Raises:
        ValueError: If method is not "isotonic" or "sigmoid".
    """
    if method not in {"isotonic", "sigmoid"}:
        raise ValueError(f"method must be 'isotonic' or 'sigmoid', got '{method}'.")

    def _make_cal() -> IsotonicRegression | LogisticRegression:
        return (
            IsotonicRegression(out_of_bounds="clip")
            if method == "isotonic"
            else LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        )

    def _fit_one(p: np.ndarray, y: np.ndarray) -> IsotonicRegression | LogisticRegression:
        cal = _make_cal()
        cal.fit(p.reshape(-1, 1) if method == "sigmoid" else p, y)
        return cal

    if len(classes) == BINARY_N_CLASSES:
        return _fit_one(oof_probas[:, 1], (y_oof == classes[1]).astype(int))
    return [_fit_one(oof_probas[:, i], (y_oof == cls).astype(int)) for i, cls in enumerate(classes)]


def _apply_calibrator(
    calibrator: IsotonicRegression | LogisticRegression | list,
    raw_probas: np.ndarray,
    classes: np.ndarray,
    method: str,
) -> np.ndarray:
    """Map raw probabilities through a fitted calibrator.

    Binary: applies calibrator to the positive-class column and reconstructs full matrix.
    Multiclass: applies each OVR calibrator and row-normalises the result.

    Calibrated probabilities are clipped away from 0 and 1 by _CALIBRATION_PROBA_EPS.
    Isotonic regression in particular emits hard 0.0/1.0 values on its extreme bins;
    left unclipped, a single confident-but-wrong prediction makes log loss diverge, and
    the deployed CalibratedModel/OOFModel would score far worse than the out-of-fold
    estimate used for method selection. Clipping keeps both consistent and finite.

    Args:
        calibrator (IsotonicRegression | LogisticRegression | list): Fitted calibrator(s).
        raw_probas (np.ndarray): Raw probabilities of shape (n_samples, n_classes).
        classes (np.ndarray): Unique class labels used during fitting.
        method (str): Calibration method used when fitting, either "isotonic" or "sigmoid".

    Returns:
        np.ndarray: Calibrated probabilities of shape (n_samples, n_classes), rows sum to 1.
    """

    def _apply_one(cal: IsotonicRegression | LogisticRegression, p: np.ndarray) -> np.ndarray:
        return cal.predict_proba(p.reshape(-1, 1))[:, 1] if method == "sigmoid" else cal.predict(p)  # ty: ignore[unresolved-attribute]

    if len(classes) == BINARY_N_CLASSES:
        p_cal = np.clip(
            _apply_one(
                cast("IsotonicRegression | LogisticRegression", calibrator), raw_probas[:, 1]
            ),
            _CALIBRATION_PROBA_EPS,
            1.0 - _CALIBRATION_PROBA_EPS,
        )
        return np.column_stack([1 - p_cal, p_cal])

    cal_matrix = np.column_stack(
        [
            _apply_one(cal, raw_probas[:, i])
            for i, cal in enumerate(
                cast("list[IsotonicRegression | LogisticRegression]", calibrator)
            )
        ]
    )
    cal_matrix = cal_matrix / np.clip(cal_matrix.sum(axis=1, keepdims=True), _PROBA_SUM_FLOOR, None)
    cal_matrix = np.clip(cal_matrix, _CALIBRATION_PROBA_EPS, None)
    return cal_matrix / cal_matrix.sum(axis=1, keepdims=True)


def _collect_oof_probas(
    cross_validator: CrossValidationTrainer,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect out-of-fold probability predictions from a fitted CrossValidationTrainer.

    Applies each fold's fitted pipeline before calling predict_proba, replicating
    the exact transformation applied at training time.

    Args:
        cross_validator (CrossValidationTrainer): Trainer already fitted via fit().
        X (pd.DataFrame): Full feature matrix (unprocessed).
        y (pd.Series): Full target vector.

    Returns:
        tuple[np.ndarray, np.ndarray]: OOF true labels of shape (n_samples,) and
            OOF probabilities of shape (n_samples, n_classes).

    Raises:
        RuntimeError: If no folds were iterated (the cross_validator may not be fitted).
    """
    n_samples = len(y)
    y_arr = y.to_numpy()
    oof_probas: np.ndarray | None = None

    for (_, val_idx), est, fold_pipeline in zip(
        cross_validator.cv.split(X, y),
        cross_validator.fitted_estimators_,
        cross_validator.fitted_pipelines_,
        strict=False,
    ):
        x_val = CrossValidationTrainer._transform_with_pipeline(fold_pipeline, X.iloc[val_idx])  # noqa: SLF001
        x_val = CrossValidationTrainer._order_x_for_estimator(x_val, est)  # noqa: SLF001
        probas = est.predict_proba(x_val)
        if probas.ndim == 1:
            probas = np.column_stack([1 - probas, probas])
        if oof_probas is None:
            oof_probas = np.zeros((n_samples, probas.shape[1]))
        oof_probas[val_idx] = probas

    if oof_probas is None:
        raise RuntimeError("No folds were iterated; cross_validator may not be fitted.")
    return y_arr, oof_probas


class OOFModel:
    """Generate out-of-fold predictions for any estimator, with optional calibration.

    Performs K-fold training to produce leakage-free out-of-fold predictions for
    stacking, model evaluation, or meta-learning. When calibrate=True, also fits a
    probability calibrator on OOF probabilities and refits the base estimator on the
    full training data using the average iteration count from the inner folds (or
    early stopping when eval_set is provided). ``classes_``, ``calibrator_`` and
    ``fitted_estimator_`` exist only after a fit with calibrate=True.
    """

    fitted_estimators_: list[Estimator]
    calibrator_: IsotonicRegression | LogisticRegression | list
    classes_: np.ndarray
    fitted_estimator_: Estimator

    def __init__(
        self,
        estimator: Estimator,
        cross_validator: CrossValidationTrainer,
        problem_type: str,
        calibrate: bool = False,
        calibration_method: str = "isotonic",
    ) -> None:
        """Initialize the OOF generator.

        Notes:
        - calibrate: If True, fit a probability calibrator on OOF probabilities and refit the base
          estimator on the full dataset. Only valid for classification. Defaults to False.

        Args:
            estimator (Estimator): Estimator to train on each fold.
            cross_validator (CrossValidationTrainer): Cross-validation trainer to use.
            problem_type (str): Problem type, either "classification" or "regression".
            calibrate (bool): Calibrate OOF probabilities and refit on all data. Defaults to False.
            calibration_method (str): "isotonic" or "sigmoid". Defaults to "isotonic".

        Raises:
            ValueError: If problem_type is invalid or calibrate=True is used with regression.
        """
        if problem_type not in {"classification", "regression"}:
            raise ValueError("problem_type must be either 'classification' or 'regression'.")
        if calibrate and problem_type != "classification":
            raise ValueError("calibrate=True is only supported for problem_type='classification'.")

        self.estimator = estimator
        self.cross_validator = cross_validator
        self.problem_type = problem_type
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.eval_metric = cross_validator.metric_fn
        self.metric_type = cross_validator.metric_type

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame | None = None,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
    ) -> OOFModel:
        """Fit estimator on each fold and optionally calibrate probabilities.

        When calibrate=True, collects OOF probabilities from the inner CV folds, fits
        a calibrator, then refits the base estimator on the full training set using the
        average iteration count from the folds. If eval_set is provided, early stopping
        is preserved for that final refit; otherwise it is disabled and the fixed
        iteration count is used.

        Notes:
        - eval_set: Validation data for early stopping during the calibrated full refit. Accepts the
          same format as EnsembleModel: [(X_train, y_train), (X_valid, y_valid)] or [(X_valid,
          y_valid)]. Only used when calibrate=True. Defaults to None.

        Args:
            X (pd.DataFrame): Feature matrix aligned with target.
            y (pd.Series): Target vector.
            X_test (pd.DataFrame | None): Test features run through the pipeline per fold.
            eval_set (list[tuple[pd.DataFrame, pd.Series]] | None): Data for early stopping.

        Returns:
            OOFModel: Fitted instance with fitted_estimators_ populated.
        """
        self.cross_validator.fit(self.estimator, X, y, X_test=X_test)
        self.fitted_estimators_ = self.cross_validator.fitted_estimators_

        if self.calibrate:
            y_oof, oof_probas = _collect_oof_probas(self.cross_validator, X, y)
            self.classes_ = np.unique(y_oof)
            self.calibrator_ = _fit_calibrator(
                oof_probas, y_oof, self.calibration_method, self.classes_
            )
            X_valid, y_valid = self._parse_eval_set(eval_set)
            n_iter_list = [
                n for est in self.fitted_estimators_ if (n := _get_n_iterations(est)) is not None
            ]
            final_est = deepcopy(self.estimator)
            if n_iter_list:
                _set_n_iterations(final_est, int(np.mean(n_iter_list)))
            if X_valid is None:
                _disable_early_stopping(final_est)
            self.fitted_estimator_ = BaseTrainer.fit_estimator(
                final_est, FitRows(X, y), FitRows.optional(X_valid, y_valid)
            )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with the calibrated or averaged fold estimators; NotFittedError before fit.

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Predicted labels for classification or values for regression.
        """
        check_fitted(self, "fitted_estimators_")

        if self.calibrate:
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

        preds = []
        for est in self.fitted_estimators_:
            x_ord = CrossValidationTrainer._order_x_for_estimator(X, est)  # noqa: SLF001
            if self.problem_type == "classification" and hasattr(est, "predict_proba"):
                preds.append(est.predict_proba(x_ord))
            else:
                preds.append(est.predict(x_ord))

        avg = np.asarray(preds, dtype=float).mean(axis=0)
        if self.problem_type == "classification":
            return (
                np.argmax(avg, axis=1)
                if avg.ndim == PROBA_MATRIX_NDIM
                else (avg >= PROBABILITY_THRESHOLD).astype(int)
            )
        return avg

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities, calibrated when enabled; NotFittedError before fit.

        When calibrate=True uses the single refitted estimator and applies the
        fitted calibrator. Otherwise averages predict_proba outputs across folds.

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).

        Raises:
            ValueError: If called for a regression problem.
        """
        if self.problem_type != "classification":
            raise ValueError("predict_proba is only available for classification problems.")
        check_fitted(self, "fitted_estimators_")

        if self.calibrate:
            x_ord = CrossValidationTrainer._order_x_for_estimator(X, self.fitted_estimator_)  # noqa: SLF001
            raw = self.fitted_estimator_.predict_proba(x_ord)
            if raw.ndim == 1:
                raw = np.column_stack([1 - raw, raw])
            return _apply_calibrator(self.calibrator_, raw, self.classes_, self.calibration_method)

        probs = [
            est.predict_proba(CrossValidationTrainer._order_x_for_estimator(X, est))  # noqa: SLF001
            for est in self.fitted_estimators_
        ]
        return np.asarray(probs, dtype=float).mean(axis=0)

    @staticmethod
    def _parse_eval_set(
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None,
    ) -> tuple[pd.DataFrame | None, pd.Series | None]:
        """Extract validation data from an eval_set in EnsembleModel format.

        Notes:
        - eval_set: [(X_train, y_train), (X_valid, y_valid)] or [(X_valid, y_valid)], or None.

        Args:
            eval_set (list[tuple] | None): Early-stopping data, or None.

        Returns:
            tuple: (X_valid, y_valid) or (None, None) if eval_set is absent.
        """
        if (
            eval_set is not None
            and isinstance(eval_set, list)
            and len(eval_set) >= 1
            and isinstance(eval_set[0], tuple)
        ):
            return eval_set[1] if len(eval_set) == TRAIN_VALID_PAIR else eval_set[0]
        return None, None


if __name__ == "__main__":
    from lightgbm import LGBMClassifier
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    def _run_demo() -> None:
        """Run OOFModel with and without calibration on a synthetic dataset."""
        x_arr, y_arr = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=17,
        )
        x_df = pd.DataFrame(x_arr, columns=[f"f{i}" for i in range(10)])
        y_ser = pd.Series(y_arr)

        cv = CrossValidationTrainer(
            metric_name="Accuracy",
            problem_type="classification",
            cv=KFold(n_splits=5, shuffle=True, random_state=17),
            verbose=False,
        )

        print("=== OOFModel without calibration ===")
        oof = OOFModel(
            estimator=LogisticRegression(max_iter=1000, solver="liblinear", random_state=17),
            cross_validator=cv,
            problem_type="classification",
        )
        oof.fit(x_df, y_ser)
        print("Fitted estimators:", len(oof.fitted_estimators_))
        print("predict_proba shape:", oof.predict_proba(x_df).shape)

        print("\n=== OOFModel with isotonic calibration ===")
        cv_cal = CrossValidationTrainer(
            metric_name="Log Loss",
            problem_type="classification",
            cv=KFold(n_splits=5, shuffle=True, random_state=17),
            verbose=False,
        )
        oof_cal = OOFModel(
            estimator=LGBMClassifier(n_estimators=50, verbose=-1, random_state=17),
            cross_validator=cv_cal,
            problem_type="classification",
            calibrate=True,
            calibration_method="isotonic",
        )
        oof_cal.fit(x_df, y_ser)
        print("Fitted estimators:", len(oof_cal.fitted_estimators_))
        print("predict_proba shape:", oof_cal.predict_proba(x_df).shape)
        print("calibrator_:", type(oof_cal.calibrator_).__name__)
        print("fitted_estimator_ type:", type(oof_cal.fitted_estimator_).__name__)

    _run_demo()
