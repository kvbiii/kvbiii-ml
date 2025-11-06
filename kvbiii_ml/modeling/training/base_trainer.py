from inspect import signature
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class BaseTrainer:
    """Common training utilities shared by trainer classes.

    Provides helpers to fit estimators with optional validation hooks and to
    obtain predictions according to a metric's required prediction type.
    """

    @staticmethod
    def fit_estimator(
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None,
    ) -> BaseEstimator:
        """Fit the estimator, adding common eval kwargs if supported."""
        sig = signature(estimator.fit)
        fit_params = {}

        if "eval_set" in sig.parameters and X_valid is not None:
            fit_params["eval_set"] = [(X_valid, y_valid)]

        if "verbose" in sig.parameters:
            fit_params["verbose"] = False

        if "X_val" in sig.parameters and "y_val" in sig.parameters:
            fit_params["X_val"] = X_valid
            fit_params["y_val"] = y_valid

        if "X_test" in sig.parameters:
            fit_params["X_test"] = X_valid

        if "sample_weight" in sig.parameters:
            fit_params["sample_weight"] = sample_weight

        estimator = BaseTrainer._set_categorical_params(estimator, X_train)
        estimator.fit(X_train, y_train, **fit_params)
        return estimator

    @staticmethod
    def fit_and_predict(
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        metric_type: str = "preds",
        X_test: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, BaseEstimator]:
        """Fit and produce predictions according to metric_type.

        Args:
                estimator (BaseEstimator): Estimator to fit.
                X_train (pd.DataFrame): Training features.
                y_train (pd.Series): Training target.
                X_valid (pd.DataFrame): Validation features.
                y_valid (pd.Series): Validation target.
                metric_type (str): Prediction mode, either "preds" or "probs". Defaults to "preds".
                X_test (pd.DataFrame | None): Optional test features to predict. Defaults to None.

        Returns:
                tuple[np.ndarray, np.ndarray, np.ndarray | None, BaseEstimator]:
                        Training predictions, validation predictions, optional test predictions, and the fitted estimator.
        """
        fitted_estimator = BaseTrainer.fit_estimator(
            estimator, X_train, y_train, X_valid, y_valid
        )

        if metric_type == "preds":
            y_train_pred = BaseTrainer.predict(fitted_estimator, X_train)
            y_valid_pred = BaseTrainer.predict(fitted_estimator, X_valid)
            y_test_pred = BaseTrainer.predict(fitted_estimator, X_test)
        elif metric_type == "probs":
            y_train_pred = BaseTrainer.predict_proba(fitted_estimator, X_train)
            y_valid_pred = BaseTrainer.predict_proba(fitted_estimator, X_valid)
            y_test_pred = BaseTrainer.predict_proba(fitted_estimator, X_test)
        else:
            raise ValueError(
                f"Unknown metric type '{metric_type}'. Must be 'preds' or 'probs'."
            )

        return y_train_pred, y_valid_pred, y_test_pred, fitted_estimator

    @staticmethod
    def predict(
        estimator: BaseEstimator,
        X: pd.DataFrame | None,
    ) -> np.ndarray | None:
        """Safely call predict if X is provided, else return None."""
        if X is None:
            return None
        return estimator.predict(X)

    @staticmethod
    def predict_proba(
        estimator: BaseEstimator,
        X: pd.DataFrame | None,
    ) -> np.ndarray | None:
        """Safely call predict_proba if X is provided; optionally return positive class only for binary."""
        if X is None:
            return None
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    @staticmethod
    def _set_categorical_params(
        estimator: BaseEstimator, X_train: pd.DataFrame
    ) -> BaseEstimator:
        """Assign categorical feature parameters for supported estimators.

        Args:
            estimator (BaseEstimator): Estimator to configure.
            X_train (pd.DataFrame): Training features to detect categorical cols.

        Returns:
            BaseEstimator: Estimator with categorical parameters set when
            applicable.
        """
        try:
            categorical_features = X_train.select_dtypes(
                include="category"
            ).columns.tolist()
        except Exception:
            categorical_features = []
        name = estimator.__class__.__name__
        if name.startswith("CatBoost"):
            estimator = estimator.set_params(cat_features=categorical_features)
        elif name.startswith("HistGradientBoosting"):
            estimator = estimator.set_params(categorical_features=categorical_features)
        return estimator
