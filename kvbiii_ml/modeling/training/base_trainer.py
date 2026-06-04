from inspect import signature

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from lightgbm import log_evaluation
import logging

logging.getLogger("lightgbm").setLevel(logging.ERROR)


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
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
        verbose: bool | int | None = None,
    ) -> BaseEstimator:
        """Fit the estimator, adding common eval kwargs if supported."""
        sig = signature(estimator.fit)
        fit_params = {}

        if verbose is not None:
            if "verbose" in sig.parameters:
                fit_params["verbose"] = verbose
            elif "verbosity" in sig.parameters:
                fit_params["verbosity"] = verbose
            if "callbacks" in sig.parameters:
                if estimator.__class__.__name__.startswith("LGBM"):
                    fit_params["callbacks"] = [log_evaluation(verbose)]

        if "eval_set" in sig.parameters and X_valid is not None and y_valid is not None:
            eval_set = [(X_train, y_train), (X_valid, y_valid)]

            if (
                estimator.__class__.__name__.startswith("CatBoost")
                and str(estimator.get_params().get("task_type", "")).upper() == "GPU"
            ):
                eval_set = eval_set[1:]

            fit_params["eval_set"] = eval_set

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
        verbose: bool | int | None = False,
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
                verbose (bool | int | None): Optional verbosity for fitting. Defaults to False.

        Returns:
                tuple[np.ndarray, np.ndarray, np.ndarray | None, BaseEstimator]:
                        Training predictions, validation predictions, optional test predictions, and the fitted estimator.
        """
        fitted_estimator = BaseTrainer.fit_estimator(
            estimator, X_train, y_train, X_valid, y_valid, verbose=verbose
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
        except (AttributeError, TypeError, ValueError):
            categorical_features = []
        name = estimator.__class__.__name__
        is_fitted = False
        if hasattr(estimator, "is_fitted") and callable(estimator.is_fitted):
            try:
                is_fitted = bool(estimator.is_fitted())
            except Exception:
                is_fitted = False

        if name.startswith("CatBoost"):
            if is_fitted:
                estimator = estimator.__class__(**estimator.get_params())
            estimator = estimator.set_params(cat_features=categorical_features)
        elif name.startswith("HistGradientBoosting"):
            estimator = estimator.set_params(categorical_features=categorical_features)
        return estimator


if __name__ == "__main__":
    print("BaseTrainer module loaded successfully.")
