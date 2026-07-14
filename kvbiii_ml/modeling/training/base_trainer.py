import logging
from copy import deepcopy
from inspect import signature

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from lightgbm import log_evaluation

logging.getLogger("lightgbm").setLevel(logging.ERROR)

LGBM_PREFIX = "LGBM"
XGB_PREFIX = "XGB"
CATBOOST_PREFIX = "CatBoost"
HISTGB_PREFIX = "HistGradientBoosting"


class BaseTrainer:
    """Common training utilities shared by trainer classes.

    Provides helpers to fit estimators with optional validation hooks and to
    obtain predictions according to a metric's required prediction type.
    """

    @staticmethod
    def _build_fit_kwargs(
        estimator: BaseEstimator,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        verbose: bool | int | None = False,
    ) -> dict:
        """Build estimator-specific fit kwargs for eval_set injection and output suppression.

        Early stopping must be configured at the estimator constructor level
        (e.g. ``LGBMClassifier(early_stopping_rounds=50)``). This method only injects
        the fold-specific eval_set and suppresses verbose output. LightGBM categorical
        features are auto-detected from X_valid column dtypes.

        Args:
            estimator (BaseEstimator): Unfitted sklearn-compatible estimator.
            X_valid (pd.DataFrame): Validation features to use as eval_set.
            y_valid (pd.Series): Validation target to use as eval_set.
            verbose (bool | int | None): Verbosity level for non-boosting estimators. Defaults to False.
        Returns:
            dict: Keyword arguments to pass to estimator.fit(), or empty dict for
                estimators not in the three recognised boosting families.
        """
        class_name = type(estimator).__name__
        if LGBM_PREFIX in class_name:

            fit_kwargs: dict = {
                "eval_set": [(X_valid, y_valid)],
                "callbacks": [log_evaluation(verbose)],
            }
            cat_cols = X_valid.select_dtypes(
                include=["category", "object"]
            ).columns.tolist()
            if cat_cols:
                fit_kwargs["categorical_feature"] = cat_cols
            return fit_kwargs

        if XGB_PREFIX in class_name:
            return {"eval_set": [(X_valid, y_valid)], "verbose": verbose}

        if CATBOOST_PREFIX in class_name:
            return {"eval_set": (X_valid, y_valid), "verbose": verbose}

        return {}

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
        """Fit the estimator, injecting eval_set and suppressing output for boosting families.

        For LightGBM, XGBoost, and CatBoost the fit kwargs are built via
        ``_build_fit_kwargs`` which injects only the validation set as eval_set
        (training set excluded) and silences all output. Early stopping must be
        configured in the estimator constructor. For all other estimators the
        existing verbose/verbosity signature-inspection path is used.

        Args:
            estimator (BaseEstimator): Estimator to fit.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            X_valid (pd.DataFrame | None): Validation features for eval_set. Defaults to None.
            y_valid (pd.Series | None): Validation target for eval_set. Defaults to None.
            sample_weight (pd.Series | None): Optional per-sample weights. Defaults to None.
            verbose (bool | int | None): Verbosity for non-boosting estimators. Defaults to None.

        Returns:
            BaseEstimator: Fitted estimator instance.
        """
        estimator = BaseTrainer._set_categorical_params(estimator, X_train)

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

        if (
            estimator.__class__.__name__.startswith(XGB_PREFIX)
            and "verbose" not in fit_params
            and "verbose" in sig.parameters
        ):
            fit_params["verbose"] = False

        if "X_val" in sig.parameters and "y_val" in sig.parameters:
            fit_params["X_val"] = X_valid
            fit_params["y_val"] = y_valid

        if "X_test" in sig.parameters:
            fit_params["X_test"] = X_valid

        if "sample_weight" in sig.parameters:
            fit_params["sample_weight"] = sample_weight

        if (
            estimator.__class__.__name__.startswith(XGB_PREFIX)
            and "eval_set" not in fit_params
        ):
            try:
                if estimator.get_params().get("early_stopping_rounds") is not None:
                    estimator.set_params(early_stopping_rounds=None)
            except (AttributeError, ValueError, KeyError) as e:
                logging.warning(
                    "Could not check early_stopping_rounds for XGB estimator; proceeding without modification. Exception: %s",
                    e,
                )

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
                Training predictions, validation predictions, optional test predictions,
                and the fitted estimator.
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
        """Safely call predict if X is provided, else return None.

        Args:
            estimator (BaseEstimator): Fitted estimator.
            X (pd.DataFrame | None): Features to predict on.

        Returns:
            np.ndarray | None: Predictions or None.
        """
        if X is None:
            return None
        return estimator.predict(X)

    @staticmethod
    def predict_proba(
        estimator: BaseEstimator,
        X: pd.DataFrame | None,
    ) -> np.ndarray | None:
        """Safely call predict_proba if X is provided; returns full probability matrix.

        Returns the full (n_samples, n_classes) matrix for both binary and multiclass
        problems. Callers that need only the positive-class column for binary
        classification must slice [:, 1] themselves.

        Args:
            estimator (BaseEstimator): Fitted estimator.
            X (pd.DataFrame | None): Features to predict on.

        Returns:
            np.ndarray | None: Full probability matrix or None.
        """
        if X is None:
            return None
        return estimator.predict_proba(X)

    @staticmethod
    def _set_categorical_params(
        estimator: BaseEstimator, X_train: pd.DataFrame
    ) -> BaseEstimator:
        """Assign categorical feature parameters for supported estimators.

        For CatBoost: only auto-detects from X_train dtypes when ``cat_features``
        is not already set in the constructor. For HistGradientBoosting: always
        auto-detects. LightGBM categorical features are handled separately via
        ``_build_fit_kwargs`` and are not touched here.

        Args:
            estimator (BaseEstimator): Estimator to configure.
            X_train (pd.DataFrame): Training features to detect categorical cols.

        Returns:
            BaseEstimator: Estimator with categorical parameters set when applicable.
        """
        try:
            categorical_features = X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        except (AttributeError, TypeError, ValueError):
            categorical_features = []

        class_name = type(estimator).__name__

        if class_name.startswith(CATBOOST_PREFIX):
            existing_cats = estimator.get_params().get("cat_features")
            if existing_cats:
                return estimator
            if categorical_features:
                estimator = deepcopy(estimator)
                estimator.set_params(cat_features=categorical_features)
        elif class_name.startswith(HISTGB_PREFIX):
            estimator = estimator.set_params(categorical_features=categorical_features)

        return estimator
