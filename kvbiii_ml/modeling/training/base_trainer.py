import logging
from copy import deepcopy
from inspect import Signature, signature

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from lightgbm import log_evaluation

logging.getLogger("lightgbm").setLevel(logging.ERROR)

LGBM_PREFIX = "LGBM"
XGB_PREFIX = "XGB"
CATBOOST_PREFIX = "CatBoost"
HISTGB_PREFIX = "HistGradientBoosting"
VERBOSE_LOG_PERIOD = 50
XGB_GPU_DEVICE_PREFIX = "cuda"
LGBM_GPU_DEVICE_PREFIX = "cuda"


class BaseTrainer:
    """Common training utilities shared by trainer classes.

    Provides helpers to fit estimators with optional validation hooks and to
    obtain predictions according to a metric's required prediction type.
    """

    @staticmethod
    def _align_xgb_device_to_cpu(estimator: BaseEstimator) -> BaseEstimator:
        """Realign a GPU-trained XGBoost estimator's inference device to CPU.

        XGBoost's sklearn wrapper predicts using whatever device is stored in its
        `device` parameter. This package always calls predict/predict_proba with
        CPU-resident pandas DataFrames, so a booster left configured with
        `device="cuda"` after training makes XGBoost's `inplace_predict` detect a
        mismatched device on every later call: it emits a UserWarning and falls
        back to a slower DMatrix-based prediction with higher memory usage.
        Realigning the already-fitted estimator's device to "cpu" right after
        training keeps GPU speed during fit while making every later prediction
        call match, at no cost since prediction never receives GPU-resident data
        in this pipeline.

        Args:
            estimator (BaseEstimator): Fitted estimator to inspect and adjust.

        Returns:
            BaseEstimator: The same estimator with `device="cpu"` set when it was
                configured for a CUDA device; unchanged otherwise.
        """
        if not type(estimator).__name__.startswith(XGB_PREFIX):
            return estimator
        device = str(estimator.get_params().get("device") or "").lower()
        if device.startswith(XGB_GPU_DEVICE_PREFIX):
            estimator = estimator.set_params(device="cpu")
        return estimator

    @staticmethod
    def _lgbm_cuda_categorical_override(estimator: BaseEstimator, sig: Signature) -> dict:
        """Disable LightGBM's categorical auto-detection when training on CUDA.

        LightGBM's CUDA tree learner has a reproducible illegal-memory-access crash
        in its categorical split finder (`cuda_best_split_finder.cu`) when it
        auto-detects a pandas "category"-dtype column with very high cardinality
        (thousands of distinct values, e.g. a raw ID-like column) as a categorical
        feature. The CPU tree learner handles the exact same columns without issue,
        so this override only fires for a CUDA-configured estimator; CPU-trained
        LGBM keeps its normal `categorical_feature="auto"` detection. Passing an
        empty list makes LightGBM treat every column as numeric for split-finding,
        which is safe here because a "WithOriginal" categorical column always has a
        frequency/target-encoded numeric counterpart already carrying its signal.

        Args:
            estimator (BaseEstimator): Estimator being fit.
            sig (Signature): Signature of `estimator.fit`, used to confirm the
                estimator actually accepts a `categorical_feature` argument.

        Returns:
            dict: `{"categorical_feature": []}` when the override applies, else `{}`.
        """
        if not estimator.__class__.__name__.startswith(LGBM_PREFIX):
            return {}
        if "categorical_feature" not in sig.parameters:
            return {}
        params = estimator.get_params()
        device = str(params.get("device") or params.get("device_type") or "").lower()
        if not device.startswith(LGBM_GPU_DEVICE_PREFIX):
            return {}
        return {"categorical_feature": []}

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
        features are auto-detected from X_valid column dtypes. Any truthy `verbose`
        is capped to once every `VERBOSE_LOG_PERIOD` rounds via `_resolve_verbose_period`
        rather than logging every round.

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
        log_period = BaseTrainer._resolve_verbose_period(verbose)
        if LGBM_PREFIX in class_name:

            fit_kwargs: dict = {"callbacks": [log_evaluation(log_period)]}
            if "eval_X" in signature(estimator.fit).parameters:
                fit_kwargs["eval_X"] = X_valid
                fit_kwargs["eval_y"] = y_valid
            else:
                fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            cat_cols = X_valid.select_dtypes(
                include=["category", "object"]
            ).columns.tolist()
            if cat_cols:
                fit_kwargs["categorical_feature"] = cat_cols
            return fit_kwargs

        if XGB_PREFIX in class_name:
            return {"eval_set": [(X_valid, y_valid)], "verbose": log_period}

        if CATBOOST_PREFIX in class_name:
            return {"eval_set": (X_valid, y_valid), "verbose": log_period}

        return {}

    @staticmethod
    def _resolve_verbose_period(verbose: bool | int | None) -> bool | int:
        """Cap a truthy verbose request to a fixed once-per-`VERBOSE_LOG_PERIOD` cadence.

        LightGBM, XGBoost, and CatBoost all print a log line on every single
        boosting round once verbose logging is truthy, which floods output for
        models trained with hundreds or thousands of estimators. All three
        libraries accept an int for their own periodic-verbosity fit parameter
        (LightGBM's ``log_evaluation(period=...)`` callback, XGBoost's and
        CatBoost's ``verbose=N`` meaning "print every N rounds"), so any truthy
        `verbose` is mapped to the fixed `VERBOSE_LOG_PERIOD` instead of being
        passed through as-is - the period is intentionally not caller-configurable.

        Args:
            verbose (bool | int | None): Verbosity as requested by the caller.

        Returns:
            bool | int: `VERBOSE_LOG_PERIOD` when verbose is truthy; `False`
                otherwise (covers an explicit falsy verbose and an unset `None`).
        """
        return VERBOSE_LOG_PERIOD if verbose else False

    @staticmethod
    def fit_estimator(
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
        verbose: bool | int | None = None,
        eval_on_train: bool = True,
    ) -> BaseEstimator:
        """Fit the estimator, injecting eval_set and suppressing output for boosting families.

        For LightGBM, XGBoost, and CatBoost the fit kwargs are built via
        ``_build_fit_kwargs`` which injects only the validation set as eval_set
        (training set excluded) and silences all output. Early stopping must be
        configured in the estimator constructor. For all other estimators the
        existing verbose/verbosity signature-inspection path is used. Any truthy
        `verbose` is capped to once every `VERBOSE_LOG_PERIOD` rounds for
        LightGBM/XGBoost/CatBoost via `_resolve_verbose_period`, rather than
        logging every round. HistGradientBoosting has no fit-time verbose
        parameter and no periodic-logging support at all (scikit-learn only
        offers a coarse 0/1/2 verbosity level set at construction time), so a
        truthy `verbose` instead sets its constructor-level `verbose=1`. After
        fitting, an XGBoost estimator configured with a CUDA `device` has its
        inference device realigned to CPU via `_align_xgb_device_to_cpu`, since
        every downstream predict call in this package receives CPU-resident
        pandas data. An LGBM estimator configured with a CUDA `device`/`device_type`
        has `categorical_feature=[]` injected via `_lgbm_cuda_categorical_override`
        unless the caller already set `categorical_feature` in `fit_params`,
        working around a reproducible LightGBM CUDA illegal-memory-access crash
        on high-cardinality auto-detected categorical columns (CPU training is
        unaffected and keeps auto-detection).

        Args:
            estimator (BaseEstimator): Estimator to fit.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            X_valid (pd.DataFrame | None): Validation features for eval_set. Defaults to None.
            y_valid (pd.Series | None): Validation target for eval_set. Defaults to None.
            sample_weight (pd.Series | None): Optional per-sample weights. Defaults to None.
            verbose (bool | int | None): Verbosity for non-boosting estimators. Defaults to None.
            eval_on_train (bool): Whether the training split is included in eval_set
                alongside the validation split. Boosting libraries score every eval_set
                entry on every iteration, so including a large training split is pure
                overhead for callers that only early-stop and never read the training
                curve. Defaults to True, preserving the historical behaviour for every
                existing caller.

        Returns:
            BaseEstimator: Fitted estimator instance.
        """
        estimator = BaseTrainer._set_categorical_params(estimator, X_train)

        sig = signature(estimator.fit)
        fit_params = {}

        if verbose is not None:
            log_period = BaseTrainer._resolve_verbose_period(verbose)
            if "verbose" in sig.parameters:
                fit_params["verbose"] = log_period
            elif "verbosity" in sig.parameters:
                fit_params["verbosity"] = verbose
            if "callbacks" in sig.parameters:
                if estimator.__class__.__name__.startswith(LGBM_PREFIX):
                    fit_params["callbacks"] = [log_evaluation(log_period)]
            if (
                estimator.__class__.__name__.startswith(HISTGB_PREFIX)
                and "verbose" in estimator.get_params()
            ):
                estimator = estimator.set_params(verbose=1 if verbose else 0)

        is_lgbm = estimator.__class__.__name__.startswith(LGBM_PREFIX)
        if (
            is_lgbm
            and "eval_X" in sig.parameters
            and "eval_y" in sig.parameters
            and X_valid is not None
            and y_valid is not None
        ):
            fit_params["eval_X"] = (X_train, X_valid) if eval_on_train else (X_valid,)
            fit_params["eval_y"] = (y_train, y_valid) if eval_on_train else (y_valid,)
        elif (
            "eval_set" in sig.parameters and X_valid is not None and y_valid is not None
        ):
            eval_set = (
                [(X_train, y_train), (X_valid, y_valid)]
                if eval_on_train
                else [(X_valid, y_valid)]
            )

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
        if "categorical_feature" not in fit_params:
            fit_params.update(
                BaseTrainer._lgbm_cuda_categorical_override(estimator, sig)
            )
        estimator.fit(X_train, y_train, **fit_params)
        estimator = BaseTrainer._align_xgb_device_to_cpu(estimator)
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
        sample_weight: pd.Series | np.ndarray | None = None,
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
            sample_weight (pd.Series | np.ndarray | None): Optional per-sample training
                weights, forwarded to ``fit_estimator`` and from there to any estimator
                whose ``fit`` accepts ``sample_weight``. Must already be restricted and
                aligned to ``X_train``; this method does no slicing of its own. Defaults
                to None.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray | None, BaseEstimator]:
                Training predictions, validation predictions, optional test predictions,
                and the fitted estimator.
        """
        fitted_estimator = BaseTrainer.fit_estimator(
            estimator,
            X_train,
            y_train,
            X_valid,
            y_valid,
            sample_weight=sample_weight,
            verbose=verbose,
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
