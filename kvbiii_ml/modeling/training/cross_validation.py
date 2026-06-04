import sys
from copy import deepcopy
from inspect import signature
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, KFold
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[3]))
from kvbiii_ml.evaluation.custom_metrics_handler import CustomMetricsHandler
from kvbiii_ml.evaluation.metrics import (
    METRICS,
    get_metric_direction,
    get_metric_function,
    get_metric_type,
)
from kvbiii_ml.modeling.training.base_trainer import BaseTrainer


class CrossValidationTrainer(BaseTrainer):
    """Cross-validation trainer with a per-fold processor pipeline.

    This class orchestrates K-Fold style training loops, optionally applying a
    sequence of processors (objects exposing fit/fit_transform and transform)
    inside each fold to avoid data leakage.
    """

    def __init__(
        self,
        problem_type: str,
        metric_name: str | None = None,
        cv: BaseCrossValidator | None = None,
        processors: list[object] | None = None,
        verbose: bool | int = True,
        custom_metric: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            problem_type (str): Problem type, either "classification" or "regression".
            metric_name (str | None): Metric name defined in METRICS or None if using a custom metric.
            cv (BaseCrossValidator | None): Cross-validation splitter. Defaults to 5-fold KFold with shuffling.
            processors (list[object] | None): Optional list of processors applied per fold. Defaults to None.
            verbose (bool | int): Verbosity flag. If set to integer 1, estimator fit verbosity is mapped to
                True for estimators that accept the fit-time verbose parameter. Defaults to True.
            custom_metric (dict[str, Any] | None): Custom metric configuration with keys 'name', 'function',
                and optionally 'metric_type', 'additional_data', and other kwargs. Defaults to None.

        Raises:
            ValueError: If problem_type is invalid or neither metric_name nor custom_metric is provided.
        """
        if problem_type not in {"classification", "regression"}:
            raise ValueError("problem_type must be 'classification' or 'regression'.")

        if cv is None:
            cv = KFold(n_splits=5, shuffle=True, random_state=17)

        if metric_name is None and custom_metric is None:
            raise ValueError("Either metric_name or custom_metric must be provided.")

        if metric_name is not None:
            if metric_name not in METRICS:
                raise ValueError(
                    f"Metric '{metric_name}' not found in available metrics: {list(METRICS.keys())}"
                )
            self.metric_fn = get_metric_function(metric_name)
            self.metric_type = get_metric_type(metric_name)
            self.metric_direction = get_metric_direction(metric_name)
            self.metric_name = metric_name

        if custom_metric is not None:
            (
                self.metric_name,
                self.metric_fn,
                self.metric_type,
                self.metric_direction,
            ) = CustomMetricsHandler.extract_metric_details(custom_metric)

        self.problem_type = problem_type
        self.cv = cv
        self.processors = processors or []
        self.verbose = verbose
        self.fit_verbose = self._resolve_fit_verbose(verbose)

        self.validate_processors()

        self.train_scores_: list[float] = []
        self.valid_scores_: list[float] = []
        self.fitted_estimators_: list[BaseEstimator] = []
        self.fitted_processors_: list[list[object]] = []

    @staticmethod
    def _resolve_fit_verbose(verbose: bool | int) -> bool | int | None:
        """Map cross-validation verbosity to estimator fit verbosity.

        Args:
            verbose (bool | int): Trainer verbosity setting.

        Returns:
            bool | int | None: Value passed to estimator.fit(..., verbose=...).
        """
        if isinstance(verbose, int):
            if verbose <= 0:
                return False
            if verbose == 1:
                return True
            return verbose
        return False

    @staticmethod
    def _clone_estimator(estimator: BaseEstimator) -> BaseEstimator:
        """Clone an estimator with constructor and deepcopy fallbacks.

        Args:
            estimator (BaseEstimator): Estimator instance to duplicate.

        Returns:
            BaseEstimator: Independent estimator instance for a CV fold.

        Raises:
            RuntimeError: If all cloning strategies fail.
        """
        try:
            return clone(estimator)
        except (TypeError, AttributeError, ValueError, RuntimeError):
            pass

        if hasattr(estimator, "get_params") and callable(estimator.get_params):
            try:
                return estimator.__class__(**estimator.get_params())
            except Exception:
                pass

        try:
            return deepcopy(estimator)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to clone estimator of type {type(estimator).__name__}."
            ) from exc

    def validate_processors(self) -> None:
        """Validate that all processors implement a compatible interface.

        Raises:
            ValueError: If any processor lacks a supported fit/transform interface.
        """
        for proc in self.processors:
            has_fit_transform = hasattr(proc, "fit_transform") and callable(proc.fit_transform)
            has_fit_and_transform = (
                hasattr(proc, "fit")
                and callable(proc.fit)
                and hasattr(proc, "transform")
                and callable(proc.transform)
            )
            has_fit_resample = hasattr(proc, "fit_resample") and callable(proc.fit_resample)

            if not (has_fit_transform or has_fit_and_transform or has_fit_resample):
                raise ValueError(
                    f"Processor {proc} must implement either (fit & transform), fit_transform, or fit_resample."
                )

    def _apply_processors(
        self,
        processors: list[object],
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None,
        y_train: pd.Series | None = None,
        y_valid: pd.Series | None = None,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame | None,
        pd.DataFrame | None,
        pd.Series | None,
        pd.Series | None,
        list[object],
    ]:
        """Clone, fit processors on training fold, and transform all splits.

        Each processor is deep-copied before fitting so every fold maintains an
        independent fitted state. Processor method signatures are introspected
        and y is passed only when accepted, allowing supervised and unsupervised
        processors to coexist without modification.

        Args:
            processors (list[object]): Ordered processor templates to clone and apply.
            X_train (pd.DataFrame): Training features for the current fold.
            X_valid (pd.DataFrame | None): Validation features. Defaults to None.
            X_test (pd.DataFrame | None): Test features to transform consistently. Defaults to None.
            y_train (pd.Series | None): Training targets. Passed to processors that accept y. Defaults to None.
            y_valid (pd.Series | None): Validation targets. Passed to transform when accepted. Defaults to None.

        Returns:
            tuple: (X_train, X_valid, X_test, y_train, y_valid, fitted_processors) where
                fitted_processors is the list of cloned and fitted processor instances for this fold.
        """
        if not processors:
            return X_train, X_valid, X_test, y_train, y_valid, []

        X_train_out = X_train
        X_valid_out = X_valid
        X_test_out = X_test
        y_train_out = y_train
        y_valid_out = y_valid
        fitted_processors: list[object] = []

        for proc_template in processors:
            proc = deepcopy(proc_template)
            fitted_processors.append(proc)

            if hasattr(proc, "fit_resample"):
                fr_sig = signature(proc.fit_resample)
                if "y" in fr_sig.parameters and y_train_out is not None:
                    X_train_out, y_train_out = proc.fit_resample(X_train_out, y=y_train_out)
                else:
                    X_train_out, y_train_out = proc.fit_resample(X_train_out, y_train_out)
                continue

            if hasattr(proc, "fit_transform"):
                ft_sig = signature(proc.fit_transform)
                ft_kwargs: dict = {}
                if "y" in ft_sig.parameters and y_train_out is not None:
                    ft_kwargs["y"] = y_train_out
                X_train_out = proc.fit_transform(X_train_out, **ft_kwargs)
            else:
                fit_sig = signature(proc.fit)
                fit_kwargs: dict = {}
                if "y" in fit_sig.parameters and y_train_out is not None:
                    fit_kwargs["y"] = y_train_out
                proc.fit(X_train_out, **fit_kwargs)

            if X_valid_out is not None and hasattr(proc, "transform"):
                tr_sig = signature(proc.transform)
                val_kwargs: dict = {}
                if "y" in tr_sig.parameters and y_valid_out is not None:
                    val_kwargs["y"] = y_valid_out
                X_valid_out = proc.transform(X_valid_out, **val_kwargs) if val_kwargs else proc.transform(X_valid_out)

            if X_test_out is not None and hasattr(proc, "transform"):
                X_test_out = proc.transform(X_test_out)

        return X_train_out, X_valid_out, X_test_out, y_train_out, y_valid_out, fitted_processors

    @staticmethod
    def _transform_with_processors(processors: list[object], X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted processors to X using transform only.

        Resamplers (fit_resample interface) are skipped — they are not applicable at inference time.

        Args:
            processors (list[object]): Fitted processor instances for a single fold.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: Transformed features.
        """
        X_out = X
        for proc in processors:
            if hasattr(proc, "fit_resample"):
                continue
            if hasattr(proc, "transform"):
                X_out = proc.transform(X_out)
        return X_out

    def fit(
        self,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame | None = None,
    ) -> tuple[list[float], list[float], np.ndarray | None]:
        """Run cross-validation and compute per-fold scores.

        Processors are cloned and re-fitted inside each fold on the training
        split to avoid leakage, then applied to the validation and optional test split.

        Args:
            estimator (BaseEstimator): Unfitted estimator to clone and train each fold.
            X (pd.DataFrame): Feature matrix aligned with target.
            y (pd.Series): Target values.
            X_test (pd.DataFrame | None): Optional test set transformed per fold. Defaults to None.

        Returns:
            tuple[list[float], list[float], np.ndarray | None]:
                Train scores, validation scores per fold, and averaged test predictions
                if X_test is provided, otherwise None.
        """
        self.train_scores_.clear()
        self.valid_scores_.clear()
        self.fitted_estimators_.clear()
        self.fitted_processors_.clear()

        iterator = self.cv.split(X, y)
        show_progress_bar = self.verbose and not bool(self.fit_verbose)
        if show_progress_bar:
            total = getattr(self.cv, "n_splits", None)
            iterator = tqdm(iterator, total=total, desc="Cross-validation")

        test_pred_sum: np.ndarray | None = None
        n_folds_for_test = 0

        for fold, (train_idx, valid_idx) in enumerate(iterator, start=1):
            X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            X_train_proc, X_valid_proc, X_test_proc, y_train_proc, y_valid_proc, fitted_processors = (
                self._apply_processors(
                    self.processors,
                    X_train,
                    X_valid,
                    X_test,
                    y_train=y_train,
                    y_valid=y_valid,
                )
            )

            y_train_eff = y_train_proc if y_train_proc is not None else y_train
            y_valid_eff = y_valid_proc if y_valid_proc is not None else y_valid

            estimator_fold = self._clone_estimator(estimator)
            y_train_pred, y_valid_pred, y_test_pred, fitted_estimator = self.fit_and_predict(
                estimator_fold,
                X_train_proc,
                y_train_eff,
                X_valid_proc,
                y_valid_eff,
                self.metric_type,
                X_test=X_test_proc,
                verbose=self.verbose,
            )

            train_score = self.metric_fn(y_train_eff, y_train_pred)
            valid_score = self.metric_fn(y_valid_eff, y_valid_pred)
            self.train_scores_.append(train_score)
            self.valid_scores_.append(valid_score)
            self.fitted_estimators_.append(fitted_estimator)
            self.fitted_processors_.append(fitted_processors)

            if y_test_pred is not None:
                fold_test_pred = np.asarray(y_test_pred, dtype=float)
                if test_pred_sum is None:
                    test_pred_sum = np.zeros_like(fold_test_pred, dtype=float)
                test_pred_sum += fold_test_pred
                n_folds_for_test += 1

            if self.verbose:
                tqdm.write(f"Fold {fold}: train={train_score:.5f} | valid={valid_score:.5f}")

        avg_test_pred = (
            (test_pred_sum / n_folds_for_test)
            if (test_pred_sum is not None and n_folds_for_test > 0)
            else None
        )
        return self.train_scores_, self.valid_scores_, avg_test_pred

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using averaged outputs across fitted fold estimators.

        Applies each fold's fitted processors to X before calling the estimator,
        then averages predictions across folds.

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Predicted labels (classification) or values (regression).

        Raises:
            RuntimeError: If called before fitting.
        """
        if not self.fitted_estimators_:
            raise RuntimeError("CrossValidationTrainer must be fitted before calling predict().")

        preds = []
        for fitted_estimator, fold_processors in zip(self.fitted_estimators_, self.fitted_processors_):
            X_proc = self._transform_with_processors(fold_processors, X)
            X_ordered = self._order_X_for_estimator(X_proc, fitted_estimator)
            if self.problem_type == "classification":
                if hasattr(fitted_estimator, "predict_proba"):
                    preds.append(fitted_estimator.predict_proba(X_ordered))
                else:
                    preds.append(fitted_estimator.predict(X_ordered))
            else:
                preds.append(fitted_estimator.predict(X_ordered))

        preds_arr = np.asarray(preds, dtype=float)
        avg = preds_arr.mean(axis=0)
        if self.problem_type == "classification":
            return np.argmax(avg, axis=1) if avg.ndim == 2 else (avg >= 0.5).astype(int)
        return avg

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities by averaging fold estimator probabilities.

        Applies each fold's fitted processors to X before calling the estimator.

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Averaged class probabilities (n_samples, n_classes) or (n_samples,) for binary.

        Raises:
            RuntimeError: If called before fitting.
            ValueError: If called for a regression problem.
        """
        if self.problem_type != "classification":
            raise ValueError("predict_proba is only available for classification problems.")
        if not self.fitted_estimators_:
            raise RuntimeError("CrossValidationTrainer must be fitted before calling predict_proba().")

        probs = []
        for fitted_estimator, fold_processors in zip(self.fitted_estimators_, self.fitted_processors_):
            X_proc = self._transform_with_processors(fold_processors, X)
            X_ordered = self._order_X_for_estimator(X_proc, fitted_estimator)
            probs.append(fitted_estimator.predict_proba(X_ordered))

        return np.asarray(probs, dtype=float).mean(axis=0)

    def predict_with_confidence(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predict with uncertainty estimates aggregated across folds.

        Applies each fold's fitted processors before calling the estimator.
        For regression: returns mean prediction, std, and 95% confidence interval.
        For classification: returns predicted class, mean probabilities,
        per-sample confidence, and cross-fold disagreement.

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            dict[str, np.ndarray]: Prediction outputs and uncertainty summaries.

        Raises:
            RuntimeError: If called before fitting.
            ValueError: If classification estimators lack predict_proba.
        """
        if not self.fitted_estimators_:
            raise RuntimeError(
                "CrossValidationTrainer must be fitted before calling predict_with_confidence()."
            )

        if self.problem_type == "regression":
            fold_preds = []
            for est, fold_processors in zip(self.fitted_estimators_, self.fitted_processors_):
                X_proc = self._transform_with_processors(fold_processors, X)
                X_ordered = self._order_X_for_estimator(X_proc, est)
                fold_preds.append(np.asarray(est.predict(X_ordered), dtype=float))
            preds = np.vstack(fold_preds)
            mean_pred = preds.mean(axis=0)
            std_pred = preds.std(axis=0, ddof=0)
            return {
                "prediction": mean_pred,
                "std": std_pred,
                "ci_95_lower": mean_pred - 1.96 * std_pred,
                "ci_95_upper": mean_pred + 1.96 * std_pred,
            }

        probs = []
        for est, fold_processors in zip(self.fitted_estimators_, self.fitted_processors_):
            X_proc = self._transform_with_processors(fold_processors, X)
            X_ordered = self._order_X_for_estimator(X_proc, est)
            if not hasattr(est, "predict_proba"):
                raise ValueError(
                    "All fold estimators must implement predict_proba for classification confidence."
                )
            p = np.asarray(est.predict_proba(X_ordered), dtype=float)
            if p.ndim == 1:
                p = np.column_stack([1 - p, p])
            probs.append(p)

        probs_arr = np.asarray(probs, dtype=float)
        mean_proba = probs_arr.mean(axis=0)
        pred_class = np.argmax(mean_proba, axis=1)
        pred_confidence = mean_proba[np.arange(len(pred_class)), pred_class]
        disagreement = probs_arr[:, np.arange(len(pred_class)), pred_class].std(axis=0)
        return {
            "prediction": pred_class,
            "confidence": pred_confidence,
            "disagreement": disagreement,
            "proba": mean_proba,
        }

    @staticmethod
    def _order_X_for_estimator(
        X: pd.DataFrame, estimator: BaseEstimator
    ) -> pd.DataFrame:
        """Reorder columns to match an estimator's expected feature order.

        Args:
            X (pd.DataFrame): Input features.
            estimator (BaseEstimator): Trained estimator.

        Returns:
            pd.DataFrame: Reordered features if the estimator exposes feature names, otherwise original X.
        """

        def _normalize(name: Any) -> str:
            """Normalize a feature name for tolerant cross-library matching."""
            return str(name).strip().replace(" ", "_")

        feature_attrs = ("feature_names_in_", "feature_names_")
        columns_set = set(X.columns)

        for attr_name in feature_attrs:
            if not hasattr(estimator, attr_name):
                continue

            expected_names = [str(n) for n in getattr(estimator, attr_name)]
            if set(expected_names).issubset(columns_set):
                return X.loc[:, expected_names]

            normalized_map: dict[str, str] = {_normalize(col): col for col in X.columns}
            normalized_expected = [_normalize(n) for n in expected_names]
            if all(n in normalized_map for n in normalized_expected):
                return X.loc[:, [normalized_map[n] for n in normalized_expected]]

        return X


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    SEED = 42

    # Test with standard metric
    iris_X, iris_y = load_iris(return_X_y=True, as_frame=True)
    model = XGBClassifier(
        n_estimators=1000,
        random_state=SEED,
        n_jobs=-1,
        enable_categorical=True,
        tree_method="hist",
        verbosity=1,
        early_stopping_rounds=100,
        eval_metric="mlogloss",
    )

    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=1000,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        early_stopping_rounds=100,
    )

    compare_metric_name = "Accuracy"
    problem_type = "classification"
    trainer = CrossValidationTrainer(
        metric_name=compare_metric_name, problem_type=problem_type, verbose=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        iris_X, iris_y, test_size=0.3, random_state=17
    )
    train_scores, validation_scores, avg_test_preds = trainer.fit(
        model, X_train, y_train, X_test=X_test
    )
    print(
        f"Train {compare_metric_name}: {np.mean(train_scores):.4f} +- {np.std(train_scores):.4f}"
    )
    print(
        f"Validation {compare_metric_name}: {np.mean(validation_scores):.4f} +- {np.std(validation_scores):.4f}"
    )
    confidence_results = trainer.predict_with_confidence(X_test)
    print("Confidence results:", confidence_results)
