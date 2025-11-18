import pandas as pd
import numpy as np
from inspect import signature
from sklearn.base import BaseEstimator
from sklearn.base import clone
from copy import deepcopy
from sklearn.model_selection import KFold, BaseCrossValidator
from tqdm import tqdm
from typing import Dict, Any

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
from kvbiii_ml.evaluation.metrics import (
    METRICS,
    get_metric_function,
    get_metric_type,
    get_metric_direction,
)
from kvbiii_ml.evaluation.custom_metrics_handler import CustomMetricsHandler
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
        verbose: bool = True,
        custom_metric: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            problem_type (str): Problem type, either "classification" or "regression".
            metric_name (str): Metric name defined in kvbiii_ml.evaluation.metrics.METRICS or None if using a custom metric.
            cv (BaseCrossValidator | None): Cross-validation splitter. If None, uses 5-fold KFold with shuffling. Defaults to None.
            processors (list[object] | None): Optional list of processors applied per fold (fit/transform, fit_transform, or fit_resample). Defaults to None.
            verbose (bool): Whether to display per-fold scores and a progress bar. Defaults to True.
            custom_metric (Dict[str, Any] | None): Custom metric configuration with keys: 'name', 'function',
                and optionally 'metric_type', 'additional_data', and other kwargs. Defaults to None.
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

        self.validate_processors()

        self.train_scores_: list[float] = []
        self.valid_scores_: list[float] = []
        self.fitted_estimators_: list[BaseEstimator] = []

    def validate_processors(self):
        """Validate that all processors are compatible with the expected interface."""
        if not self.processors:
            return

        for proc in self.processors:
            has_fit_transform = hasattr(proc, "fit_transform") and callable(
                proc.fit_transform
            )
            has_fit_and_transform = (
                hasattr(proc, "fit")
                and callable(proc.fit)
                and hasattr(proc, "transform")
                and callable(proc.transform)
            )
            has_fit_resample = hasattr(proc, "fit_resample") and callable(
                proc.fit_resample
            )

            if not (has_fit_transform or has_fit_and_transform or has_fit_resample):
                raise ValueError(
                    f"Processor {proc} must implement either (fit & transform), fit_transform, or fit_resample."
                )

    def _apply_processors(
        self,
        processors: list[object] | None,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
        y_train: pd.Series | None = None,
        y_valid: pd.Series | None = None,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame | None,
        pd.DataFrame | None,
        pd.Series | None,
        pd.Series | None,
    ]:
        """Fit processors on the training fold and transform valid/test.

        For maximum compatibility, the method introspects processor method
        signatures and passes y only if that parameter is
        accepted. This allows existing processors without y support to work
        unchanged while enabling supervised processors that require targets.

        Args:
            processors (list[object] | None): Ordered processors to apply. Processors must implement either fit/transform or fit_transform.
            train_df (pd.DataFrame): Training data for the current fold.
            valid_df (pd.DataFrame | None): Validation data for the current fold. Defaults to None.
            test_df (pd.DataFrame | None): Optional test data to transform consistently. Defaults to None.
            y_train (pd.Series | None): Training targets for the current fold. Passed to processors that accept y. Defaults to None.
            y_valid (pd.Series | None): Validation targets for the current fold. Passed to processors that accept y in transform. Defaults to None.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.Series | None, pd.Series | None]:
                Processed (train_df, valid_df, test_df, y_train_out, y_valid_out).
        """
        if not processors:
            return train_df, valid_df, test_df, y_train, y_valid

        X_train = train_df
        X_valid = valid_df
        X_test = test_df
        y_train_out = y_train
        y_valid_out = y_valid

        for proc in processors:
            if hasattr(proc, "fit_resample"):
                fr_sig = signature(proc.fit_resample)
                fr_kwargs: dict = {}
                if "y" in fr_sig.parameters and y_train_out is not None:
                    fr_kwargs["y"] = y_train_out
                if fr_kwargs:
                    X_train_res, y_train_res = proc.fit_resample(X_train, **fr_kwargs)
                else:
                    X_train_res, y_train_res = proc.fit_resample(X_train, y_train_out)
                X_train = X_train_res
                y_train_out = y_train_res
                continue

            if hasattr(proc, "fit_transform"):
                ft_sig = signature(proc.fit_transform)
                ft_kwargs: dict = {}
                if "y" in ft_sig.parameters and y_train_out is not None:
                    ft_kwargs["y"] = y_train_out
                X_train = proc.fit_transform(X_train, **ft_kwargs)
            else:
                if hasattr(proc, "fit"):
                    fit_sig = signature(proc.fit)
                    fit_kwargs: dict = {}
                    if "y" in fit_sig.parameters and y_train_out is not None:
                        fit_kwargs["y"] = y_train_out
                    proc.fit(X_train, **fit_kwargs)

            if X_valid is not None and hasattr(proc, "transform"):
                tr_sig = signature(proc.transform)
                val_kwargs: dict = {}
                if "y" in tr_sig.parameters and y_valid_out is not None:
                    val_kwargs["y"] = y_valid_out
                X_valid = (
                    proc.transform(X_valid, **val_kwargs)
                    if val_kwargs
                    else proc.transform(X_valid)
                )
            if X_test is not None and hasattr(proc, "transform"):
                X_test = proc.transform(X_test)

        return X_train, X_valid, X_test, y_train_out, y_valid_out

    def fit(
        self,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame | None = None,
    ) -> tuple[list[float], list[float], np.ndarray | None]:
        """Run cross-validation and compute per-fold scores.

        Processors are re-fitted inside each fold on the training split to avoid
        leakage, then applied to the validation (and optional test) split.

        Args:
            df (pd.DataFrame): Feature matrix aligned with target.
            target (pd.Series): Target values.
            df_test (pd.DataFrame | None): Optional test set to transform with the processors per fold. Defaults to None.

        Returns:
            tuple[list[float], list[float], np.ndarray | None]:
                Train and validation scores per fold, and averaged test predictions
                if df_test is provided, otherwise None.
        """
        # Reset state for this run
        self.train_scores_.clear()
        self.valid_scores_.clear()
        if self.fitted_estimators_ is not None:
            self.fitted_estimators_.clear()

        # Prepare iterator with optional progress bar
        iterator = self.cv.split(X, y)
        if self.verbose:
            total = getattr(self.cv, "n_splits", None)
            iterator = tqdm(iterator, total=total, desc="Cross-validation")

        test_pred_sum: np.ndarray | None = None
        n_folds_for_test = 0

        for fold, (train_idx, valid_idx) in enumerate(iterator, start=1):
            X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            # Apply processors per fold to avoid leakage
            X_train_proc, X_valid_proc, X_test_proc, y_train_proc, y_valid_proc = (
                self._apply_processors(
                    self.processors,
                    X_train,
                    X_valid,
                    X_test,
                    y_train=y_train,
                    y_valid=y_valid,
                )
            )

            # Use potentially resampled/processed targets if provided
            y_train_eff = y_train_proc if y_train_proc is not None else y_train
            y_valid_eff = y_valid_proc if y_valid_proc is not None else y_valid

            # Clone estimator per fold to avoid mutating a single shared instance
            try:
                estimator_fold = clone(estimator)
            except Exception:
                estimator_fold = deepcopy(estimator)

            y_train_pred, y_valid_pred, y_test_pred, fitted_estimator = (
                self.fit_and_predict(
                    estimator_fold,
                    X_train_proc,
                    y_train_eff,
                    X_valid_proc,
                    y_valid_eff,
                    self.metric_type,
                    X_test=X_test_proc,
                )
            )

            # Score this fold
            train_score = self.metric_fn(y_train_eff, y_train_pred)
            valid_score = self.metric_fn(y_valid_eff, y_valid_pred)
            self.train_scores_.append(train_score)
            self.valid_scores_.append(valid_score)

            # Accumulate test predictions for averaging
            if y_test_pred is not None:
                fold_test_pred = np.asarray(y_test_pred, dtype=float)
                if test_pred_sum is None:
                    test_pred_sum = np.zeros_like(fold_test_pred, dtype=float)
                test_pred_sum += fold_test_pred
                n_folds_for_test += 1

            if self.verbose:
                tqdm.write(
                    f"Fold {fold}: train={train_score:.5f} | valid={valid_score:.5f}"
                )

            if self.fitted_estimators_ is not None:
                self.fitted_estimators_.append(fitted_estimator)

        avg_test_pred = (
            (test_pred_sum / n_folds_for_test)
            if (test_pred_sum is not None and n_folds_for_test > 0)
            else None
        )
        return self.train_scores_, self.valid_scores_, avg_test_pred

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using averaged outputs across fitted fold estimators.

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Predicted labels for classification or predictions for regression.

        Raises:
            RuntimeError: If called before fitting.
        """
        if not self.fitted_estimators_:
            raise RuntimeError("OOFModel must be fitted before calling predict().")
        preds = []
        for fitted_estimator in self.fitted_estimators_:
            X_ordered = self._order_X_for_estimator(X, fitted_estimator)
            if self.problem_type == "classification":
                if hasattr(fitted_estimator, "predict_proba"):
                    proba = fitted_estimator.predict_proba(X_ordered)
                    preds.append(proba)
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

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Averaged class probabilities with shape (n_samples, n_classes) or (n_samples,) for binary.

        Raises:
            RuntimeError: If called before fitting.
            ValueError: If called for a regression problem.
        """
        if self.problem_type != "classification":
            raise ValueError(
                "predict_proba is only available for classification problems."
            )
        if not self.fitted_estimators_:
            raise RuntimeError(
                "OOFModel must be fitted before calling predict_proba()."
            )
        probs = []
        for fitted_estimator in self.fitted_estimators_:
            X_ordered = self._order_X_for_estimator(X, fitted_estimator)
            probs.append(fitted_estimator.predict_proba(X_ordered))
        probs_arr = np.asarray(probs, dtype=float)
        return probs_arr.mean(axis=0)

    def predict_with_confidence(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predict with simple uncertainty estimates aggregated across folds.

        For regression, returns mean prediction, standard deviation across fold
        estimators, and a 95% normal approximation confidence interval.

        For classification, returns predicted class labels, the mean class
        probabilities, per-sample confidence (probability of predicted class),
        and disagreement (standard deviation of the predicted class probability
        across folds).

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            dict[str, np.ndarray]: Dictionary containing prediction outputs and
                uncertainty summaries.

        Raises:
            RuntimeError: If called before fitting.
            ValueError: If classification estimators lack predict_proba.
        """
        if not self.fitted_estimators_:
            raise RuntimeError(
                "CrossValidationTrainer must be fitted before calling predict_with_confidence()."
            )

        # Regression branch
        if self.problem_type == "regression":
            fold_preds = []
            for est in self.fitted_estimators_:
                X_ordered = self._order_X_for_estimator(X, est)
                fold_preds.append(np.asarray(est.predict(X_ordered), dtype=float))
            preds = np.vstack(fold_preds)  # (n_folds, n_samples)
            mean_pred = preds.mean(axis=0)
            std_pred = preds.std(axis=0, ddof=0)
            ci_lower = mean_pred - 1.96 * std_pred
            ci_upper = mean_pred + 1.96 * std_pred
            return {
                "prediction": mean_pred,
                "std": std_pred,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
            }

        # Classification branch
        probs = []
        for est in self.fitted_estimators_:
            X_ordered = self._order_X_for_estimator(X, est)
            if not hasattr(est, "predict_proba"):
                raise ValueError(
                    "All fold estimators must implement predict_proba for classification confidence."
                )
            p = est.predict_proba(X_ordered)
            p = np.asarray(p, dtype=float)
            if p.ndim == 1:  # (n_samples,) -> assume binary and expand
                p = np.column_stack([1 - p, p])
            probs.append(p)
        probs = np.asarray(probs, dtype=float)  # (n_folds, n_samples, n_classes)
        mean_proba = probs.mean(axis=0)
        pred_class = np.argmax(mean_proba, axis=1)
        pred_confidence = mean_proba[np.arange(len(pred_class)), pred_class]
        # Std of predicted class probability across folds
        disagreement = probs[:, np.arange(len(pred_class)), pred_class].std(axis=0)
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
            pd.DataFrame: Reordered features if the estimator exposes feature names, otherwise the original X.
        """
        if hasattr(estimator, "feature_names_in_"):
            return X[estimator.feature_names_in_]
        if hasattr(estimator, "feature_names_"):
            return X[estimator.feature_names_]
        return X


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from kvbiii_ml.evaluation.custom_metrics_handler import f_beta_selection_score

    # Test with standard metric
    iris_X, iris_y = load_iris(return_X_y=True, as_frame=True)
    clf = LogisticRegression(max_iter=200, n_jobs=1)
    compare_metric_name = "Accuracy"
    problem_type = "classification"
    trainer = CrossValidationTrainer(
        metric_name=compare_metric_name, problem_type=problem_type
    )

    X_train, X_test, y_train, y_test = train_test_split(
        iris_X, iris_y, test_size=0.3, random_state=17
    )
    train_scores, validation_scores, avg_test_preds = trainer.fit(
        clf, X_train, y_train, X_test=X_test
    )
    print(
        f"Train {compare_metric_name}: {np.mean(train_scores):.4f} +- {np.std(train_scores):.4f}"
    )
    print(
        f"Validation {compare_metric_name}: {np.mean(validation_scores):.4f} +- {np.std(validation_scores):.4f}"
    )
    confidence_results = trainer.predict_with_confidence(X_test)
    print("Confidence results:", confidence_results)
