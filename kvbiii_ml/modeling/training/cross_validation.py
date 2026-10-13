from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from kvbiii_ml.evaluation.custom_metrics_handler import CustomMetricsHandler
from kvbiii_ml.evaluation.metrics import (
    METRICS,
    get_metric_direction,
    get_metric_function,
    get_metric_type,
)
from kvbiii_ml.modeling.training.base_trainer import (
    BaseTrainer,
    order_x_for_estimator,
    predict_with_offset,
    split_offset,
    to_metric_proba,
)

SampleWeightFn = Callable[[pd.DataFrame, pd.Series], pd.Series | np.ndarray]


@dataclass
class _FoldResult:
    """Everything one cross-validation fold produces."""

    train_score: Any
    valid_score: Any
    fitted_estimator: BaseEstimator
    fitted_pipeline: Pipeline | None
    test_pred: np.ndarray | None


class CrossValidationTrainer(BaseTrainer):
    """Cross-validation trainer with an optional per-fold preprocessing pipeline.

    Orchestrates K-Fold style training loops, optionally applying a single
    sklearn Pipeline inside each fold to avoid data leakage.  The pipeline is
    cloned and re-fitted on each fold's training split independently.
    """

    def __init__(
        self,
        problem_type: str,
        metric_name: str | None = None,
        cv: BaseCrossValidator | None = None,
        preprocessing_pipeline: Pipeline | None = None,
        verbose: bool | int = True,
        custom_metric: dict[str, Any] | None = None,
        n_jobs: int = 1,
        sample_weight_fn: SampleWeightFn | None = None,
        early_stopping_holdout: float | None = None,
        offset_column: str | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            problem_type (str): Problem type, either "classification" or "regression".
            metric_name (str | None): Metric name defined in METRICS or None if using
                a custom metric.
            cv (BaseCrossValidator | None): Cross-validation splitter. Defaults to
                5-fold KFold with shuffling.
            preprocessing_pipeline (Pipeline | None): Optional sklearn Pipeline applied
                per fold on the training split before fitting the estimator. Defaults to None.
            verbose (bool | int): Verbosity flag. If set to integer 1, estimator fit
                verbosity is mapped to True for estimators that accept a fit-time verbose
                parameter. Defaults to True.
            custom_metric (dict[str, Any] | None): Custom metric configuration. Required
                keys: 'name' (str), 'function' (callable ``f(y_true, y_pred, **kwargs)``),
                'metric_type' ("preds" or "probs") and 'direction' ("minimize" or
                "maximize"). Optional key: 'kwargs' (dict), extra keyword arguments
                passed to 'function' on every call. ``y_true`` arrives as a
                ``pd.Series`` that keeps the original index of X, so row-aligned extra
                data passed through 'kwargs' can be aligned with
                ``extra.loc[y_true.index]``. For "probs" metrics ``y_pred`` is the 1-D
                positive-class probability for binary problems and the
                ``(n_samples, n_classes)`` matrix for multiclass (see
                ``base_trainer.to_metric_proba``). Defaults to None.
            n_jobs (int): Number of folds trained concurrently. Folds run in threads,
                which parallelise well because the boosting libraries release the GIL
                while training; give each estimator its own modest thread count so that
                n_jobs times the estimator's threads stays within the physical cores.
                Results are collected in fold order, so every score, fitted estimator,
                fitted pipeline and averaged test prediction is identical to a
                sequential run. Defaults to 1, which trains the folds sequentially.
            sample_weight_fn (SampleWeightFn | None): Optional callable computing
                per-sample training weights inside each fold. It is called as
                ``sample_weight_fn(X_train, y_train)`` with the fold's raw training
                rows (before the preprocessing pipeline, in their original order) and
                must return one weight per row. Computing weights per fold keeps them
                relative to that fold's own training window - e.g. a recency decay
                anchored at the fold's latest row or class-balancing weights derived
                from the fold's own target - and never exposes validation rows to it.
                Validation rows are never weighted and scoring stays unweighted.
                Defaults to None, which trains without weights.
            early_stopping_holdout (float | None): Fraction in (0, 1) of each training
                fold, taken from its end (the latest rows for a time-ordered splitter),
                held out as the early-stopping eval set. The estimator is fitted on the
                remaining rows, so the validation fold is used only for scoring and
                ``valid_scores_`` are not biased by early stopping. Defaults to None,
                which keeps the previous behaviour: the validation fold is also the
                early-stopping eval set.
            offset_column (str | None): Column of X (and X_test) holding a link-scale
                offset, e.g. ``log(exposure)`` for count models. It is dropped before
                the pipeline and the estimator see X, passed to training as LightGBM
                ``init_score``/``eval_init_score``, XGBoost
                ``base_margin``/``base_margin_eval_set`` or CatBoost ``baseline``, and
                added to the raw score before the inverse link at predict time (see
                ``base_trainer.predict_with_offset``). Supported for LightGBM, XGBoost
                and CatBoost regressors and binary classifiers. Defaults to None.

        Raises:
            TypeError: If preprocessing_pipeline is not a Pipeline or None, or
                sample_weight_fn is neither callable nor None.
            ValueError: If problem_type is invalid, neither metric_name nor
                custom_metric is provided, n_jobs is not positive, or
                early_stopping_holdout is not in (0, 1).
        """
        if problem_type not in {"classification", "regression"}:
            raise ValueError("problem_type must be 'classification' or 'regression'.")

        if cv is None:
            cv = KFold(n_splits=5, shuffle=True, random_state=17)

        if metric_name is None and custom_metric is None:
            raise ValueError("Either metric_name or custom_metric must be provided.")

        if n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer.")

        if early_stopping_holdout is not None and not 0 < early_stopping_holdout < 1:
            raise ValueError(
                "early_stopping_holdout must be None or a float in (0, 1)."
            )

        if sample_weight_fn is not None and not callable(sample_weight_fn):
            raise TypeError(
                f"sample_weight_fn must be callable or None, "
                f"got {type(sample_weight_fn).__name__}."
            )

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
        self.preprocessing_pipeline = preprocessing_pipeline
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.sample_weight_fn = sample_weight_fn
        self.early_stopping_holdout = early_stopping_holdout
        self.offset_column = offset_column
        self.fit_verbose = self._resolve_fit_verbose(verbose)

        self.validate_pipeline()

        self.train_scores_: list[float] = []
        self.valid_scores_: list[float] = []
        self.fitted_estimators_: list[BaseEstimator] = []
        self.fitted_pipelines_: list[Pipeline | None] = []

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
        """Clone an estimator safely.

        Uses deepcopy for CatBoost because sklearn.clone cannot round-trip
        cat_features through CatBoost's constructor - CatBoost modifies the
        parameter internally during __init__, violating sklearn's clone contract.

        Args:
            estimator (BaseEstimator): Estimator instance to duplicate.

        Returns:
            BaseEstimator: Independent estimator instance for a CV fold.
        """
        if "CatBoost" in type(estimator).__name__:
            return deepcopy(estimator)
        try:
            return clone(estimator)
        except (RuntimeError, TypeError, AttributeError, ValueError, KeyError):
            return deepcopy(estimator)

    def validate_pipeline(self) -> None:
        """Validate that preprocessing_pipeline is a Pipeline or None.

        Raises:
            TypeError: If preprocessing_pipeline is neither a Pipeline nor None.
        """
        if self.preprocessing_pipeline is not None and not isinstance(
            self.preprocessing_pipeline, Pipeline
        ):
            raise TypeError(
                f"preprocessing_pipeline must be an sklearn Pipeline or None, "
                f"got {type(self.preprocessing_pipeline).__name__}."
            )

    @staticmethod
    def _apply_pipeline(
        pipeline: Pipeline | None,
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None,
        y_train: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, Pipeline | None]:
        """Clone, fit the pipeline on the training fold, and transform all splits.

        The pipeline is cloned before fitting so every fold maintains an independent
        fitted state.  y_train is always forwarded to the pipeline; sklearn routes it
        to each step's fit() automatically so supervised steps (e.g. MeanEncoder)
        receive labels while unsupervised steps ignore the argument.

        Args:
            pipeline (Pipeline | None): Unfitted pipeline template, or None.
            X_train (pd.DataFrame): Training features for the current fold.
            X_valid (pd.DataFrame | None): Validation features. Defaults to None.
            X_test (pd.DataFrame | None): Test features. Defaults to None.
            y_train (pd.Series | None): Training targets forwarded to pipeline.fit_transform.
                Defaults to None.

        Returns:
            tuple: (x_train_out, x_valid_out, x_test_out, fitted_pipeline) where
                fitted_pipeline is the cloned and fitted Pipeline for this fold, or None.
        """
        if pipeline is None:
            return X_train, X_valid, X_test, None

        fitted = clone(pipeline)
        x_train_out = fitted.fit_transform(X_train, y_train)
        x_valid_out = fitted.transform(X_valid) if X_valid is not None else None
        x_test_out = fitted.transform(X_test) if X_test is not None else None
        return x_train_out, x_valid_out, x_test_out, fitted

    @staticmethod
    def _transform_with_pipeline(
        pipeline: Pipeline | None, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply a fitted pipeline to X using transform only.

        Args:
            pipeline (Pipeline | None): Fitted pipeline for a single fold, or None.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: Transformed features, or the original X when pipeline is None.
        """
        if pipeline is None:
            return X
        return pipeline.transform(X)

    def fit(
        self,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame | None = None,
        preprocessing_pipeline_override: Pipeline | None = None,
    ) -> tuple[list[float], list[float], np.ndarray | None]:
        """Run cross-validation and compute per-fold scores.

        The preprocessing pipeline is cloned and re-fitted inside each fold on the
        training split to avoid leakage, then applied to the validation and optional
        test split. When ``sample_weight_fn`` is set, each fold's training weights are
        computed from that fold's own training rows.

        Args:
            estimator (BaseEstimator): Unfitted estimator to clone and train each fold.
            X (pd.DataFrame): Feature matrix aligned with target.
            y (pd.Series): Target values.
            X_test (pd.DataFrame | None): Optional test set transformed per fold.
                Defaults to None.
            preprocessing_pipeline_override (Pipeline | None): When not None, replaces
                self.preprocessing_pipeline for this call only.  Pass None explicitly to
                run with no pipeline.  Used by feature selectors to inject step-restricted
                pipelines at each elimination step without mutating stored state.
                Defaults to None.

        Returns:
            tuple[list[float], list[float], np.ndarray | None]:
                Train scores, validation scores per fold, and test predictions averaged
                over folds if X_test is provided, otherwise None. For classification the
                test predictions are always the averaged ``predict_proba`` matrix of
                shape ``(n_samples, n_classes)`` (rows sum to 1), whatever the metric
                type; averaging hard labels would give meaningless fractions. Take
                ``argmax`` (or threshold column 1) to get labels. For regression they
                are the averaged ``predict`` values.
        """
        self.train_scores_.clear()
        self.valid_scores_.clear()
        self.fitted_estimators_.clear()
        self.fitted_pipelines_.clear()

        active_pipeline = (
            preprocessing_pipeline_override
            if preprocessing_pipeline_override is not None
            else self.preprocessing_pipeline
        )

        splits = list(self.cv.split(X, y))
        show_progress_bar = self.verbose and not bool(self.fit_verbose)
        progress_bar = (
            tqdm(total=len(splits), desc="Cross-validation")
            if show_progress_bar
            else None
        )

        def run_split(train_idx: np.ndarray, valid_idx: np.ndarray) -> _FoldResult:
            """Train one fold and advance the progress bar."""
            result = self._run_fold(
                active_pipeline, estimator, X, y, X_test, train_idx, valid_idx
            )
            if progress_bar is not None:
                progress_bar.update(1)
            return result

        try:
            if self.n_jobs == 1 or len(splits) <= 1:
                fold_results = [
                    run_split(train_idx, valid_idx) for train_idx, valid_idx in splits
                ]
            else:
                with ThreadPoolExecutor(
                    max_workers=min(self.n_jobs, len(splits))
                ) as executor:
                    futures = {
                        executor.submit(run_split, train_idx, valid_idx): position
                        for position, (train_idx, valid_idx) in enumerate(splits)
                    }
                    fold_results = [None] * len(splits)
                    for future in as_completed(futures):
                        fold_results[futures[future]] = future.result()
        finally:
            if progress_bar is not None:
                progress_bar.close()

        test_pred_sum: np.ndarray | None = None
        n_folds_for_test = 0
        for fold, result in enumerate(fold_results, start=1):
            self.train_scores_.append(result.train_score)
            self.valid_scores_.append(result.valid_score)
            self.fitted_estimators_.append(result.fitted_estimator)
            self.fitted_pipelines_.append(result.fitted_pipeline)

            if result.test_pred is not None:
                fold_test_pred = np.asarray(result.test_pred, dtype=float)
                if test_pred_sum is None:
                    test_pred_sum = np.zeros_like(fold_test_pred, dtype=float)
                test_pred_sum += fold_test_pred
                n_folds_for_test += 1

            if self.verbose:
                tqdm.write(
                    f"Fold {fold}: train={result.train_score:.5f} | "
                    f"valid={result.valid_score:.5f}"
                )

        avg_test_pred = (
            (test_pred_sum / n_folds_for_test)
            if (test_pred_sum is not None and n_folds_for_test > 0)
            else None
        )
        return self.train_scores_, self.valid_scores_, avg_test_pred

    def _run_fold(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        pipeline: Pipeline | None,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame | None,
        train_idx: np.ndarray,
        valid_idx: np.ndarray,
    ) -> "_FoldResult":
        """Preprocess, train and score a single fold without touching shared state.

        Args:
            pipeline (Pipeline | None): Unfitted pipeline template, or None.
            estimator (BaseEstimator): Unfitted estimator, cloned for this fold.
            X (pd.DataFrame): Full feature matrix.
            y (pd.Series): Full target.
            X_test (pd.DataFrame | None): Optional test features.
            train_idx (np.ndarray): Positional indices of the training rows.
            valid_idx (np.ndarray): Positional indices of the validation rows.

        Returns:
            _FoldResult: Scores, fitted objects and test predictions of this fold.
        """
        X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        fold_weights = self._compute_fold_weights(X_train, y_train)
        X_train, off_train = split_offset(X_train, self.offset_column)
        X_valid, off_valid = split_offset(X_valid, self.offset_column)
        off_test = None
        if X_test is not None:
            X_test, off_test = split_offset(X_test, self.offset_column)

        x_train_proc, x_valid_proc, x_test_proc, fitted_pipeline = self._apply_pipeline(
            pipeline,
            X_train,
            X_valid,
            X_test,
            y_train=y_train,
        )

        x_fit, y_fit, w_fit, o_fit = x_train_proc, y_train, fold_weights, off_train
        x_es, y_es, o_es = x_valid_proc, y_valid, off_valid
        if self.early_stopping_holdout is not None:
            n_holdout = max(
                1, int(round(len(x_train_proc) * self.early_stopping_holdout))
            )
            x_fit, x_es = x_train_proc.iloc[:-n_holdout], x_train_proc.iloc[-n_holdout:]
            y_fit, y_es = y_train.iloc[:-n_holdout], y_train.iloc[-n_holdout:]
            w_fit = None if fold_weights is None else fold_weights[:-n_holdout]
            if off_train is not None:
                o_fit, o_es = off_train[:-n_holdout], off_train[-n_holdout:]

        fitted_estimator = self.fit_estimator(
            self._clone_estimator(estimator),
            x_fit,
            y_fit,
            x_es,
            y_es,
            sample_weight=w_fit,
            verbose=self.verbose,
            offset_train=o_fit,
            offset_valid=o_es,
        )
        test_pred = None
        if x_test_proc is not None:
            test_pred = self._estimator_output(
                fitted_estimator,
                x_test_proc,
                off_test,
                proba=self.problem_type == "classification",
            )
        return _FoldResult(
            train_score=self.metric_fn(
                y_train,
                self._metric_predictions(fitted_estimator, x_train_proc, off_train),
            ),
            valid_score=self.metric_fn(
                y_valid,
                self._metric_predictions(fitted_estimator, x_valid_proc, off_valid),
            ),
            fitted_estimator=fitted_estimator,
            fitted_pipeline=fitted_pipeline,
            test_pred=test_pred,
        )

    def _metric_predictions(
        self, estimator: BaseEstimator, X: pd.DataFrame, offset: np.ndarray | None
    ) -> np.ndarray:
        """Predict X in the form the configured metric expects (see `to_metric_proba`)."""
        if self.metric_type == "probs":
            return to_metric_proba(self._estimator_output(estimator, X, offset, True))
        if self.metric_type == "preds":
            return self._estimator_output(estimator, X, offset, False)
        raise ValueError(
            f"Unknown metric type '{self.metric_type}'. Must be 'preds' or 'probs'."
        )

    @staticmethod
    def _estimator_output(
        estimator: BaseEstimator,
        X: pd.DataFrame,
        offset: np.ndarray | None,
        proba: bool,
    ) -> np.ndarray:
        """Call predict / predict_proba, going through the offset path when an offset is set."""
        if offset is not None:
            return predict_with_offset(estimator, X, offset, proba=proba)
        return estimator.predict_proba(X) if proba else estimator.predict(X)

    def _compute_fold_weights(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> np.ndarray | None:
        """Compute one fold's training weights with ``sample_weight_fn``.

        Args:
            X_train (pd.DataFrame): Raw training features of the fold.
            y_train (pd.Series): Training target of the fold.

        Returns:
            np.ndarray | None: Float weights aligned with ``X_train``, or None when
                no ``sample_weight_fn`` is configured.

        Raises:
            ValueError: If the returned weights are not one-dimensional with one
                entry per training row.
        """
        if self.sample_weight_fn is None:
            return None
        weights = np.asarray(self.sample_weight_fn(X_train, y_train), dtype=float)
        if weights.shape != (len(X_train),):
            raise ValueError(
                f"sample_weight_fn must return one weight per training row: expected "
                f"shape ({len(X_train)},), got {weights.shape}."
            )
        return weights

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using averaged outputs across fitted fold estimators.

        Applies each fold's fitted pipeline to X before calling the estimator,
        then averages predictions across folds.

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Predicted labels (classification) or values (regression).

        Raises:
            RuntimeError: If called before fitting.
        """
        if not self.fitted_estimators_:
            raise RuntimeError(
                "CrossValidationTrainer must be fitted before calling predict()."
            )

        X, offset = split_offset(X, self.offset_column)
        preds = []
        for fitted_estimator, fold_pipeline in zip(
            self.fitted_estimators_, self.fitted_pipelines_
        ):
            x_proc = self._transform_with_pipeline(fold_pipeline, X)
            x_ordered = self._order_x_for_estimator(x_proc, fitted_estimator)
            proba = self.problem_type == "classification" and hasattr(
                fitted_estimator, "predict_proba"
            )
            preds.append(
                self._estimator_output(fitted_estimator, x_ordered, offset, proba)
            )

        preds_arr = np.asarray(preds, dtype=float)
        avg = preds_arr.mean(axis=0)
        if self.problem_type == "classification":
            return np.argmax(avg, axis=1) if avg.ndim == 2 else (avg >= 0.5).astype(int)
        return avg

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities by averaging fold estimator probabilities.

        Applies each fold's fitted pipeline to X before calling the estimator.

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Averaged class probabilities (n_samples, n_classes) or
                (n_samples,) for binary.

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
                "CrossValidationTrainer must be fitted before calling predict_proba()."
            )

        X, offset = split_offset(X, self.offset_column)
        probs = []
        for fitted_estimator, fold_pipeline in zip(
            self.fitted_estimators_, self.fitted_pipelines_
        ):
            x_proc = self._transform_with_pipeline(fold_pipeline, X)
            x_ordered = self._order_x_for_estimator(x_proc, fitted_estimator)
            probs.append(
                self._estimator_output(fitted_estimator, x_ordered, offset, True)
            )

        return np.asarray(probs, dtype=float).mean(axis=0)

    def predict_with_confidence(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predict with uncertainty estimates aggregated across folds.

        Applies each fold's fitted pipeline before calling the estimator.
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

        X, offset = split_offset(X, self.offset_column)
        if self.problem_type == "regression":
            fold_preds = []
            for est, fold_pipeline in zip(
                self.fitted_estimators_, self.fitted_pipelines_
            ):
                x_proc = self._transform_with_pipeline(fold_pipeline, X)
                x_ordered = self._order_x_for_estimator(x_proc, est)
                fold_preds.append(
                    np.asarray(
                        self._estimator_output(est, x_ordered, offset, False),
                        dtype=float,
                    )
                )
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
        for est, fold_pipeline in zip(self.fitted_estimators_, self.fitted_pipelines_):
            x_proc = self._transform_with_pipeline(fold_pipeline, X)
            x_ordered = self._order_x_for_estimator(x_proc, est)
            if not hasattr(est, "predict_proba"):
                raise ValueError(
                    "All fold estimators must implement predict_proba for classification confidence."
                )
            p = np.asarray(
                self._estimator_output(est, x_ordered, offset, True), dtype=float
            )
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
    def _order_x_for_estimator(
        X: pd.DataFrame, estimator: BaseEstimator
    ) -> pd.DataFrame:
        """Reorder columns to match an estimator's feature order (see `order_x_for_estimator`)."""
        return order_x_for_estimator(X, estimator)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from feature_engine.encoding import MeanEncoder
    from feature_engine.imputation import MeanMedianImputer
    from feature_engine.outliers import Winsorizer
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.pipeline import Pipeline

    from kvbiii_ml.data_processing.feature_engineering.categorical_aligner import (
        CategoricalAligner,
    )

    RANDOM_STATE = 42
    N_SAMPLES = 3_000
    N_FOLDS = 3
    N_FEATURES = 10
    CAT_FEATURES = ["cat_1", "cat_2"]
    EARLY_STOPPING_ROUNDS = 30

    def _make_clf_data(n_classes: int) -> tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic classification dataset with numerical and categorical features."""
        rng = np.random.RandomState(RANDOM_STATE)
        x_num, y_arr = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=6,
            n_redundant=2,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=RANDOM_STATE,
        )
        df = pd.DataFrame(x_num, columns=[f"num_{i}" for i in range(N_FEATURES)])
        df["cat_1"] = pd.Categorical(rng.choice(["A", "B", "C", "D"], size=N_SAMPLES))
        df["cat_2"] = pd.Categorical(rng.choice(["X", "Y", "Z"], size=N_SAMPLES))
        return df, pd.Series(y_arr, name="target")

    def _make_reg_data() -> tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic regression dataset with numerical and categorical features."""
        rng = np.random.RandomState(RANDOM_STATE)
        x_num, y_arr = make_regression(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=6,
            random_state=RANDOM_STATE,
        )
        df = pd.DataFrame(x_num, columns=[f"num_{i}" for i in range(N_FEATURES)])
        df["cat_1"] = pd.Categorical(rng.choice(["A", "B", "C", "D"], size=N_SAMPLES))
        df["cat_2"] = pd.Categorical(rng.choice(["X", "Y", "Z"], size=N_SAMPLES))
        return df, pd.Series(y_arr, name="target")

    def _run_scenario(
        label: str,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.Series,
        metric_name: str,
        problem_type: str,
        preprocessing_pipeline: Pipeline | None = None,
    ) -> None:
        """Run one cross-validation scenario and print a one-line summary."""
        cv_cls = StratifiedKFold if problem_type == "classification" else KFold
        trainer = CrossValidationTrainer(
            problem_type=problem_type,
            metric_name=metric_name,
            cv=cv_cls(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            preprocessing_pipeline=preprocessing_pipeline,
            verbose=True,
        )
        _, valid_s, _ = trainer.fit(estimator, X, y)
        print(f"  {label:<50} valid={np.mean(valid_s):.4f} ± {np.std(valid_s):.4f}")

    def _run_demo() -> None:
        """Run CrossValidationTrainer across the demo scenario matrix."""
        _winsorizer = Winsorizer(
            capping_method="iqr", tail="both", fold=3.0, missing_values="ignore"
        )
        _mean_encoder = MeanEncoder(variables=CAT_FEATURES, missing_values="ignore")
        _imputer = MeanMedianImputer(imputation_method="median")
        _categorial_aligner = CategoricalAligner(variables=CAT_FEATURES)
        _pipe_steps = [
            ("imputer", _imputer),
            ("winsorizer", _winsorizer),
            ("mean_encoder", _mean_encoder),
            ("categorical_aligner", _categorial_aligner),
        ]
        clf_pipeline = Pipeline(_pipe_steps)
        reg_pipeline = Pipeline(_pipe_steps)

        x_bin, y_bin = _make_clf_data(n_classes=2)
        x_multi, y_multi = _make_clf_data(n_classes=3)
        x_reg, y_reg = _make_reg_data()

        x_bin_cat = x_bin.assign(**{c: x_bin[c].astype(str) for c in CAT_FEATURES})
        x_multi_cat = x_multi.assign(
            **{c: x_multi[c].astype(str) for c in CAT_FEATURES}
        )
        x_reg_cat = x_reg.assign(**{c: x_reg[c].astype(str) for c in CAT_FEATURES})

        es = EARLY_STOPPING_ROUNDS
        _n, _rs = 300, RANDOM_STATE
        _lgbm_clf = LGBMClassifier(
            n_estimators=_n,
            early_stopping_rounds=es,
            verbose=-1,
            random_state=_rs,
        )
        _lgbm_reg = LGBMRegressor(
            n_estimators=_n,
            early_stopping_rounds=es,
            verbose=-1,
            random_state=_rs,
        )

        print("=" * 70)
        print("CrossValidationTrainer - test matrix (3 folds, LGBM only)")
        print("=" * 70)

        _run_scenario(
            "LightGBM | binary classification",
            _lgbm_clf,
            x_bin_cat,
            y_bin,
            "Balanced Accuracy",
            "classification",
            clf_pipeline,
        )
        _run_scenario(
            "LightGBM | multiclass classification",
            _lgbm_clf,
            x_multi_cat,
            y_multi,
            "Balanced Accuracy",
            "classification",
            clf_pipeline,
        )
        _run_scenario(
            "LightGBM | regression",
            _lgbm_reg,
            x_reg_cat,
            y_reg,
            "RMSE",
            "regression",
            reg_pipeline,
        )

    _run_demo()
