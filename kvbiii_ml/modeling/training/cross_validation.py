import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.pipeline import Pipeline
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
            custom_metric (dict[str, Any] | None): Custom metric configuration with keys
                'name', 'function', and optionally 'metric_type', 'additional_data', and
                other kwargs. Defaults to None.

        Raises:
            TypeError: If preprocessing_pipeline is not a Pipeline or None.
            ValueError: If problem_type is invalid or neither metric_name nor
                custom_metric is provided.
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
        self.preprocessing_pipeline = preprocessing_pipeline
        self.verbose = verbose
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
            tuple: (X_train_out, X_valid_out, X_test_out, fitted_pipeline) where
                fitted_pipeline is the cloned and fitted Pipeline for this fold, or None.
        """
        if pipeline is None:
            return X_train, X_valid, X_test, None

        fitted = clone(pipeline)
        X_train_out = fitted.fit_transform(X_train, y_train)
        X_valid_out = fitted.transform(X_valid) if X_valid is not None else None
        X_test_out = fitted.transform(X_test) if X_test is not None else None
        return X_train_out, X_valid_out, X_test_out, fitted

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
        test split.

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
                Train scores, validation scores per fold, and averaged test predictions
                if X_test is provided, otherwise None.
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

            X_train_proc, X_valid_proc, X_test_proc, fitted_pipeline = (
                self._apply_pipeline(
                    active_pipeline,
                    X_train,
                    X_valid,
                    X_test,
                    y_train=y_train,
                )
            )

            estimator_fold = self._clone_estimator(estimator)
            y_train_pred, y_valid_pred, y_test_pred, fitted_estimator = (
                self.fit_and_predict(
                    estimator_fold,
                    X_train_proc,
                    y_train,
                    X_valid_proc,
                    y_valid,
                    self.metric_type,
                    X_test=X_test_proc,
                    verbose=self.verbose,
                )
            )

            train_score = self.metric_fn(y_train, y_train_pred)
            valid_score = self.metric_fn(y_valid, y_valid_pred)
            self.train_scores_.append(train_score)
            self.valid_scores_.append(valid_score)
            self.fitted_estimators_.append(fitted_estimator)
            self.fitted_pipelines_.append(fitted_pipeline)

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

        avg_test_pred = (
            (test_pred_sum / n_folds_for_test)
            if (test_pred_sum is not None and n_folds_for_test > 0)
            else None
        )
        return self.train_scores_, self.valid_scores_, avg_test_pred

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

        preds = []
        for fitted_estimator, fold_pipeline in zip(
            self.fitted_estimators_, self.fitted_pipelines_
        ):
            X_proc = self._transform_with_pipeline(fold_pipeline, X)
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

        probs = []
        for fitted_estimator, fold_pipeline in zip(
            self.fitted_estimators_, self.fitted_pipelines_
        ):
            X_proc = self._transform_with_pipeline(fold_pipeline, X)
            X_ordered = self._order_X_for_estimator(X_proc, fitted_estimator)
            probs.append(fitted_estimator.predict_proba(X_ordered))

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

        if self.problem_type == "regression":
            fold_preds = []
            for est, fold_pipeline in zip(
                self.fitted_estimators_, self.fitted_pipelines_
            ):
                X_proc = self._transform_with_pipeline(fold_pipeline, X)
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
        for est, fold_pipeline in zip(self.fitted_estimators_, self.fitted_pipelines_):
            X_proc = self._transform_with_pipeline(fold_pipeline, X)
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
            pd.DataFrame: Reordered features if the estimator exposes feature names,
                otherwise the original X.
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
        X_num, y_arr = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=6,
            n_redundant=2,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=RANDOM_STATE,
        )
        df = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(N_FEATURES)])
        df["cat_1"] = pd.Categorical(rng.choice(["A", "B", "C", "D"], size=N_SAMPLES))
        df["cat_2"] = pd.Categorical(rng.choice(["X", "Y", "Z"], size=N_SAMPLES))
        return df, pd.Series(y_arr, name="target")

    def _make_reg_data() -> tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic regression dataset with numerical and categorical features."""
        rng = np.random.RandomState(RANDOM_STATE)
        X_num, y_arr = make_regression(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=6,
            random_state=RANDOM_STATE,
        )
        df = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(N_FEATURES)])
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

    _winsorizer = Winsorizer(
        capping_method="iqr", tail="both", fold=3.0, missing_values="ignore"
    )
    _mean_encoder = MeanEncoder(variables=CAT_FEATURES, missing_values="ignore")
    _imputer = MeanMedianImputer(imputation_method="median")
    _categorial_aligner = CategoricalAligner(categorical_features=CAT_FEATURES)
    _pipe_steps = [
        ("imputer", _imputer),
        ("winsorizer", _winsorizer),
        ("mean_encoder", _mean_encoder),
        ("categorical_aligner", _categorial_aligner),
    ]
    clf_pipeline = Pipeline(_pipe_steps)
    reg_pipeline = Pipeline(_pipe_steps)

    X_bin, y_bin = _make_clf_data(n_classes=2)
    X_multi, y_multi = _make_clf_data(n_classes=3)
    X_reg, y_reg = _make_reg_data()

    X_bin_cat = X_bin.assign(**{c: X_bin[c].astype(str) for c in CAT_FEATURES})
    X_multi_cat = X_multi.assign(**{c: X_multi[c].astype(str) for c in CAT_FEATURES})
    X_reg_cat = X_reg.assign(**{c: X_reg[c].astype(str) for c in CAT_FEATURES})

    ES = EARLY_STOPPING_ROUNDS
    _N, _RS = 300, RANDOM_STATE
    _lgbm_clf = LGBMClassifier(
        n_estimators=_N,
        early_stopping_rounds=ES,
        verbose=-1,
        random_state=_RS,
    )
    _lgbm_reg = LGBMRegressor(
        n_estimators=_N,
        early_stopping_rounds=ES,
        verbose=-1,
        random_state=_RS,
    )

    print("=" * 70)
    print("CrossValidationTrainer - test matrix (3 folds, LGBM only)")
    print("=" * 70)

    _run_scenario(
        "LightGBM | binary classification",
        _lgbm_clf,
        X_bin_cat,
        y_bin,
        "Balanced Accuracy",
        "classification",
        clf_pipeline,
    )
    _run_scenario(
        "LightGBM | multiclass classification",
        _lgbm_clf,
        X_multi_cat,
        y_multi,
        "Balanced Accuracy",
        "classification",
        clf_pipeline,
    )
    _run_scenario(
        "LightGBM | regression",
        _lgbm_reg,
        X_reg_cat,
        y_reg,
        "RMSE",
        "regression",
        reg_pipeline,
    )
