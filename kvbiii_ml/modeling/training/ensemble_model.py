import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.utils import ClassifierTags, RegressorTags, Tags

from kvbiii_ml.modeling.training.base_trainer import BaseTrainer, order_x_for_estimator

ESTIMATOR_ERROR_POLICIES = ("raise", "skip")
SKIPPABLE_ERRORS = (ValueError, AttributeError, TypeError)
DEFAULT_STACKING_SPLITS = 5
DEFAULT_STACKING_SEED = 17


class EnsembleModel(BaseEstimator):
    """Ensemble model combining multiple base estimators."""

    def __init__(
        self,
        estimators: list,
        problem_type: str,
        weights: np.ndarray | None = None,
        meta_learner: BaseEstimator | None = None,
        n_jobs: int = 1,
        cv: BaseCrossValidator | None = None,
        on_estimator_error: str = "raise",
    ) -> None:
        """
        Args:
            estimators (list): Base estimators to include in the ensemble.
            problem_type (str): Problem type, either "classification" or
                "regression".
            weights (np.ndarray | None): Weights for each estimator, stored normalised
                to sum to 1. If None, equal weights are used.
            meta_learner (BaseEstimator | None): Meta-learner for stacking.
                If None, ensemble uses blending (weighted averaging).
            n_jobs (int): Base estimators fitted concurrently, in threads. The boosting
                libraries release the GIL while training, so the threads run truly in
                parallel; keep n_jobs times each estimator's own thread count within the
                physical cores. Fitted estimators keep the order of `estimators`, so the
                result is identical to a sequential fit. Leave it at 1 when the ensemble
                is itself fitted inside parallel cross-validation folds. Defaults to 1.
            cv (BaseCrossValidator | None): Splitter used to build the stacking
                meta-features out-of-fold when ``meta_learner`` is set: each base
                estimator is refitted on every training split and predicts its held-out
                split, so the meta-learner never learns from in-sample (overfit) base
                predictions. The base estimators used at predict time are still fitted on
                all rows. Ignored for blending. Defaults to None, which means
                ``KFold(5, shuffle=True, random_state=17)``.
            on_estimator_error (str): What ``predict``/``predict_proba`` do when a base
                estimator raises ValueError, AttributeError or TypeError. ``"raise"``
                lets the exception propagate. ``"skip"`` drops that estimator, warns,
                and renormalises the remaining weights - the result is then a different
                model from the one that was fitted. Defaults to "raise".

        Raises:
            ValueError: If estimators is empty, n_jobs is not positive, problem_type or
                on_estimator_error is invalid, or weights are invalid.
        """
        if not isinstance(estimators, Iterable) or len(estimators) == 0:
            raise ValueError("estimators must be a non-empty list of estimators.")

        if n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer.")

        if problem_type not in {"classification", "regression"}:
            raise ValueError(
                "problem_type must be either 'classification' or 'regression'."
            )

        if on_estimator_error not in ESTIMATOR_ERROR_POLICIES:
            raise ValueError(
                f"on_estimator_error must be one of {ESTIMATOR_ERROR_POLICIES}."
            )

        self.estimators = estimators
        self.problem_type = problem_type
        self.meta_learner = meta_learner
        self.n_jobs = n_jobs
        self.cv = cv
        self.on_estimator_error = on_estimator_error
        self._estimators_list = list(estimators)
        self._weights_normalized = self._validate_and_normalize_weights(weights)
        self.weights = self._weights_normalized
        self.classes_: np.ndarray | None = None
        self.fitted_estimators_: list[BaseEstimator] = []
        self.fitted_meta_learner_: BaseEstimator | None = None

    def __sklearn_tags__(self) -> Tags:
        """Tag the ensemble as a classifier or regressor according to problem_type."""
        tags = super().__sklearn_tags__()
        if self.problem_type == "classification":
            tags.estimator_type = "classifier"
            tags.classifier_tags = ClassifierTags()
        else:
            tags.estimator_type = "regressor"
            tags.regressor_tags = RegressorTags()
        tags.target_tags.required = True
        return tags

    def __sklearn_clone__(self) -> "EnsembleModel":
        """Return an unfitted copy with the same parameters.

        sklearn's default clone rejects this class because ``__init__`` stores
        ``weights`` normalised rather than as passed; re-normalising already
        normalised weights is a no-op, so rebuilding from ``get_params`` is exact.
        """
        return type(self)(
            **{
                name: clone(value, safe=False)
                for name, value in self.get_params(deep=False).items()
            }
        )

    def _safe_clone_estimator(self, estimator: BaseEstimator) -> BaseEstimator:
        """
        Clone estimator with fallback for non-sklearn-clone-compatible models.

        Args:
            estimator (BaseEstimator): Estimator to clone.

        Returns:
            BaseEstimator: Cloned estimator if possible, otherwise the original.
        """
        try:
            return clone(estimator)
        except (TypeError, RuntimeError, AttributeError, ValueError, KeyError):
            try:
                return deepcopy(estimator)
            except (
                TypeError,
                RuntimeError,
                AttributeError,
                ValueError,
                KeyError,
                NotImplementedError,
            ):
                return estimator

    def _validate_and_normalize_weights(self, weights: np.ndarray | None) -> np.ndarray:
        """Validate and normalize estimator weights.

        Args:
            weights (np.ndarray | None): Input weights or None.

        Returns:
            np.ndarray: Normalized weights that sum to 1.

        Raises:
            ValueError: If weights length mismatches estimators, contains
                non-finite values, or sums to zero.
        """
        if weights is None:
            return np.ones(len(self._estimators_list), dtype=float) / len(
                self._estimators_list
            )
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != len(self._estimators_list):
            raise ValueError(
                "The number of estimators must match the number of weights."
            )
        if np.any(~np.isfinite(w)):
            raise ValueError("weights must be finite numbers.")
        s = w.sum()
        if s == 0:
            raise ValueError("weights must not sum to zero.")
        return w / s

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
        sample_weight: pd.Series | None = None,
        verbose: int | None = None,
    ) -> "EnsembleModel":
        """Fit all base estimators and optional meta-learner.

        Without an ``eval_set``, early stopping configured on a LightGBM, XGBoost or
        CatBoost base estimator is turned off (with a warning) instead of raising.
        With a ``meta_learner``, its training features are built out-of-fold with
        ``cv`` (see ``__init__``).

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            eval_set (list[tuple[pd.DataFrame, pd.Series]] | None): Optional validation
                set for early stopping; the last of up to two (X, y) pairs is used.
            sample_weight (pd.Series | None): Optional sample weights.
            verbose (int | None): Optional verbosity for fitting.

        Returns:
            EnsembleModel: Fitted ensemble instance.
        """
        X_valid, y_valid = None, None
        if isinstance(eval_set, list) and eval_set and isinstance(eval_set[0], tuple):
            X_valid, y_valid = eval_set[1] if len(eval_set) == 2 else eval_set[0]

        self.fitted_estimators_ = self._fit_base_estimators(
            X_train, y_train, X_valid, y_valid, sample_weight, verbose
        )
        if self.problem_type == "classification":
            self.classes_ = np.unique(y_train)

        if self.meta_learner is not None:
            meta_features = self._out_of_fold_meta_features(
                X_train, y_train, X_valid, y_valid, sample_weight, verbose
            )
            self.fitted_meta_learner_ = self._safe_clone_estimator(self.meta_learner)
            self.fitted_meta_learner_.fit(
                meta_features, y_train, sample_weight=sample_weight
            )
        return self

    def _fit_base_estimators(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None,
        y_valid: pd.Series | None,
        sample_weight: pd.Series | np.ndarray | None,
        verbose: int | None,
    ) -> list[BaseEstimator]:
        """Clone, configure and fit every base estimator, in `estimators` order."""

        def fit_one(base_estimator: BaseEstimator) -> BaseEstimator:
            """Clone, configure and fit one base estimator."""
            estimator = self._set_categorical_params_for_estimator(
                self._safe_clone_estimator(base_estimator), X_train
            )
            return BaseTrainer.fit_estimator(
                estimator, X_train, y_train, X_valid, y_valid, sample_weight, verbose
            )

        if self.n_jobs == 1 or len(self._estimators_list) <= 1:
            return [fit_one(base) for base in self._estimators_list]
        max_workers = min(self.n_jobs, len(self._estimators_list))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(fit_one, self._estimators_list))

    def _out_of_fold_meta_features(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None,
        y_valid: pd.Series | None,
        sample_weight: pd.Series | np.ndarray | None,
        verbose: int | None,
    ) -> np.ndarray:
        """Build stacking meta-features from base models refitted on each `cv` training split.

        Equivalent to sklearn's ``cross_val_predict``: every row's meta-features come
        from base estimators that never saw that row.

        Returns:
            np.ndarray: Out-of-fold meta-features aligned with ``X_train`` rows.

        Raises:
            ValueError: If folds disagree on the number of meta-features.
        """
        cv = self.cv or KFold(
            n_splits=DEFAULT_STACKING_SPLITS,
            shuffle=True,
            random_state=DEFAULT_STACKING_SEED,
        )
        weights = None if sample_weight is None else np.asarray(sample_weight)
        meta_features: np.ndarray | None = None
        for train_idx, valid_idx in cv.split(X_train, y_train):
            fold_estimators = self._fit_base_estimators(
                X_train.iloc[train_idx],
                y_train.iloc[train_idx],
                X_valid,
                y_valid,
                None if weights is None else weights[train_idx],
                verbose,
            )
            fold_features = self._meta_features(
                fold_estimators, X_train.iloc[valid_idx]
            )
            if meta_features is None:
                meta_features = np.zeros((len(X_train), fold_features.shape[1]))
            if fold_features.shape[1] != meta_features.shape[1]:
                raise ValueError(
                    "A stacking fold produced a different number of meta-features "
                    "(a class is missing from one training split); pass a "
                    "stratified splitter as cv."
                )
            meta_features[valid_idx] = fold_features
        return meta_features

    def _generate_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Generate meta-features from the fitted base estimators' predictions."""
        return self._meta_features(self.fitted_estimators_, X)

    def _meta_features(
        self, estimators: list[BaseEstimator], X: pd.DataFrame
    ) -> np.ndarray:
        """Stack class probabilities (classification) or predictions (regression) of estimators."""
        meta_features = []
        for estimator in estimators:
            x_ordered = self._order_x_for_estimator(X, estimator)
            if self.problem_type == "classification":
                feat = estimator.predict_proba(x_ordered)
            else:
                feat = np.asarray(estimator.predict(x_ordered)).reshape(-1, 1)
            meta_features.append(feat)
        return np.hstack(meta_features)

    def _set_categorical_params_for_estimator(
        self,
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
    ) -> BaseEstimator:
        """Assign or sanitise categorical feature parameters for supported estimators.

        For CatBoost: filters any pre-configured ``cat_features`` to exclude columns
        that are numeric in ``X_train`` (e.g. encoded to float64 by a pipeline step
        such as MeanEncoder).  Falls back to auto-detecting ``category``-dtype columns
        when no ``cat_features`` were pre-configured.

        For HistGradientBoosting: auto-detects ``category``-dtype columns.

        Args:
            estimator (BaseEstimator): Estimator to configure.
            X_train (pd.DataFrame): Training features the estimator will be fitted on.

        Returns:
            BaseEstimator: Estimator with categorical parameters set when applicable.
        """
        name = estimator.__class__.__name__

        if name.startswith("CatBoost"):
            existing_cats = estimator.get_params().get("cat_features") or []
            if existing_cats:
                active_cats = [
                    c
                    for c in existing_cats
                    if c in X_train.columns
                    and not pd.api.types.is_numeric_dtype(X_train[c])
                ]
                if set(active_cats) != set(existing_cats):
                    estimator.set_params(
                        cat_features=active_cats if active_cats else None
                    )
            else:
                auto_cats = X_train.select_dtypes(include="category").columns.tolist()
                if auto_cats:
                    estimator.set_params(cat_features=auto_cats)

        elif name.startswith("HistGradientBoosting"):
            auto_cats = X_train.select_dtypes(include="category").columns.tolist()
            if auto_cats:
                estimator.set_params(categorical_features=auto_cats)

        return estimator

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the ensemble (stacking or blending).

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Predicted labels for classification or predictions for regression.

        Raises:
            RuntimeError: If called before fitting, or if every estimator was skipped.
        """
        if not getattr(self, "fitted_estimators_", None):
            raise RuntimeError("EnsembleModel must be fitted before calling predict().")

        if self.fitted_meta_learner_ is not None:
            return self.fitted_meta_learner_.predict(self._generate_meta_features(X))

        if self.problem_type == "classification":
            proba = self.predict_proba(X)
            labels = np.argmax(proba, axis=1)
            return self.classes_[labels] if self.classes_ is not None else labels

        predictions, weights = self._collect_predictions(
            X, lambda est, x: est.predict(x)
        )
        return np.average(predictions, axis=0, weights=weights)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using the ensemble (stacking or blending).

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Averaged class probabilities.

        Raises:
            AttributeError: If problem_type is regression.
            RuntimeError: If called before fitting, or if every estimator was skipped.
        """
        if self.problem_type == "regression":
            raise AttributeError(
                "predict_proba is not available for regression problems"
            )

        if not getattr(self, "fitted_estimators_", None):
            raise RuntimeError("EnsembleModel must be fitted before calling predict().")

        if self.fitted_meta_learner_ is not None:
            meta_features = self._generate_meta_features(X)
            return self._prepare_probabilities(
                self.fitted_meta_learner_.predict_proba(meta_features)
            )

        predictions, weights = self._collect_predictions(
            X, lambda est, x: self._prepare_probabilities(est.predict_proba(x))
        )
        return self._prepare_probabilities(
            np.average(predictions, axis=0, weights=weights)
        )

    def _collect_predictions(
        self,
        X: pd.DataFrame,
        predict_fn: Callable[[BaseEstimator, pd.DataFrame], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run predict_fn on every base estimator, honouring ``on_estimator_error``.

        Args:
            X (pd.DataFrame): Features to score.
            predict_fn (Callable[[BaseEstimator, pd.DataFrame], np.ndarray]): Called
                as ``predict_fn(estimator, X_ordered)``.

        Returns:
            tuple[np.ndarray, np.ndarray]: Stacked predictions of the estimators that
                succeeded and their renormalised weights.

        Raises:
            RuntimeError: If every estimator failed under ``on_estimator_error="skip"``.
        """
        predictions = []
        used = []
        for idx, estimator in enumerate(self.fitted_estimators_):
            try:
                predictions.append(
                    predict_fn(estimator, self._order_x_for_estimator(X, estimator))
                )
            except SKIPPABLE_ERRORS as exc:
                if self.on_estimator_error == "raise":
                    raise
                warnings.warn(
                    f"Skipping base estimator {idx} ({type(estimator).__name__}) in "
                    f"EnsembleModel prediction: {exc}",
                    UserWarning,
                    stacklevel=3,
                )
                continue
            used.append(idx)

        if not predictions:
            raise RuntimeError("All estimators failed during prediction.")
        weights = self._weights_normalized[used]
        return np.asarray(predictions), weights / weights.sum()

    def predict_with_confidence(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Predict with confidence estimation for both regression and classification.

        For regression: returns mean prediction, std deviation, and 95% CI.
        For classification: returns predicted class, mean probability, and
        disagreement (std of probabilities across estimators).

        Note: When using stacking, disagreement is unavailable and returned as zeros.

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            dict[str, np.ndarray]: Dictionary with keys depending on problem type.
        """
        if not hasattr(self, "fitted_estimators_") or len(self.fitted_estimators_) == 0:
            raise RuntimeError(
                "EnsembleModel must be fitted before calling predict_with_confidence()."
            )

        if self.fitted_meta_learner_ is not None:
            if self.problem_type == "regression":
                pred = self.predict(X)
                return {
                    "prediction": pred,
                    "std": np.zeros_like(pred),
                    "ci_95_lower": pred,
                    "ci_95_upper": pred,
                }
            else:
                proba = self.predict_proba(X)
                pred_class = np.argmax(proba, axis=1)
                pred_confidence = proba[np.arange(len(pred_class)), pred_class]
                return {
                    "prediction": pred_class,
                    "confidence": pred_confidence,
                    "disagreement": np.zeros_like(pred_confidence),
                    "proba": proba,
                }

        predictions = []
        for estimator in self.fitted_estimators_:
            x_ordered = self._order_x_for_estimator(X, estimator)
            if self.problem_type == "classification":
                estimator_predictions = estimator.predict_proba(x_ordered)
                estimator_predictions = self._prepare_probabilities(
                    estimator_predictions
                )
            else:
                estimator_predictions = estimator.predict(x_ordered)
            predictions.append(estimator_predictions)
        predictions = np.array(predictions)
        if self.problem_type == "regression":
            mean_pred = np.average(
                predictions, axis=0, weights=self._weights_normalized
            )
            std_pred = np.sqrt(
                np.average(
                    (predictions - mean_pred) ** 2,
                    axis=0,
                    weights=self._weights_normalized,
                )
            )
            ci_lower = mean_pred - 1.96 * std_pred
            ci_upper = mean_pred + 1.96 * std_pred

            return {
                "prediction": mean_pred,
                "std": std_pred,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
            }

        else:
            mean_proba = np.average(
                predictions, axis=0, weights=self._weights_normalized
            )
            pred_class = np.argmax(mean_proba, axis=1)
            pred_confidence = mean_proba[np.arange(len(pred_class)), pred_class]
            disagreement = predictions[:, np.arange(len(pred_class)), pred_class].std(
                axis=0
            )

            return {
                "prediction": pred_class,
                "confidence": pred_confidence,
                "disagreement": disagreement,
                "proba": mean_proba,
            }

    def score(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None
    ) -> float:
        """Return accuracy for classification or R^2 for regression.

        Args:
            X (pd.DataFrame): Features to score.
            y (pd.Series): True target values.
            sample_weight (pd.Series | None): Optional sample weights. Defaults to None.

        Returns:
            float: Accuracy for classification, R^2 for regression.
        """
        metric = accuracy_score if self.problem_type == "classification" else r2_score
        return float(metric(y, self.predict(X), sample_weight=sample_weight))

    def _order_x_for_estimator(
        self, X: pd.DataFrame, estimator: BaseEstimator
    ) -> pd.DataFrame:
        """Reorder columns to the estimator's feature order, raising if any is missing.

        Args:
            X (pd.DataFrame): Input features.
            estimator (BaseEstimator): Trained estimator.

        Returns:
            pd.DataFrame: Reordered features (see ``order_x_for_estimator``).

        Raises:
            ValueError: If the estimator exposes feature names that X cannot supply.
        """
        return order_x_for_estimator(X, estimator, strict=True)

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Compute weighted average of feature importances from base estimators.

        Returns:
            np.ndarray: Aggregated feature importances across all estimators
            that support this attribute.

        Raises:
            RuntimeError: If called before fitting or if no estimators support
                feature importances.
        """
        if not hasattr(self, "fitted_estimators_") or len(self.fitted_estimators_) == 0:
            raise RuntimeError(
                "EnsembleModel must be fitted before accessing feature_importances_."
            )

        importances = []
        successful_indices = []

        for idx, estimator in enumerate(self.fitted_estimators_):
            if hasattr(estimator, "feature_importances_"):
                feature_importance = np.asarray(
                    estimator.feature_importances_, dtype=float
                )
                min_importance = feature_importance.min()
                max_importance = feature_importance.max()
                if max_importance > min_importance:
                    normalized_importance = (feature_importance - min_importance) / (
                        max_importance - min_importance
                    )
                else:
                    normalized_importance = np.zeros_like(feature_importance)
                importances.append(normalized_importance)
                successful_indices.append(idx)

        if not importances:
            raise RuntimeError(
                "None of the fitted estimators support feature_importances_.",
            )

        importances = np.array(importances)
        active_weights = self._weights_normalized[successful_indices]
        active_weights = active_weights / active_weights.sum()

        return np.average(importances, axis=0, weights=active_weights)

    def _prepare_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Coerce probability outputs into a finite row-normalized matrix."""
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim == 1:
            probabilities = np.column_stack((1.0 - probabilities, probabilities))
        probabilities = np.clip(probabilities, 0.0, 1.0)
        row_sums = probabilities.sum(axis=1, keepdims=True)
        return np.divide(
            probabilities,
            row_sums,
            out=np.full_like(probabilities, 1.0 / probabilities.shape[1]),
            where=row_sums > 0,
        )


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    print("=== Blending (default) ===")
    ensemble_blend = EnsembleModel(
        estimators=[
            RandomForestClassifier(n_estimators=10),
            LogisticRegression(max_iter=100),
        ],
        problem_type="classification",
    )
    ensemble_blend.fit(X, y)
    predictions_blend = ensemble_blend.predict(X)
    print(
        f"Blending predictions value counts:\n{pd.Series(predictions_blend).value_counts()}"
    )

    print("\n=== Blending with XGBClassifier (verbose must be suppressed) ===")
    ensemble_xgb = EnsembleModel(
        estimators=[
            XGBClassifier(n_estimators=50, verbosity=0),
            RandomForestClassifier(n_estimators=10),
        ],
        problem_type="classification",
    )
    ensemble_xgb.fit(X, y, eval_set=[(X, y)])
    predictions_xgb = ensemble_xgb.predict(X)
    print(
        f"XGB blending predictions value counts:\n{pd.Series(predictions_xgb).value_counts()}"
    )

    print("\n=== Stacking with LogisticRegression meta-learner ===")
    ensemble_stack = EnsembleModel(
        estimators=[
            RandomForestClassifier(n_estimators=10),
            LogisticRegression(max_iter=1000),
        ],
        problem_type="classification",
        meta_learner=LogisticRegression(max_iter=1000),
    )
    ensemble_stack.fit(X, y)
    predictions_stack = ensemble_stack.predict(X)
    print(
        f"Stacking predictions value counts:\n{pd.Series(predictions_stack).value_counts()}"
    )
