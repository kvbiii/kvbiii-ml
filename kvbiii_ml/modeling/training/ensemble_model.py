import gc
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

from kvbiii_ml.modeling.training.base_trainer import BaseTrainer


class EnsembleModel(BaseEstimator):
    """Ensemble model combining multiple base estimators."""

    def __init__(
        self,
        estimators: list,
        problem_type: str,
        weights: np.ndarray | None = None,
        meta_learner: BaseEstimator | None = None,
    ) -> None:
        """
        Args:
            estimators (list): Base estimators to include in the ensemble.
            problem_type (str): Problem type, either "classification" or
                "regression".
            weights (np.ndarray | None): Weights for each estimator. If None,
                equal weights are used.
            meta_learner (BaseEstimator | None): Meta-learner for stacking.
                If None, ensemble uses blending (weighted averaging).
        """
        if not isinstance(estimators, Iterable) or len(estimators) == 0:
            raise ValueError("estimators must be a non-empty list of estimators.")

        if problem_type not in {"classification", "regression"}:
            raise ValueError(
                "problem_type must be either 'classification' or 'regression'."
            )

        self.estimators = estimators
        self.problem_type = problem_type
        self.weights = weights
        self.meta_learner = meta_learner
        self._estimators_list = list(estimators)
        self._weights_normalized = self._validate_and_normalize_weights(weights)
        self.classes_: np.ndarray | None = None
        self.fitted_estimators_: list[BaseEstimator] = []
        self.fitted_meta_learner_: BaseEstimator | None = None
        self._x_train: pd.DataFrame | None = None

        if problem_type == "classification":
            self.__class__ = type(
                self.__class__.__name__,
                (ClassifierMixin, BaseEstimator),
                dict(self.__class__.__dict__),
            )
        else:
            self.__class__ = type(
                self.__class__.__name__,
                (RegressorMixin, BaseEstimator),
                dict(self.__class__.__dict__),
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

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            eval_set (tuple[pd.DataFrame, pd.Series] | None): Optional validation set for early stopping.
            sample_weight (pd.Series | None): Optional sample weights.
            verbose (int | None): Optional verbosity for fitting.

        Returns:
            EnsembleModel: Fitted ensemble instance.
        """
        if (
            eval_set is not None
            and isinstance(eval_set, list)
            and len(eval_set) >= 1
            and isinstance(eval_set[0], tuple)
        ):
            if len(eval_set) == 2:
                X_valid, y_valid = eval_set[1]
            else:
                X_valid, y_valid = eval_set[0]

        else:
            X_valid, y_valid = None, None
        self.fitted_estimators_ = []
        self._x_train = X_train
        for base_estimator in self._estimators_list:
            estimator = self._safe_clone_estimator(base_estimator)
            estimator = self._set_categorical_params_for_estimator(estimator)
            fitted = BaseTrainer.fit_estimator(
                estimator, X_train, y_train, X_valid, y_valid, sample_weight, verbose
            )
            self.fitted_estimators_.append(fitted)
        self._x_train = None

        if self.problem_type == "classification":
            self.classes_ = np.unique(y_train)

        if self.meta_learner is not None:
            meta_features = self._generate_meta_features(X_train)
            self.fitted_meta_learner_ = self._safe_clone_estimator(self.meta_learner)
            self.fitted_meta_learner_.fit(
                meta_features, y_train, sample_weight=sample_weight
            )
        return self

    def _generate_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Generate meta-features from base estimators' predictions.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Meta-features (base estimators' predictions stacked).
        """
        meta_features = []
        for estimator in self.fitted_estimators_:
            x_ordered = self._order_x_for_estimator(X, estimator)
            if self.problem_type == "classification":
                feat = estimator.predict_proba(x_ordered)
            else:
                feat = estimator.predict(x_ordered).reshape(-1, 1)
            meta_features.append(feat)
        return np.hstack(meta_features)

    def _set_categorical_params_for_estimator(
        self,
        estimator: BaseEstimator,
    ) -> BaseEstimator:
        """Assign or sanitise categorical feature parameters for supported estimators.

        For CatBoost: filters any pre-configured ``cat_features`` to exclude columns
        that are numeric in ``X_train`` (e.g. encoded to float64 by a pipeline step
        such as MeanEncoder).  Falls back to auto-detecting ``category``-dtype columns
        when no ``cat_features`` were pre-configured.

        For HistGradientBoosting: auto-detects ``category``-dtype columns.

        Args:
            estimator (BaseEstimator): Estimator to configure.

        Returns:
            BaseEstimator: Estimator with categorical parameters set when applicable.
        """
        if self._x_train is None:
            return estimator

        name = estimator.__class__.__name__

        if name.startswith("CatBoost"):
            existing_cats = estimator.get_params().get("cat_features") or []
            if existing_cats:
                active_cats = [
                    c
                    for c in existing_cats
                    if c in self._x_train.columns
                    and not pd.api.types.is_numeric_dtype(self._x_train[c])
                ]
                if set(active_cats) != set(existing_cats):
                    estimator.set_params(
                        cat_features=active_cats if active_cats else None
                    )
            else:
                try:
                    auto_cats = self._x_train.select_dtypes(
                        include="category"
                    ).columns.tolist()
                except (ValueError, TypeError, AttributeError):
                    auto_cats = []
                if auto_cats:
                    estimator.set_params(cat_features=auto_cats)

        elif name.startswith("HistGradientBoosting"):
            try:
                auto_cats = self._x_train.select_dtypes(
                    include="category"
                ).columns.tolist()
            except (ValueError, TypeError, AttributeError):
                auto_cats = []
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
            RuntimeError: If called before fitting.
        """
        if not hasattr(self, "fitted_estimators_") or len(self.fitted_estimators_) == 0:
            raise RuntimeError("EnsembleModel must be fitted before calling predict().")

        if self.fitted_meta_learner_ is not None:
            meta_features = self._generate_meta_features(X)
            return self.fitted_meta_learner_.predict(meta_features)

        if self.problem_type == "classification":
            proba = self.predict_proba(X)
            return (
                self.classes_[np.argmax(proba, axis=1)]
                if self.classes_ is not None
                else np.argmax(proba, axis=1)
            )

        predictions = []
        successful_indices = []
        try:
            for idx, estimator in enumerate(self.fitted_estimators_):
                try:
                    x_ordered = self._order_x_for_estimator(X, estimator)
                    if self.problem_type == "classification":
                        estimator_predictions = estimator.predict_proba(x_ordered)
                    else:
                        estimator_predictions = estimator.predict(x_ordered)
                    predictions.append(estimator_predictions)
                    successful_indices.append(idx)
                except (ValueError, AttributeError, TypeError):
                    continue

            if not predictions:
                raise RuntimeError("All estimators failed during prediction.")

            predictions = np.array(predictions)
            active_weights = self._weights_normalized[successful_indices]
            active_weights = active_weights / active_weights.sum()

            predictions = np.average(predictions, axis=0, weights=active_weights)
            if self.problem_type == "classification":
                return np.argmax(predictions, axis=1)
            return predictions
        finally:
            if "predictions" in locals() and isinstance(predictions, list):
                del predictions
            gc.collect()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using the ensemble (stacking or blending).

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Averaged class probabilities.

        Raises:
            RuntimeError: If called before fitting or if problem_type is regression.
        """
        if self.problem_type == "regression":
            raise AttributeError(
                "predict_proba is not available for regression problems"
            )

        if not hasattr(self, "fitted_estimators_") or len(self.fitted_estimators_) == 0:
            raise RuntimeError("EnsembleModel must be fitted before calling predict().")

        if self.fitted_meta_learner_ is not None:
            meta_features = self._generate_meta_features(X)
            return self._prepare_probabilities(
                self.fitted_meta_learner_.predict_proba(meta_features)
            )

        predictions = []
        successful_indices = []
        try:
            for idx, estimator in enumerate(self.fitted_estimators_):
                try:
                    x_ordered = self._order_x_for_estimator(X, estimator)
                    estimator_predictions = estimator.predict_proba(x_ordered)
                    estimator_predictions = self._prepare_probabilities(
                        estimator_predictions
                    )
                    predictions.append(estimator_predictions)
                    successful_indices.append(idx)
                except (ValueError, AttributeError, TypeError) as e:
                    print(f"Estimator {idx} failed during predict_proba: {e}")
                    continue

            if not predictions:
                raise RuntimeError("All estimators failed during prediction.")

            predictions = np.array(predictions)
            active_weights = self._weights_normalized[successful_indices]
            active_weights = active_weights / active_weights.sum()

            proba = np.average(predictions, axis=0, weights=active_weights)
            return self._prepare_probabilities(proba)
        finally:
            if "predictions" in locals() and isinstance(predictions, list):
                del predictions

            gc.collect()

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

    def _order_x_for_estimator(
        self, X: pd.DataFrame, estimator: BaseEstimator
    ) -> pd.DataFrame:
        """Reorder columns to match the estimator's expected feature order.

        Args:
            X (pd.DataFrame): Input features.
            estimator (BaseEstimator): Trained estimator.

        Returns:
            pd.DataFrame: Reordered features if the estimator exposes feature
            names; otherwise returns X unchanged.
        """
        if hasattr(estimator, "feature_names_in_"):
            return X[estimator.feature_names_in_]
        if hasattr(estimator, "feature_names_"):
            return X[estimator.feature_names_]
        return X

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
