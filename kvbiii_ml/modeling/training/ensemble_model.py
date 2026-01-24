import gc
from collections.abc import Iterable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from kvbiii_ml.modeling.training.base_trainer import BaseTrainer


class EnsembleModel(BaseTrainer, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        estimators: list,
        problem_type: str,
        weights: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            estimators (list): Base estimators to include in the ensemble.
            problem_type (str): Problem type, either "classification" or
                "regression".
            weights (np.ndarray | None): Weights for each estimator. If None,
                equal weights are used.
        """
        if not isinstance(estimators, Iterable) or len(estimators) == 0:
            raise ValueError("estimators must be a non-empty list of estimators.")

        if problem_type not in {"classification", "regression"}:
            raise ValueError(
                "problem_type must be either 'classification' or 'regression'."
            )

        self.estimators = list(estimators)
        self.problem_type = problem_type
        self.weights = self._validate_and_normalize_weights(weights)
        self.classes_: np.ndarray | None = None
        self.fitted_estimators_: list[BaseEstimator] = []

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
            return np.ones(len(self.estimators), dtype=float) / len(self.estimators)
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != len(self.estimators):
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
    ) -> "EnsembleModel":
        """Fit all base estimators.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            eval_set (tuple[pd.DataFrame, pd.Series] | None): Optional validation set for early stopping.
            sample_weight (pd.Series | None): Optional sample weights.

        Returns:
            EnsembleModel: Fitted ensemble instance.
        """
        if (
            eval_set is not None
            and isinstance(eval_set, list)
            and len(eval_set) == 1
            and isinstance(eval_set[0], tuple)
        ):
            X_valid, y_valid = eval_set[0]
        else:
            X_valid, y_valid = None, None

        for base_estimator in self.estimators:
            try:
                estimator = clone(base_estimator)
            except Exception:
                estimator = base_estimator
            estimator = self._set_categorical_params(estimator, X_train)
            fitted = BaseTrainer.fit_estimator(
                estimator, X_train, y_train, X_valid, y_valid, sample_weight
            )
            self.fitted_estimators_.append(fitted)

        if self.problem_type == "classification":
            self.classes_ = np.unique(y_train)
        return self

    def _set_categorical_params(
        self, estimator: BaseEstimator, X_train: pd.DataFrame
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
        if not categorical_features:
            return estimator
        name = estimator.__class__.__name__
        if name.startswith("CatBoost"):
            estimator.set_params(cat_features=categorical_features)
        elif name.startswith("HistGradientBoosting"):
            estimator.set_params(categorical_features=categorical_features)
        return estimator

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the ensemble.

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Predicted labels for classification or predictions for
            regression.

        Raises:
            RuntimeError: If called before fitting.
        """
        if not hasattr(self, "fitted_estimators_") or len(self.fitted_estimators_) == 0:
            raise RuntimeError("EnsembleModel must be fitted before calling predict().")

        predictions = []
        successful_indices = []
        try:
            for idx, estimator in enumerate(self.fitted_estimators_):
                try:
                    X_ordered = self._order_X_for_estimator(X, estimator)
                    if self.problem_type == "classification":
                        estimator_predictions = estimator.predict_proba(X_ordered)
                    else:
                        estimator_predictions = estimator.predict(X_ordered)
                    predictions.append(estimator_predictions)
                    successful_indices.append(idx)
                except Exception as e:
                    continue

            if not predictions:
                raise RuntimeError("All estimators failed during prediction.")

            predictions = np.array(predictions)
            active_weights = self.weights[successful_indices]
            active_weights = active_weights / active_weights.sum()

            predictions = np.average(predictions, axis=0, weights=active_weights)
            if self.problem_type == "classification":
                return np.argmax(predictions, axis=1)
            else:
                return predictions
        finally:
            if "predictions" in locals() and isinstance(predictions, list):
                del predictions
            gc.collect()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using the ensemble.

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            np.ndarray: Averaged class probabilities.

        Raises:
            RuntimeError: If called before fitting.
        """
        if not hasattr(self, "fitted_estimators_") or len(self.fitted_estimators_) == 0:
            raise RuntimeError("EnsembleModel must be fitted before calling predict().")

        predictions = []
        successful_indices = []
        try:
            for idx, estimator in enumerate(self.fitted_estimators_):
                try:
                    X_ordered = self._order_X_for_estimator(X, estimator)
                    estimator_predictions = estimator.predict_proba(X_ordered)
                    predictions.append(estimator_predictions)
                    successful_indices.append(idx)
                except Exception as e:
                    print(f"Estimator {idx} failed during predict_proba: {e}")
                    continue

            if not predictions:
                raise RuntimeError("All estimators failed during prediction.")

            predictions = np.array(predictions)
            active_weights = self.weights[successful_indices]
            active_weights = active_weights / active_weights.sum()

            return np.average(predictions, axis=0, weights=active_weights)
        finally:
            if "predictions" in locals() and isinstance(predictions, list):
                del predictions
            import gc

            gc.collect()

    def predict_with_confidence(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Predict with confidence estimation for both regression and classification.

        For regression: returns mean prediction, std deviation, and 95% CI.
        For classification: returns predicted class, mean probability, and
        disagreement (std of probabilities across estimators).

        Args:
            X (pd.DataFrame): Features to score.

        Returns:
            dict [str, np.ndarray]: Dictionary with keys:
        """
        if not hasattr(self, "fitted_estimators_") or len(self.fitted_estimators_) == 0:
            raise RuntimeError(
                "EnsembleModel must be fitted before calling predict_with_confidence()."
            )
        predictions = []
        for estimator in self.fitted_estimators_:
            X_ordered = self._order_X_for_estimator(X, estimator)
            if self.problem_type == "classification":
                estimator_predictions = estimator.predict_proba(X_ordered)
            else:
                estimator_predictions = estimator.predict(X_ordered)
            predictions.append(estimator_predictions)
        predictions = np.array(predictions)
        if self.problem_type == "regression":
            mean_pred = np.average(predictions, axis=0, weights=self.weights)
            std_pred = np.sqrt(
                np.average((predictions - mean_pred) ** 2, axis=0, weights=self.weights)
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
            mean_proba = np.average(predictions, axis=0, weights=self.weights)
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

    def _order_X_for_estimator(
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
                importances.append(estimator.feature_importances_)
                successful_indices.append(idx)

        if not importances:
            raise RuntimeError(
                "None of the fitted estimators support feature_importances_.",
            )

        importances = np.array(importances)
        active_weights = self.weights[successful_indices]
        active_weights = active_weights / active_weights.sum()

        return np.average(importances, axis=0, weights=active_weights)


if __name__ == "__main__":
    import numpy as _np
    import pandas as _pd
    from sklearn.linear_model import LogisticRegression as _LogReg
    from sklearn.naive_bayes import GaussianNB as _GNB

    X_demo = _pd.DataFrame(_np.random.randn(50, 4), columns=["f1", "f2", "f3", "f4"])
    y_demo = _pd.Series((_np.random.rand(50) > 0.5).astype(int))

    est1 = _LogReg(max_iter=200)
    est2 = _GNB()
    ensemble = EnsembleModel([est1, est2], problem_type="classification")
    ensemble.fit(X_demo, y_demo)
    y_pred = ensemble.predict(X_demo)
    y_proba = ensemble.predict_proba(X_demo)
    print("predictions shape:", y_pred.shape)
    print("probabilities shape:", y_proba.shape)
    confidence = ensemble.predict_with_confidence(X_demo)
    print("Confidence:", confidence)
    feature_importances = ensemble.feature_importances_
    print("Feature importances:", feature_importances)
