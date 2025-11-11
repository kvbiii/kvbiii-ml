from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from kvbiii_ml.evaluation.metrics import (
    METRICS,
    get_metric_function,
    get_metric_type,
)
from kvbiii_ml.modeling.training.base_trainer import BaseTrainer
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


class OOFModel:
    """Generate out-of-fold predictions for any estimator.

    This class performs K-fold training to produce leakage-free out-of-fold
    predictions for stacking, model evaluation, or meta-learning. It supports
    optional per-fold processors that are re-fitted inside each fold, reusing
    the same processor-orchestration logic as the cross-validation trainer.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cross_validator: CrossValidationTrainer,
        problem_type: str,
    ) -> None:
        """Initialize the OOF generator.

        Args:
            estimator (BaseEstimator): Estimator to train on each fold.
            cross_validator (CrossValidationTrainer): Cross-validation trainer to use for fitting.
            problem_type (str): Problem type, either "classification" or "regression".
        """
        if problem_type not in {"classification", "regression"}:
            raise ValueError(
                "problem_type must be either 'classification' or 'regression'."
            )
        self.problem_type = problem_type

        self.cross_validator = cross_validator
        if self.cross_validator.metric_name not in METRICS:
            raise ValueError(
                f"Unsupported metric: {self.cross_validator.metric_name}. Supported metrics are: {', '.join(METRICS.keys())}"
            )
        self.eval_metric = get_metric_function(self.cross_validator.metric_name)
        self.metric_type = get_metric_type(self.cross_validator.metric_name)
        self.estimator = estimator
        self.fitted_estimators_: list[BaseEstimator] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame | None = None,
    ) -> OOFModel:
        """Fit the estimator on each fold and store fitted fold estimators.

        Processors (if any) are re-fitted per fold on the training split to
        avoid leakage, then applied to the validation (and optional test) split.

        Args:
            X (pd.DataFrame): Feature matrix aligned with target.
            y (pd.Series): Target vector.
            X_test (pd.DataFrame | None): Optional test features to pass through processors per fold. Defaults to None.

        Returns:
            OOFModel: Fitted instance with ``fitted_estimators_`` available for prediction.
        """
        self.fitted_estimators_.clear()
        self.cross_validator.fit(self.estimator, X, y, X_test=X_test)
        self.fitted_estimators_ = self.cross_validator.fitted_estimators_
        return self

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
    from sklearn.model_selection import KFold
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    X_arr, y_arr = make_classification(
        n_samples=300, n_features=10, n_informative=5, n_redundant=2, random_state=17
    )
    X_df = pd.DataFrame(X_arr)
    y_ser = pd.Series(y_arr)

    clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=17)
    cv = CrossValidationTrainer(
        metric_name="Accuracy",
        cv=KFold(n_splits=5, shuffle=True, random_state=17),
        processors=None,
        verbose=False,
    )
    oof = OOFModel(estimator=clf, cross_validator=cv, problem_type="classification")
    oof.fit(X_df, y_ser, X_test=X_df)
    y_pred = oof.predict(X_df)
    y_proba = oof.predict_proba(X_df)
    print("Number of fitted estimators:", len(oof.fitted_estimators_))
    print("Predict_proba shape:", y_proba.shape)
    print("Predict shape:", y_pred.shape)
