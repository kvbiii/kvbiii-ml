import gc
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from kvbiii_ml.evaluation.metrics import (
    METRICS,
    get_metric_direction,
    get_metric_function,
    get_metric_type,
)
from kvbiii_ml.evaluation.shap_values import compute_shap_values
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


class ShapRecursiveFeatureElimination:
    """Recursive feature elimination using SHAP values.

    This strategy iteratively evaluates the impact of removing features using
    SHAP-based predictions on validation folds, then removes the least impactful
    features according to a target metric until a stopping schedule is met.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cross_validator: CrossValidationTrainer,
        problem_type: str,
        steps: int = 5,
        alpha: float = 0.95,
        verbose: bool = True,
        protected_features: list[str] | None = None,
    ) -> None:
        """Initialize the SHAP-RFE selector.

        Args:
            estimator (BaseEstimator): Estimator to fit within each fold.
            cross_validator (CrossValidationTrainer): Cross-validation trainer instance.
            problem_type (str): "classification" or "regression".
            steps (int, optional): Number of elimination iterations. Defaults to 5.
            alpha (float, optional): Weight for the final selection score mix. Defaults to 0.95.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
            protected_features (list[str], optional): Features that should never be removed.
        """
        self.estimator = estimator
        self.cross_validator = cross_validator
        self.cv = self.cross_validator.cv
        self.problem_type = problem_type
        self.steps = steps
        self.alpha = alpha
        self.verbose = verbose
        self.protected_features = protected_features or []
        self._large_dataset_warning_printed = False
        self.history_schema = {
            "step": int,
            "n_features_removed": int,
            "n_features_remaining": int,
            "removed_feature_name": object,
            "metric_value": float,
            "metric_change": float,
        }
        if self.cross_validator.metric_name not in METRICS:
            raise ValueError(
                f"Unsupported metric: {self.cross_validator.metric_name}. Supported metrics are: {', '.join(METRICS.keys())}"
            )
        self.eval_metric = get_metric_function(self.cross_validator.metric_name)
        self.metric_type = get_metric_type(self.cross_validator.metric_name)
        self.direction = get_metric_direction(self.cross_validator.metric_name)
        self._cv_cache_key = None
        self._cv_cache_result = None

    def run(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> dict[str, list | pd.DataFrame]:
        """Run SHAP-RFE and return the selection summary.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series | np.ndarray): Target array/series aligned with X.

        Returns:
            dict[str, list | pd.DataFrame]: Dictionary with keys:
                - selected_features (list): Final selected features.
                - selected_features_names (list): Alias of selected_features.
                - history (pd.DataFrame): Step-wise metrics and removals.
        """
        X = self._convert_categorical_to_codes(X)
        current_features = list(X.columns)
        missing_protected = set(self.protected_features) - set(current_features)
        if missing_protected:
            raise ValueError(
                f"Protected features not found in dataset: {missing_protected}"
            )

        summary_df = {
            "selected_features": [],
            "selected_features_names": [],
            "history": pd.DataFrame(columns=self.history_schema.keys()).astype(
                self.history_schema
            ),
        }

        avg_base_metric, _ = self._cross_val_base_metric(X, y, current_features)
        summary_df["history"] = pd.concat(
            [
                summary_df["history"],
                pd.DataFrame(
                    [
                        {
                            "step": 0,
                            "n_features_removed": 0,
                            "n_features_remaining": len(current_features),
                            "removed_feature_name": None,
                            "metric_value": avg_base_metric,
                            "metric_change": 0.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        removable_features = [
            f for f in current_features if f not in self.protected_features
        ]
        removal_schedule = self.compute_removal_schedule(len(removable_features))
        if self.verbose:
            print(
                f"🔍 Starting SHAP-RFE with {len(current_features)} features, target metric: {self.cross_validator.metric_name} ({self.direction}), steps: {self.steps}.\n📅 The number of features to remove every step: {removal_schedule}"
            )
            print(f"🛡️  Protected features: {self.protected_features}")

        for step_idx, n_features_to_remove in enumerate(removal_schedule, start=1):
            avg_metric_feature, fold_base_metric, fold_base_metric_std = (
                self._cross_val_feature_metrics(X, y, current_features)
            )
            if self.verbose:
                print(
                    f"\n🔁 Step {step_idx} | Features remaining: {len(current_features)}"
                )
                print(
                    f"📊 Average {self.cross_validator.metric_name}: {fold_base_metric:.6f} ± {fold_base_metric_std:.6f}"
                )
            removable_metrics = {
                k: v
                for k, v in avg_metric_feature.items()
                if k not in self.protected_features
            }
            feature_metric_diffs_df = pd.DataFrame(
                list(removable_metrics.items()), columns=["feature", "metric_change"]
            )
            ascending = self.direction == "minimize"
            feature_metric_diffs_df = feature_metric_diffs_df.sort_values(
                by="metric_change", ascending=ascending
            )
            n_actually_removable = min(
                n_features_to_remove, len(feature_metric_diffs_df)
            )
            features_to_remove = feature_metric_diffs_df.head(n_actually_removable)[
                "feature"
            ].tolist()
            self._log_step(
                summary_df,
                step_idx,
                X,
                current_features,
                avg_metric_feature,
                fold_base_metric,
                features_to_remove,
            )
            gc.collect()
            if len(current_features) <= len(self.protected_features):
                if self.verbose:
                    print("⏹️  Stopping early - only protected features remain")
                break

        summary_df["selected_features"], metric_selected = (
            self.select_features_weighted_score(summary_df["history"], self.alpha)
        )
        if self.verbose:
            print(f"\n🎯 Selected features: {summary_df['selected_features']}")
            diff_pct = (
                100
                * (metric_selected - summary_df["history"].iloc[0]["metric_value"])
                / summary_df["history"].iloc[0]["metric_value"]
            )
            print(
                f"📈 Final {self.cross_validator.metric_name} score (approximated): {metric_selected:.6f} | Base: {summary_df['history'].iloc[0]['metric_value']:.6f} | Δ: {metric_selected - summary_df['history'].iloc[0]['metric_value']:.6f} ({diff_pct:+.2f}%)"
            )
            n_features_initial = summary_df["history"].iloc[0]["n_features_remaining"]
            n_features_selected = len(summary_df["selected_features"])
            n_removed = n_features_initial - n_features_selected
            pct_removed = 100 * n_removed / n_features_initial
            print(
                f"🗑️ Features removed: {n_removed} of {n_features_initial} ({pct_removed:.2f}%)"
            )
        gc.collect()
        return summary_df

    def _convert_categorical_to_codes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical/object columns in X to integer codes.

        Args:
            X (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with categorical/object columns converted to codes.
        """
        X_converted = X.copy()
        for col in X_converted.select_dtypes(include=["category"]).columns:
            X_converted[col] = (
                X_converted[col].cat.codes.astype("object").astype("category")
            )
        return X_converted

    def _handle_large_dataset(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Handle large datasets by sampling and printing warning if needed.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target values.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Potentially sampled X and corresponding y.
        """
        if X.shape[0] > 100000:
            if self.verbose and not self._large_dataset_warning_printed:
                self._large_dataset_warning_printed = True
                print(
                    f"⚠️ Large dataset detected: {X.shape[0]} samples. "
                    "For computational efficiency, only a random subset of 100,000 samples will be used for SHAP-based feature elimination in each step. "
                    "This may affect the stability of feature importance estimates. "
                    "Consider running on the full dataset if reproducibility is critical."
                )
            X = X.sample(n=100000, random_state=17)
            y = y.loc[X.index]
        return X, y

    def _cross_val_base_metric(
        self, X: pd.DataFrame, y: pd.Series, current_features: list[str]
    ) -> tuple[float, float]:
        """Compute the base metric across validation folds.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target values.
            current_features (list[str]): Features to evaluate.

        Returns:
            tuple[float, float]: Average metric and its standard deviation.
        """
        X, y = self._handle_large_dataset(X, y)
        key = tuple(current_features)
        if self._cv_cache_key == key and self._cv_cache_result is not None:
            valid_scores = self._cv_cache_result["valid_scores"]
            return float(np.mean(valid_scores)), float(np.std(valid_scores))
        _, valid_scores, _ = self.cross_validator.fit(
            self.estimator, X[current_features], y
        )
        fitted = getattr(self.cross_validator, "fitted_estimators_", None)
        self._cv_cache_key = key
        self._cv_cache_result = {
            "valid_scores": valid_scores,
            "fitted_estimators": fitted,
        }
        gc.collect()
        return float(np.mean(valid_scores)), float(np.std(valid_scores))

    def compute_removal_schedule(self, total_removable_features: int) -> list[int]:
        """Compute a linear-decay schedule for features to remove per step.

        Args:
            total_removable_features (int): Number of features that can be removed.

        Returns:
            list[int]: Non-empty positive counts to remove at each step.
        """
        if total_removable_features <= 0 or self.steps <= 0:
            return []
        decay = np.linspace(1, 0.2, self.steps)
        weights = decay / decay.sum()
        removal_counts = np.round(weights * total_removable_features).astype(int)
        diff = total_removable_features - removal_counts.sum()
        for i in range(abs(diff)):
            idx = i % self.steps
            removal_counts[idx] += 1 if diff > 0 else -1
        removal_counts = removal_counts[removal_counts > 0]
        return removal_counts.tolist()

    def _cross_val_feature_metrics(
        self, X: pd.DataFrame, y: pd.Series, current_features: list[str]
    ) -> tuple[dict[str, float], float, float]:
        """Compute per-feature reduced metrics across validation folds.

        Uses a single CV run to train per-fold estimators and then computes
        SHAP-based reduced predictions by zeroing one feature's SHAP values.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target values.
            current_features (list[str]): Features to evaluate.

        Returns:
            tuple[dict[str, float], float, float]:
                - average metric per feature (dict)
                - mean base metric (float)
                - std base metric (float)
        """
        X, y = self._handle_large_dataset(X, y)
        key = tuple(current_features)
        if self._cv_cache_key == key and self._cv_cache_result is not None:
            valid_scores = self._cv_cache_result["valid_scores"]
            fitted_estimators = self._cv_cache_result.get(
                "fitted_estimators",
                getattr(self.cross_validator, "fitted_estimators_", []),
            )
        else:
            _, valid_scores, _ = self.cross_validator.fit(
                self.estimator, X[current_features], y
            )
            fitted_estimators = getattr(self.cross_validator, "fitted_estimators_", [])
            self._cv_cache_key = key
            self._cv_cache_result = {
                "valid_scores": valid_scores,
                "fitted_estimators": fitted_estimators,
            }

        fold_features_metric_monitor = {f: [] for f in current_features}

        for (train_idx, valid_idx), fitted in zip(
            self.cv.split(X[current_features], y), fitted_estimators
        ):
            X_valid = X.iloc[valid_idx][current_features]
            y_valid = y.iloc[valid_idx]

            shap_values = compute_shap_values(fitted, X_valid, current_features)
            vals = shap_values.values
            base = shap_values.base_values

            if vals.ndim == 2:
                total = vals.sum(axis=1)
                preds_matrix = None
                try:
                    reduced_all = total[None, :] - vals.T
                    if np.asarray(base).ndim == 0:
                        base_arr = float(base)
                        preds_matrix = base_arr + reduced_all
                    else:
                        base_arr = np.asarray(base)
                        preds_matrix = base_arr + reduced_all
                    if self.problem_type == "classification":
                        pred_prob = 1 / (1 + np.exp(-preds_matrix))
                        if pred_prob.ndim == 3 and y_valid.ndim == 1:
                            pred_prob = pred_prob[:, :, 1]
                        for feat_idx, feature in enumerate(current_features):
                            if self.metric_type == "probs":
                                pred = pred_prob[feat_idx]
                            else:
                                pred = (pred_prob[feat_idx] >= 0.5).astype(int)
                            metric_without_feature = self.eval_metric(y_valid, pred)
                            fold_features_metric_monitor[feature].append(
                                metric_without_feature
                            )
                    else:
                        for feat_idx, feature in enumerate(current_features):
                            pred = preds_matrix[feat_idx]
                            metric_without_feature = self.eval_metric(y_valid, pred)
                            fold_features_metric_monitor[feature].append(
                                metric_without_feature
                            )
                finally:
                    del vals, base, shap_values, preds_matrix
                    gc.collect()
            elif vals.ndim == 3:
                preds_tensor = None
                try:
                    total = vals.sum(axis=1)
                    vals_swapped = vals.swapaxes(0, 1)
                    reduced_all = total[None, :, :] - vals_swapped
                    if np.asarray(base).ndim == 0:
                        base_arr = float(base)
                        preds_tensor = base_arr + reduced_all
                    else:
                        base_arr = np.asarray(base)
                        preds_tensor = base_arr + reduced_all
                    if self.problem_type == "classification":
                        pred_prob = 1 / (1 + np.exp(-preds_tensor))
                        if pred_prob.ndim == 4 and y_valid.ndim == 1:
                            pred_prob = pred_prob[:, :, :, 1]
                        for feat_idx, feature in enumerate(current_features):
                            feat_pred = pred_prob[feat_idx]
                            if self.metric_type == "probs":
                                pred = feat_pred
                            else:
                                pred = (feat_pred >= 0.5).astype(int)
                            metric_without_feature = self.eval_metric(y_valid, pred)
                            fold_features_metric_monitor[feature].append(
                                metric_without_feature
                            )
                    else:
                        for feat_idx, feature in enumerate(current_features):
                            pred = preds_tensor[feat_idx]
                            metric_without_feature = self.eval_metric(y_valid, pred)
                            fold_features_metric_monitor[feature].append(
                                metric_without_feature
                            )
                finally:
                    del vals, base, shap_values, preds_tensor
                    gc.collect()
            else:
                try:
                    for feat_idx, feature in enumerate(current_features):
                        shap_reduced = vals.copy()
                        shap_reduced[:, feat_idx] = 0
                        if self.problem_type == "classification":
                            pred_logit = base[0] + shap_reduced.sum(axis=1)
                            pred_prob = 1 / (1 + np.exp(-pred_logit))
                            if pred_prob.ndim == 2 and y_valid.ndim == 1:
                                pred_prob = pred_prob[:, 1]
                            reduced_prediction = (
                                pred_prob
                                if self.metric_type == "probs"
                                else (pred_prob >= 0.5).astype(int)
                            )
                        else:
                            reduced_prediction = base[0] + shap_reduced.sum(axis=1)
                        metric_without_feature = self.eval_metric(
                            y_valid, reduced_prediction
                        )
                        fold_features_metric_monitor[feature].append(
                            metric_without_feature
                        )
                finally:
                    del vals, base, shap_values
                    gc.collect()

        avg_metric_feature = {
            feature: float(np.mean(fold_features_metric_monitor[feature]))
            for feature in current_features
        }
        return (
            avg_metric_feature,
            float(np.mean(self._cv_cache_result["valid_scores"])),
            float(np.std(self._cv_cache_result["valid_scores"])),
        )

    def _log_step(
        self,
        summary_df: dict[str, list | pd.DataFrame],
        step_idx: int,
        X: pd.DataFrame,
        current_features: list[str],
        avg_metric_feature: dict[str, float],
        fold_base_metric: float,
        features_to_remove: list[str],
    ) -> None:
        """Log results for the current elimination step.

        Args:
            summary_df (dict[str, list | pd.DataFrame]): Summary dict carrying history.
            step_idx (int): Current step number (1-based after the base step).
            X (pd.DataFrame): Original input features (for counts only).
            current_features (list[str]): Features remaining before removal.
            avg_metric_feature (dict[str, float]): Average metric when removing each feature.
            fold_base_metric (float): Base metric for the current set of features.
            features_to_remove (list[str]): Features to remove this step.
        """
        for feature in features_to_remove:
            if feature in self.protected_features:
                continue
            summary_df["history"] = pd.concat(
                [
                    summary_df["history"],
                    pd.DataFrame(
                        [
                            {
                                "step": step_idx,
                                "n_features_removed": len(X.columns)
                                - len(current_features)
                                + 1,
                                "n_features_remaining": len(current_features) - 1,
                                "removed_feature_name": feature,
                                "metric_value": avg_metric_feature[feature],
                                "metric_change": avg_metric_feature[feature]
                                - fold_base_metric,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            current_features.remove(feature)
        gc.collect()

    def select_features_weighted_score(
        self, history: pd.DataFrame, alpha: float | None = None
    ) -> tuple[list[str], float | None]:
        """Select features by maximizing a weighted metric/features score.

        Args:
            history (pd.DataFrame): Step-wise history with metrics and features.
            alpha (float | None, optional): Weight for metric vs. features. Defaults to self.alpha.

        Returns:
            tuple[list[str], float | None]: (selected_features, best_metric_value).
        """
        if history.empty:
            return [], None
        if alpha is None:
            alpha = self.alpha
        df = history.copy()
        metric_max = df["metric_value"].max()
        metric_min = df["metric_value"].min()
        denom_metric = metric_max - metric_min if metric_max != metric_min else 1.0
        if self.direction == "maximize":
            df["metric_norm"] = (df["metric_value"] - metric_min) / denom_metric
        else:
            df["metric_norm"] = (metric_max - df["metric_value"]) / denom_metric
        feat_max = df["n_features_remaining"].max()
        feat_min = df["n_features_remaining"].min()
        denom_feat = feat_max - feat_min if feat_max != feat_min else 1.0
        df["features_norm"] = 1 - (df["n_features_remaining"] - feat_min) / denom_feat
        df["score"] = alpha * df["metric_norm"] + (1 - alpha) * df["features_norm"]
        best_row = df.loc[df["score"].idxmax()]
        removed = set(
            history[history["n_features_removed"] < best_row["n_features_removed"]][
                "removed_feature_name"
            ].dropna()
        )
        all_features = set(history["removed_feature_name"].dropna().unique())
        all_features.update(self.protected_features)
        selected = list(all_features - removed)
        gc.collect()
        return selected, best_row["metric_value"]


if __name__ == "__main__":
    # Minimal runnable example
    from sklearn.model_selection import KFold
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier

    X_df, y_ser = load_breast_cancer(return_X_y=True, as_frame=True)
    clf = RandomForestClassifier(random_state=17, max_depth=5, n_estimators=100)

    cv = CrossValidationTrainer(
        metric_name="Accuracy",
        cv=KFold(n_splits=5, shuffle=True, random_state=17),
        processors=None,
        verbose=False,
    )
    shap_rfe = ShapRecursiveFeatureElimination(
        estimator=clf,
        cross_validator=cv,
        problem_type="classification",
        steps=5,
        alpha=0.95,
        verbose=True,
    )
    summary = shap_rfe.run(X_df, y_ser)
    print("\nSummary of SHAP-RFE:")
    print(summary["history"])
    print("Selected features:", summary["selected_features"])
