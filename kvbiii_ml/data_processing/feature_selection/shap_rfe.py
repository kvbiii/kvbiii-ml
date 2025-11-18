import gc
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
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
        steps: int = 5,
        alpha: float = 0.95,
        verbose: bool = True,
        protected_features: list[str] | None = None,
        max_samples_shap: int = 50000,
    ) -> None:
        """Initialize the SHAP-RFE selector.

        Args:
            estimator (BaseEstimator): Estimator to fit within each fold.
            cross_validator (CrossValidationTrainer): Cross-validation trainer instance.
            steps (int, optional): Number of elimination iterations. Defaults to 5.
            alpha (float, optional): Weight for the final selection score mix. Defaults to 0.95.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
            protected_features (list[str], optional): Features that should never be removed.
            max_samples_shap (int, optional): Maximum samples for SHAP computation. Defaults to 50000.
        """
        self.estimator = estimator
        self.cross_validator = cross_validator
        self.cv = self.cross_validator.cv
        self.steps = steps
        self.alpha = alpha
        self.verbose = verbose
        self.protected_features = protected_features or []
        self.max_samples_shap = max_samples_shap
        self._large_dataset_warning_printed = False
        self.history_schema = {
            "step": int,
            "n_features_removed": int,
            "n_features_remaining": int,
            "removed_feature_name": object,
            "metric_value": float,
            "metric_change": float,
        }

        self.problem_type = self.cross_validator.problem_type
        self.metric_fn = self.cross_validator.metric_fn
        self.metric_type = self.cross_validator.metric_type
        self.metric_direction = self.cross_validator.metric_direction
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
                f"🔍 Starting SHAP-RFE with {len(current_features)} features, target metric: {self.cross_validator.metric_name} ({self.metric_direction}), steps: {self.steps}.\n📅 The number of features to remove every step: {removal_schedule}"
            )
            print(f"🛡️  Protected features: {self.protected_features}")

        for step_idx, n_features_to_remove in enumerate(removal_schedule, start=1):
            avg_metric_feature, fold_base_metric, fold_base_metric_std = (
                self._cross_val_feature_metrics(X, y, current_features)
            )
            if self.verbose:
                print(
                    f"\n🔁 Step {step_idx} | Number of features remaining: {len(current_features)}"
                )
                print(f"\n🔬 Features remaining: {current_features}\n")
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
            ascending = self.metric_direction == "minimize"
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
        if X.shape[0] > self.max_samples_shap:
            if self.verbose and not self._large_dataset_warning_printed:
                self._large_dataset_warning_printed = True
                print(
                    f"⚠️ Large dataset detected: {X.shape[0]} samples. "
                    f"For computational efficiency, only a random subset of {self.max_samples_shap} samples will be used for SHAP-based feature elimination in each step. "
                    "This may affect the stability of feature importance estimates. "
                    "Consider running on the full dataset if reproducibility is critical."
                )
            X = X.sample(n=self.max_samples_shap, random_state=17)
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
            # keep the schedule length equal to self.steps even if nothing to remove
            return [0] * self.steps

        decay = np.linspace(1, 0.2, self.steps)
        weights = decay / decay.sum()
        removal_counts = np.round(weights * total_removable_features).astype(int)

        diff = total_removable_features - removal_counts.sum()
        for i in range(abs(diff)):
            idx = i % self.steps
            removal_counts[idx] += 1 if diff > 0 else -1

        # Ensure we never remove all removable features: leave at least one feature overall
        allowed_removals = max(0, total_removable_features - 1)
        current_sum = int(removal_counts.sum())
        if current_sum > allowed_removals:
            # reduce removals starting from the last steps so earlier steps keep their relative weight
            for idx in range(self.steps - 1, -1, -1):
                if current_sum <= allowed_removals:
                    break
                if removal_counts[idx] > 0:
                    dec = min(removal_counts[idx], current_sum - allowed_removals)
                    removal_counts[idx] -= dec
                    current_sum -= dec

        # ensure non-negative ints and preserve number of steps
        removal_counts = np.maximum(removal_counts, 0).astype(int)
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

        for fold_idx, ((train_idx, valid_idx), fitted) in enumerate(
            zip(self.cv.split(X[current_features], y), fitted_estimators)
        ):
            try:
                X_valid = X.iloc[valid_idx][current_features]
                y_valid = y.iloc[valid_idx]
                try:
                    shap_values = compute_shap_values(fitted, X_valid, current_features)
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ SHAP computation failed for fold {fold_idx}: {e}")
                    continue
                self._process_shap_fold(
                    shap_values,
                    y_valid,
                    current_features,
                    fold_features_metric_monitor,
                )

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error processing fold {fold_idx}: {e}")
                continue
            finally:
                gc.collect()

        avg_metric_feature = {}
        for feature in current_features:
            if fold_features_metric_monitor[feature]:
                avg_metric_feature[feature] = float(
                    np.mean(fold_features_metric_monitor[feature])
                )
            else:
                avg_metric_feature[feature] = float(
                    np.mean(self._cv_cache_result["valid_scores"])
                )

        return (
            avg_metric_feature,
            float(np.mean(self._cv_cache_result["valid_scores"])),
            float(np.std(self._cv_cache_result["valid_scores"])),
        )

    def _process_shap_fold(
        self,
        shap_values,
        y_valid: pd.Series,
        current_features: list[str],
        fold_features_metric_monitor: dict,
    ) -> None:
        """Process SHAP values for a single fold with improved memory management."""
        vals = shap_values.values
        base = shap_values.base_values

        try:
            if vals.ndim == 2:
                self._process_2d_shap(
                    vals,
                    base,
                    y_valid,
                    current_features,
                    fold_features_metric_monitor,
                )
            elif vals.ndim == 3:
                self._process_3d_shap(
                    vals,
                    base,
                    y_valid,
                    current_features,
                    fold_features_metric_monitor,
                )
            else:
                if self.verbose:
                    print(
                        f"⚠️ Unsupported SHAP values dimension: {vals.ndim}. Skipping fold."
                    )
                return
        finally:
            del vals, base, shap_values
            gc.collect()

    def _process_2d_shap(
        self,
        vals,
        base,
        y_valid,
        current_features,
        fold_features_metric_monitor,
    ):
        """Process 2D SHAP values with vectorized operations for efficiency."""
        try:
            total = vals.sum(axis=1, keepdims=True)
            reduced_vals_matrix = total - vals
            if np.asarray(base).ndim == 0:
                base_val = float(base)
                pred_matrix = base_val + reduced_vals_matrix
            else:
                base_arr = np.asarray(base)
                pred_matrix = base_arr[:, np.newaxis] + reduced_vals_matrix

            if self.problem_type == "classification":
                pred_prob_matrix = 1 / (1 + np.exp(-np.clip(pred_matrix, -500, 500)))
                if pred_prob_matrix.ndim == 3 and y_valid.ndim == 1:
                    pred_prob_matrix = pred_prob_matrix[:, :, 1]
                elif pred_prob_matrix.ndim == 2 and pred_prob_matrix.shape[1] != len(
                    current_features
                ):
                    pred_prob_matrix = (
                        pred_prob_matrix[:, 1::2]
                        if pred_prob_matrix.shape[1] % 2 == 0
                        else pred_prob_matrix
                    )
                if self.metric_type == "probs":
                    final_pred_matrix = pred_prob_matrix
                else:
                    if pred_prob_matrix.ndim == 2 and pred_prob_matrix.shape[1] == len(
                        current_features
                    ):
                        final_pred_matrix = (pred_prob_matrix >= 0.5).astype(int)
                    else:
                        final_pred_matrix = np.argmax(pred_prob_matrix, axis=-1).astype(
                            int
                        )
            else:
                final_pred_matrix = pred_matrix
            for feat_idx, feature in enumerate(current_features):
                try:
                    if final_pred_matrix.ndim == 2:
                        final_pred = final_pred_matrix[:, feat_idx]
                    else:
                        final_pred = final_pred_matrix[feat_idx]

                    metric_without_feature = self.metric_fn(y_valid, final_pred)
                    fold_features_metric_monitor[feature].append(metric_without_feature)

                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Error processing feature {feature}: {e}")
                    continue

        except Exception as e:
            if self.verbose:
                print(
                    f"⚠️ Vectorized processing failed, falling back to iterative method: {e}"
                )

        finally:
            if "total" in locals():
                del total
            if "reduced_vals_matrix" in locals():
                del reduced_vals_matrix
            if "pred_matrix" in locals():
                del pred_matrix
            if "pred_prob_matrix" in locals():
                del pred_prob_matrix
            if "final_pred_matrix" in locals():
                del final_pred_matrix
            gc.collect()

    def _process_3d_shap(
        self, vals, base, y_valid, current_features, fold_features_metric_monitor
    ):
        """Process 3D SHAP values with vectorized operations for efficiency."""
        try:
            total = vals.sum(axis=1)
            reduced_vals_matrix = total[:, np.newaxis, :] - vals
            base_arr = np.asarray(base, dtype=float)
            scalar_base = base_arr.ndim == 0

            if scalar_base:
                pred_matrix = base_arr.item() + reduced_vals_matrix
            else:
                if base_arr.ndim == 1 and reduced_vals_matrix.ndim == 3:
                    pred_matrix = (
                        base_arr[:, np.newaxis, np.newaxis] + reduced_vals_matrix
                    )
                else:
                    pred_matrix = base_arr[:, np.newaxis, :] + reduced_vals_matrix
            if self.problem_type == "classification":
                pred_prob_matrix = 1 / (1 + np.exp(-np.clip(pred_matrix, -500, 500)))
                if pred_prob_matrix.ndim == 3 and y_valid.ndim == 1:
                    if pred_prob_matrix.shape[2] == 2:
                        pred_prob_matrix = pred_prob_matrix[:, :, 1]
                    elif pred_prob_matrix.shape[2] == 1:
                        pred_prob_matrix = pred_prob_matrix[:, :, 0]
                if self.metric_type == "probs":
                    final_pred_matrix = pred_prob_matrix
                else:
                    if pred_prob_matrix.ndim == 3:
                        final_pred_matrix = np.argmax(pred_prob_matrix, axis=-1).astype(
                            int
                        )
                    else:
                        final_pred_matrix = (pred_prob_matrix >= 0.5).astype(int)
            else:
                if pred_matrix.ndim == 3:
                    final_pred_matrix = pred_matrix[:, :, 0]
                else:
                    final_pred_matrix = pred_matrix
            n = len(y_valid)
            for feat_idx, feature in enumerate(current_features):
                try:
                    if final_pred_matrix.ndim == 2:
                        final_pred = final_pred_matrix[:, feat_idx]
                    else:
                        final_pred = final_pred_matrix[feat_idx]
                    final_pred = np.asarray(final_pred)
                    if final_pred.ndim == 0:
                        final_pred = np.full(n, final_pred.item())
                    elif final_pred.shape[0] != n:
                        if final_pred.size == n:
                            final_pred = final_pred.reshape(n, -1).ravel()
                        else:
                            final_pred = np.full(n, final_pred.ravel()[0])
                    metric_without_feature = self.metric_fn(y_valid, final_pred)
                    fold_features_metric_monitor[feature].append(metric_without_feature)
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Error processing feature {feature}: {e}")
                    continue
        except Exception as e:
            if self.verbose:
                print(
                    f"⚠️ Vectorized 3D processing failed, falling back to iterative method: {e}"
                )
            self._process_3d_shap_fallback(
                vals, base, y_valid, current_features, fold_features_metric_monitor
            )

        finally:
            if "total" in locals():
                del total
            if "reduced_vals_matrix" in locals():
                del reduced_vals_matrix
            if "pred_matrix" in locals():
                del pred_matrix
            if "pred_prob_matrix" in locals():
                del pred_prob_matrix
            if "final_pred_matrix" in locals():
                del final_pred_matrix
            gc.collect()

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
        """Select features by maximizing a weighted metric/features score."""
        if history.empty:
            return [], None
        if alpha is None:
            alpha = self.alpha
        df = history.copy()
        metric_max = df["metric_value"].max()
        metric_min = df["metric_value"].min()
        denom_metric = metric_max - metric_min if metric_max != metric_min else 1.0
        if self.metric_direction == "maximize":
            df["metric_norm"] = (df["metric_value"] - metric_min) / denom_metric
        else:
            df["metric_norm"] = (metric_max - df["metric_value"]) / denom_metric
        feat_max = df["n_features_remaining"].max()
        feat_min = df["n_features_remaining"].min()
        denom_feat = feat_max - feat_min if feat_max != feat_min else 1.0
        df["features_norm"] = 1 - (df["n_features_remaining"] - feat_min) / denom_feat
        df["score"] = alpha * df["metric_norm"] + (1 - alpha) * df["features_norm"]
        best_row = df.loc[df["score"].idxmax()]
        selected = set(
            history[history["step"] >= best_row["step"]]["removed_feature_name"]
        )
        selected.update(self.protected_features)
        gc.collect()
        return list(sorted(selected)), best_row["metric_value"]


if __name__ == "__main__":
    # Minimal runnable example
    from sklearn.model_selection import KFold
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier

    X_df, y_ser = load_breast_cancer(return_X_y=True, as_frame=True)
    protected_features = ["mean radius", "mean texture"]
    y_ser = y_ser.astype("category")
    clf = RandomForestClassifier(random_state=17, max_depth=5, n_estimators=100)

    cv = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=KFold(n_splits=5, shuffle=True, random_state=17),
        processors=None,
        verbose=False,
    )
    shap_rfe = ShapRecursiveFeatureElimination(
        estimator=clf,
        cross_validator=cv,
        protected_features=protected_features,
        steps=10,
        alpha=0.95,
        verbose=True,
    )
    summary = shap_rfe.run(X_df, y_ser)
    print("\nSummary of SHAP-RFE:")
    print(summary["history"])
    print("Selected features:", summary["selected_features"])
