import gc
from typing import Any, cast

import numpy as np
import pandas as pd
import shap

from kvbiii_ml._typing import Estimator, RfeSummary
from kvbiii_ml.evaluation.shap_values import compute_shap_values
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

NDIM_2D = 2
NDIM_3D = 3
BINARY_N_CLASSES = 2
PROBABILITY_THRESHOLD = 0.5


class ShapRecursiveFeatureElimination:
    """Recursive feature elimination using SHAP values.

    This strategy iteratively evaluates the impact of removing features using
    SHAP-based predictions on validation folds, then removes the least impactful
    features according to a target metric until a stopping schedule is met.
    """

    def __init__(  # noqa: PLR0913
        self,
        estimator: Estimator,
        cross_validator: CrossValidationTrainer,
        steps: int = 5,
        alpha: float = 0.95,
        verbose: bool = True,
        protected_features: list[str] | None = None,
        max_samples_shap: int = 50000,
    ) -> None:
        """Initialize the SHAP-RFE selector.

        Args:
            estimator (Estimator): Estimator to fit within each fold.
            cross_validator (CrossValidationTrainer): Cross-validation trainer instance.
            steps (int, optional): Number of elimination iterations. Defaults to 5.
            alpha (float, optional): Weight for the final selection score mix. Defaults to 0.95.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
            protected_features (list[str], optional): Features that should never be removed.
            max_samples_shap (int): Maximum samples for SHAP computation. Defaults to 50000.
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
        self._cv_cache_result: dict[str, Any] | None = None

    def run(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> RfeSummary:
        """Run SHAP-RFE and return the selection summary.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series | np.ndarray): Target array/series aligned with X.

        Returns:
            RfeSummary: Dictionary with keys:
                - selected_features (list): Final selected features.
                - selected_features_names (list): Alias of selected_features.
                - history (pd.DataFrame): Step-wise metrics and removals.

        Raises:
            ValueError: If protected features are missing from the dataset.
        """
        X = self._convert_categorical_to_codes(X)
        current_features = list(X.columns)
        missing_protected = set(self.protected_features) - set(current_features)
        if missing_protected:
            raise ValueError(f"Protected features not found in dataset: {missing_protected}")

        avg_base_metric, _ = self._cross_val_base_metric(X, cast("pd.Series", y), current_features)
        summary_df: RfeSummary = {
            "selected_features": [],
            "selected_features_names": [],
            "history": self._init_history(len(current_features), avg_base_metric),
        }

        removable_features = [f for f in current_features if f not in self.protected_features]
        removal_schedule = self.compute_removal_schedule(len(removable_features))
        self._print_start(len(current_features), removal_schedule)

        for step_idx, n_features_to_remove in enumerate(removal_schedule, start=1):
            step_state = self._run_step(X, y, current_features, n_features_to_remove)
            self._print_step(step_idx, current_features, step_state)
            self._log_step(summary_df, step_idx, X, current_features, step_state)
            gc.collect()
            if len(current_features) <= len(self.protected_features):
                if self.verbose:
                    print("⏹️  Stopping early - only protected features remain")
                break

        summary_df["selected_features"], metric_selected = self.select_features_weighted_score(
            summary_df["history"], self.alpha
        )
        self._print_finish(summary_df, metric_selected)
        gc.collect()
        return summary_df

    def _init_history(self, n_features: int, base_metric: float) -> pd.DataFrame:
        """Build the history DataFrame holding only the step-0 baseline row.

        Args:
            n_features (int): Number of features at baseline.
            base_metric (float): Baseline CV metric value.

        Returns:
            pd.DataFrame: History with the step-0 row.
        """
        history = pd.DataFrame(columns=self.history_schema.keys()).astype(self.history_schema)  # ty: ignore[invalid-argument-type]
        return pd.concat(
            [
                history,
                pd.DataFrame(
                    [
                        {
                            "step": 0,
                            "n_features_removed": 0,
                            "n_features_remaining": n_features,
                            "removed_feature_name": None,
                            "metric_value": base_metric,
                            "metric_change": 0.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    def _run_step(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        current_features: list[str],
        n_features_to_remove: int,
    ) -> dict[str, Any]:
        """Run the SHAP-based CV for one step and pick the features to remove.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series | np.ndarray): Target aligned with X.
            current_features (list[str]): Active feature names.
            n_features_to_remove (int): Number of features this step may remove.

        Returns:
            dict[str, Any]: Step data: per-feature metrics, fold metric/std, features to remove.
        """
        avg_metric_feature, fold_base_metric, fold_base_metric_std = (
            self._cross_val_feature_metrics(X, cast("pd.Series", y), current_features)
        )
        removable_metrics = {
            k: v for k, v in avg_metric_feature.items() if k not in self.protected_features
        }
        feature_metric_diffs_df = pd.DataFrame(
            list(removable_metrics.items()), columns=["feature", "metric_change"]
        ).sort_values(by="metric_change", ascending=self.metric_direction == "minimize")
        n_actually_removable = min(n_features_to_remove, len(feature_metric_diffs_df))
        return {
            "avg_metric_feature": avg_metric_feature,
            "fold_metric": fold_base_metric,
            "fold_std": fold_base_metric_std,
            "features_to_remove": feature_metric_diffs_df.head(n_actually_removable)[
                "feature"
            ].tolist(),
        }

    def _print_start(self, n_features: int, removal_schedule: list[int]) -> None:
        """Print the initial run summary when verbose mode is enabled.

        Args:
            n_features (int): Number of features at baseline.
            removal_schedule (list[int]): Features to remove at each step.
        """
        if not self.verbose:
            return
        print(
            f"🔍 Starting SHAP-RFE with {n_features} features, "
            f"target metric: {self.cross_validator.metric_name} ({self.metric_direction}), "
            f"steps: {self.steps}.\n"
            f"📅 The number of features to remove every step: {removal_schedule}"
        )
        print(f"🛡️  Protected features: {self.protected_features}")

    def _print_step(
        self, step_idx: int, current_features: list[str], step_state: dict[str, Any]
    ) -> None:
        """Print per-step details when verbose mode is enabled.

        Args:
            step_idx (int): Current step number.
            current_features (list[str]): Active feature names.
            step_state (dict[str, Any]): Step data from ``_run_step``.
        """
        if not self.verbose:
            return
        print(f"\n🔁 Step {step_idx} | Number of features remaining: {len(current_features)}")
        print(f"\n🔬 Features remaining: {current_features}\n")
        print(
            f"📊 Average {self.cross_validator.metric_name}: "
            f"{step_state['fold_metric']:.6f} ± {step_state['fold_std']:.6f}"
        )

    def _print_finish(self, summary_df: RfeSummary, metric_selected: float | None) -> None:
        """Print the final selection summary when verbose mode is enabled.

        Args:
            summary_df (RfeSummary): Completed summary with history and selected features.
            metric_selected (float | None): Approximated metric of the selected set.
        """
        if not self.verbose:
            return
        print(f"\n🎯 Selected features: {summary_df['selected_features']}")
        diff_pct = (
            100
            * (metric_selected - summary_df["history"].iloc[0]["metric_value"])
            / summary_df["history"].iloc[0]["metric_value"]
        )
        print(
            f"📈 Final {self.cross_validator.metric_name} score (approximated): "
            f"{metric_selected:.6f} | "
            f"Base: {summary_df['history'].iloc[0]['metric_value']:.6f} | "
            f"Δ: {metric_selected - summary_df['history'].iloc[0]['metric_value']:.6f} "
            f"({diff_pct:+.2f}%)"
        )
        n_features_initial = summary_df["history"].iloc[0]["n_features_remaining"]
        n_features_selected = len(summary_df["selected_features"])
        n_removed = n_features_initial - n_features_selected
        pct_removed = 100 * n_removed / n_features_initial
        print(f"🗑️ Features removed: {n_removed} of {n_features_initial} ({pct_removed:.2f}%)")

    def _convert_categorical_to_codes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical/object columns in X to integer codes.

        Args:
            X (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with categorical/object columns converted to codes.
        """
        x_converted = X.copy()
        for col in x_converted.select_dtypes(include=["category"]).columns:
            x_converted[col] = x_converted[col].cat.codes.astype("object").astype("category")
        return x_converted

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
                    "For computational efficiency, only a random subset of "
                    f"{self.max_samples_shap} samples will be used for SHAP-based "
                    "feature elimination in each step. "
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
        _, valid_scores, _ = self.cross_validator.fit(self.estimator, X[current_features], y)
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
            # reduce removals from the last steps so earlier steps keep their relative weight
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
            _, valid_scores, _ = self.cross_validator.fit(self.estimator, X[current_features], y)
            fitted_estimators = getattr(self.cross_validator, "fitted_estimators_", [])
            self._cv_cache_key = key
            self._cv_cache_result = {
                "valid_scores": valid_scores,
                "fitted_estimators": fitted_estimators,
            }

        fold_features_metric_monitor = {f: [] for f in current_features}

        for fold_idx, ((_train_idx, valid_idx), fitted) in enumerate(
            zip(self.cv.split(X[current_features], y), fitted_estimators, strict=False)
        ):
            try:
                X_valid = X.iloc[valid_idx][current_features]
                y_valid = y.iloc[valid_idx]
                try:
                    shap_values = compute_shap_values(fitted, X_valid, current_features)
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    if self.verbose:
                        print(f"⚠️ SHAP computation failed for fold {fold_idx}: {exc}")
                    continue
                self._process_shap_fold(
                    shap_values,
                    y_valid,
                    current_features,
                    fold_features_metric_monitor,
                )

            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                if self.verbose:
                    print(f"⚠️ Error processing fold {fold_idx}: {exc}")
                continue
            finally:
                gc.collect()

        avg_metric_feature = {}
        for feature in current_features:
            if fold_features_metric_monitor[feature]:
                avg_metric_feature[feature] = float(np.mean(fold_features_metric_monitor[feature]))
            else:
                avg_metric_feature[feature] = float(np.mean(self._cv_cache_result["valid_scores"]))

        return (
            avg_metric_feature,
            float(np.mean(self._cv_cache_result["valid_scores"])),
            float(np.std(self._cv_cache_result["valid_scores"])),
        )

    def _process_shap_fold(
        self,
        shap_values: shap.Explanation,
        y_valid: pd.Series,
        current_features: list[str],
        fold_features_metric_monitor: dict,
    ) -> None:
        """Process SHAP values for a single fold with improved memory management."""
        vals = shap_values.values  # noqa: PD011
        base = shap_values.base_values

        try:
            if vals.ndim == NDIM_2D:
                self._process_2d_shap(
                    vals,
                    base,
                    y_valid,
                    current_features,
                    fold_features_metric_monitor,
                )
            elif vals.ndim == NDIM_3D:
                self._process_3d_shap(
                    vals,
                    base,
                    y_valid,
                    current_features,
                    fold_features_metric_monitor,
                )
            else:
                if self.verbose:
                    print(f"⚠️ Unsupported SHAP values dimension: {vals.ndim}. Skipping fold.")
                return
        finally:
            del vals, base, shap_values
            gc.collect()

    def _process_2d_shap(
        self,
        vals: np.ndarray,
        base: np.ndarray | float,
        y_valid: pd.Series,
        current_features: list[str],
        fold_features_metric_monitor: dict,
    ) -> None:
        """Process 2D SHAP values with vectorized operations for efficiency."""
        try:
            self._record_feature_metrics_2d(
                self._final_pred_matrix_2d(vals, base, y_valid, len(current_features)),
                y_valid,
                current_features,
                fold_features_metric_monitor,
            )
        except (FloatingPointError, OverflowError, TypeError, ValueError) as exc:
            if self.verbose:
                print(f"⚠️ Vectorized processing failed, skipping fold: {exc}")
        finally:
            gc.collect()

    @staticmethod
    def _reduced_pred_matrix_2d(vals: np.ndarray, base: np.ndarray | float) -> np.ndarray:
        """Return the 2D predictions with one feature's SHAP contribution removed per column."""
        reduced_vals_matrix = vals.sum(axis=1, keepdims=True) - vals
        if np.asarray(base).ndim == 0:
            return float(base) + reduced_vals_matrix
        return np.asarray(base)[:, np.newaxis] + reduced_vals_matrix

    def _final_pred_matrix_2d(
        self,
        vals: np.ndarray,
        base: np.ndarray | float,
        y_valid: pd.Series,
        n_features: int,
    ) -> np.ndarray:
        """Turn 2D SHAP values into the per-feature predictions the metric expects."""
        pred_matrix = self._reduced_pred_matrix_2d(vals, base)
        if self.problem_type != "classification":
            return pred_matrix
        pred_prob_matrix = 1 / (1 + np.exp(-np.clip(pred_matrix, -500, 500)))
        if pred_prob_matrix.ndim == NDIM_3D and y_valid.ndim == 1:
            pred_prob_matrix = pred_prob_matrix[:, :, 1]
        elif pred_prob_matrix.ndim == NDIM_2D and pred_prob_matrix.shape[1] != n_features:
            pred_prob_matrix = (
                pred_prob_matrix[:, 1::2]
                if pred_prob_matrix.shape[1] % 2 == 0
                else pred_prob_matrix
            )
        if self.metric_type == "probs":
            return pred_prob_matrix
        if pred_prob_matrix.ndim == NDIM_2D and pred_prob_matrix.shape[1] == n_features:
            return (pred_prob_matrix >= PROBABILITY_THRESHOLD).astype(int)
        return np.argmax(pred_prob_matrix, axis=-1).astype(int)

    def _record_feature_metrics_2d(
        self,
        final_pred_matrix: np.ndarray,
        y_valid: pd.Series,
        current_features: list[str],
        fold_features_metric_monitor: dict,
    ) -> None:
        """Append the metric without each feature to the monitor, skipping features that fail."""
        for feat_idx, feature in enumerate(current_features):
            try:
                if final_pred_matrix.ndim == NDIM_2D:
                    final_pred = final_pred_matrix[:, feat_idx]
                else:
                    final_pred = final_pred_matrix[feat_idx]
                metric_without_feature = self.metric_fn(y_valid, final_pred)
                fold_features_metric_monitor[feature].append(metric_without_feature)
            except (TypeError, ValueError) as exc:
                if self.verbose:
                    print(f"⚠️ Error processing feature {feature}: {exc}")

    def _process_3d_shap(
        self,
        vals: np.ndarray,
        base: np.ndarray | float,
        y_valid: pd.Series,
        current_features: list[str],
        fold_features_metric_monitor: dict,
    ) -> None:
        """Process 3D SHAP values with vectorized operations for efficiency."""
        try:
            self._record_feature_metrics_3d(
                self._final_pred_matrix_3d(vals, base, y_valid),
                y_valid,
                current_features,
                fold_features_metric_monitor,
            )
        except (FloatingPointError, OverflowError, TypeError, ValueError) as exc:
            if self.verbose:
                print(f"⚠️ Vectorized 3D processing failed, skipping fold: {exc}")
        finally:
            gc.collect()

    @staticmethod
    def _reduced_pred_matrix_3d(vals: np.ndarray, base: np.ndarray | float) -> np.ndarray:
        """Return the 3D predictions with one feature's SHAP contribution removed per column."""
        reduced_vals_matrix = vals.sum(axis=1)[:, np.newaxis, :] - vals
        base_arr = np.asarray(base, dtype=float)
        if base_arr.ndim == 0:
            return base_arr.item() + reduced_vals_matrix
        if base_arr.ndim == 1 and reduced_vals_matrix.ndim == NDIM_3D:
            return base_arr[:, np.newaxis, np.newaxis] + reduced_vals_matrix
        return base_arr[:, np.newaxis, :] + reduced_vals_matrix

    def _final_pred_matrix_3d(
        self, vals: np.ndarray, base: np.ndarray | float, y_valid: pd.Series
    ) -> np.ndarray:
        """Turn 3D SHAP values into the per-feature predictions the metric expects."""
        pred_matrix = self._reduced_pred_matrix_3d(vals, base)
        if self.problem_type == "classification":
            pred_prob_matrix = 1 / (1 + np.exp(-np.clip(pred_matrix, -500, 500)))
            if pred_prob_matrix.ndim == NDIM_3D and y_valid.ndim == 1:
                if pred_prob_matrix.shape[2] == BINARY_N_CLASSES:
                    pred_prob_matrix = pred_prob_matrix[:, :, 1]
                elif pred_prob_matrix.shape[2] == 1:
                    pred_prob_matrix = pred_prob_matrix[:, :, 0]
            if self.metric_type == "probs":
                return pred_prob_matrix
            if pred_prob_matrix.ndim == NDIM_3D:
                return np.argmax(pred_prob_matrix, axis=-1).astype(int)
            return (pred_prob_matrix >= PROBABILITY_THRESHOLD).astype(int)
        if pred_matrix.ndim == NDIM_3D:
            return pred_matrix[:, :, 0]
        return pred_matrix

    def _record_feature_metrics_3d(
        self,
        final_pred_matrix: np.ndarray,
        y_valid: pd.Series,
        current_features: list[str],
        fold_features_metric_monitor: dict,
    ) -> None:
        """Append the metric without each feature to the monitor, aligning predictions to y."""
        n = len(y_valid)
        for feat_idx, feature in enumerate(current_features):
            try:
                if final_pred_matrix.ndim == NDIM_2D:
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
            except (TypeError, ValueError) as exc:
                if self.verbose:
                    print(f"⚠️ Error processing feature {feature}: {exc}")

    def _log_step(
        self,
        summary_df: RfeSummary,
        step_idx: int,
        X: pd.DataFrame,
        current_features: list[str],
        step_state: dict[str, Any],
    ) -> None:
        """Log results for the current elimination step.

        Args:
            summary_df (RfeSummary): Summary dict carrying history.
            step_idx (int): Current step number (1-based after the base step).
            X (pd.DataFrame): Original input features (for counts only).
            current_features (list[str]): Features remaining before removal.
            step_state (dict[str, Any]): Step data from ``_run_step``.
        """
        avg_metric_feature = step_state["avg_metric_feature"]
        for feature in step_state["features_to_remove"]:
            if feature in self.protected_features:
                continue
            summary_df["history"] = pd.concat(
                [
                    summary_df["history"],
                    pd.DataFrame(
                        [
                            {
                                "step": step_idx,
                                "n_features_removed": len(X.columns) - len(current_features) + 1,
                                "n_features_remaining": len(current_features) - 1,
                                "removed_feature_name": feature,
                                "metric_value": avg_metric_feature[feature],
                                "metric_change": avg_metric_feature[feature]
                                - step_state["fold_metric"],
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
        selected = set(history[history["step"] >= best_row["step"]]["removed_feature_name"])
        selected.update(self.protected_features)
        gc.collect()
        return sorted(selected), best_row["metric_value"]


if __name__ == "__main__":
    # Minimal runnable example
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold

    X_df, y_ser = load_breast_cancer(return_X_y=True, as_frame=True)
    protected_features = ["mean radius", "mean texture"]
    y_ser = y_ser.astype("category")
    clf = RandomForestClassifier(random_state=17, max_depth=5, n_estimators=100)

    cv = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=KFold(n_splits=5, shuffle=True, random_state=17),
        preprocessing_pipeline=None,
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
