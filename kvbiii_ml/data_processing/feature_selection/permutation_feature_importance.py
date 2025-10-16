import gc
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[3]))
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


class PermutationRecursiveFeatureElimination:
    """Recursive feature elimination using Permutation Feature Importance.

    This strategy iteratively evaluates the impact of shuffling each feature
    on model performance across validation folds, then removes the least
    important features according to a target metric until a stopping schedule is met.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cross_validator: CrossValidationTrainer,
        steps: int = 5,
        alpha: float = 0.95,
        n_repeats: int = 5,
        verbose: bool = True,
        protected_features: list[str] | None = None,
        random_state: int = 17,
    ) -> None:
        """Initialize the Permutation Feature Importance RFE selector.

        Args:
            estimator (BaseEstimator): Estimator to fit within each fold.
            cross_validator (CrossValidationTrainer): Cross-validation trainer instance.
            steps (int, optional): Number of elimination iterations. Defaults to 5.
            alpha (float, optional): Weight for the final selection score mix. Defaults to 0.95.
            n_repeats (int, optional): Number of times to permute each feature. Defaults to 5.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
            protected_features (list[str], optional): Features that should never be removed.
            random_state (int, optional): Random state for reproducibility. Defaults to 17.
        """
        self.estimator = estimator
        self.cross_validator = cross_validator
        self.steps = steps
        self.alpha = alpha
        self.n_repeats = n_repeats
        self.verbose = verbose
        self.protected_features = protected_features or []
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

        self.metric_fn = self.cross_validator.metric_fn
        self.metric_type = self.cross_validator.metric_type
        self.metric_direction = self.cross_validator.metric_direction

        self._permutation_cache: dict[int, dict[str, list[np.ndarray]]] = {}

        self.history_schema = {
            "step": int,
            "n_features_removed": int,
            "n_features_remaining": int,
            "removed_feature_name": object,
            "metric_value": float,
            "metric_change": float,
            "importance_score": float,
        }

    def run(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> dict[str, list | pd.DataFrame]:
        """Run Permutation Feature Importance RFE and return the selection summary.

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
        self._permutation_cache = {}

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
                            "importance_score": np.nan,
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
                f"🔍 Starting Permutation Feature Importance RFE with {len(current_features)} features, target metric: {self.cross_validator.metric_name} ({self.metric_direction}), steps: {self.steps}.\n📅 The number of features to remove every step: {removal_schedule}"
            )
            print(f"🛡️  Protected features: {self.protected_features}")
            print(
                f"📊 Initial {self.cross_validator.metric_name}: {avg_base_metric:.6f} | Features: {len(current_features)}"
            )

        for step_idx, n_features_to_remove in enumerate(removal_schedule, start=1):
            importance_scores, fold_base_metric, fold_base_metric_std = (
                self._cross_val_permutation_importance(X, y, current_features)
            )
            if self.verbose:
                print(
                    f"\n🔁 Step {step_idx} | Number of features remaining: {len(current_features)}"
                )
                print(f"\n🔬 Features remaining: {current_features}\n")
                print(
                    f"📊 Average {self.cross_validator.metric_name}: {fold_base_metric:.6f} ± {fold_base_metric_std:.6f}"
                )

            removable_scores = {
                k: v
                for k, v in importance_scores.items()
                if k not in self.protected_features
            }
            importance_df = pd.DataFrame(
                list(removable_scores.items()),
                columns=["feature", "importance_score"],
            )

            importance_df = importance_df.sort_values(
                by="importance_score", ascending=True
            )

            n_actually_removable = min(n_features_to_remove, len(importance_df))
            features_to_remove = importance_df.head(n_actually_removable)[
                "feature"
            ].tolist()

            if self.verbose:
                print(f"Top 5 most important features:")
                for feat in reversed(importance_df.tail(5)["feature"].tolist()):
                    print(f"  • {feat}: {importance_scores[feat]:.6f}")
                print("\nTop 5 least important features (candidates for removal):")
                for feat in features_to_remove[:5]:
                    print(f"  • {feat}: {importance_scores[feat]:.6f}")

            self._log_step(
                summary_df,
                step_idx,
                X,
                current_features,
                importance_scores,
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
        summary_df["selected_features_names"] = summary_df["selected_features"]

        if self.verbose:
            print(f"\n🎯 Selected features: {summary_df['selected_features']}")
            base_val = summary_df["history"].iloc[0]["metric_value"]
            diff_pct = (
                100 * (metric_selected - base_val) / base_val
                if base_val != 0
                else np.nan
            )
            print(
                f"📈 Final {self.cross_validator.metric_name} score (approximated): {metric_selected:.6f} | Base: {base_val:.6f} | Δ: {metric_selected - base_val:.6f} ({diff_pct:+.2f}%)"
            )
            n_features_initial = summary_df["history"].iloc[0]["n_features_remaining"]
            n_features_selected = len(summary_df["selected_features"])
            n_removed = n_features_initial - n_features_selected
            pct_removed = (
                100 * n_removed / n_features_initial if n_features_initial else 0.0
            )
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
        _, valid_scores, _ = self.cross_validator.fit(
            self.estimator, X[current_features], y
        )
        gc.collect()
        return float(np.mean(valid_scores)), float(np.std(valid_scores))

    def _cross_val_permutation_importance(
        self, X: pd.DataFrame, y: pd.Series, current_features: list[str]
    ) -> tuple[dict[str, float], float, float]:
        """Compute permutation importance with caching of metrics, not just indices."""
        _, valid_scores, _ = self.cross_validator.fit(
            self.estimator, X[current_features], y
        )
        fitted_estimators = getattr(self.cross_validator, "fitted_estimators_", [])

        # cache structure: [fold][feature] -> list of permuted_scores (length n_repeats)
        if not hasattr(self, "_metric_cache"):
            self._metric_cache: dict[int, dict[str, list[float]]] = {}

        fold_importance_scores = {f: [] for f in current_features}

        for fold_idx, ((train_idx, valid_idx), fitted) in enumerate(
            zip(
                self.cross_validator.cv.split(X[current_features], y), fitted_estimators
            )
        ):
            X_valid = X.iloc[valid_idx][current_features].values
            y_valid = y.iloc[valid_idx]

            baseline_pred = self._predict(
                fitted, pd.DataFrame(X_valid, columns=current_features)
            )
            baseline_score = self.metric_fn(y_valid, baseline_pred)

            if fold_idx not in self._metric_cache:
                self._metric_cache[fold_idx] = {}

            feature_iterator = tqdm(
                enumerate(current_features),
                total=len(current_features),
                desc=f"Fold {fold_idx + 1} - Permuting features",
                disable=not self.verbose,
                leave=True,
            )

            for feat_idx, feature in feature_iterator:
                # If cached, just reuse
                if feature in self._metric_cache[fold_idx]:
                    permutation_scores = self._metric_cache[fold_idx][feature]
                else:
                    original_values = X_valid[:, feat_idx].copy()
                    permutation_scores = []
                    for repeat_idx in range(self.n_repeats):
                        if fold_idx not in self._permutation_cache:
                            self._permutation_cache[fold_idx] = {}
                        if feature not in self._permutation_cache[fold_idx]:
                            self._permutation_cache[fold_idx][feature] = []
                        if (
                            len(self._permutation_cache[fold_idx][feature])
                            <= repeat_idx
                        ):
                            permuted_indices = self._rng.permutation(
                                len(original_values)
                            )
                            self._permutation_cache[fold_idx][feature].append(
                                permuted_indices
                            )
                        permuted_indices = self._permutation_cache[fold_idx][feature][
                            repeat_idx
                        ]

                        X_valid[:, feat_idx] = original_values[permuted_indices]
                        permuted_pred = self._predict(
                            fitted, pd.DataFrame(X_valid, columns=current_features)
                        )
                        permutation_scores.append(
                            self.metric_fn(y_valid, permuted_pred)
                        )
                        X_valid[:, feat_idx] = original_values

                    # cache scores (not just indices)
                    self._metric_cache[fold_idx][feature] = permutation_scores

                # importance
                if self.metric_direction == "maximize":
                    importance = baseline_score - np.mean(permutation_scores)
                else:
                    importance = np.mean(permutation_scores) - baseline_score
                fold_importance_scores[feature].append(float(importance))

        avg_importance = {
            f: float(np.mean(scores)) if scores else 0.0
            for f, scores in fold_importance_scores.items()
        }

        return avg_importance, float(np.mean(valid_scores)), float(np.std(valid_scores))

    def _predict(self, estimator, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the estimator.

        Supports both regular estimators and EnsembleModel instances.

        Args:
            estimator: Fitted estimator or EnsembleModel.
            X (pd.DataFrame): Feature matrix.

        Returns:
            np.ndarray: Predictions.
        """
        if hasattr(estimator, "predict_proba") and self.metric_type == "probs":
            preds = estimator.predict_proba(X)
            if preds.ndim == 2 and preds.shape[1] == 2:
                return preds[:, 1]
            return preds
        elif hasattr(estimator, "predict"):
            return estimator.predict(X)
        else:
            raise ValueError(
                f"Estimator {type(estimator).__name__} does not have predict or predict_proba method"
            )

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

    def _log_step(
        self,
        summary_df: dict[str, list | pd.DataFrame],
        step_idx: int,
        X: pd.DataFrame,
        current_features: list[str],
        importance_scores: dict[str, float],
        fold_base_metric: float,
        features_to_remove: list[str],
    ) -> None:
        """Log results for the current elimination step.

        Args:
            summary_df (dict[str, list | pd.DataFrame]): Summary dict carrying history.
            step_idx (int): Current step number.
            X (pd.DataFrame): Original input features.
            current_features (list[str]): Features remaining before removal.
            importance_scores (dict[str, float]): Importance score for each feature.
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
                                "metric_value": fold_base_metric,
                                "metric_change": 0.0,
                                "importance_score": importance_scores.get(feature, 0.0),
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
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold

    X, y = make_classification(
        n_samples=2000,
        n_features=1000,
        n_informative=10,
        n_redundant=5,
        n_repeated=0,
        n_clusters_per_class=2,
        random_state=17,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_ser = pd.Series(y, name="target")
    clf = RandomForestClassifier(random_state=17, max_depth=5, n_estimators=100)
    cross_validator = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=KFold(n_splits=5, shuffle=True, random_state=17),
        processors=None,
        verbose=False,
    )

    selector = PermutationRecursiveFeatureElimination(
        estimator=clf,
        cross_validator=cross_validator,
        steps=5,
        alpha=0.95,
        n_repeats=5,
        verbose=True,
    )

    summary = selector.run(X_df, y_ser)
    print("\nSummary of Permutation Feature Importance RFE:")
    print(summary["history"])
    print("Selected features:", summary["selected_features"])
