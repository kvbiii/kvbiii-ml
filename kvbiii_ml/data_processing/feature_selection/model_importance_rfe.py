import gc
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parents[3]))
from kvbiii_ml.data_processing.feature_selection.pipeline_dependency import (
    PipelineDependencyGraph,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


class ModelImportanceRecursiveFeatureElimination:
    """Recursive feature elimination using model's feature_importances_ attribute.

    Removes the least-important features per a linear-decay removal schedule,
    recomputing ``feature_importances_`` from a fresh CV fit at every step.

    When the cross-validator holds a column-expansion pipeline, the elimination
    loop works entirely in *processed* feature space - the post-pipeline column
    set. Raw input features needed for each step are resolved via a
    ``PipelineDependencyGraph`` built once from a real baseline pipeline fit.

    ``protected_features`` must be specified as *processed* column names and are
    validated after the baseline CV run when the processed column set is known.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cross_validator: CrossValidationTrainer,
        steps: int = 5,
        alpha: float = 0.95,
        verbose: bool = True,
        protected_features: list[str] | None = None,
        random_state: int | None = 17,
    ) -> None:
        """
        Initialize the Model Importance RFE selector.

        Args:
            estimator (BaseEstimator): Estimator with feature_importances_ attribute.
            cross_validator (CrossValidationTrainer): Cross-validation trainer that
                optionally holds a preprocessing pipeline.  The pipeline is re-fitted
                per fold at each elimination step via a restricted clone.
            steps (int, optional): Number of elimination iterations. Defaults to 5.
            alpha (float, optional): Weight for the final selection score mix.
                Defaults to 0.95.
            verbose (bool, optional): Whether to print progress messages.
                Defaults to True.
            protected_features (list[str] | None, optional): Processed column names
                that should never be removed.  Validated after the baseline CV run
                against the actual post-pipeline column set. Defaults to None.
            random_state (int | None, optional): Random state for reproducibility.
                Defaults to 17.
        """
        self.estimator = estimator
        self.cross_validator = cross_validator
        self.steps = steps
        self.alpha = alpha
        self.verbose = verbose
        self.protected_features = protected_features or []
        self.metric_direction = self.cross_validator.metric_direction
        self.random_state = random_state
        self.all_processed_features: list[str] = []
        self.history_schema = {
            "step": int,
            "n_features_removed": int,
            "n_features_remaining": int,
            "removed_feature_name": object,
            "metric_value": float,
            "metric_change": float,
            "importance_score": float,
        }

    @staticmethod
    def _restrict_catboost_cat_features(
        estimator: BaseEstimator,
        current_features: set[str],
        post_pipeline_dtypes: pd.Series | None = None,
    ) -> BaseEstimator:
        """Return a cloned CatBoost estimator with cat_features filtered to active non-numeric columns.

        A column is kept in cat_features only when all three conditions hold:
        - it is still present in the active feature set,
        - it still exists in the post-pipeline output, and
        - its post-pipeline dtype is non-numeric.

        Returns the original estimator unchanged for non-CatBoost estimators.

        Args:
            estimator (BaseEstimator): Estimator to potentially update.
            current_features (set[str]): Feature names still present in this step.
            post_pipeline_dtypes (pd.Series | None): Column dtypes of the post-pipeline
                validation data. Defaults to None.

        Returns:
            BaseEstimator: Updated estimator for CatBoost; original for all others.
        """
        if "CatBoost" not in type(estimator).__name__:
            return estimator
        original_cats = estimator.get_params().get("cat_features") or []
        if not original_cats:
            return estimator
        if post_pipeline_dtypes is not None:
            processed_cols = set(post_pipeline_dtypes.index)
            active_cats = [
                c
                for c in original_cats
                if c in current_features
                and c in processed_cols
                and not pd.api.types.is_numeric_dtype(post_pipeline_dtypes[c])
            ]
        else:
            active_cats = [c for c in original_cats if c in current_features]
        if set(active_cats) == set(original_cats):
            return estimator
        updated = deepcopy(estimator)
        updated.set_params(cat_features=active_cats if active_cats else None)
        return updated

    def run(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> dict[str, list | pd.DataFrame]:
        """
        Run Model Importance RFE and return the selection summary.

        Algorithm:
            1. Baseline CV - pipeline fit once via PipelineDependencyGraph.build();
               processed column set and per-column raw dependencies discovered.
            2. protected_features validated against the post-pipeline column set.
            3. Each step: dependency graph pruned via restrict(), CV re-run on the
               restricted pipeline, importances extracted from feature_importances_.
            4. Least-important features removed per the linear-decay schedule.

        Args:
            X (pd.DataFrame): Feature matrix (raw columns).
            y (pd.Series | np.ndarray): Target array/series aligned with X.

        Returns:
            dict[str, list | pd.DataFrame]: Dictionary with keys:
                - selected_features (list): Final selected processed feature names.
                - selected_features_names (list): Alias of selected_features.
                - history (pd.DataFrame): Step-wise metrics and removals.

        Raises:
            ValueError: If protected features are not found in the post-pipeline
                column set.
        """
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        all_raw_features: list[str] = sorted(X.columns.tolist())

        pipeline = self.cross_validator.preprocessing_pipeline
        dependency_graph = PipelineDependencyGraph.build(
            pipeline, X[all_raw_features], y
        )

        avg_base_metric, _ = self._cross_val_base_metric(
            X[all_raw_features], y, dependency_graph
        )

        self.all_processed_features = list(dependency_graph.processed_columns)
        all_processed_set = set(self.all_processed_features)
        missing_protected = set(self.protected_features) - all_processed_set
        if missing_protected:
            raise ValueError(
                f"Protected features not found in post-pipeline column set: {missing_protected}. "
                f"Available processed columns: {sorted(all_processed_set)}"
            )

        current_features = list(self.all_processed_features)

        summary_df = {
            "selected_features": [],
            "selected_features_names": [],
            "history": pd.DataFrame(columns=self.history_schema.keys()).astype(
                self.history_schema
            ),
        }

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
                f"🔍 Starting Model Importance RFE with {len(current_features)} "
                f"features, target metric: {self.cross_validator.metric_name} "
                f"({self.metric_direction}), steps: {self.steps}.\n"
                f"📅 The number of features to remove every step: {removal_schedule}"
            )
            print(f"🛡️  Protected features: {self.protected_features}")
            print(
                f"📊 Initial {self.cross_validator.metric_name}: "
                f"{avg_base_metric:.6f} | Features: {len(current_features)}"
            )

        for step_idx, n_features_to_remove in enumerate(removal_schedule, start=1):
            current_raw_features, restricted_pipeline = dependency_graph.restrict(
                current_features
            )
            step_estimator = self._restrict_catboost_cat_features(
                self.estimator,
                set(current_features),
                dependency_graph.processed_dtypes,
            )

            importance_scores, fold_base_metric, fold_base_metric_std = (
                self._cross_val_model_importance(
                    X[current_raw_features],
                    y,
                    current_features,
                    step_estimator,
                    restricted_pipeline,
                )
            )
            if self.verbose:
                print(
                    f"\n🔁 Step {step_idx} | Number of features remaining: "
                    f"{len(current_features)}"
                )
                print(f"\n🔬 Features remaining: {current_features}\n")
                print(
                    f"📊 Average {self.cross_validator.metric_name}: "
                    f"{fold_base_metric:.6f} ± {fold_base_metric_std:.6f}"
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
                print("Most important features:")
                for feat in reversed(importance_df.tail(5)["feature"].tolist()):
                    print(f"  • {feat}: {importance_scores[feat]:.6f}")
                print("\nLeast important features (candidates for removal):")
                for feat in features_to_remove[:5]:
                    print(f"  • {feat}: {importance_scores[feat]:.6f}")

            self._log_step(
                summary_df,
                step_idx,
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
                f"📈 Final {self.cross_validator.metric_name} score (approximated): "
                f"{metric_selected:.6f} | Base: {base_val:.6f} | "
                f"Δ: {metric_selected - base_val:.6f} ({diff_pct:+.2f}%)"
            )
            n_features_initial = summary_df["history"].iloc[0]["n_features_remaining"]
            n_features_selected = len(summary_df["selected_features"])
            n_removed = n_features_initial - n_features_selected
            pct_removed = (
                100 * n_removed / n_features_initial if n_features_initial else 0.0
            )
            print(
                f"🗑️ Features removed: {n_removed} of {n_features_initial} "
                f"({pct_removed:.2f}%)"
            )
        gc.collect()
        return summary_df

    def _cross_val_base_metric(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dependency_graph: PipelineDependencyGraph,
    ) -> tuple[float, float]:
        """
        Compute the baseline metric across validation folds using the full pipeline.

        Args:
            X (pd.DataFrame): Raw feature matrix.
            y (pd.Series): Target values.
            dependency_graph (PipelineDependencyGraph): Baseline graph built from the
                full pipeline fit, used to bootstrap CatBoost's cat_features.

        Returns:
            tuple[float, float]: Average metric and its standard deviation.
        """
        baseline_estimator = self._restrict_catboost_cat_features(
            self.estimator,
            set(dependency_graph.processed_columns),
            dependency_graph.processed_dtypes,
        )
        _, valid_scores, _ = self.cross_validator.fit(baseline_estimator, X, y)
        gc.collect()
        return float(np.mean(valid_scores)), float(np.std(valid_scores))

    def _cross_val_model_importance(
        self,
        x_data: pd.DataFrame,
        y_data: pd.Series,
        current_processed_features: list[str],
        step_estimator: BaseEstimator,
        pipeline_override: Pipeline | None,
    ) -> tuple[dict[str, float], float, float]:
        """
        Compute feature importance from the model's feature_importances_ attribute.

        Args:
            x_data (pd.DataFrame): Raw features for this step.
            y_data (pd.Series): The target series.
            current_processed_features (list[str]): Processed column names the
                model sees, in the order feature_importances_ reports them.
            step_estimator (BaseEstimator): Estimator restricted to active cat features.
            pipeline_override (Pipeline | None): Restricted pipeline for this step.

        Returns:
            tuple[dict[str, float], float, float]: A tuple containing:
                - A dictionary of average importance scores for each processed feature.
                - The mean baseline validation score across folds.
                - The standard deviation of the baseline validation score.

        Raises:
            AttributeError: If the estimator does not have feature_importances_.
        """
        _, valid_scores, _ = self.cross_validator.fit(
            step_estimator,
            x_data,
            y_data,
            preprocessing_pipeline_override=pipeline_override,
        )
        importances = []
        for estimator in self.cross_validator.fitted_estimators_:
            if not hasattr(estimator, "feature_importances_"):
                raise AttributeError(
                    f"Estimator {type(estimator).__name__} does not have "
                    "'feature_importances_' attribute."
                )
            importances.append(estimator.feature_importances_)

        avg_importances = np.nan_to_num(np.mean(importances, axis=0), nan=0.0)
        avg_importance_map = {
            feature: float(imp)
            for feature, imp in zip(current_processed_features, avg_importances)
        }
        gc.collect()
        return (
            avg_importance_map,
            float(np.mean(valid_scores)),
            float(np.std(valid_scores)),
        )

    def compute_removal_schedule(self, total_removable_features: int) -> list[int]:
        """
        Compute a linear-decay schedule for features to remove per step.

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
        removal_counts = np.maximum(removal_counts, 0).astype(int)
        removal_counts = removal_counts.tolist()
        while removal_counts and removal_counts[-1] == 0:
            removal_counts.pop()
        return removal_counts

    def _log_step(
        self,
        summary_df: dict[str, list | pd.DataFrame],
        step_idx: int,
        current_features: list[str],
        importance_scores: dict[str, float],
        fold_base_metric: float,
        features_to_remove: list[str],
    ) -> None:
        """
        Log results for the current elimination step.

        Args:
            summary_df (dict[str, list | pd.DataFrame]): Summary dict carrying history.
            step_idx (int): Current step number.
            current_features (list[str]): Processed features remaining before removal.
                Mutated in place - each removed feature is popped as it is logged.
            importance_scores (dict[str, float]): Importance score for each processed feature.
            fold_base_metric (float): Base metric for the current set of features.
            features_to_remove (list[str]): Processed features to remove this step.
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
                                "n_features_removed": len(self.all_processed_features)
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
        """
        Select features by maximizing a weighted metric/features score.

        Args:
            history (pd.DataFrame): Step-wise elimination history.
            alpha (float | None, optional): Weight for metric vs. features.
                Defaults to self.alpha.

        Returns:
            tuple[list[str], float | None]: Selected features and their metric value.
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
        selected = set(
            history[history["step"] >= best_row["step"]]["removed_feature_name"]
        )
        selected.update(self.protected_features)
        gc.collect()
        return list(sorted(selected)), best_row["metric_value"]


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold

    def _run_demo() -> None:
        """Run ModelImportanceRecursiveFeatureElimination on a synthetic dataset."""
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_repeated=0,
            n_clusters_per_class=2,
            random_state=17,
        )
        x_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_ser = pd.Series(y, name="target")
        clf = RandomForestClassifier(random_state=17, max_depth=5, n_estimators=100)
        cross_validator_example = CrossValidationTrainer(
            problem_type="classification",
            metric_name="Accuracy",
            cv=KFold(n_splits=5, shuffle=True, random_state=17),
            preprocessing_pipeline=None,
            verbose=False,
        )

        selector = ModelImportanceRecursiveFeatureElimination(
            estimator=clf,
            cross_validator=cross_validator_example,
            steps=10,
            alpha=0.95,
            verbose=True,
            protected_features=["feature_0", "feature_1"],
        )

        summary = selector.run(x_df, y_ser)
        print("\nSummary of Model Importance RFE:")
        print(
            summary["history"][
                [
                    "step",
                    "removed_feature_name",
                    "n_features_remaining",
                    "metric_value",
                    "importance_score",
                ]
            ]
        )
        print("Selected features:", summary["selected_features"])
        print(f"Number of selected features: {len(summary['selected_features'])}")

    _run_demo()
