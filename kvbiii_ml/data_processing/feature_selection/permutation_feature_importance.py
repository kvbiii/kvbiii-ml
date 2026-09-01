from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
import warnings

import joblib.parallel
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[3]))
from kvbiii_ml.data_processing.feature_selection.pipeline_dependency import (
    PipelineDependencyGraph,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


@contextmanager
def _tqdm_joblib_callback(progress_bar: tqdm) -> Generator[tqdm, None, None]:
    """Report joblib batch completions to a tqdm bar for the duration of this context.

    ``permutation_importance`` dispatches one joblib task per feature column, but
    exposes no progress callback of its own. Patching joblib's batch-completion
    callback is the only way to surface live per-feature progress without
    reimplementing permutation scoring by hand. The patch is process-global for
    the duration of the ``with`` block and is always reverted in ``finally``, so
    it must not be nested with other joblib-progress hooks running concurrently
    in other threads. joblib's sequential backend (``n_jobs=1``) never invokes
    this callback, so the bar is topped up to completion on exit as a fallback.

    Args:
        progress_bar (tqdm): Progress bar to update as joblib batches complete.

    Yields:
        tqdm: The same progress bar, for convenience.
    """
    original_callback = joblib.parallel.BatchCompletionCallBack

    class _TqdmBatchCompletionCallback(original_callback):
        def __call__(self, *args: object, **kwargs: object) -> object:
            progress_bar.update(self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallback
    try:
        yield progress_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback
        remaining = progress_bar.total - progress_bar.n
        if remaining > 0:
            progress_bar.update(remaining)


class PermutationRecursiveFeatureElimination:
    """Recursive feature elimination using permutation importance.

    Iteratively fits a model on all folds once (baseline), computes permutation
    importance on the processed validation sets, then removes the least-important
    processed features according to a linear-decay schedule.

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
        n_repeats: int = 5,
        verbose: bool = True,
        protected_features: list[str] | None = None,
        random_state: int = 17,
        n_jobs: int = -1,
    ) -> None:
        """Initialize the permutation RFE selector.

        Args:
            estimator (BaseEstimator): Estimator to fit within each fold.
            cross_validator (CrossValidationTrainer): Cross-validation trainer that
                already holds the preprocessing pipeline.  The pipeline is re-fitted
                per fold during the baseline run and then injected as a step-restricted
                override at each elimination step.
            steps (int): Number of elimination iterations. Defaults to 5.
            alpha (float): Weight for metric vs feature-count balance in final
                selection. Defaults to 0.95.
            n_repeats (int): Permutation repeats per feature per fold. Defaults to 5.
            verbose (bool): Print emoji-styled step-by-step progress. Defaults to True.
            protected_features (list[str] | None): Processed feature names never
                removed.  Validated after the baseline CV against the actual
                post-pipeline column set.  Defaults to None.
            random_state (int): Random seed for reproducibility. Defaults to 17.
            n_jobs (int): Parallel jobs for permutation importance. Defaults to -1.
        """
        self.estimator = estimator
        self.cross_validator = cross_validator
        self.steps = steps
        self.alpha = alpha
        self.n_repeats = n_repeats
        self.verbose = verbose
        self.protected_features = protected_features or []
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.metric_fn = self.cross_validator.metric_fn
        self.metric_type = self.cross_validator.metric_type
        self.metric_direction = self.cross_validator.metric_direction

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
        """Run permutation RFE and return the selection summary.

        Algorithm:
            1. Baseline CV - pipeline fit once via PipelineDependencyGraph.build();
               processed column set and per-column raw dependencies discovered.
            2. Processed validation sets built from fitted pipelines.
            3. protected_features validated against processed columns.
            4. Permutation importance computed once in processed space (fixed baseline).
            5. Removal schedule built from removable processed feature count.
            6. Each step: print current state, restrict pipeline via the dependency
               graph, re-run CV, record scores.
            7. Return step-wise DataFrame for manual elbow inspection.

        Args:
            X (pd.DataFrame): Feature matrix (raw columns).
            y (pd.Series | np.ndarray): Target aligned with X.

        Returns:
            dict[str, list | pd.DataFrame]: Keys:
                - selected_features (list[str]): Final selected processed feature names.
                - selected_features_names (list[str]): Alias for selected_features.
                - history (pd.DataFrame): Step-wise metrics and removals.

        Raises:
            ValueError: When protected features are not found in the post-pipeline
                column set.
        """
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        all_raw_features: list[str] = sorted(X.columns.tolist())

        summary_df: dict[str, list | pd.DataFrame] = {
            "selected_features": [],
            "selected_features_names": [],
            "history": pd.DataFrame(columns=self.history_schema.keys()).astype(
                self.history_schema
            ),
        }

        pipeline = self.cross_validator.preprocessing_pipeline
        dependency_graph = PipelineDependencyGraph.build(
            pipeline, X[all_raw_features], y
        )

        baseline_estimator = self._restrict_catboost_cat_features(
            self.estimator,
            set(dependency_graph.processed_columns),
            dependency_graph.processed_dtypes,
        )
        _, valid_scores, _ = self.cross_validator.fit(
            baseline_estimator, X[all_raw_features], y
        )
        baseline_metric = float(np.mean(valid_scores))

        fold_splits = list(self.cross_validator.cv.split(X[all_raw_features], y))
        fold_data = [
            (
                est,
                CrossValidationTrainer._transform_with_pipeline(
                    pipe, X[all_raw_features].iloc[val_idx].copy()
                ),
                y.iloc[val_idx],
            )
            for (_, val_idx), est, pipe in zip(
                fold_splits,
                self.cross_validator.fitted_estimators_,
                self.cross_validator.fitted_pipelines_,
            )
        ]

        self.all_processed_features: list[str] = list(
            dependency_graph.processed_columns
        )
        all_processed_set = set(self.all_processed_features)

        missing_protected = set(self.protected_features) - all_processed_set
        if missing_protected:
            raise ValueError(
                f"Protected features not found in post-pipeline column set: {missing_protected}. "
                f"Available processed columns: {sorted(all_processed_set)}"
            )

        importance_scores = self._compute_fold_importances(fold_data)

        removable_processed = [
            f for f in self.all_processed_features if f not in self.protected_features
        ]
        removal_schedule = self.compute_removal_schedule(len(removable_processed))

        current_processed_features: list[str] = list(self.all_processed_features)
        current_valid_scores = valid_scores

        summary_df["history"] = pd.concat(
            [
                summary_df["history"],
                pd.DataFrame(
                    [
                        {
                            "step": 0,
                            "n_features_removed": 0,
                            "n_features_remaining": len(current_processed_features),
                            "removed_feature_name": None,
                            "metric_value": baseline_metric,
                            "metric_change": 0.0,
                            "importance_score": np.nan,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        minimize = self.metric_direction == "minimize"

        for step_idx, n_to_remove in enumerate(removal_schedule, start=1):
            if len(current_processed_features) <= len(self.protected_features):
                if self.verbose:
                    print("⏹️  Stopping early - only protected features remain.")
                break

            current_set = set(current_processed_features)
            sorted_removable = sorted(
                (
                    (k, v)
                    for k, v in importance_scores.items()
                    if k not in self.protected_features and k in current_set
                ),
                key=lambda x: x[1],
                reverse=minimize,
            )

            min_keep = max(len(self.protected_features), 1)
            n_actual = min(
                n_to_remove,
                len(sorted_removable),
                len(current_processed_features) - min_keep,
            )
            if n_actual <= 0:
                if self.verbose:
                    print("⏹️  Stopping - minimum feature count reached.")
                break

            features_to_remove = [f for f, _ in sorted_removable[:n_actual]]

            if self.verbose:
                self._print_step(
                    step_idx,
                    current_processed_features,
                    current_valid_scores,
                    sorted_removable,
                    importance_scores,
                )

            self._log_step(
                summary_df,
                step_idx,
                len(self.all_processed_features),
                current_processed_features,
                importance_scores,
                float(np.mean(current_valid_scores)),
                features_to_remove,
            )

            current_processed_features = [
                f
                for f in current_processed_features
                if f not in set(features_to_remove)
            ]
            current_raw_features, restricted_pipeline = dependency_graph.restrict(
                current_processed_features
            )
            step_estimator = self._restrict_catboost_cat_features(
                self.estimator,
                set(current_processed_features),
                dependency_graph.processed_dtypes,
            )

            _, current_valid_scores, _ = self.cross_validator.fit(
                step_estimator,
                X[current_raw_features],
                y,
                preprocessing_pipeline_override=restricted_pipeline,
            )

        protected_set = set(self.protected_features)
        logged_set = set(summary_df["history"]["removed_feature_name"].dropna())
        never_removed = [
            f
            for f in self.all_processed_features
            if f not in protected_set and f not in logged_set
        ]
        if never_removed:
            last_step = int(summary_df["history"]["step"].max())
            last_n_removed = int(summary_df["history"]["n_features_removed"].max())
            final_metric = float(np.mean(current_valid_scores))
            never_removed_sorted = sorted(
                never_removed, key=lambda f: importance_scores.get(f, 0.0)
            )
            rows = [
                {
                    "step": last_step + 1,
                    "n_features_removed": last_n_removed + i + 1,
                    "n_features_remaining": len(current_processed_features) - i - 1,
                    "removed_feature_name": feat,
                    "metric_value": final_metric,
                    "metric_change": 0.0,
                    "importance_score": importance_scores.get(feat, 0.0),
                }
                for i, feat in enumerate(never_removed_sorted)
            ]
            summary_df["history"] = pd.concat(
                [summary_df["history"], pd.DataFrame(rows)], ignore_index=True
            )

        summary_df["selected_features"], metric_selected = (
            self.select_features_weighted_score(summary_df["history"], self.alpha)
        )
        summary_df["selected_features_names"] = summary_df["selected_features"]

        if self.verbose:
            self._print_summary(
                summary_df["selected_features"],
                metric_selected,
                baseline_metric,
                len(self.all_processed_features),
            )

        return summary_df

    def _compute_fold_importances(self, fold_data: list[tuple]) -> dict[str, float]:
        """Compute mean permutation importance averaged across CV fold models.

        Args:
            fold_data (list[tuple]): List of (fitted_estimator, X_val_proc, y_val) per fold.

        Returns:
            dict[str, float]: Processed feature name → mean importance score.
        """
        is_classification = self.metric_type == "probs"

        def _scorer(estimator: BaseEstimator, X: pd.DataFrame, y: np.ndarray) -> float:
            """Sklearn-compatible scorer wrapping the user metric."""
            pred = (
                estimator.predict_proba(X)
                if is_classification
                else estimator.predict(X)
            )
            return self.metric_fn(pd.Series(y), pred)

        fold_importances: list[pd.Series] = []
        for fold_idx, (est, x_val, y_val) in enumerate(fold_data):
            with (
                tqdm(
                    total=len(x_val.columns),
                    desc=f"Fold {fold_idx + 1}/{len(fold_data)} - Permuting features",
                    disable=not self.verbose,
                    leave=True,
                ) as progress_bar,
                _tqdm_joblib_callback(progress_bar),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore")
                result = permutation_importance(
                    est,
                    x_val,
                    y_val,
                    scoring=_scorer,
                    n_repeats=self.n_repeats,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state + fold_idx,
                )
            fold_importances.append(
                pd.Series(result.importances_mean, index=x_val.columns)
            )

        return pd.DataFrame(fold_importances).mean().to_dict()

    @staticmethod
    def _restrict_catboost_cat_features(
        estimator: BaseEstimator,
        current_features: set[str],
        post_pipeline_dtypes: pd.Series | None = None,
    ) -> BaseEstimator:
        """Return a cloned CatBoost estimator with cat_features filtered to active non-numeric columns.

        A column is kept in cat_features only when all three conditions hold:
        - it is still present in the active raw feature set,
        - it still exists in the post-pipeline output (i.e. was not renamed by the pipeline,
            e.g. a MeanEncoder that outputs ``COL_PREPROCESS_MEAN_ENC`` drops ``COL``), and
        - its post-pipeline dtype is non-numeric (i.e. it was not encoded to float64).

        Returns the original estimator unchanged for non-CatBoost estimators.

        Args:
            estimator (BaseEstimator): Estimator to potentially update.
            current_features (set[str]): Raw feature names still present in this step.
            post_pipeline_dtypes (pd.Series | None): Column dtypes of the post-pipeline
                validation data (index = column names).  Defaults to None.

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

    def compute_removal_schedule(self, total_removable_features: int) -> list[int]:
        """Compute a linear-decay schedule for features to remove per step.

        Removes more features in early steps (many unimportant) and fewer in later
        steps (each removal riskier).  Schedule sums exactly to
        ``total_removable_features``.

        Args:
            total_removable_features (int): Number of features eligible for removal.

        Returns:
            list[int]: Non-zero counts to remove at each step.
        """
        if total_removable_features <= 0 or self.steps <= 0:
            return []
        decay = np.linspace(1, 0.2, self.steps)
        weights = decay / decay.sum()
        removal_counts = np.round(weights * total_removable_features).astype(int)
        diff = total_removable_features - removal_counts.sum()
        for i in range(abs(diff)):
            removal_counts[i % self.steps] += 1 if diff > 0 else -1
        removal_counts = np.maximum(removal_counts, 0).astype(int).tolist()
        while removal_counts and removal_counts[-1] == 0:
            removal_counts.pop()
        return removal_counts

    def _log_step(
        self,
        summary_df: dict[str, list | pd.DataFrame],
        step_idx: int,
        total_processed: int,
        current_processed: list[str],
        importance_scores: dict[str, float],
        metric: float,
        features_to_remove: list[str],
    ) -> None:
        """Append one row per removed processed feature to history.

        Args:
            summary_df (dict): Summary dict carrying the history DataFrame.
            step_idx (int): Current step number.
            total_processed (int): Total processed feature count at baseline.
            current_processed (list[str]): Processed features before this removal.
            importance_scores (dict[str, float]): Importance score per processed feature.
            metric (float): Metric value for the current feature set.
            features_to_remove (list[str]): Processed features being removed this step.
        """
        n_removed_base = total_processed - len(current_processed)
        protected_set = set(self.protected_features)
        rows = [
            {
                "step": step_idx,
                "n_features_removed": n_removed_base + i + 1,
                "n_features_remaining": len(current_processed) - i - 1,
                "removed_feature_name": feat,
                "metric_value": metric,
                "metric_change": 0.0,
                "importance_score": importance_scores.get(feat, 0.0),
            }
            for i, feat in enumerate(
                f for f in features_to_remove if f not in protected_set
            )
        ]
        if rows:
            summary_df["history"] = pd.concat(
                [summary_df["history"], pd.DataFrame(rows)], ignore_index=True
            )

    def select_features_weighted_score(
        self,
        history: pd.DataFrame,
        alpha: float | None = None,
    ) -> tuple[list[str], float | None]:
        """Select features by maximising a weighted metric/features score at step level.

        Aggregates history to one row per step (each step's metric is the score of
        the feature set at the *start* of that step, before any removal), then picks
        the best step and returns all features that were not removed before it.

        Args:
            history (pd.DataFrame): Step-wise history DataFrame with processed feature names.
            alpha (float | None): Weight for metric vs feature-count balance.
                If None, uses self.alpha. Defaults to None.

        Returns:
            tuple[list[str], float | None]: Selected processed feature names and the
                metric value at the selected step, or ([], None) if history is empty.
        """
        if history.empty:
            return [], None
        if alpha is None:
            alpha = self.alpha

        step_agg = (
            history.groupby("step")
            .agg(
                metric_value=("metric_value", "first"),
                n_features_remaining=("n_features_remaining", "max"),
            )
            .reset_index()
        )
        step_agg.loc[step_agg["step"] > 0, "n_features_remaining"] += 1

        metric_max = step_agg["metric_value"].max()
        metric_min = step_agg["metric_value"].min()
        denom_metric = metric_max - metric_min if metric_max != metric_min else 1.0
        step_agg["metric_norm"] = (
            (step_agg["metric_value"] - metric_min) / denom_metric
            if self.metric_direction == "maximize"
            else (metric_max - step_agg["metric_value"]) / denom_metric
        )
        feat_max = step_agg["n_features_remaining"].max()
        feat_min = step_agg["n_features_remaining"].min()
        denom_feat = feat_max - feat_min if feat_max != feat_min else 1.0
        step_agg["features_norm"] = (
            1 - (step_agg["n_features_remaining"] - feat_min) / denom_feat
        )
        step_agg["score"] = (
            alpha * step_agg["metric_norm"] + (1 - alpha) * step_agg["features_norm"]
        )

        best_idx = step_agg["score"].idxmax()
        best_step = int(step_agg.loc[best_idx, "step"])
        best_metric = float(step_agg.loc[best_idx, "metric_value"])

        removed_before_best = set(
            history[(history["step"] > 0) & (history["step"] < best_step)][
                "removed_feature_name"
            ].dropna()
        )
        selected = sorted(set(self.all_processed_features) - removed_before_best)
        return selected, best_metric

    def _print_step(
        self,
        step_idx: int,
        current_processed_features: list[str],
        valid_scores: list[float] | np.ndarray,
        sorted_removable: list[tuple[str, float]],
        importance_scores: dict[str, float],
    ) -> None:
        """Print a step header showing the current feature set, metric, and importance summary.

        Args:
            step_idx (int): Current step number.
            current_processed_features (list[str]): Active processed feature names.
            valid_scores (list[float] | np.ndarray): Per-fold validation scores.
            sorted_removable (list[tuple[str, float]]): Removable features sorted so
                the least important appear first (candidates for removal).
            importance_scores (dict[str, float]): Full importance map for top-5 lookup.
        """
        avg = float(np.mean(valid_scores))
        std = float(np.std(valid_scores))
        current_set = set(current_processed_features)

        print(
            f"\n🔁 Step {step_idx} | Number of features remaining: {len(current_processed_features)}\n"
        )
        print(f"🔬 Features remaining: {current_processed_features}\n")
        print(f"📊 Average {self.cross_validator.metric_name}: {avg:.6f} ± {std:.6f}")

        top5_most = sorted(
            ((f, s) for f, s in importance_scores.items() if f in current_set),
            key=lambda x: x[1],
            reverse=(self.metric_direction == "maximize"),
        )[:5]
        print("Top 5 most important features:")
        for feat, score in top5_most:
            print(f"  • {feat}: {score:.6f}")

        print()
        print("Top 5 least important features (candidates for removal):")
        for feat, score in sorted_removable[:5]:
            print(f"  • {feat}: {score:.6f}")
        print()

    def _print_summary(
        self,
        selected_features: list[str],
        metric_selected: float | None,
        base_metric: float,
        total_processed: int,
    ) -> None:
        """Print the final selection summary.

        Args:
            selected_features (list[str]): Selected processed feature names.
            metric_selected (float | None): Metric at the selected step.
            base_metric (float): Baseline metric value.
            total_processed (int): Total processed feature count before any elimination.
        """
        n_selected = len(selected_features)
        n_removed = total_processed - n_selected
        pct_removed = 100 * n_removed / total_processed if total_processed > 0 else 0.0

        print(f"\n🎯 Selected features: {selected_features}")

        if metric_selected is not None and base_metric != 0:
            delta = metric_selected - base_metric
            pct = 100 * delta / base_metric
            print(
                f"📈 Final {self.cross_validator.metric_name} score (approximated): "
                f"{metric_selected:.6f} | Base: {base_metric:.6f} "
                f"| Δ: {delta:+.6f} ({pct:+.2f}%)"
            )
        else:
            print(f"📈 Final metric: {metric_selected}")

        print(
            f"🗑️ Features removed: {n_removed} of {total_processed} ({pct_removed:.2f}%)"
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from feature_engine.encoding import MeanEncoder
    from feature_engine.imputation import MeanMedianImputer
    from lightgbm import LGBMClassifier, LGBMRegressor
    from catboost import CatBoostClassifier, CatBoostRegressor
    from xgboost import XGBClassifier, XGBRegressor
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.pipeline import Pipeline

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from kvbiii_ml.data_processing.preprocessing.outlier_handling.winsorizer_trimmer import (
        WinsorizerWithOriginal,
    )
    from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer
    from kvbiii_ml.modeling.training.ensemble_model import EnsembleModel

    RANDOM_STATE = 42
    N_SAMPLES = 3_000
    N_FOLDS = 3
    N_FEATURES = 12
    N_STEPS = 5
    CAT_FEATURES = ["cat_1", "cat_2"]
    NUM_FEATURES = [f"num_{i}" for i in range(N_FEATURES)]
    ES = 30

    def _make_clf_data(n_classes: int) -> tuple[pd.DataFrame, pd.Series]:
        """Generate classification dataset with numerical and categorical features."""
        rng = np.random.default_rng(RANDOM_STATE)
        x_num, y_arr = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=7,
            n_redundant=3,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=RANDOM_STATE,
        )
        df = pd.DataFrame(x_num, columns=NUM_FEATURES)
        df["cat_1"] = pd.Categorical(rng.choice(["A", "B", "C", "D"], size=N_SAMPLES))
        df["cat_2"] = pd.Categorical(rng.choice(["X", "Y", "Z"], size=N_SAMPLES))
        return df, pd.Series(y_arr, name="target")

    def _make_reg_data() -> tuple[pd.DataFrame, pd.Series]:
        """Generate regression dataset with numerical and categorical features."""
        rng = np.random.default_rng(RANDOM_STATE)
        x_num, y_arr = make_regression(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=7,
            random_state=RANDOM_STATE,
        )
        df = pd.DataFrame(x_num, columns=NUM_FEATURES)
        df["cat_1"] = pd.Categorical(rng.choice(["A", "B", "C", "D"], size=N_SAMPLES))
        df["cat_2"] = pd.Categorical(rng.choice(["X", "Y", "Z"], size=N_SAMPLES))
        return df, pd.Series(y_arr, name="target")

    def _build_pipeline(cat_features: list[str], num_features: list[str]) -> Pipeline:
        """Build the expansion pipeline used across all scenarios."""
        return Pipeline(
            [
                ("imputer", MeanMedianImputer(imputation_method="median")),
                (
                    "winsorizer_with_original",
                    WinsorizerWithOriginal(
                        variables=num_features,
                        capping_method="iqr",
                        tail="both",
                        fold=3.0,
                    ),
                ),
                (
                    "mean_encoder",
                    MeanEncoder(variables=cat_features, missing_values="ignore"),
                ),
            ]
        )

    def _run_rfe(
        label: str,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        metric_name: str,
        problem_type: str,
        pipeline: Pipeline | None,
    ) -> None:
        """Run RFE with the given pipeline and print a summary line."""
        cv_cls = StratifiedKFold if problem_type == "classification" else KFold
        trainer = CrossValidationTrainer(
            problem_type=problem_type,
            metric_name=metric_name,
            cv=cv_cls(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            preprocessing_pipeline=pipeline,
            verbose=False,
        )
        selector = PermutationRecursiveFeatureElimination(
            estimator=estimator,
            cross_validator=trainer,
            steps=N_STEPS,
            n_repeats=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=True,
        )
        summary = selector.run(X, y)

        if not len(summary["selected_features"]) > 0:
            raise AssertionError("no features selected")
        if pipeline is not None:
            removed_names = summary["history"]["removed_feature_name"].dropna().tolist()
            if not any("_PREPROCESS_" in str(f) for f in removed_names):
                raise AssertionError(
                    "no derived features appear in removal history - pipeline did not expand"
                )

        n_selected = len(summary["selected_features"])
        print(f"  {label:<55} selected={n_selected} features\n")

    def _run_demo() -> None:
        """Run PermutationRecursiveFeatureElimination across the demo scenario matrix."""
        lgbm_clf = LGBMClassifier(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=-1,
            random_state=RANDOM_STATE,
        )
        lgbm_reg = LGBMRegressor(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=-1,
            random_state=RANDOM_STATE,
        )

        x_bin, y_bin = _make_clf_data(n_classes=2)
        x_multi, y_multi = _make_clf_data(n_classes=3)
        x_reg, y_reg = _make_reg_data()

        x_bin_cat = x_bin.assign(**{c: x_bin[c].astype(str) for c in CAT_FEATURES})
        x_multi_cat = x_multi.assign(
            **{c: x_multi[c].astype(str) for c in CAT_FEATURES}
        )
        x_reg_cat = x_reg.assign(**{c: x_reg[c].astype(str) for c in CAT_FEATURES})

        clf_pipeline = _build_pipeline(CAT_FEATURES, NUM_FEATURES)
        reg_pipeline = _build_pipeline(CAT_FEATURES, NUM_FEATURES)

        print("=" * 75)
        print(
            "PermutationRecursiveFeatureElimination - full test matrix (3 folds, 5 steps)"
        )
        print("=" * 75)

        _run_rfe(
            "LightGBM | binary classification",
            lgbm_clf,
            x_bin_cat,
            y_bin,
            "Balanced Accuracy",
            "classification",
            clf_pipeline,
        )
        _run_rfe(
            "LightGBM | regression",
            lgbm_reg,
            x_reg_cat,
            y_reg,
            "RMSE",
            "regression",
            reg_pipeline,
        )
        _run_rfe(
            "LightGBM | regression (no pipeline)",
            lgbm_reg,
            x_reg_cat[NUM_FEATURES],
            y_reg,
            "RMSE",
            "regression",
            None,
        )

        xgb_clf = XGBClassifier(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbosity=0,
            random_state=RANDOM_STATE,
        )
        xgb_reg = XGBRegressor(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbosity=0,
            random_state=RANDOM_STATE,
        )
        _run_rfe(
            "XGBoost | binary classification",
            xgb_clf,
            x_bin_cat,
            y_bin,
            "Balanced Accuracy",
            "classification",
            clf_pipeline,
        )
        _run_rfe(
            "XGBoost | multiclass classification",
            xgb_clf,
            x_multi_cat,
            y_multi,
            "Balanced Accuracy",
            "classification",
            clf_pipeline,
        )
        _run_rfe(
            "XGBoost | regression",
            xgb_reg,
            x_reg_cat,
            y_reg,
            "RMSE",
            "regression",
            reg_pipeline,
        )

        cat_clf = CatBoostClassifier(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=0,
            random_state=RANDOM_STATE,
            cat_features=CAT_FEATURES,
        )
        cat_multi = CatBoostClassifier(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=0,
            random_state=RANDOM_STATE,
            cat_features=CAT_FEATURES,
            loss_function="MultiClass",
        )
        cat_reg = CatBoostRegressor(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=0,
            random_state=RANDOM_STATE,
            cat_features=CAT_FEATURES,
        )
        _run_rfe(
            "CatBoost | binary classification",
            cat_clf,
            x_bin_cat,
            y_bin,
            "Balanced Accuracy",
            "classification",
            clf_pipeline,
        )
        _run_rfe(
            "CatBoost | multiclass classification",
            cat_multi,
            x_multi_cat,
            y_multi,
            "Balanced Accuracy",
            "classification",
            clf_pipeline,
        )
        _run_rfe(
            "CatBoost | regression",
            cat_reg,
            x_reg_cat,
            y_reg,
            "RMSE",
            "regression",
            reg_pipeline,
        )

        ensemble_clf = EnsembleModel(
            estimators=[lgbm_clf, xgb_clf, cat_clf], problem_type="classification"
        )
        _run_rfe(
            "Ensemble (LGBM + XGB + CatBoost) | binary classification",
            ensemble_clf,
            x_bin_cat,
            y_bin,
            "Balanced Accuracy",
            "classification",
            clf_pipeline,
        )
        ensemble_reg = EnsembleModel(
            estimators=[lgbm_reg, xgb_reg, cat_reg], problem_type="regression"
        )
        _run_rfe(
            "Ensemble (LGBM + XGB + CatBoost) | regression",
            ensemble_reg,
            x_reg_cat,
            y_reg,
            "RMSE",
            "regression",
            reg_pipeline,
        )
        ensemble_multi = EnsembleModel(
            estimators=[lgbm_clf, xgb_clf, cat_multi], problem_type="classification"
        )
        _run_rfe(
            "Ensemble (LGBM + XGB + CatBoost MultiClass) | multiclass classification",
            ensemble_multi,
            x_multi_cat,
            y_multi,
            "Balanced Accuracy",
            "classification",
            clf_pipeline,
        )

    _run_demo()
