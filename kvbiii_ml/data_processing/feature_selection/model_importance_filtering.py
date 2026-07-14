import gc
import warnings
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

sys.path.append(str(Path(__file__).resolve().parents[3]))
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


class ModelImportanceFiltering:
    """Feature selection using stepwise model importance filtering.

    Iteratively fits a model on all folds, extracts ``feature_importances_``, and
    removes features whose mean importance falls at or below ``threshold``.  This
    repeats until no removable feature remains below the threshold or ``max_steps``
    is reached.

    When the cross-validator holds a column-expansion pipeline (steps that inherit
    from ``_FeatureExpansionBase`` and expose a ``_suffix`` attribute), the
    elimination loop works in *processed* feature space - the post-pipeline column
    set.  Raw input features are derived from the active processed set at each step.

    ``protected_features`` must be specified as *processed* column names and are
    validated after the baseline CV run when the processed column set is known.
    """

    history_schema = {
        "step": int,
        "n_features_removed": int,
        "n_features_remaining": int,
        "removed_feature_name": object,
        "metric_value": float,
        "metric_change": float,
        "importance_score": float,
    }

    def __init__(
        self,
        estimator: BaseEstimator,
        cross_validator: CrossValidationTrainer,
        threshold: float = 0.0,
        protected_features: list[str] | None = None,
        max_steps: int = 10,
        verbose: bool = False,
    ) -> None:
        """Initialize the model importance filter.

        Args:
            estimator (BaseEstimator): Estimator with ``feature_importances_`` attribute.
            cross_validator (CrossValidationTrainer): Cross-validation trainer that
                optionally holds a preprocessing pipeline.  The pipeline is re-fitted
                per fold at each elimination step via a restricted clone.
            threshold (float): Features with mean importance ≤ this value are removed
                each step.  Defaults to 0.0.
            protected_features (list[str] | None): Processed column names never removed.
                Validated after the baseline CV run against the actual post-pipeline
                column set.  Defaults to None.
            max_steps (int): Maximum elimination iterations.  Defaults to 10.
            verbose (bool): Print emoji-styled step-by-step progress.  Defaults to False.

        Raises:
            ValueError: If any parameter has an invalid type or value.
        """
        self.estimator = estimator
        self.cross_validator = cross_validator
        self.threshold = threshold
        self.protected_features = protected_features or []
        self.max_steps = max_steps
        self.verbose = verbose

        self.metric_direction = self.cross_validator.metric_direction
        self.selected_features_: list[str] = []
        self.importance_scores_: dict[str, float] = {}
        self.all_processed_features: list[str] = []
        self._validate_options()

    def _validate_options(self) -> None:
        """Validate init parameter values.

        Raises:
            ValueError: When any parameter has an invalid type or value.
        """
        if not isinstance(self.threshold, (float, int)):
            raise ValueError("threshold must be a numeric value.")
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer.")
        if not isinstance(self.verbose, bool):
            raise ValueError("verbose must be a boolean.")
        if not isinstance(self.protected_features, list) or not all(
            isinstance(item, str) for item in self.protected_features
        ):
            raise ValueError("protected_features must be a list of strings.")

    # ─────────────────────────── static pipeline helpers ────────────────────────────

    @staticmethod
    def _select_features(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        """Select specified features from X, silently skipping any absent ones."""
        return X[[f for f in features if f in X.columns]]

    @staticmethod
    def _probe_pipeline_dtypes(
        pipeline: Pipeline | None,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.Series | None:
        """Fit-transform the pipeline on a tiny sample to discover output column dtypes.

        Used to detect columns that a pipeline step (e.g. MeanEncoder) converts from
        categorical strings to numeric, so that CatBoost's cat_features can be filtered
        before the baseline CV fit.

        Args:
            pipeline (Pipeline | None): Preprocessing pipeline to probe.
            X (pd.DataFrame): Raw feature matrix.
            y (pd.Series): Target vector.

        Returns:
            pd.Series | None: Column dtype series of the probed output, or None when
                pipeline is None or the output is not a DataFrame.
        """
        if pipeline is None:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_probe = clone(pipeline).fit_transform(X.head(10), y.head(10))
        return X_probe.dtypes if isinstance(X_probe, pd.DataFrame) else None

    @staticmethod
    def _build_raw_to_derived(
        all_raw_features: list[str],
        all_processed_set: set[str],
        pipeline: Pipeline | None,
    ) -> dict[str, list[str]]:
        """Map each raw feature to its derived column names produced by expansion steps.

        Only steps that expose a non-empty ``_suffix`` class attribute (those inheriting
        from ``_FeatureExpansionBase``) contribute derived columns.

        Args:
            all_raw_features (list[str]): Raw input column names.
            all_processed_set (set[str]): Set of all post-pipeline column names.
            pipeline (Pipeline | None): Preprocessing pipeline or None.

        Returns:
            dict[str, list[str]]: raw feature → list of derived column names actually
                present in the processed output.
        """
        raw_to_derived: dict[str, list[str]] = {f: [] for f in all_raw_features}
        if pipeline is None:
            return raw_to_derived
        for _, step in pipeline.steps:
            if not (hasattr(step, "_suffix") and step._suffix):
                continue
            suffix = f"_{step._suffix}"
            for raw_feat in all_raw_features:
                derived = f"{raw_feat}{suffix}"
                if derived in all_processed_set:
                    raw_to_derived[raw_feat].append(derived)
        return raw_to_derived

    @staticmethod
    def _compute_active_raw(
        all_raw: list[str],
        current_processed: list[str],
        raw_to_derived: dict[str, list[str]],
    ) -> list[str]:
        """Compute the raw input features still needed given the active processed set.

        A raw feature is needed when either its own name appears in the processed set
        or at least one of its derived expansion columns does.

        Args:
            all_raw (list[str]): All raw input column names.
            current_processed (list[str]): Processed feature names still active.
            raw_to_derived (dict[str, list[str]]): raw → derived column names mapping.

        Returns:
            list[str]: Raw features required as pipeline input for the current step.
        """
        current_processed_set = set(current_processed)
        return [
            f
            for f in all_raw
            if f in current_processed_set
            or any(d in current_processed_set for d in raw_to_derived[f])
        ]

    @staticmethod
    def _build_restricted_pipeline(
        preprocessor: Pipeline | None,
        current_raw_features: list[str],
        current_processed_features: list[str],
        X_current: pd.DataFrame | None = None,
    ) -> Pipeline | None:
        """Clone the pipeline, restricting each step and appending a feature selector.

        Steps with an explicit ``variables`` list are filtered to contain only columns
        present in ``current_raw_features``.  Steps with ``variables=None`` are trial-fitted
        on a tiny sample to detect compatibility.  A ``FunctionTransformer`` wrapping
        ``_select_features`` is always appended as the final step so the model receives
        exactly ``current_processed_features``.

        Args:
            preprocessor (Pipeline | None): Original preprocessing pipeline.
            current_raw_features (list[str]): Raw columns still active.
            current_processed_features (list[str]): Processed columns the model should see.
            X_current (pd.DataFrame | None): Representative sample used to trial-fit steps
                with auto-detected variables.  Defaults to None.

        Returns:
            Pipeline | None: Restricted clone with feature selector appended, or None
                if preprocessor is None.
        """
        if preprocessor is None:
            return None

        raw_set = set(current_raw_features)
        new_steps: list[tuple[str, BaseEstimator]] = []

        for name, step in preprocessor.steps:
            cloned_step = clone(step)
            params = cloned_step.get_params()
            if isinstance(params.get("variables"), list):
                filtered = [v for v in params["variables"] if v in raw_set]
                if not filtered:
                    continue
                cloned_step.set_params(variables=filtered)
            elif params.get("variables") is None and X_current is not None:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        clone(step).fit(X_current.head(5))
                except (ValueError, TypeError, KeyError, AttributeError, IndexError):
                    continue
            new_steps.append((name, cloned_step))

        new_steps.append(
            (
                "_feature_selector",
                FunctionTransformer(
                    func=ModelImportanceFiltering._select_features,
                    kw_args={"features": current_processed_features},
                ),
            )
        )
        return Pipeline(new_steps)

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
                validation data.  Defaults to None.

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

    # ─────────────────────────────── summary helpers ────────────────────────────────

    def _init_summary(
        self, current_features: list[str], base_metric: float
    ) -> dict[str, Any]:
        """Initialize summary object with step-0 baseline metrics.

        Args:
            current_features (list[str]): Full processed feature set at baseline.
            base_metric (float): Baseline CV metric value.

        Returns:
            dict[str, Any]: Summary dict with ``selected_features``, ``selected_features_names``,
                and ``history`` pre-populated with the step-0 row.
        """
        history = pd.DataFrame(columns=self.history_schema.keys()).astype(
            self.history_schema
        )
        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [
                        {
                            "step": 0,
                            "n_features_removed": 0,
                            "n_features_remaining": len(current_features),
                            "removed_feature_name": None,
                            "metric_value": base_metric,
                            "metric_change": 0.0,
                            "importance_score": np.nan,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        return {
            "selected_features": [],
            "selected_features_names": [],
            "history": history,
        }

    def _append_removal_rows(
        self,
        history: pd.DataFrame,
        step_idx: int,
        current_features: list[str],
        step_state: dict[str, Any],
    ) -> pd.DataFrame:
        """Append one history row per removed feature.

        Args:
            history (pd.DataFrame): Existing history DataFrame.
            step_idx (int): Current step number.
            current_features (list[str]): Processed features before this removal.
            step_state (dict[str, Any]): Step data including importance_scores,
                fold_metric, metric_change, and features_to_remove.

        Returns:
            pd.DataFrame: Updated history with new rows appended.
        """
        start_count = len(current_features)
        total = len(self.all_processed_features)
        removed_features = step_state["features_to_remove"]
        importance_scores = step_state["importance_scores"]
        rows = [
            {
                "step": step_idx,
                "n_features_removed": total - start_count + offset,
                "n_features_remaining": start_count - offset,
                "removed_feature_name": feature_name,
                "metric_value": step_state["fold_metric"],
                "metric_change": step_state["metric_change"],
                "importance_score": importance_scores[feature_name],
            }
            for offset, feature_name in enumerate(removed_features, start=1)
        ]
        return pd.concat([history, pd.DataFrame(rows)], ignore_index=True)

    def _select_features_to_remove(
        self, importance_scores: dict[str, float]
    ) -> list[str]:
        """Select non-protected features at or below threshold, sorted ascending by importance.

        Args:
            importance_scores (dict[str, float]): Processed feature → mean importance.

        Returns:
            list[str]: Features to remove this step, least important first.
        """
        return sorted(
            (
                feature
                for feature, score in importance_scores.items()
                if feature not in self.protected_features and score <= self.threshold
            ),
            key=lambda f: importance_scores[f],
        )

    # ─────────────────────────────── logging helpers ────────────────────────────────

    def _log_start(
        self,
        current_features: list[str],
        avg_base_metric: float,
        base_std: float,
    ) -> None:
        """Print initial run summary when verbose mode is enabled.

        Args:
            current_features (list[str]): Processed features at baseline.
            avg_base_metric (float): Mean baseline CV metric.
            base_std (float): Std of baseline CV metric.
        """
        if not self.verbose:
            return
        print(
            "🔍 Starting Model Importance Stepwise Filtering with "
            f"{len(current_features)} features, target metric: "
            f"{self.cross_validator.metric_name} ({self.metric_direction}), "
            f"threshold: {self.threshold}, max_steps: {self.max_steps}.\n"
        )
        print(f"🛡️  Protected features: {self.protected_features}")
        print(
            f"📊 Initial {self.cross_validator.metric_name}: "
            f"{avg_base_metric:.6f} ± {base_std:.6f} | "
            f"Features: {len(current_features)}"
        )

    def _log_step(
        self,
        step_idx: int,
        current_features: list[str],
        step_state: dict[str, Any],
    ) -> None:
        """Print per-step details when verbose mode is enabled.

        Args:
            step_idx (int): Current step number.
            current_features (list[str]): Active processed feature names.
            step_state (dict[str, Any]): Step data including importance_scores,
                fold_metric, fold_std, and features_to_remove.
        """
        if not self.verbose:
            return

        importance_scores = step_state["importance_scores"]
        fold_metric = step_state["fold_metric"]
        fold_std = step_state["fold_std"]
        features_to_remove = step_state["features_to_remove"]

        print(f"\n🔁 Step {step_idx} | Features remaining: {len(current_features)}")
        print(
            f"📊 Average {self.cross_validator.metric_name}: "
            f"{fold_metric:.6f} ± {fold_std:.6f}"
        )

        sorted_by_importance = sorted(
            importance_scores.items(), key=lambda item: item[1], reverse=True
        )
        print(f"\n🔬 Features remaining: {current_features}\n")
        print("Most important features:")
        for feature_name, score in sorted_by_importance[:5]:
            print(f"  • {feature_name}: {score:.6f}")
        print(f"\nLeast important features (below threshold {self.threshold}):")
        for feature_name in features_to_remove[:5]:
            print(f"  • {feature_name}: {importance_scores[feature_name]:.6f}")

        pct = 100 * len(features_to_remove) / len(current_features)
        print(
            f"\n🗑️  Removing {len(features_to_remove)} "
            f"({pct:.2f}%) features this step"
        )

    def _log_finish(self, summary: dict[str, Any]) -> None:
        """Print final selection summary when verbose mode is enabled.

        Args:
            summary (dict[str, Any]): Completed summary dict.
        """
        if not self.verbose:
            return

        print(f"\n🎯 Selected features: {summary['selected_features']}")
        metric_series = summary["history"]["metric_value"].dropna()
        final_metric = (
            float(metric_series.iloc[-1]) if not metric_series.empty else np.nan
        )
        base_val = float(summary["history"].iloc[0]["metric_value"])
        diff = final_metric - base_val
        diff_pct = 100 * diff / base_val if base_val != 0 else np.nan
        print(
            f"📈 Final {self.cross_validator.metric_name} score: "
            f"{final_metric:.6f} | Base: {base_val:.6f} | "
            f"Δ: {diff:.6f} ({diff_pct:+.2f}%)"
        )

        initial_features = int(summary["history"].iloc[0]["n_features_remaining"])
        removed_count = initial_features - len(summary["selected_features"])
        pct_removed = (
            100 * removed_count / initial_features if initial_features else 0.0
        )
        print(
            f"🗑️  Features removed: {removed_count} of {initial_features} "
            f"({pct_removed:.2f}%)"
        )

    # ──────────────────────────── CV importance helper ──────────────────────────────

    def _cross_val_model_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        current_processed_features: list[str],
        step_estimator: BaseEstimator,
        pipeline_override: Pipeline | None,
    ) -> tuple[dict[str, float], float, float]:
        """Fit CV with the restricted pipeline and extract mean feature importances.

        Args:
            X (pd.DataFrame): Raw features for this step.
            y (pd.Series): Target.
            current_processed_features (list[str]): Processed column names the model sees.
            step_estimator (BaseEstimator): Estimator restricted to active cat features.
            pipeline_override (Pipeline | None): Restricted pipeline for this step.

        Returns:
            tuple[dict[str, float], float, float]: importance_map, mean_score, std_score.

        Raises:
            AttributeError: If the fitted estimator lacks ``feature_importances_``.
        """
        _, valid_scores, _ = self.cross_validator.fit(
            step_estimator,
            X,
            y,
            preprocessing_pipeline_override=pipeline_override,
        )

        importances = []
        for fitted_estimator in self.cross_validator.fitted_estimators_:
            if not hasattr(fitted_estimator, "feature_importances_"):
                raise AttributeError(
                    f"Estimator {type(fitted_estimator).__name__} does not have "
                    "'feature_importances_' attribute."
                )
            importances.append(fitted_estimator.feature_importances_)

        avg_importances = np.nan_to_num(np.mean(importances, axis=0), nan=0.0)
        importance_map = {
            feature: float(score)
            for feature, score in zip(current_processed_features, avg_importances)
        }
        gc.collect()
        return importance_map, float(np.mean(valid_scores)), float(np.std(valid_scores))

    # ──────────────────────────────── main methods ──────────────────────────────────

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """Run stepwise model importance filtering and return selection summary.

        Algorithm:
            1. Baseline CV - pipeline applied per fold; processed column set discovered.
            2. protected_features validated against post-pipeline column set.
            3. raw_to_derived mapping built from pipeline expansion steps.
            4. Each step: restricted pipeline built, CV run, importances extracted.
            5. Features at or below threshold removed; loop repeats.
            6. Never-removed non-protected features appended to history for completeness.

        Args:
            X (pd.DataFrame): Feature matrix (raw columns).
            y (pd.Series | np.ndarray): Target aligned with X.

        Returns:
            dict[str, Any]: Keys:
                - selected_features (list[str]): Final selected processed column names.
                - selected_features_names (list[str]): Alias for selected_features.
                - history (pd.DataFrame): Step-wise metrics and removals.  All
                  non-protected features appear here, including surviving ones at
                  a trailing virtual step for completeness.

        Raises:
            ValueError: When protected features are not found in the post-pipeline
                column set.
        """
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        all_raw_features: list[str] = sorted(X.columns.tolist())

        pipeline = self.cross_validator.preprocessing_pipeline
        post_pipeline_dtypes: pd.Series | None = self._probe_pipeline_dtypes(
            pipeline, X[all_raw_features], y
        )

        _baseline_feature_set = (
            set(post_pipeline_dtypes.index)
            if post_pipeline_dtypes is not None
            else set(all_raw_features)
        )
        baseline_estimator = self._restrict_catboost_cat_features(
            self.estimator, _baseline_feature_set, post_pipeline_dtypes
        )
        _, valid_scores, _ = self.cross_validator.fit(
            baseline_estimator, X[all_raw_features], y
        )
        baseline_metric = float(np.mean(valid_scores))
        baseline_std = float(np.std(valid_scores))

        fold_splits = list(self.cross_validator.cv.split(X[all_raw_features], y))
        if self.cross_validator.fitted_pipelines_:
            _, val_idx = fold_splits[0]
            first_pipe = self.cross_validator.fitted_pipelines_[0]
            X_val_proc = CrossValidationTrainer._transform_with_pipeline(
                first_pipe, X[all_raw_features].iloc[val_idx].copy()
            )
            self.all_processed_features = X_val_proc.columns.tolist()
            post_pipeline_dtypes = X_val_proc.dtypes
        else:
            self.all_processed_features = list(all_raw_features)

        all_processed_set = set(self.all_processed_features)

        missing_protected = set(self.protected_features) - all_processed_set
        if missing_protected:
            raise ValueError(
                f"Protected features not found in post-pipeline column set: {missing_protected}. "
                f"Available processed columns: {sorted(all_processed_set)}"
            )

        raw_to_derived = self._build_raw_to_derived(
            all_raw_features, all_processed_set, pipeline
        )

        current_processed_features: list[str] = list(self.all_processed_features)
        current_raw_features: list[str] = list(all_raw_features)

        summary = self._init_summary(current_processed_features, baseline_metric)
        self._log_start(current_processed_features, baseline_metric, baseline_std)

        prev_metric = baseline_metric
        importance_scores: dict[str, float] = {
            f: np.nan for f in current_processed_features
        }

        for step_idx in range(1, self.max_steps + 1):
            if len(current_processed_features) <= len(self.protected_features):
                if self.verbose:
                    print("⏹️  Stopping - only protected features remain.")
                break

            restricted_pipeline = self._build_restricted_pipeline(
                pipeline,
                current_raw_features,
                current_processed_features,
                X[current_raw_features],
            )
            step_estimator = self._restrict_catboost_cat_features(
                self.estimator, set(current_processed_features), post_pipeline_dtypes
            )

            importance_scores, fold_metric, fold_std = self._cross_val_model_importance(
                X[current_raw_features],
                y,
                current_processed_features,
                step_estimator,
                restricted_pipeline,
            )

            features_to_remove = self._select_features_to_remove(importance_scores)
            step_state = {
                "importance_scores": importance_scores,
                "fold_metric": fold_metric,
                "fold_std": fold_std,
                "features_to_remove": features_to_remove,
                "metric_change": fold_metric - prev_metric,
            }
            self._log_step(step_idx, current_processed_features, step_state)

            if not features_to_remove:
                if self.verbose:
                    print(
                        f"✅ Convergence reached - all features exceed threshold {self.threshold}"
                    )
                break

            prev_metric = fold_metric
            summary["history"] = self._append_removal_rows(
                summary["history"],
                step_idx,
                current_processed_features,
                step_state,
            )

            current_processed_features = [
                f
                for f in current_processed_features
                if f not in set(features_to_remove)
            ]
            current_raw_features = self._compute_active_raw(
                all_raw_features, current_processed_features, raw_to_derived
            )

            gc.collect()

        protected_set = set(self.protected_features)
        logged_set = set(summary["history"]["removed_feature_name"].dropna())
        never_removed = [
            f
            for f in self.all_processed_features
            if f not in protected_set and f not in logged_set
        ]
        if never_removed:
            last_step = int(summary["history"]["step"].max())
            last_n_removed = int(
                summary["history"]["n_features_removed"].fillna(0).max()
            )
            metric_series = summary["history"]["metric_value"].dropna()
            final_metric = (
                float(metric_series.iloc[-1])
                if not metric_series.empty
                else baseline_metric
            )
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
                    "importance_score": importance_scores.get(feat, np.nan),
                }
                for i, feat in enumerate(never_removed_sorted)
            ]
            summary["history"] = pd.concat(
                [summary["history"], pd.DataFrame(rows)], ignore_index=True
            )

        summary["selected_features"] = sorted(current_processed_features)
        summary["selected_features_names"] = summary["selected_features"]
        self.selected_features_ = summary["selected_features"]
        self.importance_scores_ = importance_scores

        self._log_finish(summary)
        gc.collect()
        return summary

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> "ModelImportanceFiltering":
        """Fit selector and cache selected features.

        Args:
            X (pd.DataFrame): Feature matrix (raw columns).
            y (pd.Series | np.ndarray): Target aligned with X.

        Returns:
            ModelImportanceFiltering: Self, for method chaining.
        """
        self.run(X, y)
        return self


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

    RANDOM_STATE = 42
    N_SAMPLES = 3_000
    N_FOLDS = 3
    N_FEATURES = 12
    CAT_FEATURES = ["cat_1", "cat_2"]
    NUM_FEATURES = [f"num_{i}" for i in range(N_FEATURES)]
    ES = 30

    def _make_clf_data(n_classes: int) -> tuple[pd.DataFrame, pd.Series]:
        """Generate classification dataset with numerical and categorical features."""
        rng = np.random.default_rng(RANDOM_STATE)
        X_num, y_arr = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=7,
            n_redundant=3,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=RANDOM_STATE,
        )
        df = pd.DataFrame(X_num, columns=NUM_FEATURES)
        df["cat_1"] = pd.Categorical(rng.choice(["A", "B", "C", "D"], size=N_SAMPLES))
        df["cat_2"] = pd.Categorical(rng.choice(["X", "Y", "Z"], size=N_SAMPLES))
        return df, pd.Series(y_arr, name="target")

    def _make_reg_data() -> tuple[pd.DataFrame, pd.Series]:
        """Generate regression dataset with numerical and categorical features."""
        rng = np.random.default_rng(RANDOM_STATE)
        X_num, y_arr = make_regression(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=7,
            random_state=RANDOM_STATE,
        )
        df = pd.DataFrame(X_num, columns=NUM_FEATURES)
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

    def _run_mif(
        label: str,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        metric_name: str,
        problem_type: str,
        pipeline: Pipeline | None,
        threshold: float = 0.0,
    ) -> None:
        """Run ModelImportanceFiltering and validate the summary."""
        cv_cls = StratifiedKFold if problem_type == "classification" else KFold
        trainer = CrossValidationTrainer(
            problem_type=problem_type,
            metric_name=metric_name,
            cv=cv_cls(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            preprocessing_pipeline=pipeline,
            verbose=False,
        )
        selector = ModelImportanceFiltering(
            estimator=estimator,
            cross_validator=trainer,
            threshold=threshold,
            max_steps=10,
            verbose=True,
        )
        summary = selector.run(X, y)

        if not len(summary["selected_features"]) > 0:
            raise AssertionError("no features selected")

        history_features = set(summary["history"]["removed_feature_name"].dropna())
        non_protected_selected = [
            f
            for f in summary["selected_features"]
            if f not in selector.protected_features
        ]
        for feat in non_protected_selected:
            if feat not in history_features:
                raise AssertionError(
                    f"selected feature '{feat}' missing from history - never-removed append failed"
                )

        if pipeline is not None:
            all_history_features = list(history_features)
            if all_history_features:
                if not any("_PREPROCESS_" in str(f) for f in all_history_features):
                    raise AssertionError(
                        "no derived features appear in removal history - pipeline did not expand"
                    )

        n_selected = len(summary["selected_features"])
        print(f"  {label:<60} selected={n_selected} features\n")

    _lgbm_clf = LGBMClassifier(
        n_estimators=200,
        early_stopping_rounds=ES,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    _lgbm_reg = LGBMRegressor(
        n_estimators=200,
        early_stopping_rounds=ES,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    _xgb_clf = XGBClassifier(
        n_estimators=200,
        early_stopping_rounds=ES,
        verbosity=0,
        random_state=RANDOM_STATE,
    )
    _xgb_reg = XGBRegressor(
        n_estimators=200,
        early_stopping_rounds=ES,
        verbosity=0,
        random_state=RANDOM_STATE,
    )
    _cat_clf = CatBoostClassifier(
        n_estimators=200,
        early_stopping_rounds=ES,
        verbose=0,
        random_state=RANDOM_STATE,
        cat_features=CAT_FEATURES,
    )
    _cat_multi = CatBoostClassifier(
        n_estimators=200,
        early_stopping_rounds=ES,
        verbose=0,
        random_state=RANDOM_STATE,
        cat_features=CAT_FEATURES,
        loss_function="MultiClass",
    )
    _cat_reg = CatBoostRegressor(
        n_estimators=200,
        early_stopping_rounds=ES,
        verbose=0,
        random_state=RANDOM_STATE,
        cat_features=CAT_FEATURES,
    )

    X_bin, y_bin = _make_clf_data(n_classes=2)
    X_multi, y_multi = _make_clf_data(n_classes=3)
    X_reg, y_reg = _make_reg_data()

    X_bin_cat = X_bin.assign(**{c: X_bin[c].astype(str) for c in CAT_FEATURES})
    X_multi_cat = X_multi.assign(**{c: X_multi[c].astype(str) for c in CAT_FEATURES})
    X_reg_cat = X_reg.assign(**{c: X_reg[c].astype(str) for c in CAT_FEATURES})

    clf_pipeline = _build_pipeline(CAT_FEATURES, NUM_FEATURES)
    reg_pipeline = _build_pipeline(CAT_FEATURES, NUM_FEATURES)

    print("=" * 75)
    print("ModelImportanceFiltering - full test matrix (3 folds, threshold=0.0)")
    print("=" * 75)

    _run_mif(
        "LightGBM | binary classification | with pipeline",
        _lgbm_clf,
        X_bin_cat,
        y_bin,
        "Balanced Accuracy",
        "classification",
        clf_pipeline,
        threshold=0.0,
    )
    _run_mif(
        "LightGBM | regression | with pipeline",
        _lgbm_reg,
        X_reg_cat,
        y_reg,
        "RMSE",
        "regression",
        reg_pipeline,
        threshold=0.0,
    )
    _run_mif(
        "LightGBM | regression | no pipeline",
        _lgbm_reg,
        X_reg_cat[NUM_FEATURES],
        y_reg,
        "RMSE",
        "regression",
        None,
        threshold=0.0,
    )
    _run_mif(
        "XGBoost | binary classification | with pipeline",
        _xgb_clf,
        X_bin_cat,
        y_bin,
        "Balanced Accuracy",
        "classification",
        clf_pipeline,
        threshold=0.0,
    )
    _run_mif(
        "XGBoost | multiclass classification | with pipeline",
        _xgb_clf,
        X_multi_cat,
        y_multi,
        "Balanced Accuracy",
        "classification",
        clf_pipeline,
        threshold=0.0,
    )
    _run_mif(
        "XGBoost | regression | with pipeline",
        _xgb_reg,
        X_reg_cat,
        y_reg,
        "RMSE",
        "regression",
        reg_pipeline,
        threshold=0.0,
    )
    _run_mif(
        "CatBoost | binary classification | with pipeline",
        _cat_clf,
        X_bin_cat,
        y_bin,
        "Balanced Accuracy",
        "classification",
        clf_pipeline,
        threshold=0.0,
    )
    _run_mif(
        "CatBoost | multiclass classification | with pipeline",
        _cat_multi,
        X_multi_cat,
        y_multi,
        "Balanced Accuracy",
        "classification",
        clf_pipeline,
        threshold=0.0,
    )
    _run_mif(
        "CatBoost | regression | with pipeline",
        _cat_reg,
        X_reg_cat,
        y_reg,
        "RMSE",
        "regression",
        reg_pipeline,
        threshold=0.0,
    )
