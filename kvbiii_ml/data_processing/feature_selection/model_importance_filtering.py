import gc
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parents[3]))
from kvbiii_ml.data_processing.feature_selection.pipeline_dependency import (
    PipelineDependencyGraph,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


class ModelImportanceFiltering:
    """Feature selection using stepwise model importance filtering.

    Iteratively fits a model on all folds, extracts ``feature_importances_``, and
    removes features whose mean importance falls at or below ``threshold``.  This
    repeats until no removable feature remains below the threshold or ``max_steps``
    is reached.

    When the cross-validator holds a column-expansion pipeline, the elimination
    loop works in *processed* feature space - the post-pipeline column set. Raw
    input features needed for each step are resolved via a
    ``PipelineDependencyGraph`` built once from a real baseline pipeline fit.

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
            1. Baseline CV - pipeline fit once via PipelineDependencyGraph.build();
               processed column set and per-column raw dependencies discovered.
            2. protected_features validated against post-pipeline column set.
            3. Each step: dependency graph pruned via restrict(), CV run, importances extracted.
            4. Features at or below threshold removed; loop repeats.
            5. Never-removed non-protected features appended to history for completeness.

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
        baseline_std = float(np.std(valid_scores))

        self.all_processed_features = list(dependency_graph.processed_columns)
        all_processed_set = set(self.all_processed_features)

        missing_protected = set(self.protected_features) - all_processed_set
        if missing_protected:
            raise ValueError(
                f"Protected features not found in post-pipeline column set: {missing_protected}. "
                f"Available processed columns: {sorted(all_processed_set)}"
            )

        current_processed_features: list[str] = list(self.all_processed_features)

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

            current_raw_features, restricted_pipeline = dependency_graph.restrict(
                current_processed_features
            )
            step_estimator = self._restrict_catboost_cat_features(
                self.estimator,
                set(current_processed_features),
                dependency_graph.processed_dtypes,
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
        x_num, y_arr = make_classification(
            N_SAMPLES=N_SAMPLES,
            N_FEATURES=N_FEATURES,
            n_informative=7,
            n_redundant=3,
            n_classes=n_classes,
            n_clusters_per_class=1,
            RANDOM_STATE=RANDOM_STATE,
        )
        df = pd.DataFrame(x_num, columns=NUM_FEATURES)
        df["cat_1"] = pd.Categorical(rng.choice(["A", "B", "C", "D"], size=N_SAMPLES))
        df["cat_2"] = pd.Categorical(rng.choice(["X", "Y", "Z"], size=N_SAMPLES))
        return df, pd.Series(y_arr, name="target")

    def _make_reg_data() -> tuple[pd.DataFrame, pd.Series]:
        """Generate regression dataset with numerical and categorical features."""
        rng = np.random.default_rng(RANDOM_STATE)
        x_num, y_arr = make_regression(
            N_SAMPLES=N_SAMPLES,
            N_FEATURES=N_FEATURES,
            n_informative=7,
            RANDOM_STATE=RANDOM_STATE,
        )
        df = pd.DataFrame(x_num, columns=NUM_FEATURES)
        df["cat_1"] = pd.Categorical(rng.choice(["A", "B", "C", "D"], size=N_SAMPLES))
        df["cat_2"] = pd.Categorical(rng.choice(["X", "Y", "Z"], size=N_SAMPLES))
        return df, pd.Series(y_arr, name="target")

    def _build_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
        """Build the expansion pipeline used across all scenarios."""
        return Pipeline(
            [
                ("imputer", MeanMedianImputer(imputation_method="median")),
                (
                    "winsorizer_with_original",
                    WinsorizerWithOriginal(
                        variables=num_cols,
                        capping_method="iqr",
                        tail="both",
                        fold=3.0,
                    ),
                ),
                (
                    "mean_encoder",
                    MeanEncoder(variables=cat_cols, missing_values="ignore"),
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
            cv=cv_cls(n_splits=N_FOLDS, shuffle=True, RANDOM_STATE=RANDOM_STATE),
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

    def _run_demo() -> None:
        """Run ModelImportanceFiltering across the full demo scenario matrix."""
        lgbm_clf = LGBMClassifier(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=-1,
            RANDOM_STATE=RANDOM_STATE,
        )
        lgbm_reg = LGBMRegressor(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=-1,
            RANDOM_STATE=RANDOM_STATE,
        )
        xgb_clf = XGBClassifier(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbosity=0,
            RANDOM_STATE=RANDOM_STATE,
        )
        xgb_reg = XGBRegressor(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbosity=0,
            RANDOM_STATE=RANDOM_STATE,
        )
        cat_clf = CatBoostClassifier(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=0,
            RANDOM_STATE=RANDOM_STATE,
            CAT_FEATURES=CAT_FEATURES,
        )
        cat_multi = CatBoostClassifier(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=0,
            RANDOM_STATE=RANDOM_STATE,
            CAT_FEATURES=CAT_FEATURES,
            loss_function="MultiClass",
        )
        cat_reg = CatBoostRegressor(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=0,
            RANDOM_STATE=RANDOM_STATE,
            CAT_FEATURES=CAT_FEATURES,
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
        print("ModelImportanceFiltering - full test matrix (3 folds, threshold=0.0)")
        print("=" * 75)

        _run_mif(
            "LightGBM | binary classification | with pipeline",
            lgbm_clf,
            x_bin_cat,
            y_bin,
            "Balanced Accuracy",
            "classification",
            clf_pipeline,
            threshold=0.0,
        )
        _run_mif(
            "LightGBM | regression | with pipeline",
            lgbm_reg,
            x_reg_cat,
            y_reg,
            "RMSE",
            "regression",
            reg_pipeline,
            threshold=0.0,
        )
        _run_mif(
            "LightGBM | regression | no pipeline",
            lgbm_reg,
            x_reg_cat[NUM_FEATURES],
            y_reg,
            "RMSE",
            "regression",
            None,
            threshold=0.0,
        )
        _run_mif(
            "XGBoost | binary classification | with pipeline",
            xgb_clf,
            x_bin_cat,
            y_bin,
            "Balanced Accuracy",
            "classification",
            clf_pipeline,
            threshold=0.0,
        )
        _run_mif(
            "XGBoost | multiclass classification | with pipeline",
            xgb_clf,
            x_multi_cat,
            y_multi,
            "Balanced Accuracy",
            "classification",
            clf_pipeline,
            threshold=0.0,
        )
        _run_mif(
            "XGBoost | regression | with pipeline",
            xgb_reg,
            x_reg_cat,
            y_reg,
            "RMSE",
            "regression",
            reg_pipeline,
            threshold=0.0,
        )
        _run_mif(
            "CatBoost | binary classification | with pipeline",
            cat_clf,
            x_bin_cat,
            y_bin,
            "Balanced Accuracy",
            "classification",
            clf_pipeline,
            threshold=0.0,
        )
        _run_mif(
            "CatBoost | multiclass classification | with pipeline",
            cat_multi,
            x_multi_cat,
            y_multi,
            "Balanced Accuracy",
            "classification",
            clf_pipeline,
            threshold=0.0,
        )
        _run_mif(
            "CatBoost | regression | with pipeline",
            cat_reg,
            x_reg_cat,
            y_reg,
            "RMSE",
            "regression",
            reg_pipeline,
            threshold=0.0,
        )

    _run_demo()
