import gc
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


class ModelImportanceFiltering:
    """Feature selection using stepwise model importance filtering."""

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
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the model importance filter.

        Args:
            estimator (BaseEstimator): Estimator with feature_importances_ attribute.
            cross_validator (CrossValidationTrainer): Cross-validation trainer instance.
            options (dict[str, Any] | None, optional): Runtime settings including
                threshold, protected_features, max_steps, and verbose.
            **kwargs (Any): Backward-compatible option values passed directly.

        Raises:
            ValueError: If any option has an invalid type or value.
        """
        default_options = {
            "threshold": 0.0,
            "protected_features": [],
            "max_steps": 10,
            "verbose": False,
        }
        merged_options = dict(default_options)
        if options:
            merged_options.update(options)
        if kwargs:
            merged_options.update(kwargs)

        self.estimator = estimator
        self.cross_validator = cross_validator
        self.options = merged_options
        self.selected_features_: list[str] = []
        self.importance_scores_: dict[str, float] = {}
        self._validate_options()

    @property
    def threshold(self) -> float:
        """Return the minimum score needed to keep a feature."""
        return float(self.options["threshold"])

    @property
    def protected_features(self) -> list[str]:
        """Return features that cannot be removed."""
        return list(self.options["protected_features"])

    @property
    def max_steps(self) -> int:
        """Return maximum number of elimination iterations."""
        return int(self.options["max_steps"])

    @property
    def verbose(self) -> bool:
        """Return whether verbose logging is enabled."""
        return bool(self.options["verbose"])

    @property
    def metric_direction(self) -> str:
        """Return optimization direction from cross-validator metadata."""
        return self.cross_validator.metric_direction

    def _validate_options(self) -> None:
        """Validate runtime option values."""
        if not isinstance(self.threshold, (float, int)):
            raise ValueError("threshold must be a numeric value.")
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer.")
        if not isinstance(self.verbose, bool):
            raise ValueError("verbose must be a boolean.")
        if not isinstance(self.options["protected_features"], list) or not all(
            isinstance(item, str) for item in self.options["protected_features"]
        ):
            raise ValueError("protected_features must be a list of strings.")

    def _init_summary(
        self, current_features: list[str], base_metric: float
    ) -> dict[str, Any]:
        """Initialize summary object with step-0 baseline metrics."""
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
        """Append one history row for each removed feature."""
        start_count = len(current_features)
        removed_features = step_state["features_to_remove"]
        importance_scores = step_state["importance_scores"]
        rows = []
        for offset, feature_name in enumerate(removed_features, start=1):
            rows.append(
                {
                    "step": step_idx,
                    "n_features_removed": (
                        len(self.estimator.feature_names_in_) - start_count + offset
                        if hasattr(self.estimator, "feature_names_in_")
                        else offset
                    ),
                    "n_features_remaining": start_count - offset,
                    "removed_feature_name": feature_name,
                    "metric_value": step_state["fold_metric"],
                    "metric_change": step_state["metric_change"],
                    "importance_score": importance_scores[feature_name],
                }
            )
        return pd.concat([history, pd.DataFrame(rows)], ignore_index=True)

    def _select_features_to_remove(
        self, importance_scores: dict[str, float]
    ) -> list[str]:
        """Select non-protected features below or equal to threshold."""
        removable_scores = {
            feature: score
            for feature, score in importance_scores.items()
            if feature not in self.protected_features
        }
        return [
            feature
            for feature, score in removable_scores.items()
            if score <= self.threshold
        ]

    def _log_start(
        self,
        current_features: list[str],
        avg_base_metric: float,
        base_std: float,
    ) -> None:
        """Print initial run summary when verbose mode is enabled."""
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
        """Print per-step details when verbose mode is enabled."""
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
        """Print final selection summary when verbose mode is enabled."""
        if not self.verbose:
            return

        print(f"\n🎯 Selected features: {summary['selected_features']}")
        final_metric = summary["history"].iloc[-1]["metric_value"]
        base_val = summary["history"].iloc[0]["metric_value"]
        diff = final_metric - base_val
        diff_pct = 100 * diff / base_val if base_val != 0 else np.nan
        print(
            f"📈 Final {self.cross_validator.metric_name} score: "
            f"{final_metric:.6f} | Base: {base_val:.6f} | "
            f"Δ: {diff:.6f} ({diff_pct:+.2f}%)"
        )

        initial_features = summary["history"].iloc[0]["n_features_remaining"]
        selected_features_count = len(summary["selected_features"])
        removed_count = initial_features - selected_features_count
        pct_removed = (
            100 * removed_count / initial_features if initial_features else 0.0
        )
        print(
            f"🗑️  Features removed: {removed_count} of {initial_features} "
            f"({pct_removed:.2f}%)"
        )

    def run(
        self,
        df: pd.DataFrame,
        target: pd.Series | np.ndarray,
    ) -> dict[str, list | pd.DataFrame]:
        """
        Run stepwise model importance filtering and return selection summary.

        Args:
            df (pd.DataFrame): Feature matrix.
            target (pd.Series | np.ndarray): Target array/series aligned with df.

        Returns:
            dict[str, list | pd.DataFrame]: Selected features and history table.

        Raises:
            ValueError: If protected features are missing from the dataset.
        """
        current_features = sorted(list(df.columns))
        missing_protected = set(self.protected_features) - set(current_features)
        if missing_protected:
            raise ValueError(
                f"Protected features not found in dataset: {missing_protected}"
            )

        avg_base_metric, base_std = self._cross_val_base_metric(
            df, target, current_features
        )
        summary = self._init_summary(current_features, avg_base_metric)
        self._log_start(current_features, avg_base_metric, base_std)

        prev_metric = avg_base_metric
        importance_scores: dict[str, float] = {
            feature: np.nan for feature in current_features
        }

        for step_idx in range(1, self.max_steps + 1):
            importance_scores, fold_metric, fold_std = self._cross_val_model_importance(
                df, target, current_features
            )
            step_state = {
                "importance_scores": importance_scores,
                "fold_metric": fold_metric,
                "fold_std": fold_std,
                "features_to_remove": self._select_features_to_remove(
                    importance_scores
                ),
                "metric_change": fold_metric - prev_metric,
            }
            self._log_step(step_idx, current_features, step_state)

            if not step_state["features_to_remove"]:
                if self.verbose:
                    print(
                        "✅ Convergence reached - all features exceed threshold "
                        f"{self.threshold}"
                    )
                break

            prev_metric = fold_metric
            summary["history"] = self._append_removal_rows(
                summary["history"],
                step_idx,
                current_features,
                step_state,
            )

            current_features = [
                feature
                for feature in current_features
                if feature not in set(step_state["features_to_remove"])
            ]
            if len(current_features) <= len(self.protected_features):
                if self.verbose:
                    print("⏹️  Stopping - only protected features remain")
                break

            gc.collect()

        summary["selected_features"] = sorted(current_features)
        summary["selected_features_names"] = summary["selected_features"]
        self.selected_features_ = summary["selected_features"]
        self.importance_scores_ = importance_scores

        self._log_finish(summary)
        gc.collect()
        return summary

    def fit(
        self,
        df: pd.DataFrame,
        target: pd.Series | np.ndarray,
    ) -> "ModelImportanceFiltering":
        """Fit selector and cache selected features."""
        self.run(df, target)
        return self

    def _cross_val_base_metric(
        self,
        df: pd.DataFrame,
        target: pd.Series | np.ndarray,
        current_features: list[str],
    ) -> tuple[float, float]:
        """Compute the base metric across validation folds."""
        _, valid_scores, _ = self.cross_validator.fit(
            self.estimator,
            df[current_features],
            target,
        )
        gc.collect()
        return float(np.mean(valid_scores)), float(np.std(valid_scores))

    def _cross_val_model_importance(
        self,
        df: pd.DataFrame,
        target: pd.Series | np.ndarray,
        current_features: list[str],
    ) -> tuple[dict[str, float], float, float]:
        """Compute average feature importance from fitted CV estimators."""
        _, valid_scores, _ = self.cross_validator.fit(
            self.estimator,
            df[current_features],
            target,
        )

        importances = []
        for fitted_estimator in self.cross_validator.fitted_estimators_:
            if not hasattr(fitted_estimator, "feature_importances_"):
                raise AttributeError(
                    f"Estimator {type(fitted_estimator).__name__} does not have "
                    "'feature_importances_' attribute."
                )
            importances.append(fitted_estimator.feature_importances_)

        avg_importances = np.mean(importances, axis=0)
        avg_importances = np.nan_to_num(avg_importances, nan=0.0)
        importance_map = {
            feature: float(score)
            for feature, score in zip(current_features, avg_importances)
        }

        gc.collect()
        return importance_map, float(np.mean(valid_scores)), float(np.std(valid_scores))


if __name__ == "__main__":
    demo_x, demo_y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=17,
    )
    demo_df = pd.DataFrame(
        demo_x, columns=[f"feature_{idx}" for idx in range(demo_x.shape[1])]
    )
    demo_target = pd.Series(demo_y, name="target")

    clf = RandomForestClassifier(random_state=17, max_depth=5, n_estimators=50)
    cv_trainer = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=KFold(n_splits=3, shuffle=True, random_state=17),
        processors=None,
        verbose=False,
    )

    print("Example: Stepwise Model Importance Filtering")
    mif = ModelImportanceFiltering(
        estimator=clf,
        cross_validator=cv_trainer,
        threshold=0.03,
        protected_features=["feature_0"],
        verbose=True,
    )
    demo_summary = mif.run(demo_df, demo_target)
    print("\nHistory of feature removal:")
    print(
        demo_summary["history"][
            [
                "step",
                "removed_feature_name",
                "n_features_remaining",
                "metric_value",
                "importance_score",
            ]
        ]
    )
    print(f"\nFinal selected features: {sorted(demo_summary['selected_features'])}")
    print(f"Number of selected features: {len(demo_summary['selected_features'])}")
