import gc

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


class ModelImportanceFiltering:
    """Feature selection using stepwise model importance filtering.

    This filter iteratively trains the model, computes feature importance across
    CV folds, and removes features below threshold until convergence.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cross_validator: CrossValidationTrainer,
        threshold: float = 0.0,
        protected_features: list[str] | None = None,
        max_steps: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the model importance filter.

        Args:
            estimator (BaseEstimator): Estimator with feature_importances_ attribute.
            cross_validator (CrossValidationTrainer): Cross-validation trainer instance.
            threshold (float, optional): Minimum importance value to be exceeded to
                keep a feature. Defaults to 0.0.
            protected_features (list[str] | None, optional): Features that must be
                included regardless of their importance. Defaults to None.
            max_steps (int, optional): Maximum number of filtering iterations. Process will
                stop after this many steps even if features below threshold remain.
                Defaults to 10.
            verbose (bool, optional): Whether to print selection details. Defaults to False.

        Raises:
            AttributeError: If estimator doesn't have feature_importances_ attribute.
        """
        self.estimator = estimator
        self.cross_validator = cross_validator
        self.threshold = threshold
        self.protected_features = protected_features or []
        self.max_steps = max_steps
        self.verbose = verbose
        self.metric_direction = self.cross_validator.metric_direction
        self.selected_features_ = []
        self.importance_scores_ = {}
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
        """
        Run stepwise model importance filtering and return selection summary.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series | np.ndarray): Target array/series aligned with X.

        Returns:
            dict[str, list | pd.DataFrame]: Dictionary with keys:
                - selected_features (list): Final selected features.
                - selected_features_names (list): Alias of selected_features.
                - history (pd.DataFrame): Step-wise metrics and removals.

        Raises:
            ValueError: If protected features are missing from the dataset.
        """
        current_features = sorted(list(X.columns))
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

        avg_base_metric, base_std = self._cross_val_base_metric(X, y, current_features)
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

        if self.verbose:
            print(
                f"🔍 Starting Model Importance Stepwise Filtering with "
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

        step_idx = 0
        prev_metric = avg_base_metric

        while True:
            step_idx += 1

            importance_scores, fold_metric, fold_std = self._cross_val_model_importance(
                X, y, current_features
            )

            if self.verbose:
                print(
                    f"\n🔁 Step {step_idx} | Features remaining: "
                    f"{len(current_features)}"
                )
                print(
                    f"📊 Average {self.cross_validator.metric_name}: "
                    f"{fold_metric:.6f} ± {fold_std:.6f}"
                )

            removable_scores = {
                k: v
                for k, v in importance_scores.items()
                if k not in self.protected_features
            }

            features_to_remove = [
                feat
                for feat, score in removable_scores.items()
                if score <= self.threshold
            ]

            if self.verbose:
                sorted_by_importance = sorted(
                    importance_scores.items(), key=lambda x: x[1], reverse=True
                )
                print(f"\n🔬 Features remaining: {current_features}\n")
                print("Most important features:")
                for feat, score in sorted_by_importance[:5]:
                    print(f"  • {feat}: {score:.6f}")
                print(f"\nLeast important features (below threshold {self.threshold}):")
                for feat in features_to_remove[:5]:
                    print(f"  • {feat}: {importance_scores[feat]:.6f}")
                print(
                    f"\n🗑️  Removing {len(features_to_remove)} ({len(features_to_remove) / len(current_features) * 100:.2f}%) features this step"
                )
            if not features_to_remove:
                if self.verbose:
                    print(
                        f"✅ Convergence reached - all features exceed threshold "
                        f"{self.threshold}"
                    )
                break

            metric_change = fold_metric - prev_metric
            prev_metric = fold_metric

            for feature in features_to_remove:
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
                                    "metric_value": fold_metric,
                                    "metric_change": metric_change,
                                    "importance_score": importance_scores[feature],
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
                current_features.remove(feature)

            if len(current_features) <= len(self.protected_features):
                if self.verbose:
                    print("⏹️  Stopping - only protected features remain")
                break

            if step_idx >= self.max_steps:
                if self.verbose:
                    print(
                        f"⏹️  Stopping - maximum number of steps ({self.max_steps}) reached"
                    )
                break

            gc.collect()

        summary_df["selected_features"] = sorted(current_features)
        summary_df["selected_features_names"] = summary_df["selected_features"]
        self.selected_features_ = summary_df["selected_features"]
        self.importance_scores_ = importance_scores

        if self.verbose:
            print(f"\n🎯 Selected features: {summary_df['selected_features']}")
            final_metric = summary_df["history"].iloc[-1]["metric_value"]
            base_val = summary_df["history"].iloc[0]["metric_value"]
            diff = final_metric - base_val
            diff_pct = 100 * diff / base_val if base_val != 0 else np.nan
            print(
                f"📈 Final {self.cross_validator.metric_name} score: "
                f"{final_metric:.6f} | Base: {base_val:.6f} | "
                f"Δ: {diff:.6f} ({diff_pct:+.2f}%)"
            )
            n_features_initial = summary_df["history"].iloc[0]["n_features_remaining"]
            n_features_selected = len(summary_df["selected_features"])
            n_removed = n_features_initial - n_features_selected
            pct_removed = (
                100 * n_removed / n_features_initial if n_features_initial else 0.0
            )
            print(
                f"🗑️  Features removed: {n_removed} of {n_features_initial} "
                f"({pct_removed:.2f}%)"
            )

        gc.collect()
        return summary_df

    def _cross_val_base_metric(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray, current_features: list[str]
    ) -> tuple[float, float]:
        """
        Compute the base metric across validation folds.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series | np.ndarray): Target values.
            current_features (list[str]): Features to evaluate.

        Returns:
            tuple[float, float]: Average metric and its standard deviation.
        """
        _, valid_scores, _ = self.cross_validator.fit(
            self.estimator, X[current_features], y
        )
        gc.collect()
        return float(np.mean(valid_scores)), float(np.std(valid_scores))

    def _cross_val_model_importance(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray, current_features: list[str]
    ) -> tuple[dict[str, float], float, float]:
        """
        Compute feature importance from the model's feature_importances_ attribute.

        Args:
            X (pd.DataFrame): The input feature DataFrame.
            y (pd.Series | np.ndarray): The target series or array.
            current_features (list[str]): The list of features to use.

        Returns:
            tuple[dict[str, float], float, float]: A tuple containing:
                - A dictionary of average importance scores for each feature.
                - The mean baseline validation score across folds.
                - The standard deviation of the baseline validation score.

        Raises:
            AttributeError: If the estimator does not have feature_importances_.
        """
        _, valid_scores, _ = self.cross_validator.fit(
            self.estimator, X[current_features], y
        )

        importances = []
        for estimator in self.cross_validator.fitted_estimators_:
            if not hasattr(estimator, "feature_importances_"):
                raise AttributeError(
                    f"Estimator {type(estimator).__name__} does not have "
                    "'feature_importances_' attribute."
                )
            importances.append(estimator.feature_importances_)

        avg_importances = np.mean(importances, axis=0)
        avg_importances = np.nan_to_num(avg_importances, nan=0.0)

        importance_map = {
            feat: float(score) for feat, score in zip(current_features, avg_importances)
        }

        gc.collect()
        return (
            importance_map,
            float(np.mean(valid_scores)),
            float(np.std(valid_scores)),
        )


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=17,
    )
    X_demo = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_demo = pd.Series(y, name="target")

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
    summary = mif.run(X_demo, y_demo)
    print("\nHistory of feature removal:")
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
    print(f"\nFinal selected features: {sorted(summary['selected_features'])}")
    print(f"Number of selected features: {len(summary['selected_features'])}")
