import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import KFold, BaseCrossValidator

from kvbiii_ml.evaluation.metrics import (
    METRICS,
    get_metric_direction,
    get_metric_function,
    get_metric_type,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


class MutualInformationRecursiveFeatureElimination:
    """Feature selection using Mutual Information (MI).

    This selector ranks features by MI with the target and removes the least
    informative features over multiple steps. After each removal step, the
    estimator is evaluated via cross-validation to track metric changes.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        problem_type: str,
        metric_name: str,
        steps: int = 5,
        cv: BaseCrossValidator = KFold(n_splits=5, shuffle=True, random_state=17),
        alpha: float = 0.95,
        verbose: bool = True,
    ) -> None:
        """Initialize the MI selector.

        Args:
                estimator (BaseEstimator): Estimator to fit within each fold for metric evaluation.
                problem_type (str): "classification" or "regression".
                metric_name (str): Metric name from METRICS to optimize in CV.
                steps (int): Number of elimination iterations. Defaults to 5.
                cv (BaseCrossValidator): Cross-validation splitter. Defaults to 5-fold shuffled KFold.
                alpha (float): Weight for final selection score mix. Defaults to 0.95.
                verbose (bool): Whether to print progress messages. Defaults to True.
        """
        if problem_type not in {"classification", "regression"}:
            raise ValueError("problem_type must be 'classification' or 'regression'.")

        if metric_name not in METRICS:
            raise ValueError(
                f"Unsupported metric: {metric_name}. Supported metrics are: {', '.join(METRICS.keys())}"
            )

        self.estimator = estimator
        self.problem_type = problem_type
        self.metric_name = metric_name
        self.steps = steps
        self.cv = cv
        self.alpha = alpha
        self.verbose = verbose
        self.mi_kwargs = {
            "n_neighbors": 3,
            "random_state": 17,
        }

        self.eval_metric = get_metric_function(metric_name)
        self.metric_type = get_metric_type(metric_name)
        self.direction = get_metric_direction(metric_name)

        self.history_schema = {
            "step": int,
            "n_features_removed": int,
            "n_features_remaining": int,
            "removed_feature_name": object,
            "metric_value": float,
            "metric_change": float,
            "mi_score": float,
        }

    def run(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> dict[str, list | pd.DataFrame]:
        """Run MI-based selection and return the selection summary.

        Args:
                X (pd.DataFrame): Feature matrix.
                y (pd.Series | np.ndarray): Target aligned with X.

        Returns:
                dict[str, list | pd.DataFrame]: Dictionary with keys:
                        - selected_features (list)
                        - selected_features_names (list)
                        - history (pd.DataFrame)
        """
        current_features = list(X.columns)
        summary_df = {
            "selected_features": [],
            "selected_features_names": [],
            "history": pd.DataFrame(columns=self.history_schema.keys()).astype(
                self.history_schema
            ),
        }

        # Step 0: base metric
        base_metric, _ = self._cross_val_base_metric(X, y, current_features)
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
                            "metric_value": base_metric,
                            "metric_change": 0.0,
                            "mi_score": np.nan,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        removal_schedule = self.compute_removal_schedule(len(current_features))
        if self.verbose:
            print(
                f"🧮 Starting MI selection with {len(current_features)} features, metric: {self.metric_name} ({self.direction}), steps: {self.steps}.\n📅 Remove per step: {removal_schedule}"
            )
            print(
                f"\n📊 Initial {self.metric_name}: {base_metric:.6f} | Features: {len(current_features)}"
			)

        # Compute MI scores once; recompute each step if desired by toggling below
        mi_scores = self._compute_mi_scores(X[current_features], y)
        if self.verbose:
            print("Top MI features:")
            for feat, score in sorted(
                mi_scores.items(), key=lambda kv: kv[1], reverse=True
            )[:5]:
                print(f"  • {feat}: {score:.4f}")

        prev_metric = base_metric
        for step_idx, n_remove in enumerate(removal_schedule, start=1):
            # Sort by ascending MI (least useful first)
            sorted_feats = sorted(current_features, key=lambda f: mi_scores.get(f, 0.0))
            to_remove = sorted_feats[:n_remove]
            remaining = [f for f in current_features if f not in to_remove]
            if self.verbose:
                print(f"\n🔁 Step {step_idx} | Features remaining: {len(remaining)}")
            step_metric, step_metric_std = self._cross_val_base_metric(X, y, remaining)
            if self.verbose:
                print(
                    f"📊 Average {self.metric_name}: {step_metric:.6f} ± {step_metric_std:.6f}"
                )

            # Log each removal in this step
            for feat in to_remove:
                summary_df["history"] = pd.concat(
                    [
                        summary_df["history"],
                        pd.DataFrame(
                            [
                                {
                                    "step": step_idx,
                                    "n_features_removed": len(X.columns)
                                    - len(remaining),
                                    "n_features_remaining": len(remaining),
                                    "removed_feature_name": feat,
                                    "metric_value": step_metric,
                                    "metric_change": step_metric - prev_metric,
                                    "mi_score": float(mi_scores.get(feat, 0.0)),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

            current_features = remaining
            prev_metric = step_metric

        # Final selection via weighted score (same approach as SHAP-RFE)
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
                f"📈 Final {self.metric_name}: {metric_selected:.6f} | Base: {base_val:.6f} | Δ: {metric_selected - base_val:.6f} ({diff_pct:+.2f}%)"
            )
            n_initial = summary_df["history"].iloc[0]["n_features_remaining"]
            n_selected = len(summary_df["selected_features"])
            n_removed = n_initial - n_selected
            pct_removed = 100 * n_removed / n_initial if n_initial else 0.0
            print(
                f"🗑️ Features removed: {n_removed} of {n_initial} ({pct_removed:.2f}%)"
            )

        return summary_df

    def _compute_mi_scores(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> dict[str, float]:
        """Compute mutual information scores for current features."""
        X, y = self._prepare_X_y_for_mi(X, y)
        if self.problem_type == "classification":
            mi = mutual_info_classif(X, y, **self.mi_kwargs)
        else:
            mi = mutual_info_regression(X, y, **self.mi_kwargs)
        mi = np.nan_to_num(mi, nan=0.0)
        return {feat: float(score) for feat, score in zip(X.columns, mi)}

    def _prepare_X_y_for_mi(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> tuple[pd.DataFrame, pd.Series | np.ndarray]:
        """Validate types and convert categorical features before MI.

        - Ensures X is a DataFrame and y is a Series or ndarray.
        - If X contains categorical columns, converts them to numeric and, for classification, sets a discrete_features mask in mi_kwargs when missing.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series | np.ndarray): Target.

        Returns:
            tuple[pd.DataFrame, pd.Series | np.ndarray]: Possibly transformed (X, y).
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("y must be a pandas Series or numpy array.")

        cat_cols = X.select_dtypes(include=["category"]).columns

        if len(cat_cols) > 0:
            X = X.copy()
            X[cat_cols] = X[cat_cols].apply(lambda col: col.cat.codes)
            self.mi_kwargs["discrete_features"] = X.columns.isin(cat_cols).astype(bool)
        return X, y

    def _cross_val_base_metric(
        self, X: pd.DataFrame, y: pd.Series, current_features: list[str]
    ) -> tuple[float, float]:
        """Compute the base metric across validation folds."""
        cv_trainer = CrossValidationTrainer(
            metric_name=self.metric_name, cv=self.cv, processors=None, verbose=False
        )
        _, valid_scores, _ = cv_trainer.fit(self.estimator, X[current_features], y)
        return float(np.mean(valid_scores)), float(np.std(valid_scores))

    def compute_removal_schedule(self, total_features: int) -> list[int]:
        """Compute a linear-decay schedule for features to remove per step."""
        if total_features <= 0 or self.steps <= 0:
            return []
        decay = np.linspace(1, 0.2, self.steps)
        weights = decay / decay.sum()
        removal_counts = np.round(weights * total_features).astype(int)
        diff = total_features - removal_counts.sum()
        for i in range(abs(diff)):
            idx = i % self.steps
            removal_counts[idx] += 1 if diff > 0 else -1
        removal_counts = removal_counts[removal_counts > 0]
        removal_counts[-1] = removal_counts[-1] - 1
        removal_counts = removal_counts[removal_counts > 0]
        return removal_counts.tolist()

    def select_features_weighted_score(
        self, history: pd.DataFrame, alpha: float | None = None
    ) -> tuple[list[str], float | None]:
        """Select features by maximizing a weighted metric/features score."""
        if history.empty:
            return [], None
        alpha = self.alpha if alpha is None else alpha
        df = history.copy()
        if self.direction == "maximize":
            df["metric_norm"] = (df["metric_value"] - df["metric_value"].min()) / (
                df["metric_value"].max() - df["metric_value"].min()
            )
        else:
            df["metric_norm"] = (df["metric_value"].max() - df["metric_value"]) / (
                df["metric_value"].max() - df["metric_value"].min()
            )
        df["features_norm"] = 1 - (
            df["n_features_remaining"] - df["n_features_remaining"].min()
        ) / (df["n_features_remaining"].max() - df["n_features_remaining"].min())
        df["score"] = alpha * df["metric_norm"] + (1 - alpha) * df["features_norm"]
        best_row = df.loc[df["score"].idxmax()]
        removed = set(
            history[history["step"] < best_row["step"]]["removed_feature_name"].dropna()
        )
        all_features = set(history["removed_feature_name"].dropna().unique())
        selected = list(all_features - removed)
        return selected, best_row["metric_value"]


if __name__ == "__main__":
    # Minimal runnable example
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier

    X_df, y_ser = load_breast_cancer(return_X_y=True, as_frame=True)
    clf = RandomForestClassifier(random_state=17, max_depth=5)

    selector = MutualInformationRecursiveFeatureElimination(
        estimator=clf,
        problem_type="classification",
        metric_name="Accuracy",
        steps=10,
        cv=KFold(n_splits=5, shuffle=True, random_state=17),
        alpha=0.9,
        verbose=True,
    )

    summary = selector.run(X_df.head(200), y_ser.head(200))
    print("Selected features:", summary["selected_features"])
