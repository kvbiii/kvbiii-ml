import gc
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
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
        cross_validator: CrossValidationTrainer,
        steps: int = 5,
        alpha: float = 0.95,
        verbose: bool = True,
        protected_features: list[str] | None = None,
    ) -> None:
        """Initialize the MI selector.

        Args:
            estimator (BaseEstimator): Estimator to fit within each fold for metric evaluation.
            cross_validator (CrossValidationTrainer): Cross-validation trainer instance.
            steps (int): Number of elimination iterations. Defaults to 5.
            alpha (float): Weight for final selection score mix. Defaults to 0.95.
            verbose (bool): Whether to print progress messages. Defaults to True.
            protected_features (list[str], optional): Features that should never be removed.
        """
        self.estimator = estimator
        self.cross_validator = cross_validator
        self.steps = steps
        self.alpha = alpha
        self.verbose = verbose
        self.protected_features = protected_features or []
        self.mi_kwargs = {
            "n_neighbors": 3,
            "random_state": 17,
        }

        self.metric_fn = self.cross_validator.metric_fn
        self.metric_type = self.cross_validator.metric_type
        self.metric_direction = self.cross_validator.metric_direction
        self.problem_type = self.cross_validator.problem_type

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
        X = self._convert_categorical_to_codes(X)
        current_features = list(X.columns)
        self._validate_protected_features(current_features)

        history_records = []
        removable_features = [
            f for f in current_features if f not in self.protected_features
        ]
        removal_schedule = self.compute_removal_schedule(len(removable_features))

        mi_scores = self._compute_mi_scores(X[current_features], y)
        self._print_initialization(current_features, removal_schedule, mi_scores)

        prev_metric = None
        for step_idx, n_remove in enumerate(removal_schedule, start=1):
            step_metric, step_metric_std = self._cross_val_base_metric(
                X, y, current_features
            )

            if self.verbose:
                print(
                    f"\n🔁 Step {step_idx} | Number of features remaining: {len(current_features)} | {self.cross_validator.metric_name}: {step_metric:.6f} ± {step_metric_std:.6f}\n{sorted(current_features)}"
                )

            removable_feats = [
                f for f in current_features if f not in self.protected_features
            ]
            to_remove = sorted(removable_feats, key=lambda f: mi_scores.get(f, 0.0))[
                : min(n_remove, len(removable_feats))
            ]

            for idx, feat in enumerate(to_remove):
                history_records.append(
                    {
                        "step": step_idx,
                        "n_features_removed": len(X.columns)
                        - len(current_features)
                        + idx
                        + 1,
                        "n_features_remaining": len(current_features) - idx - 1,
                        "removed_feature_name": feat,
                        "metric_value": step_metric,
                        "metric_change": (
                            0.0 if prev_metric is None else step_metric - prev_metric
                        ),
                        "mi_score": float(mi_scores.get(feat, 0.0)),
                    }
                )

            current_features = [f for f in current_features if f not in to_remove]
            prev_metric = step_metric

            if len(current_features) <= len(self.protected_features):
                if self.verbose:
                    print("⏹️  Stopping early - only protected features remain")
                break

        history_df = pd.DataFrame(history_records).astype(self.history_schema)
        selected_features, metric_selected = self.select_features_weighted_score(
            history_df, self.alpha
        )

        self._print_summary(history_df, selected_features, metric_selected)
        gc.collect()

        return {
            "selected_features": selected_features,
            "selected_features_names": selected_features,
            "history": history_df,
        }

    def _validate_protected_features(self, current_features: list[str]) -> None:
        """Validate that protected features exist in the dataset."""
        missing = set(self.protected_features) - set(current_features)
        if missing:
            raise ValueError(f"Protected features not found in dataset: {missing}")

    def _print_initialization(
        self,
        current_features: list[str],
        removal_schedule: list[int],
        mi_scores: dict[str, float],
    ) -> None:
        """Print initialization information."""
        if not self.verbose:
            return
        print(
            f"🔍 Starting MI selection with {len(current_features)} features, metric: {self.cross_validator.metric_name} ({self.metric_direction}), steps: {self.steps}."
        )
        print(f"📅 Remove per step: {removal_schedule}")
        print(f"🛡️  Protected features: {self.protected_features}")
        print("Top MI features:")
        for feat, score in sorted(
            mi_scores.items(), key=lambda kv: kv[1], reverse=True
        )[:5]:
            print(f"  • {feat}: {score:.4f}")

    def _print_summary(
        self,
        history: pd.DataFrame,
        selected_features: list[str],
        metric_selected: float,
    ) -> None:
        """Print final summary."""
        if not self.verbose or history.empty:
            return
        print(f"\n🎯 Selected features: {selected_features}")
        base_val = history.iloc[0]["metric_value"]
        diff = metric_selected - base_val
        diff_pct = 100 * diff / base_val if base_val != 0 else np.nan
        print(
            f"📈 Final {self.cross_validator.metric_name}: {metric_selected:.6f} | Base: {base_val:.6f} | Δ: {diff:.6f} ({diff_pct:+.2f}%)"
        )
        n_initial = history.iloc[0]["n_features_remaining"]
        n_removed = n_initial - len(selected_features)
        pct_removed = 100 * n_removed / n_initial if n_initial else 0.0
        print(f"🗑️ Features removed: {n_removed} of {n_initial} ({pct_removed:.2f}%)")

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
        _, valid_scores, _ = self.cross_validator.fit(
            self.estimator, X[current_features], y
        )
        print(valid_scores)
        gc.collect()
        return float(np.mean(valid_scores)), float(np.std(valid_scores))

    def compute_removal_schedule(self, total_removable_features: int) -> list[int]:
        """Compute a linear-decay schedule for features to remove per step.

        Returns a list of positive integers. If there are no
        removable features or no steps, returns an empty list.
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
        all_features = set(history["removed_feature_name"].dropna().unique())
        all_features.update(self.protected_features)
        gc.collect()
        return list(sorted(selected)), best_row["metric_value"]


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold

    X_df, y_ser = load_breast_cancer(return_X_y=True, as_frame=True)
    clf = RandomForestClassifier(random_state=17, max_depth=5, n_estimators=100)
    cross_validator = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=KFold(n_splits=5, shuffle=True, random_state=17),
        processors=None,
        verbose=False,
    )

    selector = MutualInformationRecursiveFeatureElimination(
        estimator=clf,
        cross_validator=cross_validator,
        steps=10,
        alpha=0.85,
        verbose=True,
        protected_features=["mean radius", "mean texture"],
    )

    summary = selector.run(X_df, y_ser)
    print("\nSummary of MI-RFE:")
    print(summary["history"])
    print("Selected features:", summary["selected_features"])
    print(f"Number of selected features: {len(summary['selected_features'])}")
