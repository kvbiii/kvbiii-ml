from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class MutualInformationFiltering:
    """Feature selection using mutual information filtering.

    This filter computes Mutual Information (MI) between each feature and the
    target, and selects features whose MI is at least a given threshold.
    """

    def __init__(
        self,
        problem_type: str,
        threshold: float = 0.1,
        keep_top_k: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the mutual information filter.

        Args:
            problem_type (str): Type of problem, either "classification" or "regression".
            threshold (float, optional): Minimum MI value required to keep a feature. Defaults to 0.1.
            keep_top_k (int | None, optional): If provided, select the top-k features by MI and ignore the threshold.
                Prints the implied nominal MI threshold (border) when verbose is True. Defaults to None.
            verbose (bool, optional): Whether to print selection details. Defaults to False.
        """
        if problem_type not in {"classification", "regression"}:
            raise ValueError(
                "problem_type must be either 'classification' or 'regression'."
            )
        self.problem_type = problem_type
        self.mi_kwargs = {
            "n_neighbors": 3,
            "random_state": 17,
        }
        self.threshold = threshold
        self.keep_top_k = keep_top_k
        self.verbose = verbose
        self.selected_features_ = []
        self.threshold_border_ = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> MutualInformationFiltering:
        """Fit the filter by computing MI and selecting features.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series | np.ndarray): Target variable.

        Returns:
            MutualInformationFiltering: Fitted instance with selected features.
        """
        mi_scores = self._compute_mi_scores(X, y)
        # Sort features by MI descending once
        sorted_items = sorted(mi_scores.items(), key=lambda kv: kv[1], reverse=True)

        if self.keep_top_k is not None:
            if self.keep_top_k <= 0:
                raise ValueError("keep_top_k must be a positive integer when provided.")
            k = min(self.keep_top_k, len(sorted_items))
            self.selected_features_ = [feat for feat, _ in sorted_items[:k]]
            self.threshold_border_ = float(sorted_items[k - 1][1]) if k > 0 else None
            if self.verbose:
                print(
                    f"Using top-k selection (k={self.keep_top_k}). Implied MI threshold (border): {self.threshold_border_:.6f}"
                )
        else:
            self.selected_features_ = [
                feat for feat, score in sorted_items if score >= self.threshold
            ]
            self.threshold_border_ = self.threshold
            if self.verbose:
                print(
                    f"Using nominal MI threshold: {self.threshold_border_:.6f}. Selected {len(self.selected_features_)} features."
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features whose MI met the threshold during fit.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only selected features.
        """
        if not self.selected_features_:
            raise ValueError("The filter has not been fitted yet. Call fit() first.")
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
        """Fit the filter and return the transformed features.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series | np.ndarray): Target variable.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only selected features.
        """
        self.fit(X, y)
        return self.transform(X)

    def _compute_mi_scores(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> dict[str, float]:
        """Compute Mutual Information scores for the given features.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series | np.ndarray): Target variable.

        Returns:
            dict[str, float]: Mapping of feature name to MI score.
        """
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
        """Validate inputs and encode categorical features for MI computation.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series | np.ndarray): Target variable.

        Returns:
            tuple[pd.DataFrame, pd.Series | np.ndarray]: Tuple of (X, y) with
            categorical columns encoded and MI kwargs adjusted when needed.
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


if __name__ == "__main__":
    # Minimal usage example
    rng = np.random.default_rng(17)
    X_demo = pd.DataFrame(
        {
            "num1": rng.normal(size=100),
            "num2": rng.normal(size=100),
            "cat": pd.Series(rng.choice(["A", "B", "C"], size=100), dtype="category"),
        }
    )
    y_demo = pd.Series(
        (X_demo["num1"] + rng.normal(scale=0.1, size=100) > 0).astype(int)
    )

    # Example 1: nominal threshold
    mif = MutualInformationFiltering(
        problem_type="classification", threshold=0.01, verbose=True
    )
    X_sel = mif.fit_transform(X_demo, y_demo)
    print("Selected features:", mif.selected_features_)
    print("Transformed shape:", X_sel.shape)

    # Example 2: keep top-k features (prints implied nominal threshold)
    mif_topk = MutualInformationFiltering(
        problem_type="classification", keep_top_k=2, verbose=True
    )
    X_sel_topk = mif_topk.fit_transform(X_demo, y_demo)
    print("Selected (top-k) features:", mif_topk.selected_features_)
    print("Border MI threshold:", mif_topk.threshold_border_)
