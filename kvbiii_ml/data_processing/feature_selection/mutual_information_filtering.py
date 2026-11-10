from __future__ import annotations

import numpy as np
import pandas as pd

from kvbiii_ml._validation import check_fitted
from kvbiii_ml.data_processing.eda.data_analysis import DataAnalyzer


class MutualInformationFiltering:
    """Feature selection using mutual information filtering.

    This filter computes Mutual Information (MI) between each feature and the
    target, and selects features whose MI is at least a given threshold.
    """

    selected_features_: list[str]
    threshold_border_: float | None

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
            threshold (float): Minimum MI value required to keep a feature. Defaults to 0.1.
            keep_top_k (int | None): Select top-k by MI, ignoring the threshold. Defaults to None.
            verbose (bool, optional): Whether to print selection details. Defaults to False.

        Raises:
            ValueError: If problem_type is neither "classification" nor "regression".
        """
        if problem_type not in {"classification", "regression"}:
            raise ValueError("problem_type must be either 'classification' or 'regression'.")
        self.problem_type = problem_type
        self.mi_kwargs = {
            "n_neighbors": 3,
            "random_state": 17,
        }
        self.threshold = threshold
        self.keep_top_k = keep_top_k
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> MutualInformationFiltering:
        """Fit the filter by computing MI and selecting features.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series | np.ndarray): Target variable.

        Returns:
            MutualInformationFiltering: Fitted instance with selected features.

        Raises:
            ValueError: If keep_top_k is not a positive integer.
        """
        mi_scores = self._compute_mi_scores(X, y)
        sorted_items = list(mi_scores.items())

        if self.keep_top_k is not None:
            if self.keep_top_k <= 0:
                raise ValueError("keep_top_k must be a positive integer when provided.")
            k = min(self.keep_top_k, len(sorted_items))
            self.selected_features_ = [feat for feat, _ in sorted_items[:k]]
            self.threshold_border_ = float(sorted_items[k - 1][1]) if k > 0 else None
            if self.verbose:
                print(
                    f"Using top-k selection (k={self.keep_top_k}). "
                    f"Implied MI threshold (border): {self.threshold_border_:.6f}"
                )
        else:
            self.selected_features_ = [
                feat for feat, score in sorted_items if score >= self.threshold
            ]
            self.threshold_border_ = self.threshold
            if self.verbose:
                print(
                    f"Using nominal MI threshold: {self.threshold_border_:.6f}. "
                    f"Selected {len(self.selected_features_)} features."
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select the features kept during fit, raising NotFittedError before fit.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only selected features.
        """
        check_fitted(self, "selected_features_")
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

    def _compute_mi_scores(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> dict[str, float]:
        """Compute Mutual Information scores for the given features.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series | np.ndarray): Target variable.

        Returns:
            dict[str, float]: Mapping of feature name to MI score.
        """
        return DataAnalyzer.mutual_information_scores(
            X=X,
            y=y,
            problem_type=self.problem_type,
            n_neighbors=int(self.mi_kwargs["n_neighbors"]),
            random_state=self.mi_kwargs.get("random_state"),
        )


if __name__ == "__main__":

    def _run_demo() -> None:
        """Run MutualInformationFiltering on a small synthetic dataset."""
        # Minimal usage example
        rng = np.random.default_rng(17)
        x_demo = pd.DataFrame(
            {
                "num1": rng.normal(size=100),
                "num2": rng.normal(size=100),
                "cat": pd.Series(rng.choice(["A", "B", "C"], size=100), dtype="category"),
            }
        )
        y_demo = pd.Series((x_demo["num1"] + rng.normal(scale=0.1, size=100) > 0).astype(int))

        # Example 1: nominal threshold
        mif = MutualInformationFiltering(
            problem_type="classification", threshold=0.01, verbose=True
        )
        x_sel = mif.fit_transform(x_demo, y_demo)
        print("Selected features:", mif.selected_features_)
        print("Transformed shape:", x_sel.shape)

        # Example 2: keep top-k features (prints implied nominal threshold)
        mif_topk = MutualInformationFiltering(
            problem_type="classification", keep_top_k=2, verbose=True
        )
        x_sel_topk = mif_topk.fit_transform(x_demo, y_demo)
        print("Selected (top-k) features:", mif_topk.selected_features_)
        print("Border MI threshold:", mif_topk.threshold_border_)
        print("Transformed shape:", x_sel_topk.shape)

    _run_demo()
