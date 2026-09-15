from typing import Literal

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from kvbiii_ml.data_processing.preprocessing.expansion_base import (
    _WithOriginalSubsetBase,
)


class KBinsDiscretiserWithOriginal(_WithOriginalSubsetBase):
    """Wraps scikit-learn KBinsDiscretizer to keep originals and append binned copies.

    Derived columns are named ``{original}_PREPROCESS_KBINS_{n_bins}`` - the bin
    count is baked into the suffix, since it is the defining parameter of this
    transformer, rather than using one fixed suffix like the other discretisers
    in this package. Each numeric variable is split into ``n_bins`` ordinal bins
    using the requested ``strategy`` (``"uniform"``, ``"quantile"``, or
    ``"kmeans"`` - the latter unavailable in the feature_engine-based
    discretisers already in this package). Original columns are preserved.
    """

    _suppress_warnings = True

    def __init__(
        self,
        n_bins: int = 5,
        strategy: Literal["uniform", "quantile", "kmeans"] = "quantile",
        quantile_method: str = "averaged_inverted_cdf",
        dtype: type | None = None,
        subsample: int | None = 200_000,
        random_state: int | None = None,
        variables: list[str] | None = None,
    ) -> None:
        """
        Initialize KBinsDiscretiserWithOriginal.

        Args:
            n_bins (int, optional): Number of bins to produce for every selected
                variable. Also drives the derived column suffix. Defaults to 5.
            strategy (Literal["uniform", "quantile", "kmeans"], optional): Binning
                strategy - equal-width, equal-frequency, or 1D k-means cluster
                edges. Defaults to "quantile".
            quantile_method (str, optional): Quantile interpolation method used
                when ``strategy="quantile"``. Defaults to "averaged_inverted_cdf"
                (scikit-learn's forthcoming default; avoids its deprecation
                FutureWarning).
            dtype (type | None, optional): Output dtype forwarded to
                KBinsDiscretizer. Defaults to None (inferred from input).
            subsample (int | None, optional): Maximum number of samples used to
                fit the "quantile"/"kmeans" strategies. Defaults to 200_000.
            random_state (int | None, optional): Random seed for the "kmeans"
                strategy and for subsampling. Defaults to None.
            variables (list[str] | None, optional): Numeric columns to discretise.
                Defaults to None (auto-detect all numeric columns).
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.quantile_method = quantile_method
        self.dtype = dtype
        self.subsample = subsample
        self.random_state = random_state
        self.variables = variables

    @property
    def _suffix(self) -> str:
        """Derived-column suffix, dynamically keyed on ``n_bins``.

        Returns:
            str: e.g. "PREPROCESS_KBINS_5" for ``n_bins=5``.
        """
        return f"PREPROCESS_KBINS_{self.n_bins}"

    def _build_inner(self) -> KBinsDiscretizer:
        """Return a fresh KBinsDiscretizer configured from instance attributes.

        Returns:
            KBinsDiscretizer: Unfitted discretiser, fixed to ordinal encoding so
                each variable maps to exactly one derived column.
        """
        return KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy=self.strategy,
            quantile_method=self.quantile_method,
            dtype=self.dtype,
            subsample=self.subsample,
            random_state=self.random_state,
        )

    def _default_variables(self, X: pd.DataFrame) -> list[str]:
        """Auto-detect numeric columns to discretise when ``variables`` is None.

        Args:
            X (pd.DataFrame): Training features.

        Returns:
            list[str]: Names of numeric columns in X.
        """
        return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]


__all__ = ["KBinsDiscretizer", "KBinsDiscretiserWithOriginal"]


if __name__ == "__main__":
    import numpy as np

    def _run_demo() -> None:
        """Run KBinsDiscretiserWithOriginal across a couple of n_bins settings."""
        rng = np.random.default_rng(42)
        n_rows = 300
        df = pd.DataFrame(
            {
                "age": rng.integers(18, 80, n_rows).astype(float),
                "income": rng.exponential(50_000, n_rows),
            }
        )

        enc_5 = KBinsDiscretiserWithOriginal(n_bins=5, variables=["age", "income"])
        print("=== KBinsDiscretiserWithOriginal (n_bins=5) ===")
        print(enc_5.fit_transform(df).head())

        enc_10 = KBinsDiscretiserWithOriginal(
            n_bins=10, strategy="uniform", variables=["age", "income"]
        )
        print("\n=== KBinsDiscretiserWithOriginal (n_bins=10, uniform) ===")
        print(enc_10.fit_transform(df).head())

    _run_demo()
