from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from kvbiii_ml.data_processing.preprocessing.expansion_base import (
    _WithOriginalSubsetBase,
)


class _NaNIgnoringKBinsDiscretizer:
    """Per-column KBinsDiscretizer wrapper that leaves NaN values untouched.

    Raw scikit-learn's KBinsDiscretizer rejects its entire 2D input if any
    column contains a NaN, regardless of which column it is in - so a single
    joint fit/transform across variables cannot skip missing values on a
    per-column basis. This wrapper fits one KBinsDiscretizer per variable on
    that variable's non-null rows only, and reassembles a 2D float array on
    transform with NaN left in place at the original missing positions. A
    variable that is entirely NaN at fit time has no bin edges to learn, so it
    is fitted to ``None`` and its derived column is all NaN on transform.
    Matches the plain ``fit(X, y)`` / ``transform(X)`` contract that
    ``_WithOriginalSubsetBase`` expects from ``_build_inner()``.
    """

    def __init__(
        self,
        n_bins: int,
        strategy: Literal["uniform", "quantile", "kmeans"],
        quantile_method: str,
        dtype: type | None,
        subsample: int | None,
        random_state: int | None,
    ) -> None:
        """
        Initialize _NaNIgnoringKBinsDiscretizer.

        Args:
            n_bins (int): Number of bins to produce for every variable.
            strategy (Literal["uniform", "quantile", "kmeans"]): Binning strategy
                forwarded to each per-variable KBinsDiscretizer.
            quantile_method (str): Quantile interpolation method forwarded to each
                per-variable KBinsDiscretizer.
            dtype (type | None): Output dtype forwarded to each per-variable
                KBinsDiscretizer.
            subsample (int | None): Maximum number of samples used to fit the
                "quantile"/"kmeans" strategies.
            random_state (int | None): Random seed for the "kmeans" strategy and
                for subsampling.
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.quantile_method = quantile_method
        self.dtype = dtype
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, _y: Any = None) -> "_NaNIgnoringKBinsDiscretizer":
        """Fit one KBinsDiscretizer per column, using only that column's non-null rows.

        Args:
            X (pd.DataFrame): Training features, one column per variable to discretise.
            _y (Any, optional): Unused; accepted for API consistency. Defaults to None.

        Columns with no non-null values are fitted to ``None`` instead of raising,
        so their derived column is entirely NaN on transform.

        Returns:
            _NaNIgnoringKBinsDiscretizer: Fitted instance (self).

        Raises:
            ValueError: If KBinsDiscretizer fails on a column's non-null values.
        """
        self.columns_ = list(X.columns)
        self.column_discretisers_: dict[str, KBinsDiscretizer | None] = {}
        for column in self.columns_:
            observed = X[column].dropna()
            if observed.empty:
                self.column_discretisers_[column] = None
                continue
            try:
                self.column_discretisers_[
                    column
                ] = self._build_column_discretiser().fit(
                    observed.to_numpy().reshape(-1, 1)
                )
            except ValueError as exc:
                raise ValueError(
                    f"Failed to fit KBinsDiscretizer on non-null values of column "
                    f"'{column}': {exc}"
                ) from exc
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Bin each column's non-null rows, leaving NaN rows as NaN.

        Args:
            X (pd.DataFrame): Features to transform, matching the columns seen in fit.

        Returns:
            np.ndarray: Array of shape (n_samples, n_columns) with binned values,
                NaN preserved at originally-missing positions and across every
                column that was entirely NaN at fit time.
        """
        encoded = np.full((len(X), len(self.columns_)), np.nan, dtype=np.float64)
        for column_index, column in enumerate(self.columns_):
            column_discretiser = self.column_discretisers_[column]
            if column_discretiser is None:
                continue
            series = X[column]
            not_null_mask = series.notna()
            if not not_null_mask.any():
                continue
            observed = series[not_null_mask].to_numpy().reshape(-1, 1)
            binned = column_discretiser.transform(observed).reshape(-1)
            encoded[not_null_mask.to_numpy(), column_index] = binned
        return encoded

    def _build_column_discretiser(self) -> KBinsDiscretizer:
        """Return a fresh, unfitted KBinsDiscretizer for a single variable.

        Returns:
            KBinsDiscretizer: Unfitted discretiser, fixed to ordinal encoding.
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
        missing_values: Literal["raise", "ignore"] = "raise",
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
            missing_values (Literal["raise", "ignore"], optional): How to handle
                NaN - ``"raise"`` lets KBinsDiscretizer raise its own ValueError
                on NaN input (scikit-learn's native behavior), ``"ignore"`` bins
                each variable's non-null rows and leaves NaN rows as NaN in the
                derived column (an entirely-NaN variable still fits and yields
                an all-NaN derived column). Defaults to "raise".
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.quantile_method = quantile_method
        self.dtype = dtype
        self.subsample = subsample
        self.random_state = random_state
        self.variables = variables
        self.missing_values = missing_values

    @property
    def _suffix(self) -> str:
        """Derived-column suffix, dynamically keyed on ``n_bins``.

        Returns:
            str: e.g. "PREPROCESS_KBINS_5" for ``n_bins=5``.
        """
        return f"PREPROCESS_KBINS_{self.n_bins}"

    def _build_inner(self) -> KBinsDiscretizer | _NaNIgnoringKBinsDiscretizer:
        """Return a fresh inner discretiser configured from instance attributes.

        Returns:
            KBinsDiscretizer | _NaNIgnoringKBinsDiscretizer: Unfitted discretiser,
                fixed to ordinal encoding so each variable maps to exactly one
                derived column. A plain KBinsDiscretizer when
                ``missing_values="raise"``, or the NaN-preserving per-column
                wrapper when ``missing_values="ignore"``.
        """
        if self.missing_values == "ignore":
            return _NaNIgnoringKBinsDiscretizer(
                n_bins=self.n_bins,
                strategy=self.strategy,
                quantile_method=self.quantile_method,
                dtype=self.dtype,
                subsample=self.subsample,
                random_state=self.random_state,
            )
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

        df_with_nan = df.copy()
        df_with_nan.loc[:9, "income"] = np.nan
        enc_ignore_nan = KBinsDiscretiserWithOriginal(
            n_bins=5, variables=["age", "income"], missing_values="ignore"
        )
        print(
            "\n=== KBinsDiscretiserWithOriginal (n_bins=5, missing_values='ignore') ==="
        )
        print(enc_ignore_nan.fit_transform(df_with_nan).head(12))

    _run_demo()
