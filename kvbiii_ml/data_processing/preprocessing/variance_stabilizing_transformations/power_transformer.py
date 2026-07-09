from typing import Any

import pandas as pd
from feature_engine.transformation import PowerTransformer

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class PowerTransformerWithOriginal(_WithOriginalBase):
    """Wraps PowerTransformer to keep originals and append x^exp-transformed copies.

    Derived columns are named ``{original}_PREPROCESS_POWER``. Applies x^exp; the square-root
    transform (exp=0.5) is the most common choice for moderately right-skewed data.
    """

    _suffix = "PREPROCESS_POWER"

    def __init__(
        self,
        variables: list[str] | None = None,
        exp: float | int = 0.5,
    ) -> None:
        """
        Initialize PowerTransformerWithOriginal.

        Args:
            variables (list[str] | None, optional): Numerical columns to transform.
                Defaults to None (auto-detect).
            exp (float | int, optional): Exponent applied to each value. ``0.5``
                gives the square-root transform. Defaults to 0.5.
        """
        self.variables = variables
        self.exp = exp

    def _build_inner(self) -> PowerTransformer:
        """Return a fresh PowerTransformer.

        Returns:
            PowerTransformer: Unfitted instance.
        """
        return PowerTransformer(variables=self.variables, exp=self.exp)

    def _fit_inner(
        self, inner: PowerTransformer, X: pd.DataFrame, y: Any
    ) -> PowerTransformer:
        """Fit the transformer on X.

        Args:
            inner (PowerTransformer): Unfitted transformer.
            X (pd.DataFrame): Training features.
            y (Any): Unused; accepted for API consistency.

        Returns:
            PowerTransformer: Fitted transformer.
        """
        return inner.fit(X)

    def _transform_inner(
        self, inner: PowerTransformer, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply x^exp transformation to X.

        Args:
            inner (PowerTransformer): Fitted transformer.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with power-transformed column values.
        """
        return inner.transform(X)


__all__ = ["PowerTransformer", "PowerTransformerWithOriginal"]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "count": rng.integers(0, 500, 100).astype(float),
            "duration_sec": rng.exponential(300, 100),
        }
    )

    enc = PowerTransformer(variables=["count", "duration_sec"], exp=0.5)
    print("=== PowerTransformer sqrt (replace) ===")
    print(enc.fit_transform(df).describe())

    enc_exp = PowerTransformerWithOriginal(variables=["count", "duration_sec"], exp=0.5)
    print("\n=== PowerTransformerWithOriginal sqrt (append) ===")
    print(enc_exp.fit_transform(df).describe())
