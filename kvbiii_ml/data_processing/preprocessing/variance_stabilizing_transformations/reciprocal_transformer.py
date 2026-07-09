from typing import Any

import pandas as pd
from feature_engine.transformation import ReciprocalTransformer

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class ReciprocalTransformerWithOriginal(_WithOriginalBase):
    """Wraps ReciprocalTransformer to keep originals and append 1/x-transformed copies.

    Derived columns are named ``{original}_PREPROCESS_RECIPROCAL``. Applies 1/x, which compresses
    large values and stretches small ones. Requires non-zero values.
    """

    _suffix = "PREPROCESS_RECIPROCAL"

    def __init__(self, variables: list[str] | None = None) -> None:
        """
        Initialize ReciprocalTransformerWithOriginal.

        Args:
            variables (list[str] | None, optional): Numerical columns to transform.
                Defaults to None (auto-detect).
        """
        self.variables = variables

    def _build_inner(self) -> ReciprocalTransformer:
        """Return a fresh ReciprocalTransformer.

        Returns:
            ReciprocalTransformer: Unfitted instance.
        """
        return ReciprocalTransformer(variables=self.variables)

    def _fit_inner(
        self, inner: ReciprocalTransformer, X: pd.DataFrame, y: Any
    ) -> ReciprocalTransformer:
        """Fit the transformer on X.

        Args:
            inner (ReciprocalTransformer): Unfitted transformer.
            X (pd.DataFrame): Training features.
            y (Any): Unused; accepted for API consistency.

        Returns:
            ReciprocalTransformer: Fitted transformer.
        """
        return inner.fit(X)

    def _transform_inner(
        self, inner: ReciprocalTransformer, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply 1/x transformation to X.

        Args:
            inner (ReciprocalTransformer): Fitted transformer.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with reciprocal-transformed column values.
        """
        return inner.transform(X)


__all__ = ["ReciprocalTransformer", "ReciprocalTransformerWithOriginal"]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "response_time_ms": rng.exponential(200, 100) + 1,
            "request_size_kb": rng.exponential(50, 100) + 1,
        }
    )

    enc = ReciprocalTransformer(variables=["response_time_ms", "request_size_kb"])
    print("=== ReciprocalTransformer (replace) ===")
    print(enc.fit_transform(df).describe())

    enc_exp = ReciprocalTransformerWithOriginal(
        variables=["response_time_ms", "request_size_kb"]
    )
    print("\n=== ReciprocalTransformerWithOriginal (append) ===")
    print(enc_exp.fit_transform(df).describe())
