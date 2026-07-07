from typing import Any

import pandas as pd
from feature_engine.transformation import ArcsinTransformer

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class ArcsinTransformerWithOriginal(_WithOriginalBase):
    """Wraps ArcsinTransformer to keep originals and append arcsin-transformed copies.

    Derived columns are named ``{original}_PREPROCESS_ARCSIN``. Applies arcsin(sqrt(x)), the
    variance-stabilising transformation for proportion data in [0, 1].
    """

    _suffix = "PREPROCESS_ARCSIN"

    def __init__(self, variables: list[str] | None = None) -> None:
        """
        Initialize ArcsinTransformerWithOriginal.

        Args:
            variables (list[str] | None, optional): Numerical columns in [0, 1]
                to transform. Defaults to None (auto-detect).
        """
        self.variables = variables

    def _build_inner(self) -> ArcsinTransformer:
        """Return a fresh ArcsinTransformer.

        Returns:
            ArcsinTransformer: Unfitted instance.
        """
        return ArcsinTransformer(variables=self.variables)

    def _fit_inner(
        self, inner: ArcsinTransformer, X: pd.DataFrame, y: Any
    ) -> ArcsinTransformer:
        """Fit the transformer on X.

        Args:
            inner (ArcsinTransformer): Unfitted transformer.
            X (pd.DataFrame): Training features.
            y (Any): Unused; accepted for API consistency.

        Returns:
            ArcsinTransformer: Fitted transformer.
        """
        return inner.fit(X)

    def _transform_inner(
        self, inner: ArcsinTransformer, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply arcsin(sqrt(x)) transformation to X.

        Args:
            inner (ArcsinTransformer): Fitted transformer.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with arcsin-transformed column values.
        """
        return inner.transform(X)


__all__ = ["ArcsinTransformer", "ArcsinTransformerWithOriginal"]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "click_rate": rng.uniform(0.0, 1.0, 100),
            "conversion_rate": rng.beta(2, 8, 100),
        }
    )

    enc = ArcsinTransformer(variables=["click_rate", "conversion_rate"])
    print("=== ArcsinTransformer (replace) ===")
    print(enc.fit_transform(df).describe())

    enc_exp = ArcsinTransformerWithOriginal(variables=["click_rate", "conversion_rate"])
    print("\n=== ArcsinTransformerWithOriginal (append) ===")
    print(enc_exp.fit_transform(df).describe())
