from typing import Any

import pandas as pd
from feature_engine.transformation import YeoJohnsonTransformer

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class YeoJohnsonTransformerWithOriginal(_WithOriginalBase):
    """Wraps YeoJohnsonTransformer to keep originals and append Yeo-Johnson-transformed copies.

    Derived columns are named ``{original}_PREPROCESS_YEO_JOHNSON``. The optimal lambda is estimated
    from the data. Unlike Box-Cox, this supports zero and negative values.
    """

    _suffix = "PREPROCESS_YEO_JOHNSON"

    def __init__(self, variables: list[str] | None = None) -> None:
        """
        Initialize YeoJohnsonTransformerWithOriginal.

        Args:
            variables (list[str] | None, optional): Numerical columns to transform.
                Defaults to None (auto-detect).
        """
        self.variables = variables

    def _build_inner(self) -> YeoJohnsonTransformer:
        """Return a fresh YeoJohnsonTransformer.

        Returns:
            YeoJohnsonTransformer: Unfitted instance.
        """
        return YeoJohnsonTransformer(variables=self.variables)

    def _fit_inner(
        self, inner: YeoJohnsonTransformer, X: pd.DataFrame, y: Any
    ) -> YeoJohnsonTransformer:
        """Fit the transformer on X (estimates lambda per variable).

        Args:
            inner (YeoJohnsonTransformer): Unfitted transformer.
            X (pd.DataFrame): Training features.
            y (Any): Unused; accepted for API consistency.

        Returns:
            YeoJohnsonTransformer: Fitted transformer.
        """
        return inner.fit(X)

    def _transform_inner(
        self, inner: YeoJohnsonTransformer, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply the Yeo-Johnson transformation to X.

        Args:
            inner (YeoJohnsonTransformer): Fitted transformer.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with Yeo-Johnson-transformed column values.
        """
        return inner.transform(X)


__all__ = ["YeoJohnsonTransformer", "YeoJohnsonTransformerWithOriginal"]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "score": rng.normal(0, 5, 100),
            "balance": rng.normal(1_000, 5_000, 100),
        }
    )

    enc = YeoJohnsonTransformer(variables=["score", "balance"])
    print("=== YeoJohnsonTransformer (replace) ===")
    print(enc.fit_transform(df).describe())
    print("Lambdas:", enc.lambda_dict_)

    enc_exp = YeoJohnsonTransformerWithOriginal(variables=["score", "balance"])
    print("\n=== YeoJohnsonTransformerWithOriginal (append) ===")
    print(enc_exp.fit_transform(df).describe())
