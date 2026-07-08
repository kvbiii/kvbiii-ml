from typing import Any

import pandas as pd
from feature_engine.transformation import BoxCoxTransformer

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class BoxCoxTransformerWithOriginal(_WithOriginalBase):
    """Wraps BoxCoxTransformer to keep originals and append Box-Cox-transformed copies.

    Derived columns are named ``{original}_PREPROCESS_BOX_COX``. The optimal lambda is estimated
    from the data to best normalise each variable. Requires strictly positive values.
    """

    _suffix = "PREPROCESS_BOX_COX"

    def __init__(self, variables: list[str] | None = None) -> None:
        """
        Initialize BoxCoxTransformerWithOriginal.

        Args:
            variables (list[str] | None, optional): Numerical columns to transform.
                Defaults to None (auto-detect).
        """
        self.variables = variables

    def _build_inner(self) -> BoxCoxTransformer:
        """Return a fresh BoxCoxTransformer.

        Returns:
            BoxCoxTransformer: Unfitted instance.
        """
        return BoxCoxTransformer(variables=self.variables)

    def _fit_inner(
        self, inner: BoxCoxTransformer, X: pd.DataFrame, y: Any
    ) -> BoxCoxTransformer:
        """Fit the transformer on X (estimates lambda per variable).

        Args:
            inner (BoxCoxTransformer): Unfitted transformer.
            X (pd.DataFrame): Training features.
            y (Any): Unused; accepted for API consistency.

        Returns:
            BoxCoxTransformer: Fitted transformer.
        """
        return inner.fit(X)

    def _transform_inner(
        self, inner: BoxCoxTransformer, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply the Box-Cox transformation to X.

        Args:
            inner (BoxCoxTransformer): Fitted transformer.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with Box-Cox-transformed column values.
        """
        return inner.transform(X)


__all__ = ["BoxCoxTransformer", "BoxCoxTransformerWithOriginal"]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "income": rng.exponential(50_000, 100) + 1,
            "price": rng.lognormal(3, 1, 100) + 1,
        }
    )

    enc = BoxCoxTransformer(variables=["income", "price"])
    print("=== BoxCoxTransformer (replace) ===")
    print(enc.fit_transform(df).describe())
    print("Lambdas:", enc.lambda_dict_)

    enc_exp = BoxCoxTransformerWithOriginal(variables=["income", "price"])
    print("\n=== BoxCoxTransformerWithOriginal (append) ===")
    print(enc_exp.fit_transform(df).describe())
