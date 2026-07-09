from typing import Any

import pandas as pd
from feature_engine.transformation import LogTransformer

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class LogTransformerWithOriginal(_WithOriginalBase):
    """Wraps LogTransformer to keep originals and append log-transformed copies.

    Derived columns are named ``{original}_PREPROCESS_LOG``. Applies natural log (or log base 10)
    to stabilise variance in right-skewed distributions. Requires strictly positive values.
    """

    _suffix = "PREPROCESS_LOG"

    def __init__(
        self,
        variables: list[str] | None = None,
        base: str = "e",
    ) -> None:
        """
        Initialize LogTransformerWithOriginal.

        Args:
            variables (list[str] | None, optional): Numerical columns to transform.
                Defaults to None (auto-detect).
            base (str, optional): Logarithm base - ``"e"`` for natural log or
                ``"10"`` for log base 10. Defaults to ``"e"``.
        """
        self.variables = variables
        self.base = base

    def _build_inner(self) -> LogTransformer:
        """Return a fresh LogTransformer.

        Returns:
            LogTransformer: Unfitted instance.
        """
        return LogTransformer(variables=self.variables, base=self.base)

    def _fit_inner(
        self, inner: LogTransformer, X: pd.DataFrame, y: Any
    ) -> LogTransformer:
        """Fit the transformer on X.

        Args:
            inner (LogTransformer): Unfitted transformer.
            X (pd.DataFrame): Training features.
            y (Any): Unused; accepted for API consistency.

        Returns:
            LogTransformer: Fitted transformer.
        """
        return inner.fit(X)

    def _transform_inner(self, inner: LogTransformer, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the log transformation to X.

        Args:
            inner (LogTransformer): Fitted transformer.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with log-transformed column values.
        """
        return inner.transform(X)


__all__ = ["LogTransformer", "LogTransformerWithOriginal"]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "income": rng.exponential(50_000, 100) + 1,
            "tenure_days": rng.exponential(200, 100) + 1,
        }
    )

    enc = LogTransformer(variables=["income", "tenure_days"])
    print("=== LogTransformer (replace) ===")
    print(enc.fit_transform(df).describe())

    enc_exp = LogTransformerWithOriginal(variables=["income", "tenure_days"])
    print("\n=== LogTransformerWithOriginal (append) ===")
    print(enc_exp.fit_transform(df).describe())
