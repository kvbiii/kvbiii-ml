from typing import Any

import pandas as pd
from feature_engine.transformation import LogCpTransformer

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class LogCpTransformerWithOriginal(_WithOriginalBase):
    """Wraps LogCpTransformer to keep originals and append log(x+C)-transformed copies.

    Derived columns are named ``{original}_PREPROCESS_LOG_CP``. Applies log(x + C) to handle
    zero or near-zero values in right-skewed distributions. ``C`` can be set manually
    or estimated automatically from the data.
    """

    _suffix = "PREPROCESS_LOG_CP"

    def __init__(
        self,
        variables: list[str] | None = None,
        base: str = "e",
        C: int | float | str | dict = "auto",
    ) -> None:
        """
        Initialize LogCpTransformerWithOriginal.

        Args:
            variables (list[str] | None, optional): Numerical columns to transform.
                Defaults to None (auto-detect).
            base (str, optional): Logarithm base - ``"e"`` or ``"10"``. Defaults to ``"e"``.
            C (int | float | str | dict, optional): Constant added before log. ``"auto"``
                sets C to the absolute value of the minimum plus a small epsilon.
                Defaults to ``"auto"``.
        """
        self.variables = variables
        self.base = base
        self.C = C

    def _build_inner(self) -> LogCpTransformer:
        """Return a fresh LogCpTransformer.

        Returns:
            LogCpTransformer: Unfitted instance.
        """
        return LogCpTransformer(variables=self.variables, base=self.base, C=self.C)

    def _fit_inner(
        self, inner: LogCpTransformer, X: pd.DataFrame, y: Any
    ) -> LogCpTransformer:
        """Fit the transformer on X (learns C per variable if C='auto').

        Args:
            inner (LogCpTransformer): Unfitted transformer.
            X (pd.DataFrame): Training features.
            y (Any): Unused; accepted for API consistency.

        Returns:
            LogCpTransformer: Fitted transformer.
        """
        return inner.fit(X)

    def _transform_inner(
        self, inner: LogCpTransformer, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply log(x+C) transformation to X.

        Args:
            inner (LogCpTransformer): Fitted transformer.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with log(x+C)-transformed column values.
        """
        return inner.transform(X)


__all__ = ["LogCpTransformer", "LogCpTransformerWithOriginal"]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "sales": rng.exponential(1_000, 100),
            "returns": rng.integers(0, 50, 100).astype(float),
        }
    )

    enc = LogCpTransformer(variables=["sales", "returns"], C="auto")
    print("=== LogCpTransformer (replace) ===")
    print(enc.fit_transform(df).describe())

    enc_exp = LogCpTransformerWithOriginal(variables=["sales", "returns"], C="auto")
    print("\n=== LogCpTransformerWithOriginal (append) ===")
    print(enc_exp.fit_transform(df).describe())
