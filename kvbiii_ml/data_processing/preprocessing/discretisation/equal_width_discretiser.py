from typing import Any

import pandas as pd
from feature_engine.discretisation import EqualWidthDiscretiser

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class EqualWidthDiscretiserWithOriginal(_WithOriginalBase):
    """Wraps EqualWidthDiscretiser to keep originals and append binned copies.

    Derived columns are named ``{original}_PREPROCESS_EQ_WIDTH``. Each variable is split into
    ``bins`` intervals of equal width across the observed value range.
    """

    _suffix = "PREPROCESS_EQ_WIDTH"

    def __init__(
        self,
        variables: list[str] | None = None,
        bins: int = 10,
        return_object: bool = False,
        return_boundaries: bool = False,
        precision: int = 3,
    ) -> None:
        """
        Initialize EqualWidthDiscretiserWithOriginal.

        Args:
            variables (list[str] | None, optional): Numerical columns to discretise.
                Defaults to None (auto-detect).
            bins (int, optional): Number of equal-width bins. Defaults to 10.
            return_object (bool, optional): If True, returns bin labels as strings.
                Defaults to False.
            return_boundaries (bool, optional): If True, returns interval boundaries
                as labels. Defaults to False.
            precision (int, optional): Rounding precision for bin boundaries.
                Defaults to 3.
        """
        self.variables = variables
        self.bins = bins
        self.return_object = return_object
        self.return_boundaries = return_boundaries
        self.precision = precision

    def _build_inner(self) -> EqualWidthDiscretiser:
        """Return a fresh EqualWidthDiscretiser.

        Returns:
            EqualWidthDiscretiser: Unfitted instance.
        """
        return EqualWidthDiscretiser(
            variables=self.variables,
            bins=self.bins,
            return_object=self.return_object,
            return_boundaries=self.return_boundaries,
            precision=self.precision,
        )

    def _fit_inner(
        self, inner: EqualWidthDiscretiser, X: pd.DataFrame, y: Any
    ) -> EqualWidthDiscretiser:
        """Fit the discretiser on X.

        Args:
            inner (EqualWidthDiscretiser): Unfitted discretiser.
            X (pd.DataFrame): Training features.
            y (Any): Unused; accepted for API consistency.

        Returns:
            EqualWidthDiscretiser: Fitted discretiser.
        """
        return inner.fit(X)

    def _transform_inner(
        self, inner: EqualWidthDiscretiser, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply the fitted discretiser to X.

        Args:
            inner (EqualWidthDiscretiser): Fitted discretiser.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with binned column values.
        """
        return inner.transform(X)


__all__ = ["EqualWidthDiscretiser", "EqualWidthDiscretiserWithOriginal"]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 80, 200),
            "income": rng.exponential(50_000, 200),
            "city": rng.choice(["Warsaw", "Krakow", "Gdansk"], 200),
        }
    )

    enc = EqualWidthDiscretiser(variables=["age", "income"], bins=5)
    print("=== EqualWidthDiscretiser (replace) ===")
    print(enc.fit_transform(df).head())

    enc_exp = EqualWidthDiscretiserWithOriginal(variables=["age", "income"], bins=5)
    print("\n=== EqualWidthDiscretiserWithOriginal (append) ===")
    print(enc_exp.fit_transform(df).head())
