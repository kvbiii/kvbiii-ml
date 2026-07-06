from typing import Any

import pandas as pd
from feature_engine.discretisation import GeometricWidthDiscretiser

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class GeometricWidthDiscretiserWithOriginal(_WithOriginalBase):
    """Wraps GeometricWidthDiscretiser to keep originals and append geo-binned copies.

    Derived columns are named ``{original}_PREPROCESS_GEO_WIDTH``. Bin widths grow geometrically,
    making this well-suited for heavily right-skewed distributions where a log-scale
    binning is more informative than equal-width intervals.
    """

    _suffix = "PREPROCESS_GEO_WIDTH"

    def __init__(
        self,
        variables: list[str] | None = None,
        bins: int = 10,
        return_object: bool = False,
        return_boundaries: bool = False,
        precision: int = 7,
    ) -> None:
        """
        Initialize GeometricWidthDiscretiserWithOriginal.

        Args:
            variables (list[str] | None, optional): Numerical columns to discretise.
                Defaults to None (auto-detect).
            bins (int, optional): Number of geometric-width bins. Defaults to 10.
            return_object (bool, optional): If True, returns bin labels as strings.
                Defaults to False.
            return_boundaries (bool, optional): If True, returns interval boundaries
                as labels. Defaults to False.
            precision (int, optional): Rounding precision for bin boundaries.
                Defaults to 7.
        """
        self.variables = variables
        self.bins = bins
        self.return_object = return_object
        self.return_boundaries = return_boundaries
        self.precision = precision

    def _build_inner(self) -> GeometricWidthDiscretiser:
        """Return a fresh GeometricWidthDiscretiser.

        Returns:
            GeometricWidthDiscretiser: Unfitted instance.
        """
        return GeometricWidthDiscretiser(
            variables=self.variables,
            bins=self.bins,
            return_object=self.return_object,
            return_boundaries=self.return_boundaries,
            precision=self.precision,
        )

    def _fit_inner(
        self, inner: GeometricWidthDiscretiser, X: pd.DataFrame, y: Any
    ) -> GeometricWidthDiscretiser:
        """Fit the discretiser on X.

        Args:
            inner (GeometricWidthDiscretiser): Unfitted discretiser.
            X (pd.DataFrame): Training features.
            y (Any): Unused; accepted for API consistency.

        Returns:
            GeometricWidthDiscretiser: Fitted discretiser.
        """
        return inner.fit(X)

    def _transform_inner(
        self, inner: GeometricWidthDiscretiser, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply the fitted discretiser to X.

        Args:
            inner (GeometricWidthDiscretiser): Fitted discretiser.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with geo-binned column values.
        """
        return inner.transform(X)


__all__ = ["GeometricWidthDiscretiser", "GeometricWidthDiscretiserWithOriginal"]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "income": rng.exponential(50_000, 200) + 1,
            "tenure_days": rng.exponential(500, 200) + 1,
        }
    )

    enc = GeometricWidthDiscretiser(variables=["income", "tenure_days"], bins=5)
    print("=== GeometricWidthDiscretiser (replace) ===")
    print(enc.fit_transform(df).head())

    enc_exp = GeometricWidthDiscretiserWithOriginal(
        variables=["income", "tenure_days"], bins=5
    )
    print("\n=== GeometricWidthDiscretiserWithOriginal (append) ===")
    print(enc_exp.fit_transform(df).head())
