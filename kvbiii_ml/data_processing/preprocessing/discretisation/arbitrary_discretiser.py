from typing import Any

import pandas as pd
from feature_engine.discretisation import ArbitraryDiscretiser

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class ArbitraryDiscretiserWithOriginal(_WithOriginalBase):
    """Wraps ArbitraryDiscretiser to keep originals and append user-defined binned copies.

    Derived columns are named ``{original}_PREPROCESS_ARB_DISC``. Bin boundaries are fully
    specified by the user via ``binning_dict``, making this useful when domain
    knowledge dictates exact cut-points.
    """

    _suffix = "PREPROCESS_ARB_DISC"

    def __init__(
        self,
        binning_dict: dict[str, list],
        return_object: bool = False,
        return_boundaries: bool = False,
        precision: int = 3,
        errors: str = "ignore",
    ) -> None:
        """
        Initialize ArbitraryDiscretiserWithOriginal.

        Args:
            binning_dict (dict[str, list]): Mapping from variable name to a list of
                cut-points, e.g. ``{"age": [0, 18, 35, 60, np.inf]}``.
            return_object (bool, optional): If True, returns bin labels as strings.
                Defaults to False.
            return_boundaries (bool, optional): If True, returns interval boundaries
                as labels. Defaults to False.
            precision (int, optional): Rounding precision for bin boundaries.
                Defaults to 3.
            errors (str, optional): How to handle values outside defined bins -
                ``"ignore"`` or ``"raise"``. Defaults to ``"ignore"``.
        """
        self.binning_dict = binning_dict
        self.return_object = return_object
        self.return_boundaries = return_boundaries
        self.precision = precision
        self.errors = errors

    def _build_inner(self) -> ArbitraryDiscretiser:
        """Return a fresh ArbitraryDiscretiser.

        Returns:
            ArbitraryDiscretiser: Unfitted instance.
        """
        return ArbitraryDiscretiser(
            binning_dict=self.binning_dict,
            return_object=self.return_object,
            return_boundaries=self.return_boundaries,
            precision=self.precision,
            errors=self.errors,
        )

    def _fit_inner(
        self, inner: ArbitraryDiscretiser, X: pd.DataFrame, y: Any
    ) -> ArbitraryDiscretiser:
        """Fit the discretiser on X (stores the binning dict internally).

        Args:
            inner (ArbitraryDiscretiser): Unfitted discretiser.
            X (pd.DataFrame): Training features.
            y (Any): Unused; accepted for API consistency.

        Returns:
            ArbitraryDiscretiser: Fitted discretiser.
        """
        return inner.fit(X)

    def _transform_inner(
        self, inner: ArbitraryDiscretiser, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply the fitted discretiser to X.

        Args:
            inner (ArbitraryDiscretiser): Fitted discretiser.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with user-defined binned column values.
        """
        return inner.transform(X)


__all__ = ["ArbitraryDiscretiser", "ArbitraryDiscretiserWithOriginal"]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 80, 200).astype(float),
            "income": rng.uniform(20_000, 150_000, 200),
        }
    )

    bins = {"age": [0, 25, 40, 60, np.inf], "income": [0, 40_000, 80_000, np.inf]}

    enc = ArbitraryDiscretiser(binning_dict=bins)
    print("=== ArbitraryDiscretiser (replace) ===")
    print(enc.fit_transform(df).head())

    enc_exp = ArbitraryDiscretiserWithOriginal(binning_dict=bins)
    print("\n=== ArbitraryDiscretiserWithOriginal (append) ===")
    print(enc_exp.fit_transform(df).head())
