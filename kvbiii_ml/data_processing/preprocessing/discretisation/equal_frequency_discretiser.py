from typing import Any

import pandas as pd
from feature_engine.discretisation import EqualFrequencyDiscretiser

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class EqualFrequencyDiscretiserWithOriginal(_WithOriginalBase):
    """Wraps EqualFrequencyDiscretiser to keep originals and append binned copies.

    Derived columns are named ``{original}_PREPROCESS_EQ_FREQ``. Each variable is split into
    ``q`` bins with approximately the same number of observations.
    """

    _suffix = "PREPROCESS_EQ_FREQ"

    def __init__(
        self,
        variables: list[str] | None = None,
        q: int = 10,
        return_object: bool = False,
        return_boundaries: bool = False,
        precision: int = 3,
    ) -> None:
        """
        Initialize EqualFrequencyDiscretiserWithOriginal.

        Args:
            variables (list[str] | None, optional): Numerical columns to discretise.
                Defaults to None (auto-detect).
            q (int, optional): Number of equal-frequency bins. Defaults to 10.
            return_object (bool, optional): If True, returns bin labels as strings.
                Defaults to False.
            return_boundaries (bool, optional): If True, returns interval boundaries
                as labels. Defaults to False.
            precision (int, optional): Rounding precision for bin boundaries.
                Defaults to 3.
        """
        self.variables = variables
        self.q = q
        self.return_object = return_object
        self.return_boundaries = return_boundaries
        self.precision = precision

    def _build_inner(self) -> EqualFrequencyDiscretiser:
        """Return a fresh EqualFrequencyDiscretiser.

        Returns:
            EqualFrequencyDiscretiser: Unfitted instance.
        """
        return EqualFrequencyDiscretiser(
            variables=self.variables,
            q=self.q,
            return_object=self.return_object,
            return_boundaries=self.return_boundaries,
            precision=self.precision,
        )

    def _fit_inner(
        self, inner: EqualFrequencyDiscretiser, X: pd.DataFrame, y: Any
    ) -> EqualFrequencyDiscretiser:
        """Fit the discretiser on X.

        Args:
            inner (EqualFrequencyDiscretiser): Unfitted discretiser.
            X (pd.DataFrame): Training features.
            y (Any): Unused; accepted for API consistency.

        Returns:
            EqualFrequencyDiscretiser: Fitted discretiser.
        """
        return inner.fit(X)

    def _transform_inner(
        self, inner: EqualFrequencyDiscretiser, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply the fitted discretiser to X.

        Args:
            inner (EqualFrequencyDiscretiser): Fitted discretiser.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with binned column values.
        """
        return inner.transform(X)


__all__ = ["EqualFrequencyDiscretiser", "EqualFrequencyDiscretiserWithOriginal"]


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

    enc = EqualFrequencyDiscretiser(variables=["age", "income"], q=5)
    print("=== EqualFrequencyDiscretiser (replace) ===")
    print(enc.fit_transform(df).head())

    enc_exp = EqualFrequencyDiscretiserWithOriginal(variables=["age", "income"], q=5)
    print("\n=== EqualFrequencyDiscretiserWithOriginal (append) ===")
    print(enc_exp.fit_transform(df).head())
