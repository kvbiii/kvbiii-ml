from typing import Any

import pandas as pd
from feature_engine.outliers import OutlierTrimmer

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class OutlierTrimmerWithOriginal(_WithOriginalBase):
    """Wraps OutlierTrimmer to keep originals and append trimmed copies.

    Derived columns are named ``{original}_PREPROCESS_TRIMMED``. Values beyond the computed
    boundaries are replaced with NaN in the derived columns, letting the original
    extreme values remain visible to downstream steps.
    """

    _suffix = "PREPROCESS_TRIMMED"

    def __init__(
        self,
        capping_method: str = "gaussian",
        tail: str = "right",
        fold: int | float | str = "auto",
        variables: list[str] | None = None,
        missing_values: str = "raise",
    ) -> None:
        """
        Initialize OutlierTrimmerWithOriginal.

        Args:
            capping_method (str, optional): Method to compute outlier boundaries -
                ``"gaussian"``, ``"iqr"``, ``"mad"``, or ``"quantiles"``.
                Defaults to ``"gaussian"``.
            tail (str, optional): Which tail(s) to trim - ``"right"``, ``"left"``,
                or ``"both"``. Defaults to ``"right"``.
            fold (int | float | str, optional): Multiplier for boundary calculation.
                ``"auto"`` uses the method's natural default. Defaults to ``"auto"``.
            variables (list[str] | None, optional): Numerical columns to trim.
                Defaults to None (auto-detect).
            missing_values (str, optional): How to handle NaN - ``"raise"`` or
                ``"ignore"``. Defaults to ``"raise"``.
        """
        self.capping_method = capping_method
        self.tail = tail
        self.fold = fold
        self.variables = variables
        self.missing_values = missing_values

    def _build_inner(self) -> OutlierTrimmer:
        """Return a fresh OutlierTrimmer.

        Returns:
            OutlierTrimmer: Unfitted instance.
        """
        return OutlierTrimmer(
            capping_method=self.capping_method,
            tail=self.tail,
            fold=self.fold,
            variables=self.variables,
            missing_values=self.missing_values,
        )

    def _fit_inner(
        self, inner: OutlierTrimmer, X: pd.DataFrame, y: Any
    ) -> OutlierTrimmer:
        """Fit the trimmer on X.

        Args:
            inner (OutlierTrimmer): Unfitted trimmer.
            X (pd.DataFrame): Training features.
            y (Any): Unused; accepted for API consistency.

        Returns:
            OutlierTrimmer: Fitted trimmer.
        """
        return inner.fit(X)

    def _transform_inner(self, inner: OutlierTrimmer, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted trimmer to X.

        Args:
            inner (OutlierTrimmer): Fitted trimmer.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with out-of-bound values set to NaN.
        """
        return inner.transform(X)


__all__ = ["OutlierTrimmer", "OutlierTrimmerWithOriginal"]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "salary": np.concatenate([rng.normal(60_000, 10_000, 95), [500_000]]),
            "age": np.concatenate([rng.integers(22, 65, 95).astype(float), [150.0]]),
        }
    )

    enc = OutlierTrimmer(capping_method="gaussian", tail="right")
    print("=== OutlierTrimmer (replace with NaN) ===")
    print(enc.fit_transform(df).tail())

    enc_exp = OutlierTrimmerWithOriginal(capping_method="gaussian", tail="right")
    print("\n=== OutlierTrimmerWithOriginal (append) ===")
    print(enc_exp.fit_transform(df).tail())
