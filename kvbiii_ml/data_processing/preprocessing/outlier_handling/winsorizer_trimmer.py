from typing import Any

import pandas as pd
from feature_engine.outliers import Winsorizer

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class WinsorizerWithOriginal(_WithOriginalBase):
    """Wraps feature_engine Winsorizer to keep originals and append capped copies.

    Derived columns are named ``{original}_PREPROCESS_WINSORIZER``. Keeping
    originals lets the model use extreme values as signal while the capped copies
    guard against test-time extrapolation failures.
    """

    _suffix = "PREPROCESS_WINSORIZER"

    def __init__(
        self,
        variables: list[str] | None = None,
        capping_method: str = "iqr",
        tail: str = "both",
        fold: float = 3.0,
    ) -> None:
        """
        Initialize WinsorizerWithOriginal.

        Args:
            variables (list[str] | None, optional): Numerical variables to cap.
                Defaults to None (auto-detect all numerical columns).
            capping_method (str, optional): Capping strategy - ``"iqr"``,
                ``"gaussian"``, ``"mad"``, or ``"quantiles"``. Defaults to ``"iqr"``.
            tail (str, optional): Which tail(s) to cap - ``"both"``, ``"left"``,
                or ``"right"``. Defaults to ``"both"``.
            fold (float, optional): IQR / std multiplier that determines cap
                boundaries. Defaults to 3.0.
        """
        self.variables = variables
        self.capping_method = capping_method
        self.tail = tail
        self.fold = fold

    def _build_inner(self) -> Winsorizer:
        """Return a fresh Winsorizer configured from instance attributes.

        Returns:
            Winsorizer: Unfitted Winsorizer instance.
        """
        return Winsorizer(
            variables=self.variables,
            capping_method=self.capping_method,
            tail=self.tail,
            fold=self.fold,
            missing_values="ignore",
        )

    def _fit_inner(self, inner: Winsorizer, X: pd.DataFrame, y: Any) -> Winsorizer:
        """Fit the Winsorizer on X.

        Args:
            inner (Winsorizer): Unfitted Winsorizer.
            X (pd.DataFrame): Training features.
            y (Any): Unused; accepted for API consistency.

        Returns:
            Winsorizer: Fitted Winsorizer.
        """
        return inner.fit(X)

    def _transform_inner(self, inner: Winsorizer, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted Winsorizer to X.

        Args:
            inner (Winsorizer): Fitted Winsorizer.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with capped column values.
        """
        return inner.transform(X)


__all__ = ["Winsorizer", "WinsorizerWithOriginal"]


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

    enc = Winsorizer(capping_method="gaussian", tail="right")
    print("=== Winsorizer (replace with NaN) ===")
    print(enc.fit_transform(df).tail())

    enc_exp = WinsorizerWithOriginal(capping_method="gaussian", tail="right")
    print("\n=== WinsorizerWithOriginal (append) ===")
    print(enc_exp.fit_transform(df).tail())
