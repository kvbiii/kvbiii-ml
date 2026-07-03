from typing import Any

import pandas as pd
from feature_engine.encoding import WoEEncoder

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class WoEEncoderWithOriginal(_WithOriginalBase):
    """Wraps feature_engine WoEEncoder to keep originals and append WoE-encoded copies.

    Derived columns are named ``{original}_PREPROCESS_WOE``. Weight of Evidence measures how
    much a category shifts the log-odds of the positive class, making it a natural
    encoding for binary classification. Original categorical columns are preserved.

    Requires a binary target (0/1) and at least one event and one non-event per
    category - categories violating this will cause a ValueError during fit.
    """

    _suffix = "PREPROCESS_WOE"
    _suppress_warnings = True

    def __init__(
        self,
        variables: list[str] | None = None,
        ignore_format: bool = False,
        unseen: str = "ignore",
        fill_value: int | float | None = None,
    ) -> None:
        """
        Initialize WoEEncoderWithOriginal.

        Args:
            variables (list[str] | None, optional): Categorical columns to encode.
                Defaults to None (auto-detect all object/categorical columns).
            ignore_format (bool, optional): If True, also encode numeric columns.
                Defaults to False.
            unseen (str, optional): Strategy for unseen categories at transform time -
                ``"ignore"`` or ``"raise"``. Defaults to ``"ignore"``.
            fill_value (int | float | None, optional): Value used for unseen categories
                when ``unseen="ignore"``. Defaults to None (uses 0.0).
        """
        self.variables = variables
        self.ignore_format = ignore_format
        self.unseen = unseen
        self.fill_value = fill_value

    def _build_inner(self) -> WoEEncoder:
        """Return a fresh WoEEncoder configured from instance attributes.

        Returns:
            WoEEncoder: Unfitted encoder instance.
        """
        return WoEEncoder(
            variables=self.variables,
            ignore_format=self.ignore_format,
            unseen=self.unseen,
            fill_value=self.fill_value,
        )

    def _fit_inner(self, inner: WoEEncoder, X: pd.DataFrame, y: Any) -> WoEEncoder:
        """Fit the encoder on X and y.

        Args:
            inner (WoEEncoder): Unfitted encoder.
            X (pd.DataFrame): Training features.
            y (Any): Binary target (0/1) used to compute Weight of Evidence.

        Returns:
            WoEEncoder: Fitted encoder.
        """
        return inner.fit(X, y)

    def _transform_inner(self, inner: WoEEncoder, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted encoder to X.

        Args:
            inner (WoEEncoder): Fitted encoder.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with WoE-encoded column values.
        """
        return inner.transform(X)


__all__ = ["WoEEncoder", "WoEEncoderWithOriginal"]


if __name__ == "__main__":
    import pandas as pd
    from sklearn.pipeline import Pipeline

    df = pd.DataFrame(
        {
            "city": [
                "Warsaw",
                "Warsaw",
                "Warsaw",
                "Warsaw",
                "Krakow",
                "Krakow",
                "Krakow",
                "Krakow",
                "Gdansk",
                "Gdansk",
                "Gdansk",
                "Gdansk",
            ],
            "tier": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            "revenue": [100, 120, 200, 180, 150, 140, 210, 190, 80, 75, 90, 85],
        }
    )
    y = pd.Series([1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1], name="target")

    enc = WoEEncoder(variables=["city", "tier"])
    print("=== WoEEncoder (replace) ===")
    print(enc.fit_transform(df[["city", "tier", "revenue"]], y))

    enc_exp = WoEEncoderWithOriginal(variables=["city", "tier"])
    print("\n=== WoEEncoderWithOriginal (append) ===")
    print(enc_exp.fit_transform(df[["city", "tier", "revenue"]], y))

    pipe = Pipeline([("woe", WoEEncoderWithOriginal())])
    print("\n=== Inside sklearn Pipeline (auto-detect categoricals) ===")
    print(
        pipe.fit(df[["city", "tier", "revenue"]], y).transform(
            df[["city", "tier", "revenue"]]
        )
    )
