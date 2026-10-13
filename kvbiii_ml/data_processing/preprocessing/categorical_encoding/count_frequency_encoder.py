from typing import Any

import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder

from kvbiii_ml.data_processing.preprocessing.expansion_base import (
    _WithOriginalBase,
    _cast_numeric_variables_to_object,
)


class CountFrequencyEncoderWithOriginal(_WithOriginalBase):
    """Wraps feature_engine CountFrequencyEncoder to keep originals and append encoded copies.

    Derived columns are named ``{original}_PREPROCESS_CNT_FREQ``. The original categorical
    columns are preserved so downstream steps can still access the raw labels. Any
    explicitly-named numeric column in ``variables`` (e.g. an already-coded,
    low-cardinality feature) is internally recast to object dtype before fitting/
    transforming, since feature_engine's CountFrequencyEncoder otherwise rejects
    numeric input with ``"Some of the variables are not categorical..."`` unless
    ``ignore_format=True`` is set. This keeps that default-safe behavior for
    ``variables=None`` auto-detection while still letting a numeric column be
    used directly when named explicitly. Defaults to ``unseen="encode"`` so a
    category seen only at transform time is encoded as 0 (zero) instead of
    silently becoming NaN with a ``UserWarning``.
    """

    _suffix = "PREPROCESS_CNT_FREQ"

    def __init__(
        self,
        encoding_method: str = "count",
        variables: list[str] | None = None,
        missing_values: str = "raise",
        ignore_format: bool = False,
        unseen: str = "encode",
    ) -> None:
        """
        Initialize CountFrequencyEncoderWithOriginal.

        Args:
            encoding_method (str, optional): ``"count"`` replaces each category with
                its absolute count; ``"frequency"`` uses the relative frequency.
                Defaults to ``"count"``.
            variables (list[str] | None, optional): Columns to encode. May include
                numeric columns - they are recast to object dtype internally so
                they encode without error. Defaults to None (auto-detect all
                object/categorical columns).
            missing_values (str, optional): How to handle NaN - ``"raise"`` or
                ``"ignore"``. Defaults to ``"raise"``.
            ignore_format (bool, optional): If True, also encode numeric columns.
                Defaults to False.
            unseen (str, optional): Strategy for categories seen at transform time
                but not during fit - ``"encode"`` replaces them with 0 (zero),
                ``"ignore"`` replaces them with NaN and emits a UserWarning, and
                ``"raise"`` raises a ValueError. Defaults to ``"encode"`` so unseen
                categories never silently introduce NaN downstream.
        """
        self.encoding_method = encoding_method
        self.variables = variables
        self.missing_values = missing_values
        self.ignore_format = ignore_format
        self.unseen = unseen

    def _build_inner(self) -> CountFrequencyEncoder:
        """Return a fresh CountFrequencyEncoder configured from instance attributes.

        Returns:
            CountFrequencyEncoder: Unfitted encoder instance.
        """
        return CountFrequencyEncoder(
            encoding_method=self.encoding_method,
            variables=self.variables,
            missing_values=self.missing_values,
            ignore_format=self.ignore_format,
            unseen=self.unseen,
        )

    def _fit_inner(
        self, inner: CountFrequencyEncoder, X: pd.DataFrame, y: Any
    ) -> CountFrequencyEncoder:
        """Fit the encoder on X.

        Args:
            inner (CountFrequencyEncoder): Unfitted encoder.
            X (pd.DataFrame): Training features.
            y (Any): Unused; accepted for API consistency.

        Returns:
            CountFrequencyEncoder: Fitted encoder.
        """
        with self._suppress_fe_datetime_warnings():
            return inner.fit(_cast_numeric_variables_to_object(X, self.variables), y)

    def _transform_inner(
        self, inner: CountFrequencyEncoder, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply the fitted encoder to X.

        Args:
            inner (CountFrequencyEncoder): Fitted encoder.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with encoded column values.
        """
        with self._suppress_fe_datetime_warnings():
            return inner.transform(_cast_numeric_variables_to_object(X, self.variables))


__all__ = ["CountFrequencyEncoder", "CountFrequencyEncoderWithOriginal"]


if __name__ == "__main__":
    import pandas as pd
    from sklearn.pipeline import Pipeline

    df = pd.DataFrame(
        {
            "city": ["Warsaw", "Krakow", "Warsaw", "Gdansk", "Krakow", "Warsaw"],
            "tier": ["A", "B", "A", "C", "B", "A"],
            "revenue": [100, 200, 150, 80, 220, 130],
        }
    )

    enc = CountFrequencyEncoder(encoding_method="frequency", variables=["city", "tier"])
    print("=== CountFrequencyEncoder (replace) ===")
    print(enc.fit_transform(df[["city", "tier", "revenue"]]))

    enc_exp = CountFrequencyEncoderWithOriginal(
        encoding_method="frequency", variables=["city", "tier"]
    )
    print("\n=== CountFrequencyEncoderWithOriginal (append) ===")
    print(enc_exp.fit_transform(df[["city", "tier", "revenue"]]))

    pipe = Pipeline(
        [("enc", CountFrequencyEncoderWithOriginal(encoding_method="count"))]
    )
    print("\n=== Inside sklearn Pipeline ===")
    print(pipe.fit_transform(df[["city", "tier", "revenue"]]))
