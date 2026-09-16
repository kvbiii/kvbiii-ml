from typing import Any

import pandas as pd
from feature_engine.encoding import MeanEncoder

from kvbiii_ml.data_processing.preprocessing.expansion_base import (
    _WithOriginalBase,
    _cast_numeric_variables_to_object,
)


class MeanEncoderWithOriginal(_WithOriginalBase):
    """Wraps feature_engine MeanEncoder to keep originals and append mean-encoded copies.

    Derived columns are named ``{original}_PREPROCESS_MEAN_ENC``. Each category is replaced
    by the mean of the target within that category (with optional Bayesian smoothing).
    Original categorical columns are preserved alongside the encoded ones. Any
    explicitly-named numeric column in ``variables`` (e.g. an already-coded,
    low-cardinality feature) is internally recast to object dtype before fitting/
    transforming, since feature_engine's MeanEncoder otherwise rejects numeric
    input with ``"Some of the variables are not categorical..."`` unless
    ``ignore_format=True`` is set. This keeps that default-safe behavior for
    ``variables=None`` auto-detection while still letting a numeric column be
    used directly when named explicitly. Defaults to ``unseen="encode"`` so a
    category seen only at transform time is encoded with the fitted prior
    (global target mean) instead of silently becoming NaN with a
    ``UserWarning``.
    """

    _suffix = "PREPROCESS_MEAN_ENC"
    _suppress_warnings = True

    def __init__(
        self,
        variables: list[str] | None = None,
        missing_values: str = "raise",
        ignore_format: bool = False,
        unseen: str = "encode",
        smoothing: int | float | str = 0.0,
    ) -> None:
        """
        Initialize MeanEncoderWithOriginal.

        Args:
            variables (list[str] | None, optional): Columns to encode. May include
                numeric columns - they are recast to object dtype internally so
                they encode without error. Defaults to None (auto-detect all
                object/categorical columns).
            missing_values (str, optional): How to handle NaN - ``"raise"`` or
                ``"ignore"``. Defaults to ``"raise"``.
            ignore_format (bool, optional): If True, also encode numeric columns.
                Defaults to False.
            unseen (str, optional): Strategy for categories seen at transform time
                but not during fit - ``"encode"`` replaces them with the fitted
                prior (global target mean), ``"ignore"`` replaces them with NaN
                and emits a UserWarning, and ``"raise"`` raises a ValueError.
                Defaults to ``"encode"`` so unseen categories never silently
                introduce NaN downstream.
            smoothing (int | float | str, optional): Bayesian shrinkage factor.
                ``0.0`` disables smoothing; higher values pull category means toward
                the global mean. Defaults to ``0.0``.
        """
        self.variables = variables
        self.missing_values = missing_values
        self.ignore_format = ignore_format
        self.unseen = unseen
        self.smoothing = smoothing

    def _build_inner(self) -> MeanEncoder:
        """Return a fresh MeanEncoder configured from instance attributes.

        Returns:
            MeanEncoder: Unfitted encoder instance.
        """
        return MeanEncoder(
            variables=self.variables,
            missing_values=self.missing_values,
            ignore_format=self.ignore_format,
            unseen=self.unseen,
            smoothing=self.smoothing,
        )

    def _fit_inner(self, inner: MeanEncoder, X: pd.DataFrame, y: Any) -> MeanEncoder:
        """Fit the encoder on X and y.

        Args:
            inner (MeanEncoder): Unfitted encoder.
            X (pd.DataFrame): Training features.
            y (Any): Target used to compute per-category mean values.

        Returns:
            MeanEncoder: Fitted encoder.
        """
        with self._suppress_fe_datetime_warnings():
            return inner.fit(_cast_numeric_variables_to_object(X, self.variables), y)

    def _transform_inner(self, inner: MeanEncoder, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted encoder to X.

        Args:
            inner (MeanEncoder): Fitted encoder.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with mean-encoded column values.
        """
        with self._suppress_fe_datetime_warnings():
            return inner.transform(_cast_numeric_variables_to_object(X, self.variables))


__all__ = ["MeanEncoder", "MeanEncoderWithOriginal"]


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
    y = pd.Series([1, 0, 1, 0, 1, 0], name="target")

    enc = MeanEncoder(variables=["city", "tier"])
    print("=== MeanEncoder (replace) ===")
    print(enc.fit_transform(df[["city", "tier", "revenue"]], y))

    enc_exp = MeanEncoderWithOriginal(variables=["city", "tier"])
    print("\n=== MeanEncoderWithOriginal (append) ===")
    print(enc_exp.fit_transform(df[["city", "tier", "revenue"]], y))

    pipe = Pipeline(
        [("enc", MeanEncoderWithOriginal(variables=["city", "tier"], smoothing=1.0))]
    )
    print("\n=== Inside sklearn Pipeline ===")
    print(
        pipe.fit(df[["city", "tier", "revenue"]], y).transform(
            df[["city", "tier", "revenue"]]
        )
    )
