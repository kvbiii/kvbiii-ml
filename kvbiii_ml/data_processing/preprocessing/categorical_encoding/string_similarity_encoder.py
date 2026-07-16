from typing import Any

import pandas as pd
from feature_engine.encoding import StringSimilarityEncoder

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class StringSimilarityEncoderWithOriginal(_WithOriginalBase):
    """Wraps feature_engine StringSimilarityEncoder to keep originals and append similarity columns.

    Unlike the other ``WithOriginal`` wrappers, ``StringSimilarityEncoder`` expands
    each input variable into one column per unique string seen during training
    (e.g. ``city`` → ``city_Warsaw``, ``city_Krakow``, …). Because this is a
    1-to-many mapping, ``transform`` is overridden to concatenate all generated
    columns onto the original DataFrame rather than relying on the base-class
    1-to-1 column renaming logic.

    ``_suffix`` is defined for API consistency but is not used in column naming -
    feature_engine applies its own ``{variable}_{string}`` convention.
    """

    _suffix = "PREPROCESS_STR_SIM"
    _suppress_warnings = True

    def __init__(
        self,
        top_categories: int | None = None,
        keywords: dict | None = None,
        missing_values: str = "impute",
        variables: list[str | int] | None = None,
        ignore_format: bool = False,
    ) -> None:
        """
        Initialize StringSimilarityEncoderWithOriginal.

        Args:
            top_categories (int | None, optional): Keep only the top N most frequent
                strings per variable. Defaults to None (keep all).
            keywords (dict | None, optional): Per-variable keyword overrides passed
                to the inner encoder. Defaults to None.
            missing_values (str, optional): How to handle NaN - ``"impute"`` replaces
                with an empty string; ``"raise"`` raises an error.
                Defaults to ``"impute"``.
            variables (list[str | int] | None, optional): Columns to encode.
                Defaults to None (auto-detect all object/categorical columns).
            ignore_format (bool, optional): If True, also encode numeric columns.
                Defaults to False.
        """
        self.top_categories = top_categories
        self.keywords = keywords
        self.missing_values = missing_values
        self.variables = variables
        self.ignore_format = ignore_format

    def _build_inner(self) -> StringSimilarityEncoder:
        """Return a fresh StringSimilarityEncoder configured from instance attributes.

        Returns:
            StringSimilarityEncoder: Unfitted encoder instance.
        """
        return StringSimilarityEncoder(
            top_categories=self.top_categories,
            keywords=self.keywords,
            missing_values=self.missing_values,
            variables=self.variables,
            ignore_format=self.ignore_format,
        )

    def _fit_inner(
        self, inner: StringSimilarityEncoder, X: pd.DataFrame, y: Any
    ) -> StringSimilarityEncoder:
        """Fit the encoder on X.

        Args:
            inner (StringSimilarityEncoder): Unfitted encoder.
            X (pd.DataFrame): Training features.
            y (Any): Unused; accepted for API consistency.

        Returns:
            StringSimilarityEncoder: Fitted encoder.
        """
        return inner.fit(X)

    def _transform_inner(
        self, inner: StringSimilarityEncoder, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply the fitted encoder to X.

        Args:
            inner (StringSimilarityEncoder): Fitted encoder.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with original columns replaced by similarity columns.
        """
        return inner.transform(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append string-similarity columns to X, preserving all original columns.

        Overrides the base-class transform because ``StringSimilarityEncoder``
        produces multiple output columns per input variable. Generated columns follow
        feature_engine's naming convention: ``{variable}_{string}``.

        Args:
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: Original columns plus one column per unique string per
                encoded variable.
        """
        x_orig = X.copy()
        with self._maybe_suppress_warnings():
            x_transformed = self._transform_inner(self._inner, X.copy())
        new_cols = [c for c in x_transformed.columns if c not in x_orig.columns]
        return pd.concat([x_orig, x_transformed[new_cols]], axis=1)

    def get_feature_names_out(self) -> list[str]:
        """Return output column names produced by transform().

        Returns:
            list[str]: All column names in the transformed DataFrame.
        """
        return self._inner.get_feature_names_out()


__all__ = ["StringSimilarityEncoder", "StringSimilarityEncoderWithOriginal"]


if __name__ == "__main__":
    import pandas as pd
    from sklearn.pipeline import Pipeline

    df = pd.DataFrame(
        {
            "product": [
                "apple",
                "orange",
                "apple_juice",
                "orange_juice",
                "apple",
                "grape",
            ],
            "brand": [
                "FreshCo",
                "FreshCo",
                "JuiceLab",
                "JuiceLab",
                "OrganicFarm",
                "FreshCo",
            ],
            "price": [1.5, 2.0, 3.5, 4.0, 1.8, 2.5],
        }
    )

    enc = StringSimilarityEncoder(variables=["product", "brand"])
    print("=== StringSimilarityEncoder (replace) ===")
    print(enc.fit_transform(df[["product", "brand", "price"]]))

    enc_exp = StringSimilarityEncoderWithOriginal(
        variables=["product", "brand"], top_categories=3
    )
    print("\n=== StringSimilarityEncoderWithOriginal (append) ===")
    print(enc_exp.fit_transform(df[["product", "brand", "price"]]))

    pipe = Pipeline(
        [("str_sim", StringSimilarityEncoderWithOriginal(top_categories=2))]
    )
    print("\n=== Inside sklearn Pipeline ===")
    print(pipe.fit_transform(df[["product", "brand", "price"]]))
