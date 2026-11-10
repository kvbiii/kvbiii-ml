from typing import cast

import pandas as pd
from feature_engine.encoding import RareLabelEncoder

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class RareLabelEncoderWithOriginal(_WithOriginalBase):
    """Wraps feature_engine RareLabelEncoder to keep originals and append consolidated copies.

    Derived columns are named ``{original}_PREPROCESS_RARE``. Categories whose frequency falls
    below ``tol`` are grouped into a single ``replace_with`` label (default ``"Rare"``).
    Original categorical columns are preserved so downstream steps can still access
    the uncollapsed labels.
    """

    _suffix = "PREPROCESS_RARE"

    def __init__(  # noqa: PLR0913
        self,
        tol: float = 0.05,
        n_categories: int = 10,
        max_n_categories: int | None = None,
        replace_with: str | int | float = "Rare",
        variables: list[str] | None = None,
        missing_values: str = "raise",
        ignore_format: bool = False,
    ) -> None:
        """
        Initialize RareLabelEncoderWithOriginal.

        Args:
            tol (float): Min relative frequency; rarer categories are grouped. Defaults to 0.05.
            n_categories (int): Min distinct categories before rare grouping. Defaults to 10.
            max_n_categories (int | None): Categories kept (excl. rare label). Defaults to no cap.
            replace_with (str | int | float): Label for rare categories. Defaults to "Rare".
            variables (list[str] | None): Categorical columns to encode. Defaults to auto-detect.
            missing_values (str): How to handle NaN - "raise" or "ignore". Defaults to "raise".
            ignore_format (bool, optional): If True, also encode numeric columns. Defaults to False.
        """
        self.tol = tol
        self.n_categories = n_categories
        self.max_n_categories = max_n_categories
        self.replace_with = replace_with
        self.variables = variables
        self.missing_values = missing_values
        self.ignore_format = ignore_format

    def _build_inner(self) -> RareLabelEncoder:
        """Return a fresh RareLabelEncoder configured from instance attributes.

        Returns:
            RareLabelEncoder: Unfitted encoder instance.
        """
        return RareLabelEncoder(
            tol=self.tol,
            n_categories=self.n_categories,
            max_n_categories=self.max_n_categories,
            replace_with=self.replace_with,
            variables=cast("list[str | int] | None", self.variables),
            missing_values=self.missing_values,
            ignore_format=self.ignore_format,
        )

    def _fit_inner(self, inner: RareLabelEncoder, X: pd.DataFrame, y: object) -> RareLabelEncoder:
        """Fit the encoder on X.

        Args:
            inner (RareLabelEncoder): Unfitted encoder.
            X (pd.DataFrame): Training features.
            y (object): Unused; accepted for API consistency.

        Returns:
            RareLabelEncoder: Fitted encoder.
        """
        return inner.fit(X)

    def _transform_inner(self, inner: RareLabelEncoder, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted encoder to X.

        Args:
            inner (RareLabelEncoder): Fitted encoder.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with rare categories consolidated.
        """
        return inner.transform(X)


__all__ = ["RareLabelEncoder", "RareLabelEncoderWithOriginal"]


if __name__ == "__main__":
    import pandas as pd
    from sklearn.pipeline import Pipeline

    df = pd.DataFrame(
        {
            "city": [
                "Warsaw",
                "Warsaw",
                "Warsaw",
                "Krakow",
                "Krakow",
                "Gdansk",
                "Poznan",
                "Wroclaw",
                "Lodz",
                "Warsaw",
            ],
            "tier": ["A", "A", "B", "A", "B", "C", "C", "D", "D", "A"],
            "revenue": [100, 120, 130, 200, 210, 80, 75, 60, 55, 115],
        }
    )

    enc = RareLabelEncoder(tol=0.15, n_categories=3, variables=["city", "tier"])
    print("=== RareLabelEncoder (replace) ===")
    print(enc.fit_transform(df[["city", "tier", "revenue"]]))
    print("Encoder mapping:", enc.encoder_dict_)

    enc_exp = RareLabelEncoderWithOriginal(tol=0.15, n_categories=3, variables=["city", "tier"])
    print("\n=== RareLabelEncoderWithOriginal (append) ===")
    print(enc_exp.fit_transform(df[["city", "tier", "revenue"]]))

    pipe = Pipeline([("rare", RareLabelEncoderWithOriginal(tol=0.2))])
    print("\n=== Inside sklearn Pipeline ===")
    print(pipe.fit_transform(df[["city", "tier", "revenue"]]))
