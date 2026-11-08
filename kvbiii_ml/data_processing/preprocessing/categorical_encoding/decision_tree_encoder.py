from typing import Any

import pandas as pd
from feature_engine.encoding import DecisionTreeEncoder

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class DecisionTreeEncoderWithOriginal(_WithOriginalBase):
    """Wraps feature_engine DecisionTreeEncoder to keep originals and append tree-encoded copies.

    Derived columns are named ``{original}_PREPROCESS_DT_ENC``. Each category is replaced by
    the prediction of a decision tree fitted on that variable vs. the target. The
    tree depth is tuned via cross-validation, capturing non-linear relationships
    between categories and the target. Original categorical columns are preserved.
    """

    _suffix = "PREPROCESS_DT_ENC"

    def __init__(
        self,
        encoding_method: str = "arbitrary",
        cv: int = 3,
        scoring: str = "neg_mean_squared_error",
        param_grid: dict | None = None,
        regression: bool = True,
        random_state: int | None = None,
        variables: list[str] | None = None,
        ignore_format: bool = False,
        precision: int | None = None,
        unseen: str = "ignore",
        fill_value: float | None = None,
    ) -> None:
        """
        Initialize DecisionTreeEncoderWithOriginal.

        Args:
            encoding_method (str): "arbitrary" uses the leaf prediction. Defaults to "arbitrary".
            cv (int): Number of cross-validation folds used to tune tree depth. Defaults to 3.
            scoring (str): sklearn scoring metric used in CV. Defaults to "neg_mean_squared_error".
            param_grid (dict | None): Tree parameter grid. Defaults to feature_engine's grid.
            regression (bool): Regression tree if True, else classification. Defaults to True.
            random_state (int | None, optional): Random seed for reproducibility. Defaults to None.
            variables (list[str] | None): Categorical columns to encode. Defaults to auto-detect.
            ignore_format (bool, optional): If True, also encode numeric columns. Defaults to False.
            precision (int | None): Decimal places to round encoded values. Defaults to no rounding.
            unseen (str): "ignore" or "raise" for unseen categories. Defaults to "ignore".
            fill_value (float | None): Unseen-category value when unseen="ignore". Defaults to None.
        """
        self.encoding_method = encoding_method
        self.cv = cv
        self.scoring = scoring
        self.param_grid = param_grid
        self.regression = regression
        self.random_state = random_state
        self.variables = variables
        self.ignore_format = ignore_format
        self.precision = precision
        self.unseen = unseen
        self.fill_value = fill_value

    def _build_inner(self) -> DecisionTreeEncoder:
        """Return a fresh DecisionTreeEncoder configured from instance attributes.

        Returns:
            DecisionTreeEncoder: Unfitted encoder instance.
        """
        return DecisionTreeEncoder(
            encoding_method=self.encoding_method,
            cv=self.cv,
            scoring=self.scoring,
            param_grid=self.param_grid,
            regression=self.regression,
            random_state=self.random_state,
            variables=self.variables,
            ignore_format=self.ignore_format,
            precision=self.precision,
            unseen=self.unseen,
            fill_value=self.fill_value,
        )

    def _fit_inner(
        self, inner: DecisionTreeEncoder, X: pd.DataFrame, y: Any
    ) -> DecisionTreeEncoder:
        """Fit the encoder on X and y.

        Args:
            inner (DecisionTreeEncoder): Unfitted encoder.
            X (pd.DataFrame): Training features.
            y (Any): Target used to fit the per-variable decision trees.

        Returns:
            DecisionTreeEncoder: Fitted encoder.
        """
        return inner.fit(X, y)

    def _transform_inner(self, inner: DecisionTreeEncoder, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted encoder to X.

        Args:
            inner (DecisionTreeEncoder): Fitted encoder.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with tree-encoded column values.
        """
        return inner.transform(X)


__all__ = ["DecisionTreeEncoder", "DecisionTreeEncoderWithOriginal"]


if __name__ == "__main__":
    import pandas as pd
    from sklearn.pipeline import Pipeline

    df = pd.DataFrame(
        {
            "city": [
                "Warsaw",
                "Krakow",
                "Warsaw",
                "Gdansk",
                "Krakow",
                "Warsaw",
                "Gdansk",
                "Krakow",
            ],
            "tier": ["A", "B", "A", "C", "B", "A", "C", "B"],
            "revenue": [100.0, 200.0, 150.0, 80.0, 220.0, 130.0, 90.0, 210.0],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 0, 0, 1], name="target")

    enc = DecisionTreeEncoder(variables=["city", "tier"], regression=False, cv=2, random_state=42)
    print("=== DecisionTreeEncoder (replace) ===")
    print(enc.fit_transform(df[["city", "tier", "revenue"]], y))

    enc_exp = DecisionTreeEncoderWithOriginal(
        variables=["city", "tier"], regression=False, cv=2, random_state=42
    )
    print("\n=== DecisionTreeEncoderWithOriginal (append) ===")
    print(enc_exp.fit_transform(df[["city", "tier", "revenue"]], y))

    pipe = Pipeline(
        [
            (
                "dt_enc",
                DecisionTreeEncoderWithOriginal(regression=False, cv=2, random_state=42),
            )
        ]
    )
    print("\n=== Inside sklearn Pipeline ===")
    print(pipe.fit(df[["city", "tier", "revenue"]], y).transform(df[["city", "tier", "revenue"]]))
