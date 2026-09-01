import warnings
from typing import Literal

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_FILL_VALUE = "Unknown"
FILL_STRATEGIES = ("mode", "constant")


class CategoricalAligner(BaseEstimator, TransformerMixin):
    """
    Transformer to align categorical features between train and test sets.
    Stores train categories and applies them to test data during transform.
    """

    def __init__(
        self,
        categorical_features: list[str] | None = None,
        fill_values: dict[str, str] | None = None,
        fill_strategy: Literal["mode", "constant"] = "constant",
        default_value: str = DEFAULT_FILL_VALUE,
        warn_on_unknown: bool = True,
    ):
        """
        Initialize the CategoricalAligner.

        Args:
            categorical_features (list[str] | None, optional):
                List of categorical features names. If None, auto-detects object
                and category columns during fit. Defaults to None.
            fill_values (dict[str, str] | None, optional):
                Feature-specific fill values for unknown categories, keyed by feature
                name (e.g. {"color": "Puste"}). Features listed here always use their
                configured value, regardless of fill_strategy. Defaults to None.
            fill_strategy (Literal["mode", "constant"], optional):
                Strategy used for features not present in fill_values. "mode" fills
                with each feature's training-data mode; "constant" fills with
                default_value. Defaults to "constant".
            default_value (str, optional):
                Constant fill value used by the "constant" fill_strategy, and as the
                fallback for "mode" when a feature has no observed mode (e.g. all-NaN
                training column). Defaults to "Unknown".
            warn_on_unknown (bool, optional):
                Whether to raise warnings when unknown categories are found.
                Defaults to True.
        """
        self.categorical_features = categorical_features
        self.fill_values = fill_values
        self.fill_strategy = fill_strategy
        self.default_value = default_value
        self.warn_on_unknown = warn_on_unknown
        self.categories_ = {}
        self.modes_ = {}

    @classmethod
    def create_fill_values(
        cls,
        df: pd.DataFrame,
        categorical_features: list[str] | None = None,
        custom_fills: dict[str, str] | None = None,
        default_value: str = DEFAULT_FILL_VALUE,
    ) -> dict[str, str]:
        """
        Helper method to create fill_values dictionary for categorical features.

        Args:
            df (pd.DataFrame): DataFrame to analyze.
            categorical_features (list[str] | None, optional): Categorical features.
                If None, auto-detects category dtype features. Defaults to None.
            custom_fills (dict[str, str] | None, optional): Custom fill values for specific
                features. Defaults to None.
            default_value (str, optional): Default fill value for features not in custom_fills.
                Defaults to "Unknown".

        Returns:
            dict[str, str]: Dictionary mapping feature names to fill values.
        """
        custom_fills = custom_fills or {}
        categorical_features = categorical_features or []
        return {
            feature: custom_fills.get(feature, default_value)
            for feature in categorical_features
            if feature in df.columns
        }

    def fit(
        self, df: pd.DataFrame, _y: pd.Series | None = None
    ) -> "CategoricalAligner":
        """
        Fit the transformer by learning categories and modes from training data.

        Args:
            df (pd.DataFrame): Training data.
            _y (pd.Series | None, optional): Target (unused). Defaults to None.

        Returns:
            CategoricalAligner: Fitted transformer.

        Raises:
            ValueError: If fill_strategy is not one of "mode" or "constant".
        """
        if self.fill_strategy not in FILL_STRATEGIES:
            raise ValueError(
                f"fill_strategy must be one of {FILL_STRATEGIES}, got "
                f"'{self.fill_strategy}'."
            )
        df = df.copy()
        if self.categorical_features is None:
            self.categorical_features = df.select_dtypes(
                include=["object", "string", "category"]
            ).columns.tolist()
        for feature in self.categorical_features:
            if feature not in df.columns:
                continue
            mode_series = df[feature].mode(dropna=True)
            self.modes_[feature] = (
                str(mode_series[0]) if not mode_series.empty else self.default_value
            )
            fill_value = self._get_fill_value(feature)
            df[feature] = self._fillna_as_str(df[feature], fill_value).astype(
                "category"
            )
            cats = df[feature].cat.categories.tolist()
            if fill_value not in cats:
                cats = sorted(cats + [fill_value])
            self.categories_[feature] = cats
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by aligning categories with fitted training categories.

        Args:
            df (pd.DataFrame): Data to transform.

        Returns:
            pd.DataFrame: Transformed data with aligned categories.
        """
        df = df.copy()
        for feature, categories in self.categories_.items():
            if feature not in df.columns:
                continue
            fill_value = self._get_fill_value(feature)
            df[feature] = self._fillna_as_str(df[feature], fill_value)
            unknown_mask = ~df[feature].isin(categories)
            unknown_count = unknown_mask.sum()
            if self.warn_on_unknown and unknown_count > 0:
                unknown_values = df.loc[unknown_mask, feature].unique()
                n_unique_unknown = len(unknown_values)
                pct_unknown = unknown_count / len(df) * 100
                unknown_preview = list(unknown_values[:5])
                suffix = "..." if n_unique_unknown > 5 else ""
                category_word = "y" if n_unique_unknown == 1 else "ies"
                warnings.warn(
                    (
                        f"\n[{self.__class__.__name__}.transform]\n"
                        f"  Feature '{feature}':\n"
                        f"  Found {unknown_count} rows ({pct_unknown:.2f}%) "
                        f"with {n_unique_unknown} unknown categor{category_word} "
                        "not seen during training.\n"
                        f"  Unknown values: {unknown_preview}{suffix}\n"
                        f"  Replacing with '{fill_value}'."
                    ),
                    UserWarning,
                    stacklevel=0,
                )
            df[feature] = df[feature].where(~unknown_mask, fill_value)
            df[feature] = df[feature].astype("category").cat.set_categories(categories)
        return df

    def _fillna_as_str(self, series: pd.Series, fill_value: str) -> pd.Series:
        """Fill NaNs with fill_value and cast to string dtype.

        Adds fill_value as a category first when series is already categorical,
        since pandas forbids filling a Categorical with a value outside its
        existing categories.

        Args:
            series (pd.Series): Series to fill, possibly of categorical dtype.
            fill_value (str): Value to replace NaNs with.

        Returns:
            pd.Series: String-dtype series with NaNs replaced by fill_value.
        """
        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            and fill_value not in series.cat.categories
        ):
            series = series.cat.add_categories([fill_value])
        return series.fillna(fill_value).astype("str")

    def _get_fill_value(self, feature: str) -> str:
        """Get the fill value for a feature.

        Args:
            feature (str): Feature name.

        Returns:
            str: Fill value to use for unknown categories.
        """
        fill_values = self.fill_values or {}
        if feature in fill_values:
            return fill_values[feature]
        if self.fill_strategy == "mode":
            return self.modes_.get(feature, self.default_value)
        return self.default_value


if __name__ == "__main__":
    df_train = pd.DataFrame(
        {
            "color": pd.Categorical(["red", "blue", "green", "red"]),
            "fuel": pd.Categorical(["petrol", "diesel", "electric", "diesel"]),
            "mileage": [10000, 15000, 12000, 18000],
        }
    )
    df_test = pd.DataFrame(
        {
            "color": ["blue", "purple", "yellow"],
            "fuel": ["diesel", "hydrogen", "petrol"],
            "mileage": [13000, 17000, 9000],
        }
    )

    categorical_aligner = CategoricalAligner(fill_values={"color": "Unknown"})
    categorical_aligner.fit(df_train)

    print("--- test transform ---")
    transformed_test = categorical_aligner.transform(df_test)
    print(transformed_test)
    print(transformed_test["color"].unique())

    print("\n--- train transform (fill category present even without unknowns) ---")
    transformed_train = categorical_aligner.transform(df_train)
    print(transformed_train)
    print(transformed_train["color"].unique())
