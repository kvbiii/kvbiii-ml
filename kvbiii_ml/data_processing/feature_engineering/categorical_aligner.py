import warnings

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalAligner(BaseEstimator, TransformerMixin):
    """
    Transformer to align categorical features between train and test sets.
    Stores train categories and applies them to test data during transform.
    """

    def __init__(
        self,
        categorical_features: list[str] | None = None,
        fill_values: dict[str, str] | None = None,
        warn_on_unknown: bool = True,
    ):
        """
        Initialize the CategoricalAligner.

        Args:
            categorical_features (list[str] | None, optional):
                List of categorical features names. If None, auto-detects object
                and category columns during fit. Defaults to None.
            fill_values (dict[str, str] | None, optional):
                Feature-specific fill values for unknown categories. Defaults to None.
            warn_on_unknown (bool, optional):
                Whether to raise warnings when unknown categories are found.
                Defaults to True.
        """
        self.categorical_features = categorical_features
        self.fill_values = fill_values or {}
        self.warn_on_unknown = warn_on_unknown
        self.categories_ = {}
        self.modes_ = {}

    @classmethod
    def create_fill_values(
        cls,
        df: pd.DataFrame,
        categorical_features: list[str] | None = None,
        custom_fills: dict[str, str] | None = None,
        default_value: str = "mode",
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
                Use "mode" for mode-based filling or any string value. Defaults to "mode".

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
        """
        df = df.copy()
        if self.categorical_features is None:
            self.categorical_features = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        for feature in self.categorical_features:
            if feature not in df.columns:
                continue
            mode_series = df[feature].mode(dropna=True)
            self.modes_[feature] = (
                str(mode_series[0]) if not mode_series.empty else "Unknown"
            )
            fill_value = self._get_fill_value(feature)
            df[feature] = (
                df[feature].fillna(fill_value).astype("str").astype("category")
            )
            self.categories_[feature] = df[feature].cat.categories.tolist()
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
            df[feature] = df[feature].fillna(fill_value).astype("str")
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

    def _get_fill_value(self, feature: str) -> str:
        """
        Get the appropriate fill value for a feature.

        Args:
            feature (str): feature name.

        Returns:
            str: Fill value to use for unknown categories.
        """
        if feature in self.fill_values:
            return self.fill_values[feature]
        return self.modes_[feature]


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

    assigner = CategoricalAligner(["color", "fuel"]).fit(df_train)
    transformed = assigner.transform(df_test)
    print(transformed)
    print(transformed.dtypes)
    print(df_train["color"].unique())
    print(transformed["color"].unique())
