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
        categorical_features: list[str],
        fill_values: dict[str, str] | None = None,
        warn_on_unknown: bool = True,
    ):
        """
        Initialize the CategoricalAligner.

        Args:
            categorical_features (list[str]): List of categorical features names.
            fill_values (dict[str, str] | None, optional): Dictionary mapping feature names to custom fill values for unknown categories. Defaults to None.
            warn_on_unknown (bool, optional): Whether to raise warnings when unknown categories are found. Defaults to True.
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
        return {
            feature: custom_fills.get(feature, default_value)
            for feature in categorical_features
            if feature in df.columns
        }

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CategoricalAligner":
        """
        Fit the transformer by learning categories and modes from training data.

        Args:
            X (pd.DataFrame): Training data.
            y (pd.Series | None, optional): Target (unused). Defaults to None.

        Returns:
            CategoricalAligner: Fitted transformer.
        """
        X = X.copy()
        for feature in self.categorical_features:
            if feature not in X.columns:
                continue
            mode_series = X[feature].mode(dropna=True)
            self.modes_[feature] = (
                str(mode_series[0]) if not mode_series.empty else "Brak"
            )
            fill_value = self._get_fill_value(feature)
            X[feature] = X[feature].fillna(fill_value).astype("str").astype("category")
            self.categories_[feature] = X[feature].cat.categories.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by aligning categories with fitted training categories.

        Args:
            X (pd.DataFrame): Data to transform.

        Returns:
            pd.DataFrame: Transformed data with aligned categories.
        """
        X = X.copy()
        for feature, categories in self.categories_.items():
            if feature not in X.columns:
                continue
            fill_value = self._get_fill_value(feature)
            X[feature] = X[feature].fillna(fill_value).astype("str")
            unknown_mask = ~X[feature].isin(categories)
            unknown_count = unknown_mask.sum()
            if self.warn_on_unknown and unknown_count > 0:
                unknown_values = X.loc[unknown_mask, feature].unique()
                n_unique_unknown = len(unknown_values)
                warnings.warn(
                    (
                        f"\n[{self.__class__.__name__}.transform]\n"
                        f"  Feature '{feature}':\n"
                        f"  Found {unknown_count} rows ({unknown_count/len(X)*100:.2f}%) with {n_unique_unknown} unknown categor{'y' if n_unique_unknown == 1 else 'ies'} not seen during training.\n"
                        f"  Unknown values: {list(unknown_values[:5])}{'...' if n_unique_unknown > 5 else ''}\n"
                        f"  Replacing with '{fill_value}'."
                    ),
                    UserWarning,
                    stacklevel=0,
                )
            X[feature] = X[feature].where(~unknown_mask, fill_value)
            X[feature] = X[feature].astype("category").cat.set_categories(categories)
        return X

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
    # Minimal example
    df_train = pd.DataFrame(
        {
            "color": pd.Categorical(["red", "blue", "green", "red"]),
            "fuel": pd.Categorical(["petrol", "diesel", "electric", "diesel"]),
            "mileage": [10000, 15000, 12000, 18000],
        }
    )
    df_test = pd.DataFrame(
        {
            "color": ["blue", "purple", "yellow"],  # purple, yellow unseen
            "fuel": ["diesel", "hydrogen", "petrol"],  # hydrogen unseen
            "mileage": [13000, 17000, 9000],
        }
    )

    assigner = CategoricalAligner(["color", "fuel"]).fit(df_train)
    transformed = assigner.transform(df_test)
    print(transformed)
    print(transformed.dtypes)
    print(df_train["color"].unique())
    print(transformed["color"].unique())
