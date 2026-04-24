import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalCleaner(BaseEstimator, TransformerMixin):
    """Replace designated placeholder values with NaN and cast columns to category.

    The transformer receives a mapping of placeholder value -> list of columns. For each
    listed column present in the input DataFrame, occurrences of the placeholder value
    are replaced with NaN and the column is converted to pandas 'category' dtype.
    """

    def __init__(self, feature_groups: dict[str, list[str]]):
        """Initialize cleaner with placeholder mappings.

        Args:
            feature_groups (dict[str, list[str]]): Mapping of placeholder value to list of
                column names where that value should be treated as missing.
        """
        self.feature_groups = feature_groups
        self.input_features_: pd.Index = pd.Index([])

    def fit(self, df: pd.DataFrame, _y: object | None = None) -> "CategoricalCleaner":
        """Store input feature names.

        Args:
            df (pd.DataFrame): Input feature matrix.
            _y (object | None, optional): Ignored. Present for API compatibility.

        Returns:
            CategoricalCleaner: Fitted transformer.
        """
        self.input_features_ = df.columns
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace configured placeholder values with NaN and cast to category.

        Args:
            df (pd.DataFrame): Input feature matrix.

        Returns:
            pd.DataFrame: Transformed DataFrame with placeholders replaced and
            affected columns cast to category dtype.
        """
        df = df.copy()
        for fill_value, features in self.feature_groups.items():
            for feature in features:
                if feature in df.columns:
                    df[feature] = df[feature].astype("object")
                    df[feature] = df[feature].apply(
                        lambda x: str(x) if pd.notna(x) else x
                    )
                    df[feature] = (
                        df[feature]
                        .replace(fill_value, np.nan)
                        .infer_objects(copy=False)
                    )
                    df[feature] = df[feature].astype("category")
        return df

    def get_feature_names_out(
        self, _input_features: list[str] | None = None
    ) -> pd.Index:
        """Return feature names seen during fit.

        Args:
            _input_features (list[str] | None, optional): Unused. For API consistency.

        Returns:
            pd.Index: Original feature names.
        """
        return self.input_features_


if __name__ == "__main__":
    data = {
        "details_body_type": ["sedan", "hatchback", "Not Provided"],
        "details_color": ["red", "blue", "Not Provided"],
        "fuel": ["petrol", "diesel", "Unknown"],
        "model_code": ["A123", "B456", "Unknown"],
        "verified_car": [True, False, "False"],
    }
    demo_df = pd.DataFrame(data)

    demo_feature_groups = {
        "Not Provided": ["details_body_type", "details_color"],
        "Unknown": ["fuel", "model_code"],
        "False": ["verified_car"],
    }

    cleaner = CategoricalCleaner(demo_feature_groups)
    cleaned_df = cleaner.fit_transform(demo_df)
    print(cleaned_df.dtypes)
    print(cleaned_df)
