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

    def fit(self, X: pd.DataFrame, y: object | None = None) -> "CategoricalCleaner":
        """Store input feature names.

        Args:
            X (pd.DataFrame): Input feature matrix.
            y (object | None, optional): Ignored. Present for API compatibility.

        Returns:
            CategoricalCleaner: Fitted transformer.
        """
        self.input_features_ = X.columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace configured placeholder values with NaN and cast to category.

        Args:
            X (pd.DataFrame): Input feature matrix.

        Returns:
            pd.DataFrame: Transformed DataFrame with placeholders replaced and
            affected columns cast to category dtype.
        """
        X = X.copy()
        for fill_value, features in self.feature_groups.items():
            for feature in features:
                if feature in X.columns:
                    X[feature] = X[feature].astype("object")
                    X[feature] = X[feature].apply(lambda x: str(x) if pd.notna(x) else x)
                    X[feature] = X[feature].replace(fill_value, np.nan).infer_objects(copy=False)
                    X[feature] = X[feature].astype("category")
        return X

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> pd.Index:
        """Return feature names seen during fit.

        Args:
            input_features (list[str] | None, optional): Unused. For API consistency.

        Returns:
            pd.Index: Original feature names.
        """
        return self.input_features_


if __name__ == "__main__":
    import pandas as pd

    data = {
        "details_body_type": ["sedan", "hatchback", "Not Provided"],
        "details_color": ["red", "blue", "Not Provided"],
        "fuel": ["petrol", "diesel", "Unknown"],
        "model_code": ["A123", "B456", "Unknown"],
        "verified_car": [True, False, "False"],
    }
    df = pd.DataFrame(data)

    feature_groups = {
        "Not Provided": ["details_body_type", "details_color"],
        "Unknown": ["fuel", "model_code"],
        "False": ["verified_car"],
    }

    cleaner = CategoricalCleaner(feature_groups)
    cleaned_df = cleaner.fit_transform(df)
    print(cleaned_df.dtypes)
    print(cleaned_df)
