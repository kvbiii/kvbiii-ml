import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NumericalDowncaster(BaseEstimator, TransformerMixin):
    """Downcast numerical features to more efficient dtypes.

    For each feature in numerical_features, the transformer applies pandas
    to_numeric with downcast='integer' for integer types and downcast='float'
    for float types, reducing memory usage while preserving values.
    """

    def __init__(self, numerical_features: list[str]):
        """Initialize the downcaster.

        Args:
            numerical_features (list[str]): List of column names to downcast.
        """
        self.numerical_features = numerical_features

    def fit(self, X: pd.DataFrame, y: object | None = None) -> "NumericalDowncaster":
        """Store input feature names.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (object | None, optional): Ignored. Present for compatibility.

        Returns:
            NumericalDowncaster: Fitted transformer.
        """
        self.input_features_ = X.columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Downcast specified numerical features to smaller dtypes.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with downcasted numerical features.
        """
        X = X.copy()
        for col in self.numerical_features:
            if col in X.columns:
                col_type = X[col].dtype
                if pd.api.types.is_integer_dtype(col_type):
                    X[col] = pd.to_numeric(X[col], downcast="integer")
                elif pd.api.types.is_float_dtype(col_type):
                    X[col] = pd.to_numeric(X[col], downcast="float")
        return X

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> pd.Index:
        """Return feature names seen during fit.

        Args:
            input_features (list[str] | None, optional): Unused; for API symmetry.

        Returns:
            pd.Index: Original feature names.
        """
        return self.input_features_


if __name__ == "__main__":
    import numpy as np

    # Create sample DataFrame with various numerical types
    df_train = pd.DataFrame(
        {
            "price": np.array([10000, 15000, 12000, 18000], dtype=np.int64),
            "mileage": np.array([50000, 75000, 60000, 90000], dtype=np.int64),
            "rating": np.array([4.5, 3.8, 4.2, 4.9], dtype=np.float64),
            "efficiency": np.array([15.5, 12.3, 14.1, 16.8], dtype=np.float64),
            "category": ["A", "B", "A", "C"],
        }
    )

    df_test = pd.DataFrame(
        {
            "price": np.array([11000, 16000, 13000], dtype=np.int64),
            "mileage": np.array([55000, 80000, 65000], dtype=np.int64),
            "rating": np.array([4.3, 3.9, 4.6], dtype=np.float64),
            "efficiency": np.array([14.8, 13.1, 15.9], dtype=np.float64),
            "category": ["B", "A", "C"],
        }
    )

    print("Original dtypes:")
    print(df_train.dtypes)
    print(f"\nOriginal memory: {df_train.memory_usage(deep=True).sum() / 1024:.2f} KB")

    downcaster = NumericalDowncaster(["price", "mileage", "rating", "efficiency"]).fit(
        df_train
    )
    transformed_train = downcaster.transform(df_train)
    transformed_test = downcaster.transform(df_test)

    print("\nDowncasted dtypes:")
    print(transformed_train.dtypes)
    print(
        f"\nDowncasted memory: {transformed_train.memory_usage(deep=True).sum() / 1024:.2f} KB"
    )

    print("\nTransformed train data:")
    print(transformed_train)
    print("\nTransformed test data:")
    print(transformed_test)
