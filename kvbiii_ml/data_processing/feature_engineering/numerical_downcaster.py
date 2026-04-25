import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NumericalDowncaster(BaseEstimator, TransformerMixin):
    """
    Transformer to downcast numerical columns to smaller dtypes to reduce memory usage.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        int_downcast: bool = True,
        float_downcast: bool = True,
    ):
        """
        Initialize the NumericalDowncaster.

        Args:
            columns (list[str] | None, optional): List of column names to downcast.
                If None, all numeric columns will be downcasted. Defaults to None.
            int_downcast (bool, optional): Whether to downcast integer types. Defaults to True.
            float_downcast (bool, optional):
                Whether to downcast float types to float32. Defaults to True.
        """
        self.columns = columns
        self.int_downcast = int_downcast
        self.float_downcast = float_downcast
        self.dtype_map_ = None

    def fit(
        self, df: pd.DataFrame, _y: pd.Series | None = None
    ) -> "NumericalDowncaster":
        """
        Fit method that determines optimal dtypes for numerical columns.

        Args:
            df (pd.DataFrame): Training data.
            _y (pd.Series | None, optional): Target (unused). Defaults to None.

        Returns:
            NumericalDowncaster: Fitted transformer.
        """
        columns_to_process = self.columns if self.columns is not None else df.columns
        self.dtype_map_ = {
            col: self._get_optimal_dtype(df[col])
            for col in columns_to_process
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by downcasting numerical columns to smaller dtypes.

        Args:
            df (pd.DataFrame): Data to transform.

        Returns:
            pd.DataFrame: Transformed data with downcasted dtypes.
        """
        df = df.copy()
        for column, target_dtype in self.dtype_map_.items():
            if column in df.columns:
                try:
                    df[column] = df[column].astype(target_dtype)
                except (ValueError, TypeError) as e:
                    warnings.warn(
                        f"Failed to downcast column '{column}' to {target_dtype}: {e}",
                        UserWarning,
                        stacklevel=2,
                    )
        return df

    def _get_optimal_dtype(self, series: pd.Series) -> Any:
        """
        Determine optimal dtype for a numerical series with downcasting.

        Args:
            series (pd.Series): Series to analyze.

        Returns:
            Any: Optimal dtype for the series.
        """
        dtype = series.dtype

        if pd.api.types.is_integer_dtype(dtype) and self.int_downcast:
            return self._downcast_integer(series)
        if pd.api.types.is_float_dtype(dtype) and self.float_downcast:
            return self._downcast_float(series)
        return dtype

    def _downcast_integer(self, series: pd.Series) -> Any:
        """
        Downcast integer series to smallest possible signed integer type.

        Args:
            series (pd.Series): Integer series to downcast.

        Returns:
            Any: Optimal signed integer dtype.
        """
        if series.isna().any():
            return np.float32

        min_val, max_val = series.min(), series.max()

        return next(
            (
                dtype
                for dtype, info in [
                    (np.int8, np.iinfo(np.int8)),
                    (np.int16, np.iinfo(np.int16)),
                    (np.int32, np.iinfo(np.int32)),
                ]
                if info.min <= min_val and max_val <= info.max
            ),
            np.int64,
        )

    def _downcast_float(self, _series: pd.Series) -> Any:
        """
        Downcast float series to float32.

        Args:
            _series (pd.Series): Float series to downcast.

        Returns:
            Any: float32 dtype.
        """
        return np.float32


if __name__ == "__main__":
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
