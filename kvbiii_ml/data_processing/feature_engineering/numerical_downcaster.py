import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

FLOAT32_EXACT_INT_LIMIT = 2**24
FLOAT_PRECISIONS = ("lossless", "float32")
LOSSLESS_RTOL = 0.0
INTEGER_DTYPES = (np.int8, np.int16, np.int32)


class NumericalDowncaster(BaseEstimator, TransformerMixin):
    """
    Transformer to downcast numerical columns to smaller dtypes to reduce memory usage.

    Downcasting never changes a value by default: floats move to float32 only when
    every value survives the round trip, and integer columns with missing values
    become float32 only while float32 still represents every integer exactly
    (absolute value up to 2**24), float64 otherwise.
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        int_downcast: bool = True,
        float_downcast: bool = True,
        float_precision: str = "lossless",
        columns: list[str] | None = None,
    ):
        """
        Initialize the NumericalDowncaster.

        Args:
            variables (list[str] | None, optional): Column names to downcast. If None,
                all numeric columns are downcast. Named ``variables`` so
                ``build_restricted_pipeline`` can narrow it. Defaults to None.
            int_downcast (bool, optional): Whether to downcast integer types. Defaults to True.
            float_downcast (bool, optional): Whether to downcast float types. Defaults to True.
            float_precision (str, optional): ``"lossless"`` downcasts a float column to
                float32 only when ``np.allclose(s, s.astype(np.float32), rtol=0, atol=0,
                equal_nan=True)`` holds, i.e. no value changes, otherwise it stays as is
                (checked again on every transformed frame). ``"float32"`` always
                downcasts, accepting ~7 significant digits. Defaults to "lossless".
            columns (list[str] | None, optional): Deprecated alias of ``variables``,
                used only when ``variables`` is None. Defaults to None.
        """
        self.variables = variables
        self.int_downcast = int_downcast
        self.float_downcast = float_downcast
        self.float_precision = float_precision
        self.columns = columns

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

        Raises:
            ValueError: If float_precision is not "lossless" or "float32".
        """
        if self.float_precision not in FLOAT_PRECISIONS:
            raise ValueError(f"float_precision must be one of {FLOAT_PRECISIONS}.")
        variables = self.variables
        if variables is None and self.columns is not None:
            warnings.warn(
                "NumericalDowncaster(columns=...) is deprecated; use variables=... instead.",
                FutureWarning,
                stacklevel=2,
            )
            variables = self.columns
        columns_to_process = variables if variables is not None else df.columns
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

        Raises:
            NotFittedError: If called before fit.
        """
        check_is_fitted(self, "dtype_map_")
        df = df.copy()
        for column, target_dtype in self.dtype_map_.items():
            if column in df.columns:
                safe_dtype = self._get_safe_dtype(df[column], target_dtype)
                try:
                    df[column] = df[column].astype(safe_dtype)
                except (ValueError, TypeError) as e:
                    warnings.warn(
                        f"Failed to downcast column '{column}' to {safe_dtype}: {e}",
                        UserWarning,
                        stacklevel=2,
                    )
        return df

    def _get_safe_dtype(self, series: pd.Series, target_dtype: Any) -> Any:
        """
        Adjust a learned dtype so the series being transformed loses no value.

        An integer dtype becomes a float when the series has missing values (see
        `_nan_integer_dtype`); a float32 target in lossless mode is skipped when this
        series' values would not survive the cast.

        Args:
            series (pd.Series): Column data being transformed.
            target_dtype (Any): Dtype learned during fit for this column.

        Returns:
            Any: Dtype safe to cast the series to.
        """
        if pd.api.types.is_integer_dtype(target_dtype) and series.isna().any():
            return self._nan_integer_dtype(series)
        if (
            target_dtype is np.float32
            and self.float_precision == "lossless"
            and not self._is_float32_lossless(series)
        ):
            return series.dtype
        return target_dtype

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
            Any: Optimal signed integer dtype, or a float dtype when the (nullable)
                series has missing values.
        """
        if series.isna().any():
            return self._nan_integer_dtype(series)

        min_val, max_val = series.min(), series.max()
        return next(
            (
                dtype
                for dtype in INTEGER_DTYPES
                if np.iinfo(dtype).min <= min_val and max_val <= np.iinfo(dtype).max
            ),
            np.int64,
        )

    def _downcast_float(self, series: pd.Series) -> Any:
        """
        Downcast a float series to float32 when allowed by ``float_precision``.

        Args:
            series (pd.Series): Float series to downcast.

        Returns:
            Any: float32, or the original dtype when lossless mode would lose precision.
        """
        if self.float_precision == "float32" or self._is_float32_lossless(series):
            return np.float32
        return series.dtype

    @staticmethod
    def _is_float32_lossless(series: pd.Series) -> bool:
        """Whether every value of series survives a float32 round trip unchanged."""
        values = series.to_numpy(dtype=np.float64, na_value=np.nan)
        with np.errstate(over="ignore"):
            as_float32 = values.astype(np.float32)
        return bool(
            np.allclose(
                values, as_float32, rtol=LOSSLESS_RTOL, atol=0.0, equal_nan=True
            )
        )

    @staticmethod
    def _nan_integer_dtype(series: pd.Series) -> Any:
        """Float dtype that holds an integer series with NaN exactly (float32 up to 2**24)."""
        abs_max = series.abs().max()
        if pd.isna(abs_max) or abs_max <= FLOAT32_EXACT_INT_LIMIT:
            return np.float32
        return np.float64


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

    downcaster = NumericalDowncaster(
        variables=["price", "mileage", "rating", "efficiency"]
    ).fit(df_train)
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
