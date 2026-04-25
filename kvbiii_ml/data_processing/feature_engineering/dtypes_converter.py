import warnings
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DtypesConverter(BaseEstimator, TransformerMixin):
    """
    Transformer to convert DataFrame columns to specified dtypes.
    """

    def __init__(self, dtype_map: dict[str, Any] | None = None):
        """
        Initialize the DtypesConverter.

        Args:
            dtype_map (dict[str, Any] | None, optional):
                Dictionary mapping column names to target dtypes. If None,
                dtypes are learned during fit. Defaults to None.
        """
        self.dtype_map = dtype_map
        self.dtype_map_ = None

    def fit(self, df: pd.DataFrame, _y: pd.Series | None = None) -> "DtypesConverter":
        """
        Fit method that learns dtypes from training data.

        Args:
            df (pd.DataFrame): Training data.
            _y (pd.Series | None, optional): Target (unused). Defaults to None.

        Returns:
            DtypesConverter: Fitted transformer.
        """
        self.dtype_map_ = (
            self.dtype_map
            if self.dtype_map is not None
            else {col: df[col].dtype for col in df.columns}
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by converting columns to specified dtypes.

        Args:
            df (pd.DataFrame): Data to transform.

        Returns:
            pd.DataFrame: Transformed data with converted dtypes.
        """
        df = df.copy()
        for column, target_dtype in self.dtype_map_.items():
            if column not in df.columns:
                continue
            current_dtype = df[column].dtype
            if current_dtype != target_dtype:
                warnings.warn(
                    (
                        f"\n[{self.__class__.__name__}.transform]\n"
                        f"  Column '{column}':\n"
                        f"  Current dtype '{current_dtype}' differs from "
                        f"expected dtype '{target_dtype}'.\n"
                        f"  Converting to '{target_dtype}'."
                    ),
                    UserWarning,
                    stacklevel=2,
                )
                try:
                    df[column] = df[column].astype(target_dtype)
                except (ValueError, TypeError) as e:
                    warnings.warn(
                        f"Failed to convert column '{column}' to {target_dtype}: {e}",
                        UserWarning,
                        stacklevel=2,
                    )
        return df


if __name__ == "__main__":
    df_train = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1.5, 2.5, 3.5],
            "C": ["a", "b", "c"],
        }
    )

    demo_dtype_map = {
        "A": "float32",
        "B": "float64",
        "C": "category",
    }

    converter = DtypesConverter(dtype_map=demo_dtype_map)
    converter.fit(df_train)

    df_transformed = converter.transform(df_train)

    print("Original dtypes:")
    print(df_train.dtypes)
    print("\nTransformed dtypes:")
    print(df_transformed.dtypes)
