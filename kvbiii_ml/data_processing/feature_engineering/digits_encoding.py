import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DigitsEncodingFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Extracts individual digits from numeric features.

    From the number 123.45:
    - Position 2 extracts '1' (hundreds place)
    - Position 1 extracts '2' (tens place)
    - Position 0 extracts '3' (ones place)
    - Position -1 extracts '4' (first decimal place)
    - Position -2 extracts '5' (second decimal place)

    The relevant digit positions per feature are determined automatically from
    the min/max values seen during fitting.
    """

    def __init__(
        self,
        features_names: list[str] | None = None,
        fill_value: int = -1,
        dtype: str = "int8",
        min_digits: int = 2,
        max_digits: int = 6,
    ) -> None:
        """
        Initialize the DigitsEncodingFeatureGenerator.

        Args:
            features_names (list[str] | None, optional): Feature names to extract
                digits from. If None, all numeric columns are used. Defaults to None.
            fill_value (int, optional): Value used for NaN entries. Defaults to -1.
            dtype (str, optional): Output dtype for the generated digit columns.
                Defaults to "int8".
            min_digits (int, optional): Minimum number of digit positions to
                extract per feature. Defaults to 2.
            max_digits (int, optional): Maximum number of digit positions to
                extract per feature. Defaults to 6.
        """
        self.features_names = features_names
        self.fill_value = fill_value
        self.dtype = dtype
        self.min_digits = min_digits
        self.max_digits = max_digits

    def fit(
        self, df: pd.DataFrame, _y: pd.Series | None = None
    ) -> "DigitsEncodingFeatureGenerator":
        """
        Determine which digit positions to extract for each numeric feature.

        Args:
            df (pd.DataFrame): Feature DataFrame.
            _y (pd.Series | None, optional): Target (unused). Defaults to None.

        Returns:
            DigitsEncodingFeatureGenerator: The fitted generator instance.
        """
        columns = self.features_names if self.features_names else list(df.columns)
        columns = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
        self.feature_names_in_ = np.asarray(df.columns, dtype=object)
        self.feature_configs_: dict[str, tuple[int, int]] = {
            col: self._digit_range(df[col]) for col in columns
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append extracted digit columns to the original features.

        Args:
            df (pd.DataFrame): Feature DataFrame to transform.

        Returns:
            pd.DataFrame: Original features plus one column per extracted digit.
        """
        digit_data = {
            f"{col}_d{position}": self._extract_digit(df[col], position)
            for col, (start, end) in self.feature_configs_.items()
            for position in range(start, end)
        }
        if not digit_data:
            return df.copy()
        digits_df = pd.DataFrame(digit_data, index=df.index).astype(self.dtype)
        return pd.concat([df, digits_df], axis=1)

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> np.ndarray:
        """
        Get output feature names: original columns plus generated digit columns.

        Args:
            input_features (list[str] | None, optional): Unused, kept for sklearn
                API compatibility. Defaults to None.

        Returns:
            np.ndarray: Output feature names in the order produced by ``transform``.
        """
        generated = [
            f"{col}_d{position}"
            for col, (start, end) in self.feature_configs_.items()
            for position in range(start, end)
        ]
        return np.asarray(list(self.feature_names_in_) + generated, dtype=object)

    def _digit_range(self, series: pd.Series) -> tuple[int, int]:
        """
        Determine the (start, end) digit positions to extract for one feature.

        Args:
            series (pd.Series): Numeric series to analyze.

        Returns:
            tuple[int, int]: Half-open ``[start, end)`` range of digit positions.
        """
        clean = series.dropna()
        if clean.empty:
            return (-1, 1)

        abs_values = clean.abs()
        max_val = abs_values.max()
        max_power = int(np.floor(np.log10(max_val))) + 1 if max_val > 0 else 1

        min_power = 0
        if (clean % 1 != 0).any():
            decimal_places = abs_values.astype(str).str.split(".").str[-1].str.len()
            min_power = -int(decimal_places.max())

        start = max(min_power, -self.max_digits // 2)
        end = min(max_power, self.max_digits // 2)

        span = end - start
        if span < self.min_digits:
            center = (start + end) // 2
            start = center - self.min_digits // 2
            end = start + self.min_digits
        elif span > self.max_digits:
            end = start + self.max_digits

        return (start, end)

    def _extract_digit(self, series: pd.Series, position: int) -> pd.Series:
        """
        Extract a single digit position from a numeric series.

        Args:
            series (pd.Series): Numeric series to extract the digit from.
            position (int): Power of 10 identifying the digit (see class docstring).

        Returns:
            pd.Series: Extracted digit per row, with NaNs filled by ``fill_value``.
        """
        digit = (series.abs() * 10 ** (-position)) % 10
        return digit.fillna(self.fill_value)


if __name__ == "__main__":
    data = {
        "annual_income": [45000.50, 75000.00, 32000.75, 120000.25],
        "interest_rate": [3.75, 4.25, 5.50, 2.99],
        "string_feature": ["A", "B", "C", "D"],
    }
    demo_df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(demo_df)
    print()

    digits_generator = DigitsEncodingFeatureGenerator(fill_value=-1)

    transformed_df = digits_generator.fit_transform(demo_df)
    print("Transformed DataFrame:")
    print("\nColumns containing 'annual_income':")
    annual_income_cols = [
        col for col in transformed_df.columns if "annual_income" in col
    ]
    print(transformed_df[annual_income_cols])

    print("\nColumns containing 'interest_rate':")
    interest_rate_cols = [
        col for col in transformed_df.columns if "interest_rate" in col
    ]
    print(transformed_df[interest_rate_cols])

    print("\nget_feature_names_out():")
    print(digits_generator.get_feature_names_out())
