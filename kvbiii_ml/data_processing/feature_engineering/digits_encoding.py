from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

AUTO_DIGITS = "auto"
MAX_AUTO_DECIMAL_DIGITS = 3


class DigitsEncodingFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Extracts individual digits from numeric features.

    From the number 123.45:
    - Position 2 extracts '1' (hundreds place)
    - Position 1 extracts '2' (tens place)
    - Position 0 extracts '3' (ones place)
    - Position -1 extracts '4' (first decimal place)
    - Position -2 extracts '5' (second decimal place)

    Derived columns are named ``{original}_PREPROCESS_DIGIT_{n}`` for
    non-negative positions and ``{original}_PREPROCESS_DIGIT_N{n}`` for
    negative (decimal) positions - an ``N`` prefix replaces the minus sign so
    generated feature names stay hyphen-free.

    The digit positions extracted per feature can either be set explicitly
    (``min_digits``/``max_digits`` as integers) or, by default, determined
    automatically from the min/max values seen during fitting
    (``min_digits`` and ``max_digits`` both ``"auto"``). In "auto" mode the
    integer-part range is sized to the feature's actual magnitude with no
    forced minimum, and the decimal-part range is capped at
    ``MAX_AUTO_DECIMAL_DIGITS`` places, so a feature with values up to the
    millions and five observed decimal places still only yields digits down
    to the thousandths place.
    """

    _suffix = "PREPROCESS_DIGIT"

    def __init__(
        self,
        features_names: list[str] | None = None,
        fill_value: int = -1,
        dtype: str = "int8",
        min_digits: int | Literal["auto"] = AUTO_DIGITS,
        max_digits: int | Literal["auto"] = AUTO_DIGITS,
    ) -> None:
        """
        Initialize the DigitsEncodingFeatureGenerator.

        Args:
            features_names (list[str] | None, optional): Feature names to extract
                digits from. If None, all numeric columns are used. Defaults to None.
            fill_value (int, optional): Value used for NaN entries. Defaults to -1.
            dtype (str, optional): Output dtype for the generated digit columns.
                Defaults to "int8".
            min_digits (int | Literal["auto"], optional): Minimum number of digit
                positions to extract per feature, or ``"auto"`` to derive it from
                the fitted data with no forced minimum. Must be ``"auto"`` iff
                ``max_digits`` is too. Defaults to ``"auto"``.
            max_digits (int | Literal["auto"], optional): Maximum number of digit
                positions to extract per feature, or ``"auto"`` to derive it from
                the fitted data, capping decimal places at
                ``MAX_AUTO_DECIMAL_DIGITS``. Must be ``"auto"`` iff ``min_digits``
                is too. Defaults to ``"auto"``.
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

        Raises:
            ValueError: When only one of min_digits/max_digits is "auto".
        """
        if (self.min_digits == AUTO_DIGITS) != (self.max_digits == AUTO_DIGITS):
            raise ValueError(
                'min_digits and max_digits must either both be "auto" '
                "or both be explicit integers."
            )
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
            self._column_name(col, position): self._extract_digit(df[col], position)
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
            self._column_name(col, position)
            for col, (start, end) in self.feature_configs_.items()
            for position in range(start, end)
        ]
        return np.asarray(list(self.feature_names_in_) + generated, dtype=object)

    def _column_name(self, col: str, position: int) -> str:
        """
        Build the derived column name for one digit position of one feature.

        Args:
            col (str): Source feature name.
            position (int): Power of 10 identifying the digit (see class docstring).

        Returns:
            str: Derived column name, e.g. "price_PREPROCESS_DIGIT_2" for the
                hundreds place or "price_PREPROCESS_DIGIT_N1" for the first
                decimal place.
        """
        label = f"N{abs(position)}" if position < 0 else str(position)
        return f"{col}_{self._suffix}_{label}"

    def _digit_range(self, series: pd.Series) -> tuple[int, int]:
        """
        Determine the (start, end) digit positions to extract for one feature.

        When ``max_digits`` is ``"auto"`` the range is derived purely from the
        observed data: the integer-part bound matches the feature's actual
        magnitude and the decimal-part bound is capped at
        ``MAX_AUTO_DECIMAL_DIGITS`` places. Otherwise the explicit
        ``min_digits``/``max_digits`` bounds are applied.

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

        if self.max_digits == AUTO_DIGITS:
            return (max(min_power, -MAX_AUTO_DECIMAL_DIGITS), max_power)

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
