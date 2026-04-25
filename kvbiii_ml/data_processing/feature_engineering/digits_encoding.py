import numpy as np
import pandas as pd
from tqdm import tqdm


class DigitsEncodingFeatureGenerator:
    """
    Class for extracting individual digits from numeric features.

    This feature generator automatically determines and extracts relevant digits from
    numeric columns based on the data distribution. For example, from the number 123.45:
    - Position 2 extracts '1' (hundreds place)
    - Position 1 extracts '2' (tens place)
    - Position 0 extracts '3' (ones place)
    - Position -1 extracts '4' (first decimal place)
    - Position -2 extracts '5' (second decimal place)

    The class automatically determines which digit positions are relevant based on the
    min/max values in the data during fitting.
    """

    def __init__(
        self,
        features_names: list[str] | None = None,
        fill_value: int = -1,
        dtype: str = "int8",
        digit_limits: tuple[int, int] | None = None,
        **kwargs: int,
    ) -> None:
        """
        Initialize the DigitsFeatureGenerator.

        Args:
            features_names (list[str] | None, optional): List of feature names to
                extract digits from. If None or empty, all numeric features will
                be used. Defaults to None.
            fill_value (int, optional): Value to use for NaN entries. Defaults to -1.
            dtype (str, optional): Data type for output columns. Defaults to "int8".
            digit_limits (tuple[int, int] | None, optional):
                `(min_digits, max_digits)` limits for extracted positions.
            **kwargs (int): Compatibility kwargs "min_digits" and "max_digits".

        Raises:
            ValueError: If parameters are invalid.
        """
        if "min_digits" in kwargs or "max_digits" in kwargs:
            min_digits = int(kwargs.pop("min_digits", 2))
            max_digits = int(kwargs.pop("max_digits", 6))
            digit_limits = (min_digits, max_digits)
        if kwargs:
            invalid_args = ", ".join(sorted(kwargs))
            raise ValueError(f"Unexpected arguments: {invalid_args}")

        min_digits, max_digits = digit_limits if digit_limits is not None else (2, 6)
        self.features_names = list(features_names) if features_names is not None else []
        self.fill_value = fill_value
        self.dtype = dtype
        self.min_digits = int(min_digits)
        self.max_digits = int(max_digits)
        self.feature_configs_: dict[str, tuple[int, int]] = {}
        self.generated_feature_names_: list[str] = []
        self._validate_init_params()

    def _validate_init_params(self) -> None:
        """
        Validate the initialization parameters.

        Raises:
            ValueError: If parameters are invalid.
        """
        if self.features_names is not None and (
            not isinstance(self.features_names, list)
            or not all(isinstance(name, str) for name in self.features_names)
        ):
            raise ValueError("features_names must be None or a list of strings.")

        if not isinstance(self.fill_value, int):
            raise ValueError("fill_value must be an integer.")

        if self.dtype not in ["int8", "int16", "int32", "int64"]:
            raise ValueError(
                f"dtype must be one of ['int8', 'int16', 'int32', 'int64'], "
                f"got '{self.dtype}'."
            )

        if not isinstance(self.min_digits, int) or self.min_digits < 1:
            raise ValueError("min_digits must be a positive integer.")

        if not isinstance(self.max_digits, int) or self.max_digits < self.min_digits:
            raise ValueError(
                f"max_digits must be >= min_digits. Got max_digits={self.max_digits}, "
                f"min_digits={self.min_digits}."
            )

    def _determine_digit_range(self, series: pd.Series) -> tuple[int, int]:
        """
        Determine the range of digit positions to extract based on data values.

        Args:
            series (pd.Series): Numeric series to analyze.

        Returns:
            tuple[int, int]: (start_position, end_position) for digit extraction.
        """
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return (-1, 1)

        abs_series = clean_series.abs()
        max_val = abs_series.max()

        max_power = int(np.floor(np.log10(max_val))) + 1 if max_val > 0 else 1

        min_power = 0
        has_decimals = (clean_series % 1 != 0).any()
        if has_decimals:
            decimal_strings = abs_series.astype(str)
            max_decimal_places = max(
                len(s.split(".")[-1]) if "." in s else 0 for s in decimal_strings
            )
            min_power = -max_decimal_places if max_decimal_places > 0 else 0

        start_pos = max(min_power, -self.max_digits // 2)
        end_pos = min(max_power, self.max_digits // 2)

        total_digits = end_pos - start_pos
        if total_digits < self.min_digits:
            center = (start_pos + end_pos) // 2
            start_pos = center - self.min_digits // 2
            end_pos = start_pos + self.min_digits

        if total_digits > self.max_digits:
            end_pos = start_pos + self.max_digits

        return (start_pos, end_pos)

    def _extract_digit(
        self, series: pd.Series, position: int, feature_name: str
    ) -> pd.Series:
        """
        Extract a specific digit from a numeric series.

        Args:
            series (pd.Series): Numeric series to extract digit from.
            position (int): Power of 10 indicating which digit to extract.
                Positive values extract digits left of decimal point,
                negative values extract digits right of decimal point.
            feature_name (str): Original feature name for generating column name.

        Returns:
            pd.Series: Series containing extracted digits.
        """
        column_name = f"{feature_name}_d{position}"
        abs_series = series.abs()
        extracted = (abs_series * (10 ** (-position))) % 10
        extracted = extracted.fillna(self.fill_value).astype(int)
        return extracted.astype(self.dtype).rename(column_name)

    def fit(
        self,
        df: pd.DataFrame,
        _y: pd.Series | None = None,
    ) -> "DigitsEncodingFeatureGenerator":
        """
        Fit the digits feature generator.

        Automatically determines which digit positions to extract for each numeric
        feature based on the data distribution. The y parameter is ignored but kept
        for sklearn compatibility.

        Args:
            df (pd.DataFrame): Feature DataFrame.
            _y (optional): Target variable (ignored). Defaults to None.

        Returns:
            DigitsEncodingFeatureGenerator: The fitted generator instance.

        Raises:
            TypeError: If `df` is not a DataFrame.
            KeyError: If specified features are missing from `df`.
        """
        self._validate_fit_inputs(df)
        if not self.features_names:
            self.features_names = [
                col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
            ]
        else:
            self.features_names = [
                col
                for col in self.features_names
                if pd.api.types.is_numeric_dtype(df[col])
            ]

        self.feature_configs_ = {}
        self.generated_feature_names_ = []

        for feature_name in tqdm(self.features_names, desc="Fitting digits encoding"):
            start_pos, end_pos = self._determine_digit_range(df[feature_name])
            self.feature_configs_[feature_name] = (start_pos, end_pos)

            for k in range(start_pos, end_pos):
                self.generated_feature_names_.append(f"{feature_name}_d{k}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by extracting digits from configured features.

        Args:
            df (pd.DataFrame): Feature DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with original features and extracted digit
                features.

        Raises:
            ValueError: If the generator has not been fitted yet.
            KeyError: If configured features are missing from `df`.
        """
        if not self.feature_configs_:
            raise ValueError("DigitsFeatureGenerator must be fitted before transform.")

        self._validate_transform_inputs(df)

        digit_columns = []

        pbar = tqdm(
            total=len(self.feature_configs_), desc="Transforming with digits encoding"
        )

        for feature_name, (start_pos, end_pos) in self.feature_configs_.items():
            for k in range(start_pos, end_pos):
                digit_series = self._extract_digit(df[feature_name], k, feature_name)
                digit_columns.append(digit_series)
            pbar.update(1)

        pbar.close()

        if digit_columns:
            return pd.concat([df] + digit_columns, axis=1)
        return df.copy()

    def fit_transform(
        self,
        df: pd.DataFrame,
        _y: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Fit the generator and transform the data.

        Args:
            df (pd.DataFrame): Feature DataFrame.
            _y (optional): Target variable (ignored). Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with original features and extracted digit
                features.
        """
        return self.fit(df, _y).transform(df)

    def get_feature_names_out(self) -> list[str]:
        """
        Get the names of generated digit features.

        Returns:
            list[str]: List of generated feature names.

        Raises:
            ValueError: If the generator has not been fitted yet.
        """
        if not self.generated_feature_names_:
            raise ValueError(
                "DigitsFeatureGenerator must be fitted before getting feature names."
            )
        return self.generated_feature_names_.copy()

    def get_feature_configs(self) -> dict[str, tuple[int, int]]:
        """
        Get the automatically determined digit extraction configurations.

        Returns:
            dict[str, tuple[int, int]]: Dictionary mapping feature names to
                (start_position, end_position) tuples.

        Raises:
            ValueError: If the generator has not been fitted yet.
        """
        if not self.feature_configs_:
            raise ValueError(
                "DigitsFeatureGenerator must be fitted before getting configs."
            )
        return self.feature_configs_.copy()

    def _validate_fit_inputs(self, df: pd.DataFrame) -> None:
        """
        Validate inputs for fit and fit_transform.

        Args:
            df (pd.DataFrame): Feature DataFrame.

        Raises:
            TypeError: If `df` is not a DataFrame.
            KeyError: If specified features are missing from `df`.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        if self.features_names is not None:
            missing = [
                feature for feature in self.features_names if feature not in df.columns
            ]
            if missing:
                raise KeyError(f"Missing features in df: {missing}")

    def _validate_transform_inputs(self, df: pd.DataFrame) -> None:
        """
        Validate inputs for transform.

        Args:
            df (pd.DataFrame): Feature DataFrame.

        Raises:
            TypeError: If `df` is not a DataFrame.
            KeyError: If configured features are missing from `df`.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        missing = [
            feature for feature in self.feature_configs_ if feature not in df.columns
        ]
        if missing:
            raise KeyError(f"Missing features in df: {missing}")


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
