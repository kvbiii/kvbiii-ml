"""Tests for digits encoding feature generation."""

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.feature_engineering.digits_encoding import (
    MAX_AUTO_DECIMAL_DIGITS,
    DigitsEncodingFeatureGenerator,
)


def _digit_columns(frame: pd.DataFrame) -> list[str]:
    """Return generated digit columns for the numeric feature used in tests."""
    return [column for column in frame.columns if column.startswith("num_PREPROCESS_DIGIT")]


def test_transform_handles_nan_rows_with_fill_value() -> None:
    """Transform should encode NaN rows using the configured fill value."""
    train = pd.DataFrame({"num": [123.45, 67.89, 10.01]})
    test = pd.DataFrame({"num": [np.nan, 44.56, np.nan]})

    generator = DigitsEncodingFeatureGenerator(
        features_names=["num"],
        fill_value=-1,
        min_digits=2,
        max_digits=4,
    )

    generator.fit(train)
    transformed = generator.transform(test)

    digit_columns = _digit_columns(transformed)
    missing_rows = test["num"].isna()

    if not digit_columns:
        raise AssertionError()
    for column in digit_columns:
        if not transformed.loc[missing_rows, column].eq(-1).all():
            raise AssertionError()


def test_fit_transform_handles_nan_rows_with_fill_value() -> None:
    """fit_transform should preserve NaN safety for generated digit columns."""
    frame = pd.DataFrame({"num": [12.34, np.nan, 56.78]})

    generator = DigitsEncodingFeatureGenerator(
        features_names=["num"],
        fill_value=-1,
        min_digits=2,
        max_digits=4,
    )

    transformed = generator.fit_transform(frame)
    digit_columns = _digit_columns(transformed)
    missing_rows = frame["num"].isna()

    if not digit_columns:
        raise AssertionError()
    for column in digit_columns:
        if not transformed.loc[missing_rows, column].eq(-1).all():
            raise AssertionError()


def test_column_names_use_preprocess_digit_suffix_without_hyphen() -> None:
    """Generated column names should follow the PREPROCESS_DIGIT_{n}/N{n} convention."""
    frame = pd.DataFrame({"num": [123.45, 67.89, 10.01]})

    generator = DigitsEncodingFeatureGenerator(
        features_names=["num"], min_digits=3, max_digits=5
    )
    transformed = generator.fit_transform(frame)

    digit_columns = _digit_columns(transformed)
    if not digit_columns:
        raise AssertionError()
    for column in digit_columns:
        if "-" in column:
            raise AssertionError(f"column name should not contain a hyphen: {column}")
    if "num_PREPROCESS_DIGIT_1" not in digit_columns:
        raise AssertionError()
    if "num_PREPROCESS_DIGIT_N1" not in digit_columns:
        raise AssertionError()


def test_auto_mode_is_the_default() -> None:
    """min_digits/max_digits should default to auto range detection."""
    generator = DigitsEncodingFeatureGenerator()

    if generator.min_digits != "auto":
        raise AssertionError()
    if generator.max_digits != "auto":
        raise AssertionError()


def test_auto_mode_derives_range_from_data_and_caps_decimals() -> None:
    """Auto mode should size the integer part to the data and cap decimals."""
    frame = pd.DataFrame({"num": [12345.123456, 200.5]})

    generator = DigitsEncodingFeatureGenerator(features_names=["num"])
    transformed = generator.fit_transform(frame)

    digit_columns = set(_digit_columns(transformed))
    expected_integer_columns = {
        "num_PREPROCESS_DIGIT_4",
        "num_PREPROCESS_DIGIT_3",
        "num_PREPROCESS_DIGIT_2",
        "num_PREPROCESS_DIGIT_1",
        "num_PREPROCESS_DIGIT_0",
    }
    expected_decimal_columns = {
        f"num_PREPROCESS_DIGIT_N{n}" for n in range(1, MAX_AUTO_DECIMAL_DIGITS + 1)
    }

    if not expected_integer_columns.issubset(digit_columns):
        raise AssertionError()
    if not expected_decimal_columns.issubset(digit_columns):
        raise AssertionError()
    if f"num_PREPROCESS_DIGIT_N{MAX_AUTO_DECIMAL_DIGITS + 1}" in digit_columns:
        raise AssertionError("auto mode should not exceed MAX_AUTO_DECIMAL_DIGITS places")


def test_auto_mode_does_not_pad_integer_only_feature() -> None:
    """Auto mode should not force extra decimal columns for integer-only data."""
    frame = pd.DataFrame({"num": [7, 42, 999]})

    generator = DigitsEncodingFeatureGenerator(features_names=["num"])
    transformed = generator.fit_transform(frame)

    digit_columns = _digit_columns(transformed)
    if any(column.startswith("num_PREPROCESS_DIGIT_N") for column in digit_columns):
        raise AssertionError("integer-only feature should not generate decimal digit columns")


def test_mismatched_auto_and_explicit_digits_raises() -> None:
    """Mixing "auto" with an explicit integer across min/max_digits should raise."""
    frame = pd.DataFrame({"num": [1.5, 2.5, 3.5]})

    generator = DigitsEncodingFeatureGenerator(
        features_names=["num"], min_digits="auto", max_digits=4
    )

    with pytest.raises(ValueError):
        generator.fit(frame)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
