import pytest

"""Tests for digits encoding feature generation."""

import numpy as np
import pandas as pd

from kvbiii_ml.data_processing.feature_engineering.digits_encoding import (
    DigitsEncodingFeatureGenerator,
)


def _digit_columns(frame: pd.DataFrame) -> list[str]:
    """Return generated digit columns for the numeric feature used in tests."""
    return [column for column in frame.columns if column.startswith("num_d")]


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

    assert digit_columns
    for column in digit_columns:
        assert transformed.loc[missing_rows, column].eq(-1).all()


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

    assert digit_columns
    for column in digit_columns:
        assert transformed.loc[missing_rows, column].eq(-1).all()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
