import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.categorical_encoding.count_frequency_encoder import (
    CountFrequencyEncoderWithOriginal,
)


@pytest.fixture
def city_revenue_data() -> pd.DataFrame:
    """Provides a DataFrame with a categorical column of known category counts.

    Warsaw occurs 3 times, Krakow 2 times, and Gdansk once, out of 6 rows.

    Returns:
        pd.DataFrame: DataFrame with a `city` categorical column and a numeric
            `revenue` column.
    """
    return pd.DataFrame(
        {
            "city": ["Warsaw", "Warsaw", "Warsaw", "Krakow", "Krakow", "Gdansk"],
            "revenue": [100, 150, 130, 200, 220, 80],
        }
    )


def test_countfrequencyencoderwithoriginal_transform_encodes_counts(city_revenue_data):
    """Tests that encoding_method='count' encodes each category to its raw count.

    Args:
        city_revenue_data: DataFrame with known category counts.

    Asserts:
        - Derived column values match the absolute occurrence count per category.
    """
    encoder = CountFrequencyEncoderWithOriginal(
        encoding_method="count", variables=["city"]
    )
    encoder.fit(city_revenue_data)
    result = encoder.transform(city_revenue_data)
    expected = [3, 3, 3, 2, 2, 1]
    if not np.allclose(
        result["city_PREPROCESS_CNT_FREQ"].to_numpy(dtype=float), expected
    ):
        raise AssertionError("Count encoding did not match expected occurrence counts.")


def test_countfrequencyencoderwithoriginal_transform_encodes_frequencies(
    city_revenue_data,
):
    """Tests that encoding_method='frequency' encodes each category to its relative frequency.

    Args:
        city_revenue_data: DataFrame with known category counts.

    Asserts:
        - Derived column values match the relative frequency per category.
    """
    encoder = CountFrequencyEncoderWithOriginal(
        encoding_method="frequency", variables=["city"]
    )
    encoder.fit(city_revenue_data)
    result = encoder.transform(city_revenue_data)
    expected = [0.5, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 6]
    if not np.allclose(
        result["city_PREPROCESS_CNT_FREQ"].to_numpy(dtype=float), expected
    ):
        raise AssertionError(
            "Frequency encoding did not match expected relative frequencies."
        )


def test_countfrequencyencoderwithoriginal_fit_missing_values_raise_rejects_nan(
    city_revenue_data,
):
    """Tests that missing_values='raise' raises when NaN is present at fit time.

    Args:
        city_revenue_data: DataFrame with known category counts.

    Asserts:
        - fit() raises ValueError when the configured variable contains NaN.
    """
    data_with_nan = city_revenue_data.copy()
    data_with_nan.loc[0, "city"] = None
    encoder = CountFrequencyEncoderWithOriginal(
        variables=["city"], missing_values="raise"
    )
    with pytest.raises(ValueError):
        encoder.fit(data_with_nan)


def test_countfrequencyencoderwithoriginal_fit_missing_values_ignore_accepts_nan(
    city_revenue_data,
):
    """Tests that missing_values='ignore' tolerates NaN at fit and transform time.

    Args:
        city_revenue_data: DataFrame with known category counts.

    Asserts:
        - fit() and transform() complete without raising when NaN is present.
    """
    data_with_nan = city_revenue_data.copy()
    data_with_nan.loc[0, "city"] = None
    encoder = CountFrequencyEncoderWithOriginal(
        variables=["city"], missing_values="ignore"
    )
    encoder.fit(data_with_nan)
    result = encoder.transform(data_with_nan)
    if "city_PREPROCESS_CNT_FREQ" not in result.columns:
        raise AssertionError("Derived column missing after tolerating NaN input.")


def test_countfrequencyencoderwithoriginal_transform_unseen_raise_raises_valueerror(
    city_revenue_data,
):
    """Tests that unseen='raise' raises on a category not present at fit time.

    Args:
        city_revenue_data: DataFrame with known category counts.

    Asserts:
        - transform() raises ValueError when given an unseen category.
    """
    encoder = CountFrequencyEncoderWithOriginal(variables=["city"], unseen="raise")
    encoder.fit(city_revenue_data)
    unseen_data = pd.DataFrame({"city": ["Poznan"], "revenue": [90]})
    with pytest.raises(ValueError):
        encoder.transform(unseen_data)


def test_countfrequencyencoderwithoriginal_transform_unseen_ignore_encodes_as_nan(
    city_revenue_data,
):
    """Tests that unseen='ignore' encodes an unseen category as NaN.

    Args:
        city_revenue_data: DataFrame with known category counts.

    Asserts:
        - transform() does not raise for an unseen category.
        - The derived value for the unseen category is NaN.
    """
    encoder = CountFrequencyEncoderWithOriginal(variables=["city"], unseen="ignore")
    encoder.fit(city_revenue_data)
    unseen_data = pd.DataFrame({"city": ["Poznan"], "revenue": [90]})
    result = encoder.transform(unseen_data)
    if not pd.isna(result["city_PREPROCESS_CNT_FREQ"].iloc[0]):
        raise AssertionError(
            "Unseen category was not encoded as NaN under unseen='ignore'."
        )


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
