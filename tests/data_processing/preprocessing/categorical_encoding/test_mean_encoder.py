import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.categorical_encoding.mean_encoder import (
    MeanEncoderWithOriginal,
)


@pytest.fixture
def mean_encoder_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a categorical column with a target whose category means diverge sharply.

    Warsaw has a target mean of 2/3 and Krakow has a target mean of 0, against a
    global mean of 1/6, making Bayesian shrinkage clearly observable.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix with a `city` column and a
            binary target.
    """
    X = pd.DataFrame({"city": ["Warsaw"] * 3 + ["Krakow"] * 3})
    y = pd.Series([1, 1, 0, 0, 0, 0], name="target")
    return X, y


def test_meanencoderwithoriginal_transform_without_smoothing_matches_group_means(
    mean_encoder_data,
):
    """Tests that smoothing=0.0 encodes categories to their exact target mean.

    Args:
        mean_encoder_data: DataFrame and target with diverging category means.

    Asserts:
        - Derived values equal the raw per-category target means.
    """
    X, y = mean_encoder_data
    encoder = MeanEncoderWithOriginal(variables=["city"], smoothing=0.0)
    encoder.fit(X, y)
    result = encoder.transform(X)
    warsaw_value = result.loc[X["city"] == "Warsaw", "city_PREPROCESS_MEAN_ENC"].iloc[0]
    krakow_value = result.loc[X["city"] == "Krakow", "city_PREPROCESS_MEAN_ENC"].iloc[0]
    if not np.isclose(warsaw_value, 2 / 3):
        raise AssertionError(f"Expected Warsaw mean 2/3, got {warsaw_value}.")
    if not np.isclose(krakow_value, 0.0):
        raise AssertionError(f"Expected Krakow mean 0.0, got {krakow_value}.")


def test_meanencoderwithoriginal_transform_higher_smoothing_shrinks_toward_global_mean(
    mean_encoder_data,
):
    """Tests that increasing smoothing pulls encoded values closer to the global mean.

    Args:
        mean_encoder_data: DataFrame and target with diverging category means.

    Asserts:
        - The high-smoothing encoding for Warsaw is closer to the global mean
          than the low-smoothing encoding.
    """
    X, y = mean_encoder_data
    global_mean = y.mean()
    low_smoothing = MeanEncoderWithOriginal(variables=["city"], smoothing=0.0)
    high_smoothing = MeanEncoderWithOriginal(variables=["city"], smoothing=50.0)
    low_smoothing.fit(X, y)
    high_smoothing.fit(X, y)
    low_result = low_smoothing.transform(X)
    high_result = high_smoothing.transform(X)
    low_warsaw = low_result.loc[X["city"] == "Warsaw", "city_PREPROCESS_MEAN_ENC"].iloc[
        0
    ]
    high_warsaw = high_result.loc[
        X["city"] == "Warsaw", "city_PREPROCESS_MEAN_ENC"
    ].iloc[0]
    if not abs(high_warsaw - global_mean) < abs(low_warsaw - global_mean):
        raise AssertionError(
            "Higher smoothing did not shrink the encoded value toward the global mean."
        )


@pytest.fixture
def numeric_category_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a low-cardinality numeric column with a target whose category means diverge.

    Category 1 has a target mean of 2/3, category 2 has 1/3, and category 3 has 0.5.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix with an integer `num_cat`
            column and a binary target.
    """
    X = pd.DataFrame({"num_cat": [1, 2, 1, 3, 2, 1, 3, 2]})
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0], name="target")
    return X, y


def test_meanencoderwithoriginal_transform_accepts_explicit_numeric_variable(
    numeric_category_data,
):
    """Tests that an explicitly-named numeric column encodes without a dtype error.

    feature_engine's MeanEncoder normally rejects numeric input with "Some of
    the variables are not categorical..." unless ignore_format=True; this
    wrapper instead recasts explicitly-configured numeric variables to object
    dtype internally so they encode directly.

    Args:
        numeric_category_data: DataFrame and target with diverging category means.

    Asserts:
        - transform() succeeds and encodes the exact per-category target means.
        - The original numeric column's dtype is left untouched in the output.
    """
    X, y = numeric_category_data
    encoder = MeanEncoderWithOriginal(variables=["num_cat"], smoothing=0.0)
    encoder.fit(X, y)
    result = encoder.transform(X)
    expected = [2 / 3, 1 / 3, 2 / 3, 0.5, 1 / 3, 2 / 3, 0.5, 1 / 3]
    if not np.allclose(result["num_cat_PREPROCESS_MEAN_ENC"].to_numpy(), expected):
        raise AssertionError("Mean encoding did not match expected per-category means.")
    if not pd.api.types.is_numeric_dtype(result["num_cat"]):
        raise AssertionError(
            "Original numeric column dtype was altered by transform()."
        )


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
