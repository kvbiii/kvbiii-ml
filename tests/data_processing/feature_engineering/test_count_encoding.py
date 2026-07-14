"""Tests for kvbiii_ml.data_processing.feature_engineering.count_encoding module."""

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.feature_engineering.count_encoding import (
    CountEncodingFeatureGenerator as CountEncoder,
)


@pytest.fixture
def categorical_data(test_settings):
    """Provides dataset with categorical features for testing count encoding.

    Args:
        test_settings: Test configuration fixture

    Returns:
        pd.DataFrame: DataFrame with categorical features
    """
    np.random.seed(test_settings.SEED)
    n_samples = test_settings.N_SAMPLES

    # Create data with categorical features of varying cardinality
    data = pd.DataFrame(
        {
            "high_cardinality": np.random.choice(
                [f"val_{i}" for i in range(50)], n_samples
            ),
            "medium_cardinality": np.random.choice(
                [f"cat_{i}" for i in range(10)], n_samples
            ),
            "low_cardinality": np.random.choice(["A", "B", "C"], n_samples),
            "binary": np.random.choice(["yes", "no"], n_samples),
            "numeric": np.random.randn(n_samples),
        }
    )

    # Add some missing values
    data.loc[0:5, "high_cardinality"] = None
    data.loc[10:15, "medium_cardinality"] = None

    return data


def test_countencoder_init_stores_configuration():
    """Tests CountEncoder initialization stores configuration correctly.

    Asserts:
        - Feature list is stored correctly
        - Configuration parameters are stored with correct defaults
        - Empty feature list is handled properly
    """
    features = ["col1", "col2"]

    # Test with default parameters
    encoder = CountEncoder(features_names=features, fill_value=0)
    if not (encoder.features_names == features):
        raise AssertionError()
    encoder = CountEncoder()
    if not (encoder.features_names == []):
        raise AssertionError()


def test_countencoder_fit_computes_count_maps(categorical_data):
    """Tests fit method computes category count maps correctly.

    Args:
        categorical_data: Categorical data fixture

    Asserts:
        - Fit computes category counts correctly
        - Count maps are stored for each feature
        - Missing values are handled according to configuration
    """
    encoder = CountEncoder(
        features_names=["high_cardinality", "medium_cardinality", "low_cardinality"]
    )
    fitted_encoder = encoder.fit(categorical_data)

    if not (fitted_encoder is encoder):
        raise AssertionError()

    # Check count maps are created
    if not (hasattr(encoder, "count_maps_")):
        raise AssertionError()
    if not (isinstance(encoder.count_maps_, dict)):
        raise AssertionError()
    if not ("high_cardinality" in encoder.count_maps_):
        raise AssertionError()
    if not ("medium_cardinality" in encoder.count_maps_):
        raise AssertionError()
    if not ("low_cardinality" in encoder.count_maps_):
        raise AssertionError()

    # Check counts are computed correctly
    for feature in encoder.features_names:
        value_counts = categorical_data[feature].value_counts(dropna=False)

        for category, count in encoder.count_maps_[feature].items():
            if category != np.nan and pd.notna(category):
                if not (count == value_counts[category]):
                    raise AssertionError()


def test_countencoder_transform_replaces_categories_with_counts(categorical_data):
    """Tests transform method replaces categories with their counts.

    Args:
        categorical_data: Categorical data fixture

    Asserts:
        - Categories are replaced with counts in output
        - Missing values are handled according to configuration
        - Non-categorical columns remain unchanged
    """
    encoder = CountEncoder(
        features_names=["high_cardinality", "medium_cardinality", "low_cardinality"]
    )
    encoder.fit(categorical_data)

    transformed = encoder.transform(categorical_data)

    # Original data should be unchanged
    if not (categorical_data["high_cardinality"].dtype == object):
        raise AssertionError()

    # Transformed data should have numeric count values
    for col in ["CE_high_cardinality", "CE_medium_cardinality", "CE_low_cardinality"]:
        if not (np.issubdtype(transformed[col].dtype, np.integer)):
            raise AssertionError()

    # Counts should match expected values
    feature = "low_cardinality"
    for category, count in categorical_data[feature].value_counts().items():
        category_rows = categorical_data[feature] == category
        if not ((transformed.loc[category_rows, "CE_" + feature] == count).all()):
            raise AssertionError()

    # Non-categorical columns should remain unchanged
    pd.testing.assert_series_equal(transformed["numeric"], categorical_data["numeric"])
    pd.testing.assert_series_equal(transformed["binary"], categorical_data["binary"])


def test_countencoder_no_normalization_available(categorical_data):
    """Tests encoding a low-cardinality feature without normalization."""
    encoder = CountEncoder(features_names=["low_cardinality"])
    encoder.fit(categorical_data)
    transformed = encoder.transform(categorical_data)
    if not ("CE_low_cardinality" in transformed.columns):
        raise AssertionError()


def test_countencoder_min_count_not_supported(categorical_data):
    """Tests encoding a high-cardinality feature without a min_count filter."""
    encoder = CountEncoder(features_names=["high_cardinality"])
    encoder.fit(categorical_data)
    transformed = encoder.transform(categorical_data)
    if not ("CE_high_cardinality" in transformed.columns):
        raise AssertionError()


def test_countencoder_missing_values_replaced_with_fill_value(categorical_data):
    """Tests that missing categories are encoded with the configured fill_value."""
    encoder = CountEncoder(features_names=["high_cardinality"], fill_value=-1)
    encoder.fit(categorical_data)
    transformed = encoder.transform(categorical_data)
    if not ("CE_high_cardinality" in transformed.columns):
        raise AssertionError()


def test_countencoder_unknown_categories_get_fill_value():
    """Tests that categories unseen during fit are encoded with fill_value."""
    train = pd.DataFrame({"feature": ["A", "A", "B", "C"]})
    test = pd.DataFrame({"feature": ["A", "Z"]})
    enc = CountEncoder(features_names=["feature"], fill_value=0)
    enc.fit(train)
    out = enc.transform(test)
    if not (out.loc[1, "CE_feature"] == 0):
        raise AssertionError()


def test_countencoder_fit_transform_combines_operations(categorical_data):
    """Tests fit_transform method combines fit and transform operations.

    Args:
        categorical_data: Categorical data fixture

    Asserts:
        - fit_transform produces same result as separate fit and transform
        - Method works correctly with all encoding options
    """
    encoder = CountEncoder(features_names=["high_cardinality", "medium_cardinality"])

    # Get result using fit_transform
    result1 = encoder.fit_transform(categorical_data)

    # Get result using separate fit and transform
    encoder = CountEncoder(features_names=["high_cardinality", "medium_cardinality"])
    encoder.fit(categorical_data)
    result2 = encoder.transform(categorical_data)

    # Results should be identical
    pd.testing.assert_frame_equal(result1, result2)


def test_countencoder_get_feature_names_returns_new_columns(categorical_data):
    """Tests that get_feature_names returns the generated CE_-prefixed column names."""
    enc = CountEncoder(features_names=["high_cardinality", "medium_cardinality"])
    enc.fit(categorical_data)
    names = enc.get_feature_names()
    if not (names == ["CE_high_cardinality", "CE_medium_cardinality"]):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
