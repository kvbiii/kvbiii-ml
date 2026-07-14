"""Tests for kvbiii_ml.data_processing.feature_engineering.categorical_cleaner module."""

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.feature_engineering.categorical_cleaner import (
    CategoricalCleaner,
)


@pytest.fixture
def messy_categorical_data(test_settings):
    """Provides DataFrame with messy categorical data for testing.

    Args:
        test_settings: Test configuration fixture

    Returns:
        pd.DataFrame: DataFrame with messy categorical features
    """
    np.random.seed(test_settings.SEED)
    n_samples = test_settings.N_SAMPLES

    # Create messy categorical data
    data = pd.DataFrame(
        {
            "clean_cat": np.random.choice(["A", "B", "C"], n_samples),
            "messy_spaces": np.random.choice([" A", "B ", " C "], n_samples),
            "messy_case": np.random.choice(["a", "A", "b", "B"], n_samples),
            "messy_special": np.random.choice(["a!", "a@", "a#", "a$"], n_samples),
            "numeric": np.random.randn(n_samples),
        }
    )

    # Add some missing values
    data.loc[0:5, "messy_spaces"] = None
    data.loc[10:15, "messy_case"] = None

    return data


def test_categoricalcleaner_init_stores_configuration():
    """Initialization stores provided feature_groups mapping."""
    groups = {"Not Provided": ["col1", "col2"], "Unknown": ["col3"]}
    cleaner = CategoricalCleaner(groups)
    if cleaner.feature_groups != groups:
        raise AssertionError()


def test_categoricalcleaner_fit_returns_self(messy_categorical_data):
    """Tests fit method returns self for scikit-learn compatibility.

    Args:
        messy_categorical_data: Messy categorical data fixture

    Asserts:
        - fit method returns self
        - No error occurs during fitting
    """
    cleaner = CategoricalCleaner(
        {"None": ["messy_spaces"], "None2": ["messy_case"], "a!": ["messy_special"]}
    )
    result = cleaner.fit(messy_categorical_data)

    if result is not cleaner:
        raise AssertionError()


def test_categoricalcleaner_transform_replaces_placeholders(messy_categorical_data):
    """Tests that transform replaces configured placeholder tokens with NaN."""
    groups = {"Not Provided": ["messy_spaces"], "a!": ["messy_special"]}
    # Insert placeholder tokens
    messy_categorical_data.loc[0, "messy_spaces"] = "Not Provided"
    messy_categorical_data.loc[1, "messy_special"] = "a!"
    cleaner = CategoricalCleaner(groups)
    cleaner.fit(messy_categorical_data)
    result = cleaner.transform(messy_categorical_data)
    if not pd.isna(result.loc[0, "messy_spaces"]):
        raise AssertionError()
    if not pd.isna(result.loc[1, "messy_special"]):
        raise AssertionError()
    # Non configured columns unchanged
    pd.testing.assert_series_equal(result["numeric"], messy_categorical_data["numeric"])


def test_categoricalcleaner_no_op_for_missing_group(messy_categorical_data):
    """Tests that transform is a no-op when configured columns don't exist."""
    cleaner = CategoricalCleaner({"Placeholder": ["nonexistent"]})
    cleaner.fit(messy_categorical_data)
    result = cleaner.transform(messy_categorical_data)
    # No listed columns exist; should be an exact copy (no dtype/category conversion)
    pd.testing.assert_frame_equal(result, messy_categorical_data)


def test_categoricalcleaner_fit_transform_combines_operations(messy_categorical_data):
    """Tests fit_transform method combines fit and transform operations.

    Args:
        messy_categorical_data: Messy categorical data fixture

    Asserts:
        - fit_transform produces same result as separate fit and transform
        - Method works correctly with all cleaning options
    """
    cleaner = CategoricalCleaner(
        {"Not Provided": ["messy_spaces", "messy_case", "messy_special"]}
    )

    # Get result using fit_transform
    result1 = cleaner.fit_transform(messy_categorical_data)

    # Get result using separate fit and transform
    cleaner.fit(messy_categorical_data)
    result2 = cleaner.transform(messy_categorical_data)

    # Results should be identical
    pd.testing.assert_frame_equal(result1, result2)


def test_categoricalcleaner_transform_handles_new_features(messy_categorical_data):
    """Tests transform handles features not present in training data.

    Args:
        messy_categorical_data: Messy categorical data fixture

    Asserts:
        - No error when transform data has new columns
        - Only configured features are processed
    """
    cleaner = CategoricalCleaner({"Not Provided": ["messy_spaces", "messy_case"]})
    cleaner.fit(messy_categorical_data)

    # Create test data with a new column
    test_data = messy_categorical_data.copy()
    test_data["new_column"] = " NEW "

    result = cleaner.transform(test_data)

    # New column should be in result but unchanged
    if "new_column" not in result.columns:
        raise AssertionError()
    if result["new_column"].iloc[0] != " NEW ":
        raise AssertionError()

    # Configured columns should be cast to category dtype even if no placeholder matches
    if str(result["messy_spaces"].dtype) != "category":
        raise AssertionError()
    if str(result["messy_case"].dtype) != "category":
        raise AssertionError()
    # Original raw values (with spaces / case) remain since cleaner only replaces placeholders
    if set(result["messy_spaces"].cat.categories) != set(
        messy_categorical_data["messy_spaces"].dropna().unique()
    ):
        raise AssertionError()


def test_categoricalcleaner_transform_handles_missing_features(messy_categorical_data):
    """Tests transform handles missing features gracefully.

    Args:
        messy_categorical_data: Messy categorical data fixture

    Asserts:
        - No error when configured feature is missing
        - Available features are still processed
    """
    cleaner = CategoricalCleaner(
        {"Not Provided": ["messy_spaces", "messy_case", "non_existent_feature"]}
    )
    cleaner.fit(messy_categorical_data)

    result = cleaner.transform(messy_categorical_data)

    # Available features converted to category; values unchanged except dtype
    if str(result["messy_spaces"].dtype) != "category":
        raise AssertionError()
    if " A" not in result["messy_spaces"].cat.categories:
        raise AssertionError()
    # Case not modified by cleaner
    if set(result["messy_case"].cat.categories) != set(
        messy_categorical_data["messy_case"].dropna().unique()
    ):
        raise AssertionError()

    # Should not add the non-existent feature
    if "non_existent_feature" in result.columns:
        raise AssertionError()


def test_categoricalcleaner_get_feature_names_out_returns_input_features(
    messy_categorical_data,
):
    """Tests get_feature_names_out returns original feature names.

    Args:
        messy_categorical_data: Messy categorical data fixture

    Asserts:
        - Returns all input feature names
        - Ignores any input parameter
    """
    cleaner = CategoricalCleaner({"Not Provided": ["messy_spaces", "messy_case"]})
    cleaner.fit(messy_categorical_data)

    feature_names = cleaner.get_feature_names_out()

    # Implementation returns pd.Index
    if list(feature_names) != list(messy_categorical_data.columns):
        raise AssertionError()

    # Should ignore input parameter
    feature_names_with_param = cleaner.get_feature_names_out(["ignored"])
    if list(feature_names_with_param) != list(messy_categorical_data.columns):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
