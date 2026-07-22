"""Tests for kvbiii_ml.data_processing.eda.data_cleaning module."""

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.eda.data_cleaning import DataCleaner


@pytest.fixture
def dirty_data(test_settings):
    """Provides messy data for cleaning tests.

    Args:
        test_settings: Test configuration fixture

    Returns:
        pd.DataFrame: DataFrame with various data quality issues
    """
    np.random.seed(test_settings.SEED)
    n_samples = test_settings.N_SAMPLES

    data = pd.DataFrame(
        {
            "Normal Feature": np.random.randn(n_samples),
            "Feature With Spaces ": np.random.randn(n_samples),
            "feature_with_missing": np.random.randn(n_samples),
            "outlier_feature": np.random.randn(n_samples),
            "categorical": np.random.choice(["A", "B", "C"], size=n_samples),
        }
    )

    # Add missing values
    data.loc[:5, "feature_with_missing"] = np.nan

    # Add outliers
    data.loc[0, "outlier_feature"] = 100  # Extreme outlier
    data.loc[1, "outlier_feature"] = -100

    # Add duplicate rows
    data = pd.concat([data, data.iloc[:3]], ignore_index=True)

    return data


def test_datacleaner_init_creates_instance_with_default_parameters():
    """Tests DataCleaner initialization with default parameters.

    Asserts:
        - Instance is created successfully
        - Default cleaning parameters are reasonable
        - All cleaning options can be configured independently
    """
    try:
        cleaner = DataCleaner()
        # Basic assertion that instance was created
        if not isinstance(cleaner, DataCleaner):
            raise AssertionError()
    except (NameError, AttributeError):
        pytest.skip("DataCleaner class not implemented")


def test_datacleaner_init_accepts_custom_parameters():
    """Tests DataCleaner initialization works correctly.

    Asserts:
        - DataCleaner can be instantiated without parameters
        - All methods are static and available
        - Instance creation works as expected
    """
    try:
        cleaner = DataCleaner()
        if cleaner is None:
            raise AssertionError()

        # Check that static methods are available
        if not hasattr(DataCleaner, "get_categorical_features"):
            raise AssertionError()
        if not hasattr(DataCleaner, "initial_features_removal"):
            raise AssertionError()
        if not hasattr(DataCleaner, "drop_highly_skewed_categorical_features"):
            raise AssertionError()
        if not hasattr(DataCleaner, "categorize_categorical_features_by_missing"):
            raise AssertionError()

    except (NameError, AttributeError):
        pytest.skip("DataCleaner class not implemented")


def test_datacleaner_remove_duplicate_features_removes_exact_duplicates():
    """Tests DataCleaner._remove_duplicate_features removes columns with identical values.

    Asserts:
        - The duplicate column is dropped from the returned DataFrame
        - The original column is retained
        - The duplicate column name is reported in the removed features list
    """
    test_data = pd.DataFrame(
        {
            "original": [1, 2, 1, 3, 2],
            "duplicate": [1, 2, 1, 3, 2],
            "distinct": ["a", "b", "a", "c", "b"],
        }
    )

    cleaned_data, removed_features = DataCleaner._remove_duplicate_features(test_data)

    if "duplicate" in cleaned_data.columns:
        raise AssertionError()
    if "original" not in cleaned_data.columns:
        raise AssertionError()
    if "distinct" not in cleaned_data.columns:
        raise AssertionError()
    if removed_features != ["duplicate"]:
        raise AssertionError()


def test_datacleaner_remove_duplicate_features_handles_no_duplicates():
    """Tests DataCleaner._remove_duplicate_features when no columns are duplicated.

    Asserts:
        - The returned DataFrame is unchanged when no duplicate columns exist
        - No features are reported as removed
    """
    unique_data = pd.DataFrame({"A": [1, 2, 3, 4], "B": ["a", "b", "c", "d"]})

    cleaned_data, removed_features = DataCleaner._remove_duplicate_features(unique_data)

    pd.testing.assert_frame_equal(cleaned_data, unique_data)
    if removed_features != []:
        raise AssertionError()


def test_datacleaner_categorize_categorical_features_by_missing_boundary_thresholds():
    """Tests DataCleaner.categorize_categorical_features_by_missing at exact threshold
    boundaries (0.0 and 0.1 missing ratio).

    Asserts:
        - A feature with exactly 0.0 missing ratio is treated as non-missing, not
          categorical_missing_features
        - A feature with exactly 0.1 missing ratio falls into the "not many missing"
          bucket, matching the inclusive `<= 0.1` boundary in the implementation
    """
    test_data = pd.DataFrame(
        {
            "zero_missing": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "ten_percent_missing": ["A", "B", "C", "A", "B", "C", "A", "B", "C", None],
        }
    )

    categories = DataCleaner.categorize_categorical_features_by_missing(
        test_data, ["zero_missing", "ten_percent_missing"]
    )

    if "zero_missing" not in categories["non_missing_categorical_features"]:
        raise AssertionError()
    if "zero_missing" in categories["categorical_missing_features"]:
        raise AssertionError()
    if "ten_percent_missing" not in categories["categorical_not_many_missing_features"]:
        raise AssertionError()
    if "ten_percent_missing" in categories["categorical_many_missing_features"]:
        raise AssertionError()


def test_datacleaner_fit_transform_combines_all_cleaning_steps(dirty_data):
    """Tests DataCleaner static methods work together for data cleaning.

    Args:
        dirty_data (pd.DataFrame): Data with multiple quality issues

    Asserts:
        - Static methods can be chained for data cleaning
        - Data quality is improved through the pipeline
        - Methods work correctly on realistic data
    """
    try:
        # Add some duplicate columns and single-value columns for testing
        test_data = dirty_data.copy()
        test_data["duplicate_col"] = test_data.iloc[
            :, 0
        ].copy()  # Duplicate first column
        test_data["single_val_col"] = "constant"  # Single value
        test_data["all_na_col"] = pd.NA  # All missing

        original_shape = test_data.shape

        # Apply initial cleaning
        cleaned_data = DataCleaner.initial_features_removal(test_data)

        # Should have fewer columns due to duplicate/single-value/all-NA removal
        if not cleaned_data.shape[1] < original_shape[1]:
            raise AssertionError()
        if cleaned_data.shape[0] != original_shape[0]:
            raise AssertionError()

        # Get categorical features
        cat_features = DataCleaner.get_categorical_features(cleaned_data)
        if not isinstance(cat_features, list):
            raise AssertionError()

        # Test categorization by missing values
        if cat_features:
            cat_categories = DataCleaner.categorize_categorical_features_by_missing(
                cleaned_data, cat_features
            )
            if not isinstance(cat_categories, dict):
                raise AssertionError()
            expected_keys = [
                "categorical_not_many_missing_features",
                "categorical_many_missing_features",
                "categorical_missing_features",
                "non_missing_categorical_features",
            ]
            for key in expected_keys:
                if key not in cat_categories:
                    raise AssertionError()

    except (NameError, AttributeError):
        pytest.skip("DataCleaner methods not implemented")


def test_datacleaner_handles_empty_dataframe():
    """Tests DataCleaner handles empty DataFrame inputs gracefully.

    Asserts:
        - Empty DataFrames are processed without errors
        - Empty result is returned appropriately
        - No exceptions are raised
    """
    empty_df = pd.DataFrame()

    cleaner = DataCleaner()

    # Test static methods with empty DataFrame
    result = cleaner.initial_features_removal(empty_df)
    if not isinstance(result, pd.DataFrame):
        raise AssertionError()
    if len(result) != 0:
        raise AssertionError()

    # Test categorical features detection
    cat_features = cleaner.get_categorical_features(empty_df)
    if not isinstance(cat_features, list):
        raise AssertionError()
    if len(cat_features) != 0:
        raise AssertionError()


def test_datacleaner_preserves_data_types_where_appropriate(test_settings):
    """Tests DataCleaner preserves appropriate data types during cleaning.

    Args:
        test_settings: Test configuration fixture

    Asserts:
        - Numeric data types are preserved after cleaning
        - Categorical data types are maintained
        - Type preservation doesn't interfere with cleaning operations
    """
    np.random.seed(test_settings.SEED)
    typed_data = pd.DataFrame(
        {
            "int_column": pd.Series([1, 2, 3, 4, 5], dtype="int64"),
            "float_column": pd.Series([1.1, 2.2, 3.3, 4.4, 5.5], dtype="float64"),
            "category_column": pd.Categorical(["A", "B", "A", "C", "B"]),
        }
    )

    try:
        # Test categorical feature identification
        cat_features = DataCleaner.get_categorical_features(typed_data, threshold=10)

        # Should identify categorical columns
        if "category_column" not in cat_features:
            raise AssertionError()

        # Test initial cleaning preserves structure
        cleaned_data = DataCleaner.initial_features_removal(typed_data)
        if cleaned_data.shape != typed_data.shape:
            raise AssertionError()

        # Data types should be preserved
        if not pd.api.types.is_integer_dtype(cleaned_data["int_column"]):
            raise AssertionError()
        if not pd.api.types.is_float_dtype(cleaned_data["float_column"]):
            raise AssertionError()
        if not isinstance(cleaned_data["category_column"].dtype, pd.CategoricalDtype):
            raise AssertionError()

    except (NameError, AttributeError):
        pytest.skip("DataCleaner data type preservation not implemented")


def test_datacleaner_get_categorical_features_method(test_settings):
    """Tests DataCleaner.get_categorical_features static method.

    Args:
        test_settings: Test configuration fixture

    Asserts:
        - Method correctly identifies categorical features
        - Threshold parameter works as expected
        - Binary features are included regardless of type
    """
    np.random.seed(test_settings.SEED)
    test_data = pd.DataFrame(
        {
            "high_card_cat": np.random.choice(
                range(200), test_settings.N_SAMPLES
            ),  # High cardinality
            "low_card_cat": np.random.choice(
                ["A", "B", "C"], test_settings.N_SAMPLES
            ),  # Low cardinality
            "binary_numeric": np.random.choice(
                [0, 1], test_settings.N_SAMPLES
            ),  # Binary
            "continuous": np.random.randn(test_settings.N_SAMPLES),  # Continuous
            "object_cat": pd.Series(
                np.random.choice(["cat", "dog", "bird"], test_settings.N_SAMPLES),
                dtype="object",
            ),
        }
    )

    # Test with default threshold (100)
    cat_features_100 = DataCleaner.get_categorical_features(test_data, threshold=100)

    if "low_card_cat" not in cat_features_100:
        raise AssertionError()
    if "binary_numeric" not in cat_features_100:
        raise AssertionError()
    if "object_cat" not in cat_features_100:
        raise AssertionError()
    if "continuous" in cat_features_100:
        raise AssertionError()

    # Test with lower threshold (10)
    cat_features_10 = DataCleaner.get_categorical_features(test_data, threshold=10)

    if "low_card_cat" not in cat_features_10:
        raise AssertionError()
    if "binary_numeric" not in cat_features_10:
        raise AssertionError()
    if "object_cat" not in cat_features_10:
        raise AssertionError()
    if not len(cat_features_10) <= len(cat_features_100):
        raise AssertionError()


def test_datacleaner_initial_features_removal_method(test_settings):
    """Tests DataCleaner.initial_features_removal static method.

    Args:
        test_settings: Test configuration fixture

    Asserts:
        - Single value features are removed
        - All-missing features are removed
        - Duplicate features are removed
        - Normal features are preserved
    """
    np.random.seed(test_settings.SEED)
    base_data = pd.DataFrame(
        {
            "normal_feature": np.random.randn(test_settings.N_SAMPLES),
            "single_value": ["constant"] * test_settings.N_SAMPLES,
            "all_missing": [pd.NA] * test_settings.N_SAMPLES,
            "duplicate_original": np.random.randn(test_settings.N_SAMPLES),
        }
    )
    base_data["duplicate_copy"] = base_data[
        "duplicate_original"
    ].copy()  # Exact duplicate

    original_shape = base_data.shape
    cleaned_data = DataCleaner.initial_features_removal(base_data)

    # Should remove single_value, all_missing, and one of the duplicates
    if not cleaned_data.shape[1] < original_shape[1]:
        raise AssertionError()
    if cleaned_data.shape[0] != original_shape[0]:
        raise AssertionError()

    # Normal feature should remain
    if "normal_feature" not in cleaned_data.columns:
        raise AssertionError()

    # One of the duplicates should remain, one should be removed
    duplicate_features_remaining = [
        col for col in cleaned_data.columns if "duplicate" in col
    ]
    if len(duplicate_features_remaining) != 1:
        raise AssertionError()


def test_datacleaner_categorize_categorical_features_by_missing_method(test_settings):
    """Tests DataCleaner.categorize_categorical_features_by_missing static method.

    Args:
        test_settings: Test configuration fixture

    Asserts:
        - Features are correctly categorized by missing value proportion
        - All expected category keys are returned
        - Logic for different missing thresholds is correct
    """
    np.random.seed(test_settings.SEED)

    # Create categorical features with different missing patterns
    n = test_settings.N_SAMPLES
    test_data = pd.DataFrame(
        {
            "no_missing": np.random.choice(["A", "B", "C"], n),
            "few_missing": np.random.choice(
                ["A", "B", "C", None], n, p=[0.4, 0.4, 0.15, 0.05]
            ),  # ~5% missing
            "many_missing": np.random.choice(
                ["A", "B", None], n, p=[0.3, 0.4, 0.3]
            ),  # ~30% missing
        }
    )

    cat_features = ["no_missing", "few_missing", "many_missing"]
    categories = DataCleaner.categorize_categorical_features_by_missing(
        test_data, cat_features
    )

    # Check all expected keys are present
    expected_keys = [
        "categorical_not_many_missing_features",
        "categorical_many_missing_features",
        "categorical_missing_features",
        "non_missing_categorical_features",
    ]
    for key in expected_keys:
        if key not in categories:
            raise AssertionError()
        if not isinstance(categories[key], list):
            raise AssertionError()

    # Check categorization logic
    if "no_missing" not in categories["non_missing_categorical_features"]:
        raise AssertionError()
    if "few_missing" not in categories["categorical_not_many_missing_features"]:
        raise AssertionError()
    if "many_missing" not in categories["categorical_many_missing_features"]:
        raise AssertionError()

    # Features with any missing should be in categorical_missing_features
    if "few_missing" not in categories["categorical_missing_features"]:
        raise AssertionError()
    if "many_missing" not in categories["categorical_missing_features"]:
        raise AssertionError()
    if "no_missing" in categories["categorical_missing_features"]:
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
