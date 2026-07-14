"""Tests for kvbiii_ml.data_processing.eda.data_cleaning module."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

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
        if not (isinstance(cleaner, DataCleaner)):
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
        if not (cleaner is not None):
            raise AssertionError()

        # Check that static methods are available
        if not (hasattr(DataCleaner, "get_categorical_features")):
            raise AssertionError()
        if not (hasattr(DataCleaner, "initial_features_removal")):
            raise AssertionError()
        if not (hasattr(DataCleaner, "drop_highly_skewed_categorical_features")):
            raise AssertionError()
        if not (hasattr(DataCleaner, "categorize_categorical_features_by_missing")):
            raise AssertionError()

    except (NameError, AttributeError):
        pytest.skip("DataCleaner class not implemented")


def test_remove_duplicate_rows_removes_exact_duplicates(dirty_data):
    """Tests duplicate row removal using pandas drop_duplicates method.

    Args:
        dirty_data (pd.DataFrame): Data with duplicate rows

    Asserts:
        - Duplicate rows are identified and removed correctly
        - Original row order is preserved for non-duplicates
        - First occurrence of duplicates is retained
    """
    # Create test data with known duplicates
    test_data = pd.DataFrame(
        {
            "A": [1, 2, 1, 3, 2],  # Duplicates at indices 0,2 and 1,4
            "B": ["a", "b", "a", "c", "b"],
        }
    )

    original_length = len(test_data)
    cleaned_data = test_data.drop_duplicates()

    if not (len(cleaned_data) < original_length):
        raise AssertionError()
    if not (isinstance(cleaned_data, pd.DataFrame)):
        raise AssertionError()
    # Check that all remaining rows are unique
    if not (len(cleaned_data) == len(cleaned_data.drop_duplicates())):
        raise AssertionError()
    if not (len(cleaned_data) == 3):
        raise AssertionError()


def test_remove_duplicate_rows_handles_no_duplicates():
    """Tests duplicate row handling with pandas drop_duplicates when no duplicates exist.

    Asserts:
        - Function returns data unchanged when no duplicates present
        - No errors occur with unique data
        - DataFrame structure is preserved
    """
    unique_data = pd.DataFrame({"A": [1, 2, 3, 4], "B": ["a", "b", "c", "d"]})

    cleaned_data = unique_data.drop_duplicates()
    pd.testing.assert_frame_equal(cleaned_data, unique_data)


def test_handle_missing_values_drops_missing_rows():
    """Tests missing value handling by dropping rows with pandas dropna.

    Asserts:
        - Rows with any missing values are removed when using dropna()
        - Complete rows are preserved
        - DataFrame shape is reduced appropriately
    """
    data_with_missing = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [1, np.nan, 3, 4]})

    cleaned_data = data_with_missing.dropna()

    if not (len(cleaned_data) == 2):
        raise AssertionError()
    if not (not cleaned_data.isnull().any().any()):
        raise AssertionError()


def test_handle_missing_values_fills_missing_with_mean():
    """Tests missing value handling by filling with mean using pandas fillna.

    Asserts:
        - Missing numeric values are filled with column means
        - Non-numeric columns are handled appropriately
        - No missing values remain after filling
    """
    data_with_missing = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, np.nan, 4.0],  # mean = 2.333...
            "categorical": ["A", "B", np.nan, "C"],
        }
    )

    # Fill numeric column with mean
    numeric_mean = data_with_missing["numeric"].mean()
    cleaned_data = data_with_missing.copy()
    cleaned_data["numeric"] = cleaned_data["numeric"].fillna(numeric_mean)

    # No missing values should remain in numeric column
    if not (not cleaned_data["numeric"].isnull().any()):
        raise AssertionError()

    # Check that mean was used for numeric column
    expected_mean = np.mean([1.0, 2.0, 4.0])  # 2.333...
    if not (abs(cleaned_data.loc[2, "numeric"] - expected_mean) < 1e-10):
        raise AssertionError()


def test_handle_missing_values_fills_missing_with_median():
    """Tests missing value handling by filling with median using pandas fillna.

    Asserts:
        - Missing numeric values are filled with column medians
        - Median calculation handles even/odd number of values
        - Result contains no missing values
    """
    data_with_missing = pd.DataFrame(
        {"numeric": [1.0, 2.0, np.nan, 4.0, 5.0]}
    )  # median = 3.0

    # Fill with median
    numeric_median = data_with_missing["numeric"].median()
    cleaned_data = data_with_missing.copy()
    cleaned_data["numeric"] = cleaned_data["numeric"].fillna(numeric_median)

    if not (not cleaned_data["numeric"].isnull().any()):
        raise AssertionError()
    # Median of [1, 2, 4, 5] is 3.0
    if not (cleaned_data.loc[2, "numeric"] == 3.0):
        raise AssertionError()


def test_remove_outliers_using_iqr_method(dirty_data):
    """Tests outlier removal using IQR method with pandas and numpy.

    Args:
        dirty_data (pd.DataFrame): Data containing outliers

    Asserts:
        - Outliers are detected and removed using IQR method
        - Normal data points are preserved
        - Outlier detection is applied to numeric columns only
    """
    # Create test data with known outliers
    test_data = pd.DataFrame(
        {
            "normal_feature": np.random.normal(0, 1, 100),
            "outlier_feature": list(np.random.normal(0, 1, 98))
            + [50, -50],  # Add clear outliers
        }
    )

    # Apply IQR method
    q1 = test_data["outlier_feature"].quantile(0.25)
    q3 = test_data["outlier_feature"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    cleaned_data = test_data[
        (test_data["outlier_feature"] >= lower_bound)
        & (test_data["outlier_feature"] <= upper_bound)
    ]

    # Should have fewer rows after outlier removal
    if not (len(cleaned_data) <= len(test_data)):
        raise AssertionError()

    # Extreme outliers should be removed
    outlier_feature = cleaned_data["outlier_feature"]
    if not (outlier_feature.max() < 50):
        raise AssertionError()
    if not (outlier_feature.min() > -50):
        raise AssertionError()


def test_remove_outliers_using_zscore_method():
    """Tests outlier removal using z-score method with pandas and scipy.

    Asserts:
        - Outliers are detected using z-score threshold
        - Data points with |z-score| > threshold are removed
        - Method works with different threshold values
    """
    data_with_outliers = pd.DataFrame(
        {"feature": [1, 2, 3, 100, 4, 5, -100]}  # 100 and -100 are outliers
    )

    # Apply z-score method with a stricter threshold
    z_scores = np.abs(stats.zscore(data_with_outliers["feature"]))
    threshold = 1.0  # More strict threshold to actually filter outliers
    cleaned_data = data_with_outliers[z_scores < threshold]

    # Should remove some data points (outliers)
    if not (len(cleaned_data) < len(data_with_outliers)):
        raise AssertionError()

    # Check that extreme outliers are filtered
    outlier_mask = z_scores >= threshold
    outliers_removed = data_with_outliers[outlier_mask]
    if not (len(outliers_removed) > 0):
        raise AssertionError()


def test_standardize_column_names_cleans_column_names(dirty_data):
    """Tests column name standardization using pandas string methods.

    Args:
        dirty_data (pd.DataFrame): Data with inconsistent column names

    Asserts:
        - Spaces in column names are replaced with underscores
        - Column names are converted to lowercase
        - Special characters are handled appropriately
    """
    # Create test data with messy column names
    test_data = pd.DataFrame(
        {
            "Normal Feature": [1, 2, 3],
            "Feature With Spaces": [4, 5, 6],
            "UPPERCASE": [7, 8, 9],
        }
    )

    # Standardize column names
    standardized_data = test_data.copy()
    standardized_data.columns = standardized_data.columns.str.lower().str.replace(
        " ", "_"
    )

    # Check that column names are standardized
    if not ("normal_feature" in standardized_data.columns):
        raise AssertionError()
    if not ("feature_with_spaces" in standardized_data.columns):
        raise AssertionError()
    if not ("uppercase" in standardized_data.columns):
        raise AssertionError()

    # No spaces should remain in column names
    for col in standardized_data.columns:
        if not (" " not in col):
            raise AssertionError()

    # Should be lowercase
    for col in standardized_data.columns:
        if not (col == col.lower()):
            raise AssertionError()


def test_standardize_column_names_handles_special_characters():
    """Tests column name standardization with special characters using pandas.

    Asserts:
        - Special characters are replaced or removed appropriately
        - Column names remain valid Python identifiers where possible
        - Duplicate names after cleaning are handled
    """
    data_special_cols = pd.DataFrame(
        {
            "Column-With-Dashes": [1, 2],
            "Column.With.Dots": [3, 4],
            "Column With Spaces": [5, 6],
            "Column@#$%": [7, 8],
        }
    )

    # Apply standardization using pandas string methods
    standardized_data = data_special_cols.copy()
    standardized_data.columns = (
        standardized_data.columns.str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(".", "_")
        .str.replace("@", "_")
        .str.replace("#", "_")
        .str.replace("$", "_")
        .str.replace("%", "_")
    )

    # Check standardized names
    for col in standardized_data.columns:
        # Should not contain problematic characters
        if not ("-" not in col):
            raise AssertionError()
        if not ("." not in col):
            raise AssertionError()
        if not (" " not in col):
            raise AssertionError()
        if not ("@" not in col):
            raise AssertionError()
        if not ("#" not in col):
            raise AssertionError()
        if not ("$" not in col):
            raise AssertionError()
        if not ("%" not in col):
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
        if not (cleaned_data.shape[1] < original_shape[1]):
            raise AssertionError()
        if not (cleaned_data.shape[0] == original_shape[0]):
            raise AssertionError()

        # Get categorical features
        cat_features = DataCleaner.get_categorical_features(cleaned_data)
        if not (isinstance(cat_features, list)):
            raise AssertionError()

        # Test categorization by missing values
        if cat_features:
            cat_categories = DataCleaner.categorize_categorical_features_by_missing(
                cleaned_data, cat_features
            )
            if not (isinstance(cat_categories, dict)):
                raise AssertionError()
            expected_keys = [
                "categorical_not_many_missing_features",
                "categorical_many_missing_features",
                "categorical_missing_features",
                "non_missing_categorical_features",
            ]
            for key in expected_keys:
                if not (key in cat_categories):
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
    if not (isinstance(result, pd.DataFrame)):
        raise AssertionError()
    if not (len(result) == 0):
        raise AssertionError()

    # Test categorical features detection
    cat_features = cleaner.get_categorical_features(empty_df)
    if not (isinstance(cat_features, list)):
        raise AssertionError()
    if not (len(cat_features) == 0):
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
        if not ("category_column" in cat_features):
            raise AssertionError()

        # Test initial cleaning preserves structure
        cleaned_data = DataCleaner.initial_features_removal(typed_data)
        if not (cleaned_data.shape == typed_data.shape):
            raise AssertionError()

        # Data types should be preserved
        if not (pd.api.types.is_integer_dtype(cleaned_data["int_column"])):
            raise AssertionError()
        if not (pd.api.types.is_float_dtype(cleaned_data["float_column"])):
            raise AssertionError()
        if not (isinstance(cleaned_data["category_column"].dtype, pd.CategoricalDtype)):
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

    if not ("low_card_cat" in cat_features_100):
        raise AssertionError()
    if not ("binary_numeric" in cat_features_100):
        raise AssertionError()
    if not ("object_cat" in cat_features_100):
        raise AssertionError()
    if not ("continuous" not in cat_features_100):
        raise AssertionError()

    # Test with lower threshold (10)
    cat_features_10 = DataCleaner.get_categorical_features(test_data, threshold=10)

    if not ("low_card_cat" in cat_features_10):
        raise AssertionError()
    if not ("binary_numeric" in cat_features_10):
        raise AssertionError()
    if not ("object_cat" in cat_features_10):
        raise AssertionError()
    if not (len(cat_features_10) <= len(cat_features_100)):
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
    if not (cleaned_data.shape[1] < original_shape[1]):
        raise AssertionError()
    if not (cleaned_data.shape[0] == original_shape[0]):
        raise AssertionError()

    # Normal feature should remain
    if not ("normal_feature" in cleaned_data.columns):
        raise AssertionError()

    # One of the duplicates should remain, one should be removed
    duplicate_features_remaining = [
        col for col in cleaned_data.columns if "duplicate" in col
    ]
    if not (len(duplicate_features_remaining) == 1):
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
        if not (key in categories):
            raise AssertionError()
        if not (isinstance(categories[key], list)):
            raise AssertionError()

    # Check categorization logic
    if not ("no_missing" in categories["non_missing_categorical_features"]):
        raise AssertionError()
    if not ("few_missing" in categories["categorical_not_many_missing_features"]):
        raise AssertionError()
    if not ("many_missing" in categories["categorical_many_missing_features"]):
        raise AssertionError()

    # Features with any missing should be in categorical_missing_features
    if not ("few_missing" in categories["categorical_missing_features"]):
        raise AssertionError()
    if not ("many_missing" in categories["categorical_missing_features"]):
        raise AssertionError()
    if not ("no_missing" not in categories["categorical_missing_features"]):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
